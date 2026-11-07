// Minimal GLB reader for training: positions, indices and TEXCOORD_0 per glTF mesh,
// baked with the world transform of the first node that instances it.
import { readFileSync } from 'node:fs';

const COMP = { 5120: [Int8Array, 1], 5121: [Uint8Array, 1], 5122: [Int16Array, 2], 5123: [Uint16Array, 2], 5125: [Uint32Array, 4], 5126: [Float32Array, 4] };
const NCOMP = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };

function readAccessor(gltf, bin, idx) {
  const acc = gltf.accessors[idx];
  const view = gltf.bufferViews[acc.bufferView];
  const [Arr, size] = COMP[acc.componentType];
  const nc = NCOMP[acc.type];
  const stride = view.byteStride || size * nc;
  const base = (view.byteOffset || 0) + (acc.byteOffset || 0);
  const out = new Float64Array(acc.count * nc);
  const dv = new DataView(bin.buffer, bin.byteOffset, bin.byteLength);
  const norm = acc.normalized;
  for (let i = 0; i < acc.count; i++) for (let c = 0; c < nc; c++) {
    const o = base + i * stride + c * size;
    let v;
    switch (acc.componentType) {
      case 5126: v = dv.getFloat32(o, true); break;
      case 5125: v = dv.getUint32(o, true); break;
      case 5123: v = dv.getUint16(o, true); if (norm) v /= 65535; break;
      case 5122: v = dv.getInt16(o, true); if (norm) v = Math.max(v / 32767, -1); break;
      case 5121: v = dv.getUint8(o); if (norm) v /= 255; break;
      case 5120: v = dv.getInt8(o); if (norm) v = Math.max(v / 127, -1); break;
    }
    out[i * nc + c] = v;
  }
  void Arr;
  return out;
}

function mat4FromNode(n) {
  if (n.matrix) return n.matrix.slice();
  const [tx, ty, tz] = n.translation || [0, 0, 0];
  const [qx, qy, qz, qw] = n.rotation || [0, 0, 0, 1];
  const [sx, sy, sz] = n.scale || [1, 1, 1];
  const xx = qx * qx, yy = qy * qy, zz = qz * qz, xy = qx * qy, xz = qx * qz, yz = qy * qz, wx = qw * qx, wy = qw * qy, wz = qw * qz;
  return [
    (1 - 2 * (yy + zz)) * sx, 2 * (xy + wz) * sx, 2 * (xz - wy) * sx, 0,
    2 * (xy - wz) * sy, (1 - 2 * (xx + zz)) * sy, 2 * (yz + wx) * sy, 0,
    2 * (xz + wy) * sz, 2 * (yz - wx) * sz, (1 - 2 * (xx + yy)) * sz, 0,
    tx, ty, tz, 1,
  ];
}
function mul(a, b) { // column-major a*b
  const o = new Array(16).fill(0);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) for (let k = 0; k < 4; k++) o[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
  return o;
}

export function readGLB(path) {
  const buf = readFileSync(path);
  if (buf.readUInt32LE(0) !== 0x46546C67) throw new Error('not a GLB');
  let off = 12, gltf = null, bin = null;
  while (off < buf.length) {
    const len = buf.readUInt32LE(off), type = buf.readUInt32LE(off + 4);
    const chunk = buf.subarray(off + 8, off + 8 + len);
    if (type === 0x4E4F534A) gltf = JSON.parse(chunk.toString('utf8'));
    else if (type === 0x004E4942) bin = chunk;
    off += 8 + len;
  }
  const req = gltf.extensionsRequired || [];
  if (req.some(e => /draco|meshopt|quantiz/i.test(e))) throw new Error('compressed: ' + req.join(','));
  // world matrices, first instance per mesh
  const meshXf = new Map();
  const visit = (ni, parent) => {
    const n = gltf.nodes[ni];
    const m = mul(parent, mat4FromNode(n));
    if (n.mesh !== undefined && !meshXf.has(n.mesh)) meshXf.set(n.mesh, m);
    for (const c of n.children || []) visit(c, m);
  };
  const I = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const scene = gltf.scenes ? gltf.scenes[gltf.scene || 0] : { nodes: gltf.nodes.map((_, i) => i) };
  for (const r of scene.nodes) visit(r, I);

  const out = [];
  gltf.meshes.forEach((mesh, mi) => {
    const M = meshXf.get(mi) || I;
    const P = [], IDX = [], CUV = [];
    let hasUV = true;
    for (const prim of mesh.primitives) {
      if ((prim.mode ?? 4) !== 4 || prim.attributes.POSITION === undefined) continue;
      if (prim.attributes.TEXCOORD_0 === undefined) { hasUV = false; continue; }
      const pos = readAccessor(gltf, bin, prim.attributes.POSITION);
      const uv = readAccessor(gltf, bin, prim.attributes.TEXCOORD_0);
      const n = pos.length / 3;
      const ind = prim.indices !== undefined ? readAccessor(gltf, bin, prim.indices) : Float64Array.from({ length: n }, (_, i) => i);
      const base = P.length / 3;
      for (let i = 0; i < n; i++) {
        const x = pos[3 * i], y = pos[3 * i + 1], z = pos[3 * i + 2];
        P.push(M[0] * x + M[4] * y + M[8] * z + M[12], M[1] * x + M[5] * y + M[9] * z + M[13], M[2] * x + M[6] * y + M[10] * z + M[14]);
      }
      const det = M[0] * (M[5] * M[10] - M[9] * M[6]) - M[4] * (M[1] * M[10] - M[9] * M[2]) + M[8] * (M[1] * M[6] - M[5] * M[2]);
      const order = det < 0 ? [0, 2, 1] : [0, 1, 2];
      for (let t = 0; t + 2 < ind.length; t += 3) for (const k of order) {
        const vi = ind[t + k];
        IDX.push(base + vi);
        CUV.push(uv[2 * vi], uv[2 * vi + 1]);
      }
    }
    if (IDX.length && hasUV) out.push({ name: mesh.name || `mesh${mi}`, positions: Float32Array.from(P), indices: Uint32Array.from(IDX), cornerUV: Float32Array.from(CUV) });
  });
  return out;
}
