// Mesh topology: vertex welding, edges, face/edge adjacency, ground-truth seams from UVs.
// Pure JS (no three.js) so it runs in the browser, a Web Worker and Node.

const QBITS = 131072; // quantisation levels per axis (17 bits -> 51-bit keys)

/**
 * @param {Float32Array|number[]} posIn  xyz per input vertex
 * @param {Uint32Array|number[]} idxIn   3 indices per triangle
 * @param {Float32Array|null} cornerUVIn optional uv per *input triangle corner* (6 per tri)
 */
export function buildMesh(posIn, idxIn, cornerUVIn = null) {
  const nIn = posIn.length / 3;
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < nIn; i++) for (let k = 0; k < 3; k++) {
    const v = posIn[3 * i + k];
    if (v < min[k]) min[k] = v;
    if (v > max[k]) max[k] = v;
  }
  const ext = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2]) || 1;
  const diag = Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]) || 1;

  // --- weld by quantised position
  const remap = new Int32Array(nIn);
  const keyMap = new Map();
  const posArr = [];
  const s = (QBITS - 1) / ext;
  for (let i = 0; i < nIn; i++) {
    const qx = Math.round((posIn[3 * i] - min[0]) * s);
    const qy = Math.round((posIn[3 * i + 1] - min[1]) * s);
    const qz = Math.round((posIn[3 * i + 2] - min[2]) * s);
    const key = (qx * QBITS + qy) * QBITS + qz;
    let id = keyMap.get(key);
    if (id === undefined) {
      id = posArr.length / 3;
      keyMap.set(key, id);
      posArr.push(posIn[3 * i], posIn[3 * i + 1], posIn[3 * i + 2]);
    }
    remap[i] = id;
  }
  const nV = posArr.length / 3;
  const pos = Float64Array.from(posArr);

  // --- faces (drop degenerate)
  const nTriIn = idxIn.length / 3;
  const facesArr = [];
  const faceSrc = [];
  const minArea = 1e-14 * diag * diag;
  for (let t = 0; t < nTriIn; t++) {
    const a = remap[idxIn[3 * t]], b = remap[idxIn[3 * t + 1]], c = remap[idxIn[3 * t + 2]];
    if (a === b || b === c || a === c) continue;
    const ux = pos[3 * b] - pos[3 * a], uy = pos[3 * b + 1] - pos[3 * a + 1], uz = pos[3 * b + 2] - pos[3 * a + 2];
    const vx = pos[3 * c] - pos[3 * a], vy = pos[3 * c + 1] - pos[3 * a + 1], vz = pos[3 * c + 2] - pos[3 * a + 2];
    const cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
    if (0.5 * Math.hypot(cx, cy, cz) < minArea) continue;
    facesArr.push(a, b, c);
    faceSrc.push(t);
  }
  const faces = Int32Array.from(facesArr);
  const nF = faces.length / 3;

  // --- face normals / areas
  const fn = new Float64Array(3 * nF), farea = new Float64Array(nF);
  for (let f = 0; f < nF; f++) {
    const a = faces[3 * f], b = faces[3 * f + 1], c = faces[3 * f + 2];
    const ux = pos[3 * b] - pos[3 * a], uy = pos[3 * b + 1] - pos[3 * a + 1], uz = pos[3 * b + 2] - pos[3 * a + 2];
    const vx = pos[3 * c] - pos[3 * a], vy = pos[3 * c + 1] - pos[3 * a + 1], vz = pos[3 * c + 2] - pos[3 * a + 2];
    let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const l = Math.hypot(nx, ny, nz) || 1;
    farea[f] = 0.5 * l;
    fn[3 * f] = nx / l; fn[3 * f + 1] = ny / l; fn[3 * f + 2] = nz / l;
  }

  // --- edges
  const edgeMap = new Map();
  const ev = [];
  const faceEdges = new Int32Array(3 * nF);
  const efList = []; // flat pairs, -1 when missing
  const nonManifoldList = [];
  for (let f = 0; f < nF; f++) {
    for (let k = 0; k < 3; k++) {
      const a = faces[3 * f + k], b = faces[3 * f + (k + 1) % 3];
      const lo = a < b ? a : b, hi = a < b ? b : a;
      const key = lo * nV + hi;
      let e = edgeMap.get(key);
      if (e === undefined) {
        e = ev.length / 2;
        edgeMap.set(key, e);
        ev.push(lo, hi);
        efList.push(f, -1);
        nonManifoldList.push(0);
      } else if (efList[2 * e + 1] === -1) {
        efList[2 * e + 1] = f;
      } else {
        nonManifoldList[e] = 1;
      }
      faceEdges[3 * f + k] = e;
    }
  }
  const nE = ev.length / 2;
  const edgeV = Int32Array.from(ev);
  const edgeF = Int32Array.from(efList);
  const nonManifold = Uint8Array.from(nonManifoldList);
  const boundary = new Uint8Array(nE);
  const edgeLen = new Float64Array(nE);
  let lenSum = 0;
  for (let e = 0; e < nE; e++) {
    if (edgeF[2 * e + 1] < 0) boundary[e] = 1;
    const a = edgeV[2 * e], b = edgeV[2 * e + 1];
    edgeLen[e] = Math.hypot(pos[3 * a] - pos[3 * b], pos[3 * a + 1] - pos[3 * b + 1], pos[3 * a + 2] - pos[3 * b + 2]);
    lenSum += edgeLen[e];
  }

  // --- vertex -> edge adjacency (CSR)
  const vDeg = new Int32Array(nV + 1);
  for (let e = 0; e < nE; e++) { vDeg[edgeV[2 * e] + 1]++; vDeg[edgeV[2 * e + 1] + 1]++; }
  for (let i = 0; i < nV; i++) vDeg[i + 1] += vDeg[i];
  const vEdgeStart = vDeg;
  const vEdges = new Int32Array(2 * nE);
  const fill = vEdgeStart.slice(0, nV);
  for (let e = 0; e < nE; e++) {
    vEdges[fill[edgeV[2 * e]]++] = e;
    vEdges[fill[edgeV[2 * e + 1]]++] = e;
  }

  // --- corner UVs carried over (after degenerate removal)
  let cornerUV = null;
  if (cornerUVIn) {
    cornerUV = new Float32Array(6 * nF);
    for (let f = 0; f < nF; f++) for (let j = 0; j < 6; j++) cornerUV[6 * f + j] = cornerUVIn[6 * faceSrc[f] + j];
  }

  return {
    nV, nF, nE, pos, faces, faceSrc: Int32Array.from(faceSrc), fn, farea,
    edgeV, edgeF, faceEdges, nonManifold, boundary, edgeLen,
    meanEdgeLen: lenSum / Math.max(1, nE), vEdgeStart, vEdges,
    bboxMin: min, bboxMax: max, diag, cornerUV,
  };
}

/** Index (0..2) of vertex v inside face f, or -1. */
export function cornerOf(mesh, f, v) {
  const F = mesh.faces;
  if (F[3 * f] === v) return 0;
  if (F[3 * f + 1] === v) return 1;
  if (F[3 * f + 2] === v) return 2;
  return -1;
}

/**
 * Ground-truth seams: an interior edge is a seam if the UVs of its two faces disagree
 * at either endpoint. Boundary / non-manifold edges get label 255 (ignore).
 */
export function seamsFromUV(mesh, eps = 1e-4) {
  const { nE, edgeV, edgeF, cornerUV, boundary, nonManifold } = mesh;
  const lab = new Uint8Array(nE);
  if (!cornerUV) return null;
  for (let e = 0; e < nE; e++) {
    if (boundary[e] || nonManifold[e]) { lab[e] = 255; continue; }
    const f0 = edgeF[2 * e], f1 = edgeF[2 * e + 1];
    let seam = 0;
    for (let s = 0; s < 2 && !seam; s++) {
      const v = edgeV[2 * e + s];
      const c0 = cornerOf(mesh, f0, v), c1 = cornerOf(mesh, f1, v);
      const du = cornerUV[6 * f0 + 2 * c0] - cornerUV[6 * f1 + 2 * c1];
      const dv = cornerUV[6 * f0 + 2 * c0 + 1] - cornerUV[6 * f1 + 2 * c1 + 1];
      if (Math.abs(du) > eps || Math.abs(dv) > eps || !isFinite(du) || !isFinite(dv)) seam = 1;
    }
    lab[e] = seam;
  }
  return lab;
}

/** Simple union-find on typed arrays. */
export class UnionFind {
  constructor(n) { this.p = new Int32Array(n); for (let i = 0; i < n; i++) this.p[i] = i; }
  find(x) { const p = this.p; while (p[x] !== x) { p[x] = p[p[x]]; x = p[x]; } return x; }
  union(a, b) { a = this.find(a); b = this.find(b); if (a !== b) this.p[a] = b; }
}
