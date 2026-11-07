// V5: builds a "voxel-style" evaluation set with transferred artist seams.
// Each hand-unwrapped artist mesh is voxelised, re-surfaced with Surface Nets, smoothed, and the
// artist's UV charts are transferred to the new surface by closest point. Chart borders become
// the reference seams. Output: OBJ files whose texture coordinates encode the charts only
// ("labels-only": one vt per (vertex, chart) pair), so tools/bench.mjs can score seams.
//
//   node tools/voxelize.mjs [--res 32,48,64] [--out data/voxel] [--filter REGEX] [--smooth 6]
import { mkdirSync, readdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { readGLB } from '../training/glb.mjs';
import { readOBJ } from './load.mjs';
import { buildMesh } from '../src/core/mesh.js';
import { uvSeamsAndCharts } from '../src/core/metrics.js';

const argv = process.argv.slice(2);
const opt = (n, d) => { const i = argv.indexOf('--' + n); return i >= 0 ? argv[i + 1] : d; };
const RES = opt('res', '32,48,64').split(',').map(Number);
const OUT = opt('out', 'data/voxel');
const FILTER = new RegExp(opt('filter', '.'));
const SMOOTH = +opt('smooth', 6);
const SRC = opt('data', 'training/models');
const MIN_FACES_PER_CHART = 60, MAX_SRC_FACES = 60000;

// ------------------------------------------------------------------ geometry helpers
/** Closest point on triangle (Ericson, Real-Time Collision Detection 5.1.5); returns squared distance. */
function pointTriDist2(px, py, pz, P, a, b, c) {
  const ax = P[3 * a], ay = P[3 * a + 1], az = P[3 * a + 2];
  const abx = P[3 * b] - ax, aby = P[3 * b + 1] - ay, abz = P[3 * b + 2] - az;
  const acx = P[3 * c] - ax, acy = P[3 * c + 1] - ay, acz = P[3 * c + 2] - az;
  const apx = px - ax, apy = py - ay, apz = pz - az;
  const d1 = abx * apx + aby * apy + abz * apz, d2 = acx * apx + acy * apy + acz * apz;
  let qx, qy, qz;
  const set = (u, v) => { qx = ax + abx * u + acx * v; qy = ay + aby * u + acy * v; qz = az + abz * u + acz * v; };
  if (d1 <= 0 && d2 <= 0) set(0, 0);
  else {
    const bpx = px - P[3 * b], bpy = py - P[3 * b + 1], bpz = pz - P[3 * b + 2];
    const d3 = abx * bpx + aby * bpy + abz * bpz, d4 = acx * bpx + acy * bpy + acz * bpz;
    const cpx = px - P[3 * c], cpy = py - P[3 * c + 1], cpz = pz - P[3 * c + 2];
    const d5 = abx * cpx + aby * cpy + abz * cpz, d6 = acx * cpx + acy * cpy + acz * cpz;
    const vc = d1 * d4 - d3 * d2, vb = d5 * d2 - d1 * d6, va = d3 * d6 - d5 * d4;
    if (d3 >= 0 && d4 <= d3) set(1, 0);
    else if (d6 >= 0 && d5 <= d6) set(0, 1);
    else if (vc <= 0 && d1 >= 0 && d3 <= 0) set(d1 / (d1 - d3), 0);
    else if (vb <= 0 && d2 >= 0 && d6 <= 0) set(0, d2 / (d2 - d6));
    else if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) { const w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); set(1 - w, w); }
    else { const den = 1 / (va + vb + vc); set(vb * den, vc * den); }
  }
  return (px - qx) ** 2 + (py - qy) ** 2 + (pz - qz) ** 2;
}

/** Uniform-grid closest-triangle queries. */
class TriGrid {
  constructor(P, F, cell) {
    this.P = P; this.F = F; this.cell = cell;
    const n = F.length / 3;
    this.min = [Infinity, Infinity, Infinity];
    for (let i = 0; i < P.length; i += 3) for (let k = 0; k < 3; k++) this.min[k] = Math.min(this.min[k], P[i + k]);
    this.map = new Map();
    for (let t = 0; t < n; t++) {
      const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) { const v = P[3 * F[3 * t + j] + k]; lo[k] = Math.min(lo[k], v); hi[k] = Math.max(hi[k], v); }
      const a = lo.map((v, k) => Math.floor((v - this.min[k]) / cell)), b = hi.map((v, k) => Math.floor((v - this.min[k]) / cell));
      for (let x = a[0]; x <= b[0]; x++) for (let y = a[1]; y <= b[1]; y++) for (let z = a[2]; z <= b[2]; z++) {
        const key = (x * 4096 + y) * 4096 + z;
        let l = this.map.get(key); if (!l) this.map.set(key, l = []); l.push(t);
      }
    }
  }
  closest(px, py, pz) {
    const c = [px, py, pz].map((v, k) => Math.floor((v - this.min[k]) / this.cell));
    let best = -1, bestD = Infinity;
    const seen = new Set();
    for (let r = 0; r < 64; r++) {
      if (best >= 0 && (r - 1) * this.cell > Math.sqrt(bestD)) break;
      for (let x = c[0] - r; x <= c[0] + r; x++) for (let y = c[1] - r; y <= c[1] + r; y++) for (let z = c[2] - r; z <= c[2] + r; z++) {
        if (Math.max(Math.abs(x - c[0]), Math.abs(y - c[1]), Math.abs(z - c[2])) !== r) continue;
        const l = this.map.get((x * 4096 + y) * 4096 + z); if (!l) continue;
        for (const t of l) {
          if (seen.has(t)) continue; seen.add(t);
          const d = pointTriDist2(px, py, pz, this.P, this.F[3 * t], this.F[3 * t + 1], this.F[3 * t + 2]);
          if (d < bestD) { bestD = d; best = t; }
        }
      }
    }
    return { tri: best, dist: Math.sqrt(bestD) };
  }
}

// ------------------------------------------------------------------ voxelisation + Surface Nets
function voxelSurface(mesh, N) {
  const { pos, faces, nF } = mesh;
  const lo = mesh.bboxMin, hi = mesh.bboxMax;
  const L = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
  const h = L / N, pad = 3;
  const dim = [0, 1, 2].map(k => Math.ceil((hi[k] - lo[k]) / h) + 2 * pad + 1);
  const org = [0, 1, 2].map(k => lo[k] - pad * h);
  const [nx, ny, nz] = dim;
  const id = (i, j, k) => (k * ny + j) * nx + i;
  const near = new Uint8Array(nx * ny * nz);
  const r = 0.9 * h, r2 = r * r;
  // 1) voxels whose centre lies within r of the surface
  for (let t = 0; t < nF; t++) {
    const a = faces[3 * t], b = faces[3 * t + 1], c = faces[3 * t + 2];
    const i0 = [0, 1, 2].map(k => Math.max(0, Math.floor((Math.min(pos[3 * a + k], pos[3 * b + k], pos[3 * c + k]) - r - org[k]) / h)));
    const i1 = [0, 1, 2].map(k => Math.min(dim[k] - 1, Math.ceil((Math.max(pos[3 * a + k], pos[3 * b + k], pos[3 * c + k]) + r - org[k]) / h)));
    for (let k = i0[2]; k <= i1[2]; k++) for (let j = i0[1]; j <= i1[1]; j++) for (let i = i0[0]; i <= i1[0]; i++) {
      const v = id(i, j, k); if (near[v]) continue;
      if (pointTriDist2(org[0] + i * h, org[1] + j * h, org[2] + k * h, pos, a, b, c) <= r2) near[v] = 1;
    }
  }
  // 2) flood the outside from the border; whatever the flood cannot reach is inside
  const outside = new Uint8Array(nx * ny * nz);
  const stack = [id(0, 0, 0)]; outside[stack[0]] = 1;
  while (stack.length) {
    const v = stack.pop();
    const i = v % nx, j = Math.floor(v / nx) % ny, k = Math.floor(v / (nx * ny));
    for (const [di, dj, dk] of [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]) {
      const a = i + di, b = j + dj, c = k + dk;
      if (a < 0 || b < 0 || c < 0 || a >= nx || b >= ny || c >= nz) continue;
      const w = id(a, b, c);
      if (outside[w] || near[w]) continue;
      outside[w] = 1; stack.push(w);
    }
  }
  // 3) occupancy, lightly blurred into a scalar field (iso = 0.5)
  const occ = new Float32Array(nx * ny * nz);
  for (let v = 0; v < occ.length; v++) occ[v] = outside[v] ? 0 : 1;
  const field = new Float32Array(occ.length);
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    let s = 0, n = 0;
    for (let dk = -1; dk <= 1; dk++) for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) {
      const a = i + di, b = j + dj, c = k + dk;
      if (a < 0 || b < 0 || c < 0 || a >= nx || b >= ny || c >= nz) continue;
      const w = (di === 0) + (dj === 0) + (dk === 0) + 1; // centre-weighted
      s += w * occ[id(a, b, c)]; n += w;
    }
    field[id(i, j, k)] = s / n;
  }
  // 4) Surface Nets: one vertex per cell with a sign change, one quad per crossing grid edge
  const iso = 0.5, inside = v => field[v] >= iso;
  const cellVert = new Int32Array((nx - 1) * (ny - 1) * (nz - 1)).fill(-1);
  const cid = (i, j, k) => (k * (ny - 1) + j) * (nx - 1) + i;
  const V = [];
  const corners = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]];
  const edges = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
  for (let k = 0; k < nz - 1; k++) for (let j = 0; j < ny - 1; j++) for (let i = 0; i < nx - 1; i++) {
    const vals = corners.map(([a, b, c]) => field[id(i + a, j + b, k + c)]);
    let any = 0, all = 1;
    for (const x of vals) { if (x >= iso) any = 1; else all = 0; }
    if (!any || all) continue;
    let sx = 0, sy = 0, sz = 0, n = 0;
    for (const [p, q] of edges) {
      if ((vals[p] >= iso) === (vals[q] >= iso)) continue;
      const t = (iso - vals[p]) / (vals[q] - vals[p]);
      const A = corners[p], B = corners[q];
      sx += A[0] + t * (B[0] - A[0]); sy += A[1] + t * (B[1] - A[1]); sz += A[2] + t * (B[2] - A[2]); n++;
    }
    cellVert[cid(i, j, k)] = V.length / 3;
    V.push(org[0] + (i + sx / n) * h, org[1] + (j + sy / n) * h, org[2] + (k + sz / n) * h);
  }
  const T = [];
  const quad = (q, flip) => {
    if (q.some(x => x < 0)) return;
    const [a, b, c, d] = flip ? [q[3], q[2], q[1], q[0]] : q;
    const P = (x, y) => (V[3 * x] - V[3 * y]) ** 2 + (V[3 * x + 1] - V[3 * y + 1]) ** 2 + (V[3 * x + 2] - V[3 * y + 2]) ** 2;
    if (P(a, c) <= P(b, d)) T.push(a, b, c, a, c, d); else T.push(a, b, d, b, c, d);
  };
  for (let k = 1; k < nz - 1; k++) for (let j = 1; j < ny - 1; j++) for (let i = 1; i < nx - 1; i++) {
    const p = id(i, j, k), ins = inside(p);
    // x edge: cells around it in (y, z)
    if (i < nx - 1 && ins !== inside(id(i + 1, j, k))) quad([cid(i, j - 1, k - 1), cid(i, j, k - 1), cid(i, j, k), cid(i, j - 1, k)].map(c => cellVert[c]), !ins);
    // y edge: cells around it in (z, x)
    if (j < ny - 1 && ins !== inside(id(i, j + 1, k))) quad([cid(i - 1, j, k - 1), cid(i - 1, j, k), cid(i, j, k), cid(i, j, k - 1)].map(c => cellVert[c]), !ins);
    // z edge: cells around it in (x, y)
    if (k < nz - 1 && ins !== inside(id(i, j, k + 1))) quad([cid(i - 1, j - 1, k), cid(i, j - 1, k), cid(i, j, k), cid(i - 1, j, k)].map(c => cellVert[c]), !ins);
  }
  const P = Float64Array.from(V), F = Int32Array.from(T);
  // make the surface face outwards (positive enclosed volume)
  let vol = 0;
  for (let t = 0; t < F.length; t += 3) {
    const a = 3 * F[t], b = 3 * F[t + 1], c = 3 * F[t + 2];
    vol += P[a] * (P[b + 1] * P[c + 2] - P[b + 2] * P[c + 1]) - P[a + 1] * (P[b] * P[c + 2] - P[b + 2] * P[c]) + P[a + 2] * (P[b] * P[c + 1] - P[b + 1] * P[c]);
  }
  if (vol < 0) for (let t = 0; t < F.length; t += 3) { const x = F[t + 1]; F[t + 1] = F[t + 2]; F[t + 2] = x; }
  return { P, F, h };
}

/** Taubin λ/μ smoothing (removes voxel staircase without shrinking). */
function taubin(P, F, iters) {
  const n = P.length / 3, nb = Array.from({ length: n }, () => new Set());
  for (let t = 0; t < F.length; t += 3) for (let k = 0; k < 3; k++) { const a = F[t + k], b = F[t + (k + 1) % 3]; nb[a].add(b); nb[b].add(a); }
  const step = lam => {
    const Q = Float64Array.from(P);
    for (let v = 0; v < n; v++) {
      if (!nb[v].size) continue;
      let x = 0, y = 0, z = 0; for (const w of nb[v]) { x += P[3 * w]; y += P[3 * w + 1]; z += P[3 * w + 2]; }
      const s = nb[v].size;
      Q[3 * v] += lam * (x / s - P[3 * v]); Q[3 * v + 1] += lam * (y / s - P[3 * v + 1]); Q[3 * v + 2] += lam * (z / s - P[3 * v + 2]);
    }
    P.set(Q);
  };
  for (let i = 0; i < iters; i++) { step(0.5); step(-0.53); }
}

// ------------------------------------------------------------------ main
mkdirSync(OUT, { recursive: true });
const manifest = [];
for (const file of readdirSync(SRC).filter(f => /\.(glb|obj)$/i.test(f) && FILTER.test(f)).sort()) {
  const model = file.replace(/\.(glb|obj)$/i, '');
  // GLB: artist meshes with real UVs; OBJ: labels-only meshes (e.g. tools/gen-garments.mjs)
  let meshes; try { meshes = /\.obj$/i.test(file) ? readOBJ(join(SRC, file)) : readGLB(join(SRC, file)); } catch (e) { console.log('skip', model, e.message); continue; }
  meshes.forEach((m, mi) => {
    const src = buildMesh(m.positions, m.indices, m.cornerUV);
    if (src.nF < 60 || src.nF > MAX_SRC_FACES) return;
    const ref = uvSeamsAndCharts(src, src.cornerUV);
    if (ref.charts < 2 || src.nF / ref.charts < MIN_FACES_PER_CHART) return; // only hand-made layouts
    const grid = new TriGrid(src.pos, src.faces, src.diag / 40);
    for (const N of RES) {
      const t0 = Date.now();
      const { P, F, h } = voxelSurface(src, N);
      taubin(P, F, SMOOTH);
      // transfer charts: each new face takes the chart of the closest source face to its centroid
      const nf = F.length / 3, faceChart = new Int32Array(nf);
      let far = 0;
      for (let t = 0; t < nf; t++) {
        const a = 3 * F[3 * t], b = 3 * F[3 * t + 1], c = 3 * F[3 * t + 2];
        const q = grid.closest((P[a] + P[b] + P[c]) / 3, (P[a + 1] + P[b + 1] + P[c + 1]) / 3, (P[a + 2] + P[b + 2] + P[c + 2]) / 3);
        faceChart[t] = ref.faceChart[q.tri];
        if (q.dist > 2 * h) far++;
      }
      // OBJ with labels-only texture coordinates: vt per (vertex, chart)
      const meshTag = /\.obj$/i.test(file) ? String(mi) : `${mi}_${(m.name || 'mesh').replace(/[^\w-]+/g, '_')}`;
      const name = `${model}__${meshTag}__v${N}`; // index keeps names unique
      const lines = ['# seamline: labels-only', `# source ${model}/${m.name} voxels ${N} smooth ${SMOOTH} charts ${ref.charts}`];
      for (let v = 0; v < P.length; v += 3) lines.push(`v ${+P[v].toFixed(6)} ${+P[v + 1].toFixed(6)} ${+P[v + 2].toFixed(6)}`);
      const vt = new Map(); const vtl = [];
      const vtOf = (v, ch) => { const k = v * 65536 + ch; let i = vt.get(k); if (i === undefined) { i = vt.size + 1; vt.set(k, i); vtl.push(`vt ${ch} ${(v % 1000) / 1000}`); } return i; };
      const fl = [];
      for (let t = 0; t < nf; t++) {
        const ch = faceChart[t];
        fl.push('f ' + [0, 1, 2].map(k => `${F[3 * t + k] + 1}/${vtOf(F[3 * t + k], ch)}`).join(' '));
      }
      writeFileSync(join(OUT, name + '.obj'), [...lines, ...vtl, ...fl].join('\n') + '\n');
      const out = buildMesh(Float32Array.from(P), Uint32Array.from(F));
      const info = { file: name + '.obj', source: `${model}/${m.name}`, voxels: N, faces: out.nF, nonManifold: out.nonManifold.reduce((s, x) => s + x, 0), boundary: out.boundary.reduce((s, x) => s + x, 0), farFaces: far, ms: Date.now() - t0 };
      manifest.push(info);
      console.log(`${name.padEnd(56)} faces ${String(info.faces).padStart(6)} nonManifold ${info.nonManifold} boundary ${info.boundary} far ${far} ${info.ms}ms`);
    }
  });
}
writeFileSync(join(OUT, 'manifest.json'), JSON.stringify({ res: RES, smooth: SMOOTH, items: manifest }, null, 1));
console.log(`wrote ${manifest.length} meshes to ${OUT}`);
