// Seam probabilities -> charts -> LSCM -> packing.
import { UnionFind } from './mesh.js';
import { lscm, chartDistortion } from './lscm.js';
import { packCharts } from './pack.js';
import { rasterOverlap } from './metrics.js';

export const DEFAULT_PARAMS = {
  threshold: 0.5,        // seam probability threshold
  maxDistortion: 1.25,   // area-distortion tolerance per chart (1 = perfect)
  minFragment: 3,        // seam fragments shorter than this many edges are dropped
  extendRatio: 0.7,      // dangling seam ends may continue along edges with prob >= threshold * extendRatio
  extendSteps: 0,        // ... for at most this many edges (0 = off; experimental gap closing, not validated)
  seamMode: 'absolute',  // 'absolute': prob >= threshold. 'adaptive': also edges that stand out on this mesh
                         //   (robust z-score >= adaptiveZ, prob >= adaptiveFloor) and are the local maximum across the line
  adaptiveZ: 4,
  adaptiveFloor: 0.3,
  padding: 4,            // texels between charts
  resolution: 1024,      // texture size used to convert padding to UV units
  maxDepth: 14,
  dihedralWeight: 0.2,   // how much geometry (vs AI probability) steers automatic cuts
  rejectOverlap: true,   // split charts whose UVs overlap themselves (LSCM does not prevent this)
  overlapTolerance: 0.0005, // ... when the overlapped share of the chart's UV area exceeds this
                           //     (kept below the 0.1% validity limit used by the metrics)
  flipTolerance: 0.00005,  // flipped share of the chart's 3D area that triggers a repair
  maxSlits: 4,             // local repairs (cut from the problem area to the border) before splitting the chart
};

/** Heap of (key, id) with float keys. */
class MinHeap {
  constructor() { this.k = []; this.v = []; }
  get size() { return this.k.length; }
  push(key, val) {
    const k = this.k, v = this.v; let i = k.length; k.push(key); v.push(val);
    while (i > 0) { const p = (i - 1) >> 1; if (k[p] <= key) break; k[i] = k[p]; v[i] = v[p]; i = p; }
    k[i] = key; v[i] = val;
  }
  pop() {
    const k = this.k, v = this.v; const top = v[0], lastK = k.pop(), lastV = v.pop();
    const n = k.length;
    if (n) {
      let i = 0;
      for (;;) {
        let c = 2 * i + 1; if (c >= n) break;
        if (c + 1 < n && k[c + 1] < k[c]) c++;
        if (k[c] >= lastK) break;
        k[i] = k[c]; v[i] = v[c]; i = c;
      }
      k[i] = lastK; v[i] = lastV;
    }
    return top;
  }
  topKey() { return this.k[0]; }
}

/** Build initial seam set from probabilities and user overrides, then prune tiny fragments. */
export function seamsFromProbabilities(mesh, prob, params, userSeam = null) {
  const { nE, edgeV, boundary, nonManifold, nV, vEdgeStart, vEdges } = mesh;
  const seam = new Uint8Array(nE);
  const adaptive = params.seamMode === 'adaptive' ? adaptiveSeams(mesh, prob, params) : null;
  for (let e = 0; e < nE; e++) {
    let s = prob[e] >= params.threshold || (adaptive && adaptive[e]) ? 1 : 0;
    if (userSeam && userSeam[e]) s = userSeam[e] > 0 ? 1 : 0;
    if (boundary[e] || nonManifold[e]) s = 1;
    seam[e] = s;
  }
  extendDanglingEnds(mesh, prob, seam, params, userSeam);
  // prune small connected seam fragments that are not user-forced and don't touch a boundary
  const visited = new Uint8Array(nE);
  const stack = [];
  for (let e0 = 0; e0 < nE; e0++) {
    if (!seam[e0] || visited[e0] || boundary[e0]) continue;
    const comp = [];
    let forced = false, touchesBoundary = false;
    stack.push(e0); visited[e0] = 1;
    while (stack.length) {
      const e = stack.pop(); comp.push(e);
      if (userSeam && userSeam[e] > 0) forced = true;
      for (let s = 0; s < 2; s++) {
        const v = edgeV[2 * e + s];
        for (let i = vEdgeStart[v]; i < vEdgeStart[v + 1]; i++) {
          const o = vEdges[i];
          if (!seam[o] || visited[o]) continue;
          if (boundary[o] || nonManifold[o]) { touchesBoundary = true; continue; }
          visited[o] = 1; stack.push(o);
        }
      }
    }
    if (!forced && !touchesBoundary && comp.length < params.minFragment) for (const e of comp) seam[e] = 0;
  }
  return seam;
}

/**
 * Threshold-free seam candidates: edges whose probability stands out on *this* mesh
 * (robust z-score against the mesh's own median / MAD) and that are the local maximum across the
 * seam line (non-maximum suppression: no roughly parallel edge at the opposite vertices of the two
 * adjacent faces scores higher). Thin ridges thus give one line of edges instead of a band.
 */
export function adaptiveSeams(mesh, prob, params) {
  const { nE, edgeV, edgeF, faces, pos, boundary, vEdgeStart, vEdges } = mesh;
  const vals = []; for (let e = 0; e < nE; e++) if (!boundary[e]) vals.push(prob[e]);
  vals.sort((a, b) => a - b);
  const med = vals[vals.length >> 1] ?? 0;
  const dev = vals.map(v => Math.abs(v - med)).sort((a, b) => a - b);
  const mad = Math.max(1e-6, (dev[dev.length >> 1] ?? 0) * 1.4826);
  const out = new Uint8Array(nE);
  const dir = e => {
    const a = edgeV[2 * e], b = edgeV[2 * e + 1];
    const x = pos[3 * b] - pos[3 * a], y = pos[3 * b + 1] - pos[3 * a + 1], z = pos[3 * b + 2] - pos[3 * a + 2];
    const l = Math.hypot(x, y, z) || 1; return [x / l, y / l, z / l];
  };
  for (let e = 0; e < nE; e++) {
    if (boundary[e]) continue;
    const p = prob[e];
    if (p < params.adaptiveFloor || (p - med) / mad < params.adaptiveZ) continue;
    const d = dir(e);
    let isMax = true;
    for (let j = 0; j < 2 && isMax; j++) {
      const f = edgeF[2 * e + j]; if (f < 0) continue;
      let opp = faces[3 * f]; if (opp === edgeV[2 * e] || opp === edgeV[2 * e + 1]) opp = faces[3 * f + 1];
      if (opp === edgeV[2 * e] || opp === edgeV[2 * e + 1]) opp = faces[3 * f + 2];
      for (let i = vEdgeStart[opp]; i < vEdgeStart[opp + 1]; i++) {
        const o = vEdges[i]; if (o === e) continue;
        const q = dir(o);
        if (Math.abs(d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) < 0.7) continue; // only edges running alongside
        if (prob[o] > p) { isMax = false; break; }
      }
    }
    if (isMax) out[e] = 1;
  }
  return out;
}

/**
 * Hysteresis-style gap closing: from every dangling seam end, keep walking along the most
 * seam-like straight-ish continuation while its probability stays above a lower threshold.
 * Stops on reaching another seam or the mesh boundary; walks that reach nothing are undone.
 */
function extendDanglingEnds(mesh, prob, seam, params, userSeam) {
  const { nV, edgeV, pos, vEdgeStart, vEdges } = mesh;
  const low = params.threshold * (params.extendRatio ?? 0.7), maxSteps = params.extendSteps ?? 10;
  if (!(maxSteps > 0)) return;
  const deg = new Int32Array(nV);
  for (let e = 0; e < mesh.nE; e++) if (seam[e]) { deg[edgeV[2 * e]]++; deg[edgeV[2 * e + 1]]++; }
  const other = (e, v) => edgeV[2 * e] === v ? edgeV[2 * e + 1] : edgeV[2 * e];
  for (let v0 = 0; v0 < nV; v0++) {
    if (deg[v0] !== 1) continue;
    let v = v0, prevE = -1;
    for (let i = vEdgeStart[v]; i < vEdgeStart[v + 1]; i++) if (seam[vEdges[i]]) prevE = vEdges[i];
    const path = [];
    let reached = false;
    for (let step = 0; step < maxSteps; step++) {
      const u = other(prevE, v);
      const dx = pos[3 * v] - pos[3 * u], dy = pos[3 * v + 1] - pos[3 * u + 1], dz = pos[3 * v + 2] - pos[3 * u + 2];
      const dl = Math.hypot(dx, dy, dz) || 1;
      let best = -1, bestP = low;
      for (let i = vEdgeStart[v]; i < vEdgeStart[v + 1]; i++) {
        const e = vEdges[i];
        if (seam[e] || (userSeam && userSeam[e] < 0)) continue;
        const w = other(e, v);
        const ex = pos[3 * w] - pos[3 * v], ey = pos[3 * w + 1] - pos[3 * v + 1], ez = pos[3 * w + 2] - pos[3 * v + 2];
        const cos = (dx * ex + dy * ey + dz * ez) / (dl * (Math.hypot(ex, ey, ez) || 1));
        if (cos < 0.2) continue; // don't turn back or sideways
        if (prob[e] >= bestP) { bestP = prob[e]; best = e; }
      }
      if (best < 0) break;
      path.push(best); seam[best] = 1;
      const w = other(best, v);
      deg[v]++; deg[w]++;
      if (deg[w] > 1) { reached = true; break; }
      prevE = best; v = w;
    }
    if (!reached) for (const e of path) { seam[e] = 0; deg[edgeV[2 * e]]--; deg[edgeV[2 * e + 1]]--; }
  }
}

/** Faces connected through non-seam edges -> list of face arrays. */
function componentsOf(mesh, faceList, seam, inSet) {
  const { edgeF, faceEdges } = mesh;
  for (const f of faceList) inSet[f] = 1;
  const seen = new Map();
  const comps = [];
  for (const f0 of faceList) {
    if (seen.has(f0)) continue;
    const comp = [f0]; seen.set(f0, 1);
    for (let qi = 0; qi < comp.length; qi++) {
      const f = comp[qi];
      for (let k = 0; k < 3; k++) {
        const e = faceEdges[3 * f + k];
        if (seam[e]) continue;
        const g = edgeF[2 * e] === f ? edgeF[2 * e + 1] : edgeF[2 * e];
        if (g < 0 || !inSet[g] || seen.has(g)) continue;
        seen.set(g, 1); comp.push(g);
      }
    }
    comps.push(comp);
  }
  for (const f of faceList) inSet[f] = 0;
  return comps;
}

/** Local vertices of a chart: corners glued across non-seam edges. */
function buildWedges(mesh, faces, seam, faceLocal) {
  const { edgeV, edgeF, faceEdges, faces: F, pos } = mesh;
  const nf = faces.length;
  for (let i = 0; i < nf; i++) faceLocal[faces[i]] = i;
  const uf = new UnionFind(3 * nf);
  for (let i = 0; i < nf; i++) {
    const f = faces[i];
    for (let k = 0; k < 3; k++) {
      const e = faceEdges[3 * f + k];
      if (seam[e]) continue;
      const g = edgeF[2 * e] === f ? edgeF[2 * e + 1] : edgeF[2 * e];
      if (g < 0 || faceLocal[g] < 0) continue;
      const j = faceLocal[g];
      for (let s = 0; s < 2; s++) {
        const v = edgeV[2 * e + s];
        const ci = F[3 * f] === v ? 0 : F[3 * f + 1] === v ? 1 : 2;
        const cj = F[3 * g] === v ? 0 : F[3 * g + 1] === v ? 1 : 2;
        uf.union(3 * i + ci, 3 * j + cj);
      }
    }
  }
  const rootId = new Map();
  const T = new Int32Array(3 * nf);
  const P = [];
  const vert = [];
  for (let c = 0; c < 3 * nf; c++) {
    const r = uf.find(c);
    let id = rootId.get(r);
    if (id === undefined) {
      id = vert.length; rootId.set(r, id);
      const v = F[3 * faces[Math.floor(c / 3)] + (c % 3)];
      vert.push(v); P.push(pos[3 * v], pos[3 * v + 1], pos[3 * v + 2]);
    }
    T[c] = id;
  }
  // wedge edges & boundary
  const eMap = new Map();
  const nv = vert.length;
  for (let i = 0; i < nf; i++) for (let k = 0; k < 3; k++) {
    const a = T[3 * i + k], b = T[3 * i + (k + 1) % 3];
    const key = a < b ? a * nv + b : b * nv + a;
    eMap.set(key, (eMap.get(key) || 0) + 1);
  }
  const bVerts = new Set();
  for (const [key, cnt] of eMap) if (cnt === 1) { bVerts.add(Math.floor(key / nv)); bVerts.add(key % nv); }
  for (let i = 0; i < nf; i++) faceLocal[faces[i]] = -1;
  const chi = nv - eMap.size + nf;
  const bEdges = [];
  for (const [key, cnt] of eMap) if (cnt === 1) bEdges.push(Math.floor(key / nv), key % nv);
  return { P: Float64Array.from(P), T, vert, chi, boundaryVerts: [...bVerts], bEdges };
}

function farthestPins(P, cand) {
  const d2 = (a, b) => (P[3 * a] - P[3 * b]) ** 2 + (P[3 * a + 1] - P[3 * b + 1]) ** 2 + (P[3 * a + 2] - P[3 * b + 2]) ** 2;
  let a = cand[0], b = a, best = -1;
  for (const c of cand) { const d = d2(a, c); if (d > best) { best = d; b = c; } }
  best = -1; let c2 = b;
  for (const c of cand) { const d = d2(b, c); if (d > best) { best = d; c2 = c; } }
  if (c2 === b) c2 = cand.find(x => x !== b) ?? (b === 0 ? 1 : 0);
  return [b, c2];
}

/** Dijkstra over face centroids inside a chart; returns farthest face from start. */
function farthestFace(mesh, faces, start, faceLocal, centroid) {
  const { edgeF, faceEdges } = mesh;
  for (let i = 0; i < faces.length; i++) faceLocal[faces[i]] = i;
  const dist = new Float64Array(faces.length).fill(Infinity);
  const h = new MinHeap();
  dist[faceLocal[start]] = 0; h.push(0, start);
  let far = start, farD = 0;
  while (h.size) {
    const d = h.topKey(); const f = h.pop();
    const li = faceLocal[f];
    if (d > dist[li]) continue;
    if (d > farD) { farD = d; far = f; }
    for (let k = 0; k < 3; k++) {
      const e = faceEdges[3 * f + k];
      const g = edgeF[2 * e] === f ? edgeF[2 * e + 1] : edgeF[2 * e];
      if (g < 0 || faceLocal[g] < 0) continue;
      const nd = d + Math.hypot(centroid[3 * f] - centroid[3 * g], centroid[3 * f + 1] - centroid[3 * g + 1], centroid[3 * f + 2] - centroid[3 * g + 2]);
      const lj = faceLocal[g];
      if (nd < dist[lj]) { dist[lj] = nd; h.push(nd, g); }
    }
  }
  for (let i = 0; i < faces.length; i++) faceLocal[faces[i]] = -1;
  return far;
}

/**
 * Faces of a chart whose layout is invalid: covered twice in UV (raster test) or flipped.
 * Returns null when the chart is within tolerance, else the local indices of the offending faces.
 */
function chartProblems(P, T, uv, overlapTol, flipTol) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < uv.length; i += 2) { x0 = Math.min(x0, uv[i]); x1 = Math.max(x1, uv[i]); y0 = Math.min(y0, uv[i + 1]); y1 = Math.max(y1, uv[i + 1]); }
  const size = Math.max(x1 - x0, y1 - y0) || 1;
  const nf = T.length / 3;
  const res = Math.max(256, Math.min(1024, Math.round(Math.sqrt(nf) * 16)));
  const tri = new Float64Array(6 * nf);
  for (let f = 0; f < nf; f++) for (let k = 0; k < 3; k++) { tri[6 * f + 2 * k] = uv[2 * T[3 * f + k]]; tri[6 * f + 2 * k + 1] = uv[2 * T[3 * f + k] + 1]; }
  const r = rasterOverlap(tri, res, [x0, y0, size], true);
  const bad = new Set(r.overlapped > Math.max(2, r.covered * overlapTol) ? r.overlapTris : []);
  // flipped faces (the layout has already been mirrored so that most of the area is positive)
  let A3 = 0, flipA = 0; const flipped = [];
  for (let f = 0; f < nf; f++) {
    const i = T[3 * f], j = T[3 * f + 1], k = T[3 * f + 2];
    const a2 = (uv[2 * j] - uv[2 * i]) * (uv[2 * k + 1] - uv[2 * i + 1]) - (uv[2 * k] - uv[2 * i]) * (uv[2 * j + 1] - uv[2 * i + 1]);
    const ux = P[3 * j] - P[3 * i], uy = P[3 * j + 1] - P[3 * i + 1], uz = P[3 * j + 2] - P[3 * i + 2];
    const vx = P[3 * k] - P[3 * i], vy = P[3 * k + 1] - P[3 * i + 1], vz = P[3 * k + 2] - P[3 * i + 2];
    const a3 = Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx);
    A3 += a3;
    if (a2 <= 0) { flipA += a3; flipped.push(f); }
  }
  if (flipA > flipTol * A3) for (const f of flipped) bad.add(f);
  return bad.size ? { faces: [...bad], pairs: bad.size && r.pairs.length && r.overlapped > Math.max(2, r.covered * overlapTol) ? r.pairs : [] } : null;
}

/**
 * Local repair: a cut ("dart") from the middle of the problem area to the chart border along the
 * most seam-like edges. The chart stays one disk-shaped piece, but the fold can open up.
 */
function slitPath(mesh, W, problemFaces, cost, minLen = 3) {
  const nv = W.vert.length, T = W.T, P = W.P;
  const onB = new Uint8Array(nv);
  for (let i = 0; i < W.bEdges.length; i++) onB[W.bEdges[i]] = 1;
  // area-weighted centre of the problem area, then the closest interior vertex of those faces
  let cx = 0, cy = 0, cz = 0, n = 0;
  const cand = new Set();
  for (const f of problemFaces) for (let k = 0; k < 3; k++) { const v = T[3 * f + k]; cx += P[3 * v]; cy += P[3 * v + 1]; cz += P[3 * v + 2]; n++; if (!onB[v]) cand.add(v); }
  if (!cand.size) return null;
  cx /= n; cy /= n; cz /= n;
  let start = -1, bd = Infinity;
  for (const v of cand) { const d = (P[3 * v] - cx) ** 2 + (P[3 * v + 1] - cy) ** 2 + (P[3 * v + 2] - cz) ** 2; if (d < bd) { bd = d; start = v; } }
  const adj = Array.from({ length: nv }, () => []);
  for (let t = 0; t < T.length; t += 3) for (let k = 0; k < 3; k++) { const a = T[t + k], b = T[t + (k + 1) % 3]; adj[a].push(b); adj[b].push(a); }
  const dist = new Float64Array(nv).fill(Infinity), prev = new Int32Array(nv).fill(-1);
  const h = new MinHeap(); dist[start] = 0; h.push(0, start);
  let target = -1;
  while (h.size) {
    const d = h.topKey(), v = h.pop();
    if (d > dist[v]) continue;
    if (onB[v]) { target = v; break; }
    for (const w of adj[v]) {
      const e = meshEdge(mesh, W.vert[v], W.vert[w]);
      if (e < 0) continue;
      const nd = d + (mesh.edgeLen[e] / mesh.meanEdgeLen) * (0.08 + Math.max(0, 1 - cost[e]) ** 2);
      if (nd < dist[w]) { dist[w] = nd; prev[w] = v; h.push(nd, w); }
    }
  }
  if (target < 0) return null;
  const path = [];
  for (let v = target; prev[v] >= 0; v = prev[v]) { const e = meshEdge(mesh, W.vert[v], W.vert[prev[v]]); if (e >= 0) path.push(e); }
  // a dart that starts next to the border cannot open a fold: the problem is the border itself
  return path.length >= minLen ? path : null;
}

/** Mesh edge between mesh vertices a and b (or -1). */
function meshEdge(mesh, a, b) {
  const { vEdgeStart, vEdges, edgeV } = mesh;
  for (let i = vEdgeStart[a]; i < vEdgeStart[a + 1]; i++) {
    const e = vEdges[i];
    if (edgeV[2 * e] === b || edgeV[2 * e + 1] === b) return e;
  }
  return -1;
}

/**
 * A chart with several boundary loops (holes, or seam fragments ending inside the chart) is
 * made disk-like by cutting the cheapest seam-like path from one loop to another.
 * Returns the mesh edges of the path, or null when the chart has a single boundary loop.
 */
function connectBoundaryLoops(mesh, W, cost) {
  const nv = W.vert.length;
  const uf = new UnionFind(nv);
  const onB = new Uint8Array(nv);
  for (let i = 0; i < W.bEdges.length; i += 2) { uf.union(W.bEdges[i], W.bEdges[i + 1]); onB[W.bEdges[i]] = 1; onB[W.bEdges[i + 1]] = 1; }
  const loopOf = new Int32Array(nv).fill(-1);
  const loops = new Map();
  for (let v = 0; v < nv; v++) if (onB[v]) { const r = uf.find(v); if (!loops.has(r)) loops.set(r, loops.size); loopOf[v] = loops.get(r); }
  if (loops.size < 2) return null;
  // the outer loop is the longest one; start from the smallest other loop
  const loopLen = new Float64Array(loops.size);
  for (let i = 0; i < W.bEdges.length; i += 2) {
    const a = W.bEdges[i], b = W.bEdges[i + 1];
    loopLen[loopOf[a]] += Math.hypot(W.P[3 * a] - W.P[3 * b], W.P[3 * a + 1] - W.P[3 * b + 1], W.P[3 * a + 2] - W.P[3 * b + 2]);
  }
  let src = 0;
  for (let l = 1; l < loops.size; l++) if (loopLen[l] < loopLen[src]) src = l;
  // wedge adjacency through interior edges
  const adj = Array.from({ length: nv }, () => []);
  const T = W.T;
  for (let t = 0; t < T.length; t += 3) for (let k = 0; k < 3; k++) {
    const a = T[t + k], b = T[t + (k + 1) % 3];
    adj[a].push(b); adj[b].push(a);
  }
  const dist = new Float64Array(nv).fill(Infinity), prev = new Int32Array(nv).fill(-1);
  const h = new MinHeap();
  for (let v = 0; v < nv; v++) if (loopOf[v] === src) { dist[v] = 0; h.push(0, v); }
  let target = -1;
  while (h.size) {
    const d = h.topKey(), v = h.pop();
    if (d > dist[v]) continue;
    if (loopOf[v] >= 0 && loopOf[v] !== src) { target = v; break; }
    for (const w of adj[v]) {
      const e = meshEdge(mesh, W.vert[v], W.vert[w]);
      if (e < 0) continue;
      const len = mesh.edgeLen[e] / mesh.meanEdgeLen;
      const nd = d + len * (0.08 + Math.max(0, 1 - cost[e]) ** 2);
      if (nd < dist[w]) { dist[w] = nd; prev[w] = v; h.push(nd, w); }
    }
  }
  if (target < 0) return null;
  const path = [];
  for (let v = target; prev[v] >= 0; v = prev[v]) {
    const e = meshEdge(mesh, W.vert[v], W.vert[prev[v]]);
    if (e >= 0) path.push(e);
  }
  return path.length ? path : null;
}

/**
 * Split a chart in two with a watershed flood from two far-apart seeds. Crossing an edge costs
 * its seam likelihood, so the meeting front settles on the most seam-like edges.
 */
function splitChart(mesh, faces, cost, seam, faceLocal, centroid, seeds = null) {
  const { edgeF, faceEdges, diag } = mesh;
  const a = seeds ? seeds[0] : farthestFace(mesh, faces, faces[0], faceLocal, centroid);
  const b = seeds ? seeds[1] : farthestFace(mesh, faces, a, faceLocal, centroid);
  if (a === b) return null;
  for (let i = 0; i < faces.length; i++) faceLocal[faces[i]] = i;
  const label = new Int8Array(faces.length).fill(-1);
  const h = new MinHeap();
  const LEVELS = 48;
  const push = (f, lab, level, dist) => h.push(level * 1e4 + dist / diag, f * 2 + lab);
  const level = new Float64Array(faces.length);
  const dist = new Float64Array(faces.length);
  push(a, 0, 0, 0); push(b, 1, 0, 0);
  const pendingLevel = new Map(), pendingDist = new Map();
  pendingLevel.set(a * 2, 0); pendingDist.set(a * 2, 0);
  pendingLevel.set(b * 2 + 1, 0); pendingDist.set(b * 2 + 1, 0);
  while (h.size) {
    const item = h.pop();
    const f = item >> 1, lab = item & 1;
    const li = faceLocal[f];
    if (label[li] >= 0) continue;
    label[li] = lab;
    const lv = pendingLevel.get(item), ds = pendingDist.get(item);
    level[li] = lv; dist[li] = ds;
    for (let k = 0; k < 3; k++) {
      const e = faceEdges[3 * f + k];
      const g = edgeF[2 * e] === f ? edgeF[2 * e + 1] : edgeF[2 * e];
      if (g < 0 || faceLocal[g] < 0 || label[faceLocal[g]] >= 0) continue;
      const c = seam[e] ? LEVELS : Math.round(cost[e] * LEVELS);
      const nl = Math.max(lv, c);
      const nd = ds + Math.hypot(centroid[3 * f] - centroid[3 * g], centroid[3 * f + 1] - centroid[3 * g + 1], centroid[3 * f + 2] - centroid[3 * g + 2]);
      const key = g * 2 + lab;
      const old = pendingLevel.get(key);
      if (old === undefined || nl < old || (nl === old && nd < pendingDist.get(key))) {
        pendingLevel.set(key, nl); pendingDist.set(key, nd);
        push(g, lab, nl, nd);
      }
    }
  }
  // smooth the cut: a face whose neighbours mostly belong to the other side switches sides,
  // which removes the zig-zag a flood leaves on evenly tessellated surfaces
  for (let pass = 0; pass < 4; pass++) {
    let changed = 0;
    for (let i = 0; i < faces.length; i++) {
      const f = faces[i];
      if (f === a || f === b) continue;
      let other = 0, same = 0;
      for (let k = 0; k < 3; k++) {
        const e = faceEdges[3 * f + k];
        const g = edgeF[2 * e] === f ? edgeF[2 * e + 1] : edgeF[2 * e];
        if (g < 0 || faceLocal[g] < 0 || seam[e]) continue;
        if (label[faceLocal[g]] === label[i]) same++; else if (cost[e] < 0.9) other++;
      }
      if (other >= 2 && same <= 1) { label[i] = 1 - label[i]; changed++; }
    }
    if (!changed) break;
  }
  const newSeams = [];
  for (let i = 0; i < faces.length; i++) {
    const f = faces[i];
    for (let k = 0; k < 3; k++) {
      const e = faceEdges[3 * f + k];
      const g = edgeF[2 * e] === f ? edgeF[2 * e + 1] : edgeF[2 * e];
      if (g < 0 || faceLocal[g] < 0) continue;
      if (label[faceLocal[g]] !== label[i] && !seam[e]) { seam[e] = 1; newSeams.push(e); }
    }
  }
  for (let i = 0; i < faces.length; i++) faceLocal[faces[i]] = -1;
  return newSeams;
}

/**
 * Main entry. Returns per-corner UVs in [0,1], chart id per face, final seams and stats.
 * @param {(stage:string, frac:number)=>void} onProgress
 */
export function unwrap(mesh, prob, dihedral, paramsIn = {}, userSeam = null, onProgress = () => {}) {
  const params = { ...DEFAULT_PARAMS, ...paramsIn };
  const { nF, nE, faces: F, pos } = mesh;
  const seam = seamsFromProbabilities(mesh, prob, params, userSeam);
  const cost = new Float64Array(nE);
  for (let e = 0; e < nE; e++) {
    let c = (1 - params.dihedralWeight) * prob[e] + params.dihedralWeight * Math.min(1, dihedral[e] * 2);
    if (userSeam && userSeam[e] < 0) c = 0; // user said "never cut here": flood crosses it freely
    cost[e] = c;
  }
  const centroid = new Float64Array(3 * nF);
  for (let f = 0; f < nF; f++) for (let k = 0; k < 3; k++) {
    centroid[3 * f + k] = (pos[3 * F[3 * f] + k] + pos[3 * F[3 * f + 1] + k] + pos[3 * F[3 * f + 2] + k]) / 3;
  }
  const faceLocal = new Int32Array(nF).fill(-1);
  const inSet = new Uint8Array(nF);
  const all = Array.from({ length: nF }, (_, i) => i);
  const queue = componentsOf(mesh, all, seam, inSet).map(c => ({ faces: c, depth: 0, slits: 0 }));
  const charts = [];
  let processedFaces = 0;
  let autoCuts = 0, overlapSplits = 0, slitRepairs = 0;

  while (queue.length) {
    const { faces, depth, slits = 0 } = queue.pop();
    const W = buildWedges(mesh, faces, seam, faceLocal);
    let ok = W.chi === 1 && W.boundaryVerts.length >= 2;
    let overlapBad = false, problemFaces = null;
    let res = null, dist = null;
    if (ok) {
      const pins = farthestPins(W.P, W.boundaryVerts);
      res = lscm(W.P, W.T, pins);
      dist = chartDistortion(W.P, W.T, res.uv);
      if (dist.mirrored) for (let i = 0; i < res.uv.length; i += 2) res.uv[i] = -res.uv[i];
      if (!isFinite(dist.distortion)) ok = false;
      else if (dist.distortion > params.maxDistortion && faces.length >= 6 && depth < params.maxDepth) ok = false;
      else if (params.rejectOverlap && faces.length >= 2 && (problemFaces = chartProblems(W.P, W.T, res.uv, params.overlapTolerance, params.flipTolerance))) { ok = false; overlapBad = true; }
    }
    // topology problems (not a disk) are always cut further: every split strictly shrinks the chart,
    // so this terminates; distortion-driven splits are limited by maxDepth
    const topoBad = !(W.chi === 1 && W.boundaryVerts.length >= 2);
    const mustSplit = topoBad || overlapBad; // invalid layouts are always cut further (each split shrinks the chart)
    if (topoBad && W.chi < 1 && W.bEdges.length) {
      const path = connectBoundaryLoops(mesh, W, cost);
      if (path) {
        let added = 0;
        for (const e of path) if (!seam[e]) { seam[e] = 1; added++; }
        if (added) {
          autoCuts += added;
          for (const c of componentsOf(mesh, faces, seam, inSet)) queue.push({ faces: c, depth, slits });
          continue;
        }
      }
    }
    // invalid layout: first try local darts from the problem area to the border, then split
    if (overlapBad && slits < params.maxSlits) {
      const path = slitPath(mesh, W, problemFaces.faces, cost);
      let added = 0;
      if (path) for (const e of path) if (!seam[e]) { seam[e] = 1; added++; }
      if (added) {
        autoCuts += added; slitRepairs++;
        for (const c of componentsOf(mesh, faces, seam, inSet)) queue.push({ faces: c, depth, slits: slits + 1 });
        continue;
      }
    }
    if (overlapBad) overlapSplits++;
    if (!ok && faces.length > 1 && (mustSplit || depth < params.maxDepth)) {
      // NOTE: the flood barrier is the seam *likelihood*: high cost = good place for a seam
      // for an overlap, separate the two pieces of surface that land on each other
      let seeds = null;
      if (overlapBad && problemFaces.pairs.length) {
        let best = -1;
        for (let i = 0; i < problemFaces.pairs.length; i += 2) {
          const fa = faces[problemFaces.pairs[i]], fb = faces[problemFaces.pairs[i + 1]];
          const d = (centroid[3 * fa] - centroid[3 * fb]) ** 2 + (centroid[3 * fa + 1] - centroid[3 * fb + 1]) ** 2 + (centroid[3 * fa + 2] - centroid[3 * fb + 2]) ** 2;
          if (d > best && fa !== fb) { best = d; seeds = [fa, fb]; }
        }
      }
      const cut = splitChart(mesh, faces, cost, seam, faceLocal, centroid, seeds);
      if (cut && cut.length) {
        for (const e of cut) seam[e] = 1;
        autoCuts += cut.length;
        for (const c of componentsOf(mesh, faces, seam, inSet)) queue.push({ faces: c, depth: depth + 1 });
        continue;
      }
    }
    if (!res) {
      // last resort (e.g. a closed piece that could not be cut): planar projection
      const pins = W.boundaryVerts.length >= 2 ? farthestPins(W.P, W.boundaryVerts) : [0, Math.min(1, W.vert.length - 1)];
      res = lscm(W.P, W.T, pins, { maxIter: 0 });
      dist = chartDistortion(W.P, W.T, res.uv);
      if (dist.mirrored) for (let i = 0; i < res.uv.length; i += 2) res.uv[i] = -res.uv[i];
    }
    charts.push({ faces, T: W.T, uv: res.uv, area3d: dist.area3d, area2d: dist.area2d, distortion: dist.distortion, flips: dist.flips });
    processedFaces += faces.length;
    onProgress('unwrap', processedFaces / nF);
  }

  onProgress('pack', 0);
  const packed = packCharts(charts, params);
  const cornerUV = new Float32Array(6 * nF);
  const faceChart = new Int32Array(nF);
  let distSum = 0, areaSum = 0, flips = 0;
  charts.forEach((c, ci) => {
    const uv = packed.uvs[ci];
    for (let i = 0; i < c.faces.length; i++) {
      const f = c.faces[i];
      faceChart[f] = ci;
      for (let k = 0; k < 3; k++) {
        const w = c.T[3 * i + k];
        cornerUV[6 * f + 2 * k] = uv[2 * w];
        cornerUV[6 * f + 2 * k + 1] = uv[2 * w + 1];
      }
    }
    distSum += c.distortion * c.area3d; areaSum += c.area3d; flips += c.flips;
  });
  return {
    cornerUV, faceChart, seam, chartCount: charts.length,
    stats: {
      charts: charts.length,
      seamEdges: seam.reduce((s, x, e) => s + (x && !mesh.boundary[e] ? 1 : 0), 0),
      autoCuts, overlapSplits, slitRepairs,
      distortion: distSum / (areaSum || 1),
      flips,
      packingEfficiency: packed.efficiency,
    },
  };
}
