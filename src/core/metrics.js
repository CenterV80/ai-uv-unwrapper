// UV quality metrics for research evaluation. Pure JS; works on any per-corner UV layout
// (ours, xatlas, artist UVs), so different methods are measured the same way.
import { UnionFind, cornerOf } from './mesh.js';

/**
 * Coverage raster: counts how many triangles cover each pixel centre (strict inside test, so
 * triangles that only share an edge are not counted twice).
 * @param {Float64Array|Float32Array} tri  6 numbers (u0 v0 u1 v1 u2 v2) per triangle
 * @param {number} res   raster resolution, @param box [u0, v0, size] region to rasterise
 * @returns {{covered: number, overlapped: number, overlapTris: Set<number>}}
 */
export function rasterOverlap(tri, res, box = [0, 0, 1], wantTris = false) {
  const [bu, bv, bs] = box;
  const n = tri.length / 6;
  const cnt = new Uint8Array(res * res);
  const owner = wantTris ? new Int32Array(res * res).fill(-1) : null;
  const overlapTris = new Set();
  const sc = res / bs;
  for (let t = 0; t < n; t++) {
    const x0 = (tri[6 * t] - bu) * sc, y0 = (tri[6 * t + 1] - bv) * sc;
    const x1 = (tri[6 * t + 2] - bu) * sc, y1 = (tri[6 * t + 3] - bv) * sc;
    const x2 = (tri[6 * t + 4] - bu) * sc, y2 = (tri[6 * t + 5] - bv) * sc;
    const area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
    if (!(Math.abs(area) > 1e-12)) continue;
    const s = area > 0 ? 1 : -1;
    const i0 = Math.max(0, Math.ceil(Math.min(x0, x1, x2) - 0.5)), i1 = Math.min(res - 1, Math.floor(Math.max(x0, x1, x2) - 0.5));
    const j0 = Math.max(0, Math.ceil(Math.min(y0, y1, y2) - 0.5)), j1 = Math.min(res - 1, Math.floor(Math.max(y0, y1, y2) - 0.5));
    for (let j = j0; j <= j1; j++) {
      const py = j + 0.5;
      for (let i = i0; i <= i1; i++) {
        const px = i + 0.5;
        const w0 = s * ((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py));
        const w1 = s * ((x2 - px) * (y0 - py) - (x0 - px) * (y2 - py));
        const w2 = s * ((x0 - px) * (y1 - py) - (x1 - px) * (y0 - py));
        if (w0 <= 0 || w1 <= 0 || w2 <= 0) continue;
        const k = j * res + i;
        if (cnt[k] < 255) cnt[k]++;
        if (owner) { if (cnt[k] === 1) owner[k] = t; else { overlapTris.add(t); overlapTris.add(owner[k]); } }
      }
    }
  }
  let covered = 0, overlapped = 0;
  for (let k = 0; k < cnt.length; k++) { if (cnt[k]) covered++; if (cnt[k] > 1) overlapped++; }
  return { covered, overlapped, overlapTris };
}

/**
 * Consistent 3D orientation: +1/-1 per face so that neighbours across manifold edges agree
 * (input meshes often mix windings; a valid UV map must not be blamed for that).
 */
export function faceOrientation(mesh) {
  const { nF, faces, edgeV, edgeF, faceEdges, nonManifold } = mesh;
  const orient = new Int8Array(nF);
  const dir = (f, a, b) => { for (let k = 0; k < 3; k++) if (faces[3 * f + k] === a) return faces[3 * f + (k + 1) % 3] === b ? 1 : -1; return 0; };
  for (let s = 0; s < nF; s++) {
    if (orient[s]) continue;
    orient[s] = 1;
    const q = [s];
    while (q.length) {
      const f = q.pop();
      for (let k = 0; k < 3; k++) {
        const e = faceEdges[3 * f + k];
        if (nonManifold[e]) continue;
        const g = edgeF[2 * e] === f ? edgeF[2 * e + 1] : edgeF[2 * e];
        if (g < 0 || orient[g]) continue;
        const a = edgeV[2 * e], b = edgeV[2 * e + 1];
        // consistent neighbours traverse the shared edge in opposite directions
        orient[g] = dir(f, a, b) === dir(g, a, b) ? -orient[f] : orient[f];
        q.push(g);
      }
    }
  }
  return orient;
}

/** UV islands: faces joined across interior edges whose UVs agree at both endpoints. */
export function uvSeamsAndCharts(mesh, cornerUV, eps = 1e-6) {
  const { nE, nF, edgeV, edgeF, boundary } = mesh;
  const seam = new Uint8Array(nE);
  const uf = new UnionFind(nF);
  for (let e = 0; e < nE; e++) {
    const f0 = edgeF[2 * e], f1 = edgeF[2 * e + 1];
    if (boundary[e] || f1 < 0 || mesh.nonManifold[e]) { seam[e] = 1; continue; }
    let split = false;
    for (let s = 0; s < 2 && !split; s++) {
      const v = edgeV[2 * e + s];
      const c0 = cornerOf(mesh, f0, v), c1 = cornerOf(mesh, f1, v);
      if (Math.abs(cornerUV[6 * f0 + 2 * c0] - cornerUV[6 * f1 + 2 * c1]) > eps ||
          Math.abs(cornerUV[6 * f0 + 2 * c0 + 1] - cornerUV[6 * f1 + 2 * c1 + 1]) > eps) split = true;
    }
    if (split) seam[e] = 1; else uf.union(f0, f1);
  }
  const ids = new Map(), faceChart = new Int32Array(nF);
  for (let f = 0; f < nF; f++) { const r = uf.find(f); if (!ids.has(r)) ids.set(r, ids.size); faceChart[f] = ids.get(r); }
  return { seam, faceChart, charts: ids.size };
}

function weightedPercentile(values, weights, q) {
  const idx = Array.from(values.keys()).sort((a, b) => values[a] - values[b]);
  const total = weights.reduce((s, w) => s + w, 0);
  let acc = 0;
  for (const i of idx) { acc += weights[i]; if (acc >= q * total) return values[i]; }
  return values[idx[idx.length - 1]] ?? NaN;
}

/**
 * Full metric set for one UV layout.
 * - areaDist: per face max(r, 1/r), r = UV/3D area ratio after global scale normalisation (1 = ideal)
 * - angleDist: per face σ1/σ2 of the 3D→UV Jacobian (1 = conformal)
 * - flips: faces whose orientation disagrees with the majority of their chart
 * - overlapRatio: share of covered UV area covered more than once (raster at `overlapRes`)
 */
export function computeMetrics(mesh, cornerUV, { overlapRes = 1024 } = {}) {
  const { nF, faces, pos, edgeLen, boundary } = mesh;
  const { seam, faceChart, charts } = uvSeamsAndCharts(mesh, cornerUV);
  let A3 = 0, A2 = 0, oob = 0, nonFinite = 0;
  const a2s = new Float64Array(nF), a3s = new Float64Array(nF);
  const J = new Float64Array(4 * nF);
  for (let f = 0; f < nF; f++) {
    const ia = faces[3 * f], ib = faces[3 * f + 1], ic = faces[3 * f + 2];
    const ex = [pos[3 * ib] - pos[3 * ia], pos[3 * ib + 1] - pos[3 * ia + 1], pos[3 * ib + 2] - pos[3 * ia + 2]];
    const ey = [pos[3 * ic] - pos[3 * ia], pos[3 * ic + 1] - pos[3 * ia + 1], pos[3 * ic + 2] - pos[3 * ia + 2]];
    const l1 = Math.hypot(...ex) || 1e-30;
    const x1 = l1, x2 = (ex[0] * ey[0] + ex[1] * ey[1] + ex[2] * ey[2]) / l1;
    const cr = [ex[1] * ey[2] - ex[2] * ey[1], ex[2] * ey[0] - ex[0] * ey[2], ex[0] * ey[1] - ex[1] * ey[0]];
    const y2 = Math.hypot(...cr) / l1 || 1e-30;
    a3s[f] = 0.5 * x1 * y2;
    const u = [cornerUV[6 * f], cornerUV[6 * f + 2], cornerUV[6 * f + 4]], v = [cornerUV[6 * f + 1], cornerUV[6 * f + 3], cornerUV[6 * f + 5]];
    for (let k = 0; k < 3; k++) {
      if (!isFinite(u[k]) || !isFinite(v[k])) nonFinite++;
      if (u[k] < -1e-6 || u[k] > 1 + 1e-6 || v[k] < -1e-6 || v[k] > 1 + 1e-6) oob++;
    }
    a2s[f] = 0.5 * ((u[1] - u[0]) * (v[2] - v[0]) - (u[2] - u[0]) * (v[1] - v[0]));
    // Jacobian of (x,y) -> (u,v): columns d/dx, d/dy
    const du1 = u[1] - u[0], dv1 = v[1] - v[0], du2 = u[2] - u[0], dv2 = v[2] - v[0];
    const ux = du1 / x1, vx = dv1 / x1;
    J[4 * f] = ux; J[4 * f + 1] = (du2 - ux * x2) / y2; J[4 * f + 2] = vx; J[4 * f + 3] = (dv2 - vx * x2) / y2;
    A3 += a3s[f]; A2 += Math.abs(a2s[f]);
  }
  // orientation per chart = sign of its summed signed area, after making 3D windings consistent
  const orient = faceOrientation(mesh);
  for (let f = 0; f < nF; f++) a2s[f] *= orient[f];
  const chartSign = new Float64Array(charts);
  for (let f = 0; f < nF; f++) chartSign[faceChart[f]] += a2s[f];
  const scale = A2 > 0 ? Math.sqrt(A3 / A2) : 1;
  const areaD = [], angleD = [], wts = [];
  let flips = 0, flipArea = 0;
  for (let f = 0; f < nF; f++) {
    const sgn = chartSign[faceChart[f]] >= 0 ? 1 : -1;
    if (sgn * a2s[f] <= 0) { flips++; flipArea += a3s[f]; continue; }
    const a = J[4 * f] * scale, b = J[4 * f + 1] * scale, c = J[4 * f + 2] * scale, d = J[4 * f + 3] * scale;
    // singular values of [[a b][c d]]
    const s1sq = a * a + b * b + c * c + d * d, det = Math.abs(a * d - b * c);
    const disc = Math.sqrt(Math.max(0, s1sq * s1sq / 4 - det * det));
    const sMax = Math.sqrt(s1sq / 2 + disc), sMin = Math.sqrt(Math.max(1e-30, s1sq / 2 - disc));
    const r = sMax * sMin;
    areaD.push(Math.max(r, 1 / r)); angleD.push(sMax / sMin); wts.push(a3s[f]);
  }
  const wmean = arr => { let s = 0, w = 0; for (let i = 0; i < arr.length; i++) { s += arr[i] * wts[i]; w += wts[i]; } return w ? s / w : NaN; };
  // seams: interior edges that are UV-discontinuous
  let seamLen = 0, seamEdges = 0;
  for (let e = 0; e < mesh.nE; e++) if (seam[e] && !boundary[e]) { seamLen += edgeLen[e]; seamEdges++; }
  // overlap over the unit square (out-of-range parts are clipped)
  const tri = new Float64Array(6 * nF);
  for (let i = 0; i < 6 * nF; i++) tri[i] = cornerUV[i];
  const ov = rasterOverlap(tri, overlapRes);
  const overlapRatio = ov.covered ? ov.overlapped / ov.covered : 0;
  const m = {
    faces: nF, charts, seamEdges,
    seamLengthNorm: seamLen / Math.sqrt(A3 || 1),
    areaDistMean: wmean(areaD), areaDistP95: weightedPercentile(areaD, wts, 0.95),
    angleDistMean: wmean(angleD), angleDistP95: weightedPercentile(angleD, wts, 0.95),
    flips, flipAreaRatio: flipArea / (A3 || 1),
    overlapRatio, outOfBounds: oob, nonFinite,
    utilization: A2, // UV area inside the unit square (UVs are normalised to [0,1])
  };
  // valid: no out-of-range UVs, flipped area ≤ 0.01% (single slivers in the input), overlap ≤ 0.1%
  m.valid = nonFinite === 0 && oob === 0 && m.flipAreaRatio <= 1e-4 && overlapRatio <= 1e-3;
  return m;
}

/**
 * Seam agreement with reference (artist) seams. labels: 0/1, 255 = ignore (boundary).
 * Exact P/R/F1, plus a tolerant version: an edge counts as matched if both endpoints lie within
 * `tolFrac` x bounding-box diagonal (geodesic, along mesh edges) of the other seam set.
 */
export function seamAgreement(mesh, predSeam, labels, tolFrac = 0.01, withClasses = false) {
  const { nE, nV, edgeV, edgeLen, vEdgeStart, vEdges, diag } = mesh;
  const tol = tolFrac * diag;
  const P = [], G = [];
  for (let e = 0; e < nE; e++) {
    if (labels[e] === 255) continue;
    if (predSeam[e]) P.push(e);
    if (labels[e] === 1) G.push(e);
  }
  const distFrom = edges => { // multi-source Dijkstra over vertices, cut off at tol
    const d = new Float64Array(nV).fill(Infinity);
    const heap = [];
    const push = (k, v) => { heap.push([k, v]); let i = heap.length - 1; while (i > 0) { const p = (i - 1) >> 1; if (heap[p][0] <= heap[i][0]) break; [heap[p], heap[i]] = [heap[i], heap[p]]; i = p; } };
    const pop = () => { const top = heap[0], last = heap.pop(); if (heap.length) { heap[0] = last; let i = 0; for (;;) { let c = 2 * i + 1; if (c >= heap.length) break; if (c + 1 < heap.length && heap[c + 1][0] < heap[c][0]) c++; if (heap[c][0] >= heap[i][0]) break; [heap[c], heap[i]] = [heap[i], heap[c]]; i = c; } } return top; };
    for (const e of edges) for (let s = 0; s < 2; s++) { const v = edgeV[2 * e + s]; if (d[v] > 0) { d[v] = 0; push(0, v); } }
    while (heap.length) {
      const [dv, v] = pop();
      if (dv > d[v] || dv > tol) continue;
      for (let i = vEdgeStart[v]; i < vEdgeStart[v + 1]; i++) {
        const e = vEdges[i], w = edgeV[2 * e] === v ? edgeV[2 * e + 1] : edgeV[2 * e];
        const nd = dv + edgeLen[e];
        if (nd < d[w] && nd <= tol) { d[w] = nd; push(nd, w); }
      }
    }
    return d;
  };
  const isG = new Uint8Array(nE); for (const e of G) isG[e] = 1;
  const tp = P.filter(e => isG[e]).length;
  const prec = P.length ? tp / P.length : (G.length ? 0 : 1), rec = G.length ? tp / G.length : 1;
  const near = (d, e) => d[edgeV[2 * e]] <= tol && d[edgeV[2 * e + 1]] <= tol;
  const dG = distFrom(G), dP = distFrom(P);
  const precT = P.length ? P.filter(e => near(dG, e)).length / P.length : (G.length ? 0 : 1);
  const recT = G.length ? G.filter(e => near(dP, e)).length / G.length : 1;
  const f1 = (p, r) => (p + r) ? 2 * p * r / (p + r) : 0;
  const out = { seamPrecision: prec, seamRecall: rec, seamF1: f1(prec, rec), seamPrecisionTol: precT, seamRecallTol: recT, seamF1Tol: f1(precT, recT), predSeams: P.length, refSeams: G.length };
  if (withClasses) {
    // per edge: 1 = predicted seam near a reference seam (match), 2 = predicted, no reference nearby (extra),
    //           3 = reference seam with no predicted seam nearby (missed), 0 = neither
    const cls = new Uint8Array(nE);
    for (const e of P) cls[e] = near(dG, e) ? 1 : 2;
    for (const e of G) if (!near(dP, e) && !cls[e]) cls[e] = 3;
    let lenP = 0, lenG = 0, lenMatch = 0, lenExtra = 0, lenMissed = 0;
    for (const e of P) { lenP += edgeLen[e]; if (cls[e] === 1) lenMatch += edgeLen[e]; else lenExtra += edgeLen[e]; }
    for (const e of G) { lenG += edgeLen[e]; if (!near(dP, e)) lenMissed += edgeLen[e]; }
    out.classes = cls;
    out.lengths = { pred: lenP, ref: lenG, match: lenMatch, extra: lenExtra, missed: lenMissed };
  }
  return out;
}
