// Per-edge geometric features for seam prediction. All features are scale invariant.

export const FEATURE_NAMES = [
  'dihedral', 'convex', 'concave', 'logLen', 'logAreaRatio',
  'curvMin', 'curvMax', 'curvSum', 'valence',
  'nbr1', 'nbr2', 'nbr4', 'ridge', 'lineMin', 'lineMax',
  'normalY', 'height', 'symmetryX', 'nearBoundary',
  'creaseS', 'creaseL', 'creaseRidgeS',
];
export const NF = FEATURE_NAMES.length;
export const CONTEXT_NAMES = ['p', 'pNbr1', 'pNbr2', 'pLineMin', 'pLineMax', 'pNbrMax'];
export const NF2 = NF + CONTEXT_NAMES.length;

/** Dihedral angle per edge, normalised to [0,1] (0 = flat, 1 = folded back), with sign array. */
export function dihedrals(mesh) {
  const { nE, edgeV, edgeF, faces, fn, pos, boundary, nonManifold } = mesh;
  const a = new Float64Array(nE), sign = new Int8Array(nE);
  for (let e = 0; e < nE; e++) {
    if (boundary[e]) continue;
    const f0 = edgeF[2 * e], f1 = edgeF[2 * e + 1];
    const va = edgeV[2 * e], vb = edgeV[2 * e + 1];
    // detect inconsistent winding: in an oriented pair the edge runs opposite ways
    const d0 = dirInFace(faces, f0, va, vb), d1 = dirInFace(faces, f1, va, vb);
    const flip = d0 === d1 ? -1 : 1;
    let dot = flip * (fn[3 * f0] * fn[3 * f1] + fn[3 * f0 + 1] * fn[3 * f1 + 1] + fn[3 * f0 + 2] * fn[3 * f1 + 2]);
    dot = Math.max(-1, Math.min(1, dot));
    a[e] = Math.acos(dot) / Math.PI;
    if (nonManifold[e]) a[e] = 1;
    // opposite vertex of f1 relative to plane of f0
    let opp = faces[3 * f1];
    if (opp === va || opp === vb) opp = faces[3 * f1 + 1];
    if (opp === va || opp === vb) opp = faces[3 * f1 + 2];
    const h = fn[3 * f0] * (pos[3 * opp] - pos[3 * va]) + fn[3 * f0 + 1] * (pos[3 * opp + 1] - pos[3 * va + 1]) + fn[3 * f0 + 2] * (pos[3 * opp + 2] - pos[3 * va + 2]);
    sign[e] = h < 0 ? 1 : -1; // 1 convex, -1 concave
  }
  return { a, sign };
}

function dirInFace(faces, f, va, vb) {
  for (let k = 0; k < 3; k++) if (faces[3 * f + k] === va) return faces[3 * f + (k + 1) % 3] === vb ? 1 : -1;
  return 0;
}

/** Mean of a per-edge signal over the 4 face-adjacent edges. */
export function diffuse(mesh, x) {
  const { nE, edgeF, faceEdges } = mesh;
  const out = new Float64Array(nE);
  for (let e = 0; e < nE; e++) {
    let s = 0, c = 0;
    for (let j = 0; j < 2; j++) {
      const f = edgeF[2 * e + j];
      if (f < 0) continue;
      for (let k = 0; k < 3; k++) { const o = faceEdges[3 * f + k]; if (o !== e) { s += x[o]; c++; } }
    }
    out[e] = c ? s / c : x[e];
  }
  return out;
}

/** Max over the 4 face-adjacent edges. */
export function neighbourMax(mesh, x) {
  const { nE, edgeF, faceEdges } = mesh;
  const out = new Float64Array(nE);
  for (let e = 0; e < nE; e++) {
    let m = 0;
    for (let j = 0; j < 2; j++) {
      const f = edgeF[2 * e + j];
      if (f < 0) continue;
      for (let k = 0; k < 3; k++) { const o = faceEdges[3 * f + k]; if (o !== e && x[o] > m) m = x[o]; }
    }
    out[e] = m;
  }
  return out;
}

/**
 * "Line continuation": at each endpoint, the strongest signal on an edge that continues
 * this edge in a straight line. Returns [min over endpoints, max over endpoints].
 */
export function lineContinuation(mesh, x) {
  const { nE, edgeV, pos, vEdgeStart, vEdges } = mesh;
  const lo = new Float64Array(nE), hi = new Float64Array(nE);
  for (let e = 0; e < nE; e++) {
    let best = [0, 0];
    for (let s = 0; s < 2; s++) {
      const v = edgeV[2 * e + s], w = edgeV[2 * e + 1 - s];
      let dx = pos[3 * w] - pos[3 * v], dy = pos[3 * w + 1] - pos[3 * v + 1], dz = pos[3 * w + 2] - pos[3 * v + 2];
      const l = Math.hypot(dx, dy, dz) || 1; dx /= l; dy /= l; dz /= l;
      let m = 0;
      for (let i = vEdgeStart[v]; i < vEdgeStart[v + 1]; i++) {
        const o = vEdges[i];
        if (o === e) continue;
        const u = edgeV[2 * o] === v ? edgeV[2 * o + 1] : edgeV[2 * o];
        let ox = pos[3 * u] - pos[3 * v], oy = pos[3 * u + 1] - pos[3 * v + 1], oz = pos[3 * u + 2] - pos[3 * v + 2];
        const ol = Math.hypot(ox, oy, oz) || 1;
        const straight = Math.max(0, -(dx * ox + dy * oy + dz * oz) / ol);
        const sc = straight * straight * x[o];
        if (sc > m) m = sc;
      }
      best[s] = m;
    }
    lo[e] = Math.min(best[0], best[1]);
    hi[e] = Math.max(best[0], best[1]);
  }
  return [lo, hi];
}

/** Face normals averaged over face neighbourhoods (area weighted), `iters` diffusion steps. */
export function smoothedFaceNormals(mesh, iters) {
  const { nF, edgeF, faceEdges, fn, farea, faces } = mesh;
  let cur = new Float64Array(3 * nF);
  for (let f = 0; f < nF; f++) for (let k = 0; k < 3; k++) cur[3 * f + k] = fn[3 * f + k] * farea[f];
  let next = new Float64Array(3 * nF);
  for (let it = 0; it < iters; it++) {
    for (let f = 0; f < nF; f++) {
      let x = cur[3 * f], y = cur[3 * f + 1], z = cur[3 * f + 2];
      for (let k = 0; k < 3; k++) {
        const e = faceEdges[3 * f + k];
        const g = edgeF[2 * e] === f ? edgeF[2 * e + 1] : edgeF[2 * e];
        if (g < 0) continue;
        // keep orientation consistent with f for inconsistently wound neighbours
        const va = mesh.edgeV[2 * e], vb = mesh.edgeV[2 * e + 1];
        const sg = dirInFace(faces, f, va, vb) === dirInFace(faces, g, va, vb) ? -1 : 1;
        x += sg * cur[3 * g]; y += sg * cur[3 * g + 1]; z += sg * cur[3 * g + 2];
      }
      next[3 * f] = x; next[3 * f + 1] = y; next[3 * f + 2] = z;
    }
    [cur, next] = [next, cur];
    // renormalise magnitudes to avoid overflow while keeping area weighting relative
    let m = 0; for (let i = 0; i < cur.length; i++) m = Math.max(m, Math.abs(cur[i]));
    if (m > 0) for (let i = 0; i < cur.length; i++) cur[i] /= m;
  }
  for (let f = 0; f < nF; f++) {
    const l = Math.hypot(cur[3 * f], cur[3 * f + 1], cur[3 * f + 2]) || 1;
    cur[3 * f] /= l; cur[3 * f + 1] /= l; cur[3 * f + 2] /= l;
  }
  return cur;
}

function normalDihedral(mesh, N) {
  const { nE, edgeF, edgeV, faces, boundary } = mesh;
  const out = new Float64Array(nE);
  for (let e = 0; e < nE; e++) {
    if (boundary[e]) continue;
    const f0 = edgeF[2 * e], f1 = edgeF[2 * e + 1];
    const va = edgeV[2 * e], vb = edgeV[2 * e + 1];
    const sg = dirInFace(faces, f0, va, vb) === dirInFace(faces, f1, va, vb) ? -1 : 1;
    const d = sg * (N[3 * f0] * N[3 * f1] + N[3 * f0 + 1] * N[3 * f1 + 1] + N[3 * f0 + 2] * N[3 * f1 + 2]);
    out[e] = Math.acos(Math.max(-1, Math.min(1, d))) / Math.PI;
  }
  return out;
}

/** Stage-1 feature matrix (row-major, nE x NF). */
export function computeEdgeFeatures(mesh) {
  const { nV, nE, nF, edgeV, edgeF, faces, pos, fn, farea, edgeLen, meanEdgeLen, boundary, vEdgeStart, bboxMin, bboxMax } = mesh;
  const { a, sign } = dihedrals(mesh);

  // angle defect per vertex
  const angSum = new Float64Array(nV);
  for (let f = 0; f < nF; f++) {
    for (let k = 0; k < 3; k++) {
      const i = faces[3 * f + k], j = faces[3 * f + (k + 1) % 3], l = faces[3 * f + (k + 2) % 3];
      const ux = pos[3 * j] - pos[3 * i], uy = pos[3 * j + 1] - pos[3 * i + 1], uz = pos[3 * j + 2] - pos[3 * i + 2];
      const vx = pos[3 * l] - pos[3 * i], vy = pos[3 * l + 1] - pos[3 * i + 1], vz = pos[3 * l + 2] - pos[3 * i + 2];
      const c = (ux * vx + uy * vy + uz * vz) / ((Math.hypot(ux, uy, uz) * Math.hypot(vx, vy, vz)) || 1);
      angSum[i] += Math.acos(Math.max(-1, Math.min(1, c)));
    }
  }
  const onB = new Uint8Array(nV);
  for (let e = 0; e < nE; e++) if (boundary[e]) { onB[edgeV[2 * e]] = 1; onB[edgeV[2 * e + 1]] = 1; }
  const defect = new Float64Array(nV);
  for (let v = 0; v < nV; v++) {
    const d = ((onB[v] ? Math.PI : 2 * Math.PI) - angSum[v]) / Math.PI;
    defect[v] = Math.max(-2, Math.min(2, d));
  }

  const n1 = diffuse(mesh, a), n2 = diffuse(mesh, n1);
  const n3 = diffuse(mesh, n2), n4 = diffuse(mesh, n3);
  const [lineLo, lineHi] = lineContinuation(mesh, a);
  const creaseS = normalDihedral(mesh, smoothedFaceNormals(mesh, 3));
  const creaseL = normalDihedral(mesh, smoothedFaceNormals(mesh, 10));
  const creaseSn = diffuse(mesh, diffuse(mesh, creaseS));

  const rangeY = (bboxMax[1] - bboxMin[1]) || 1, rangeX = (bboxMax[0] - bboxMin[0]) || 1;
  const cx = 0.5 * (bboxMin[0] + bboxMax[0]);

  const X = new Float32Array(nE * NF);
  for (let e = 0; e < nE; e++) {
    const va = edgeV[2 * e], vb = edgeV[2 * e + 1];
    const f0 = edgeF[2 * e], f1 = edgeF[2 * e + 1];
    const o = e * NF;
    X[o] = a[e];
    X[o + 1] = sign[e] > 0 ? a[e] : 0;
    X[o + 2] = sign[e] < 0 ? a[e] : 0;
    X[o + 3] = Math.max(-4, Math.min(4, Math.log((edgeLen[e] + 1e-12) / meanEdgeLen)));
    X[o + 4] = f1 >= 0 ? Math.min(6, Math.abs(Math.log((farea[f0] + 1e-20) / (farea[f1] + 1e-20)))) : 0;
    X[o + 5] = Math.min(Math.abs(defect[va]), Math.abs(defect[vb]));
    X[o + 6] = Math.max(Math.abs(defect[va]), Math.abs(defect[vb]));
    X[o + 7] = defect[va] + defect[vb];
    X[o + 8] = ((vEdgeStart[va + 1] - vEdgeStart[va]) + (vEdgeStart[vb + 1] - vEdgeStart[vb]) - 12) / 12;
    X[o + 9] = n1[e];
    X[o + 10] = n2[e];
    X[o + 11] = n4[e];
    X[o + 12] = a[e] - n1[e];
    X[o + 13] = lineLo[e];
    X[o + 14] = lineHi[e];
    let ny = fn[3 * f0 + 1];
    if (f1 >= 0) ny = 0.5 * (ny + fn[3 * f1 + 1]);
    X[o + 15] = ny;
    const my = 0.5 * (pos[3 * va + 1] + pos[3 * vb + 1]);
    const mx = 0.5 * (pos[3 * va] + pos[3 * vb]);
    X[o + 16] = (my - bboxMin[1]) / rangeY;
    X[o + 17] = Math.max(0, 1 - Math.abs(mx - cx) / (0.5 * rangeX));
    X[o + 18] = 0.5 * (onB[va] + onB[vb]);
    X[o + 19] = Math.min(1, creaseS[e] * 4);
    X[o + 20] = Math.min(1, creaseL[e] * 8);
    X[o + 21] = Math.max(-1, Math.min(1, (creaseS[e] - creaseSn[e]) * 4));
  }
  return { X, dihedral: a };
}

/** Stage-2 features: stage-1 features + context from stage-1 predictions. */
export function contextFeatures(mesh, X1, p) {
  const nE = mesh.nE;
  const q1 = diffuse(mesh, p), q2 = diffuse(mesh, q1);
  const [lo, hi] = lineContinuation(mesh, p);
  const mx = neighbourMax(mesh, p);
  const X2 = new Float32Array(nE * NF2);
  for (let e = 0; e < nE; e++) {
    X2.set(X1.subarray(e * NF, e * NF + NF), e * NF2);
    const o = e * NF2 + NF;
    X2[o] = p[e]; X2[o + 1] = q1[e]; X2[o + 2] = q2[e]; X2[o + 3] = lo[e]; X2[o + 4] = hi[e]; X2[o + 5] = mx[e];
  }
  return X2;
}
