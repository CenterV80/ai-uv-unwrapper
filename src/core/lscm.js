// Least Squares Conformal Maps (Lévy et al. 2002), solved with Jacobi-preconditioned CG on the
// normal equations. Works on a chart described by local vertices ("wedges").

/**
 * @param {Float64Array} P     xyz per local vertex
 * @param {Int32Array}   T     3 local vertex indices per face
 * @param {number[]}     pins  two local vertex indices to fix
 * @returns {Float64Array} uv per local vertex
 */
export function lscm(P, T, pins, { maxIter = 800, tol = 1e-5 } = {}) {
  const nv = P.length / 3, nf = T.length / 3;
  // per-face coefficients: rows  r1 = Σ gx_i u_i - gy_i v_i ,  r2 = Σ gy_i u_i + gx_i v_i
  const G = new Float64Array(nf * 6);
  for (let f = 0; f < nf; f++) {
    const i0 = T[3 * f], i1 = T[3 * f + 1], i2 = T[3 * f + 2];
    const ex = [P[3 * i1] - P[3 * i0], P[3 * i1 + 1] - P[3 * i0 + 1], P[3 * i1 + 2] - P[3 * i0 + 2]];
    const ey0 = [P[3 * i2] - P[3 * i0], P[3 * i2 + 1] - P[3 * i0 + 1], P[3 * i2 + 2] - P[3 * i0 + 2]];
    const l1 = Math.hypot(ex[0], ex[1], ex[2]) || 1e-20;
    const x1 = l1;
    const x2 = (ex[0] * ey0[0] + ex[1] * ey0[1] + ex[2] * ey0[2]) / l1;
    const cr = [ex[1] * ey0[2] - ex[2] * ey0[1], ex[2] * ey0[0] - ex[0] * ey0[2], ex[0] * ey0[1] - ex[1] * ey0[0]];
    const y2 = Math.hypot(cr[0], cr[1], cr[2]) / l1;
    const area2 = x1 * y2 || 1e-20; // twice the area
    // local 2D points: p0=(0,0) p1=(x1,0) p2=(x2,y2); g_i = perp(p_{i+2}-p_{i+1}) / (2A), perp(x,y)=(-y,x)
    const px = [0, x1, x2], py = [0, 0, y2];
    const w = Math.sqrt(area2 / 2);
    for (let k = 0; k < 3; k++) {
      const a = (k + 1) % 3, b = (k + 2) % 3;
      const dx = px[b] - px[a], dy = py[b] - py[a];
      G[6 * f + 2 * k] = (-dy / area2) * w;
      G[6 * f + 2 * k + 1] = (dx / area2) * w;
    }
  }

  // initial guess: projection on the best-fit plane, similarity-aligned to the pins
  const x = initialGuess(P, T, pins);
  const n = 2 * nv;
  const fi = new Int32Array(n).fill(-1);
  let nfree = 0;
  for (let i = 0; i < nv; i++) if (i !== pins[0] && i !== pins[1]) { fi[2 * i] = nfree++; fi[2 * i + 1] = nfree++; }

  // assemble M = AᵀA over free variables (sparse, symmetric) and rhs b = -Aᵀ A x_pinned
  const rows = Array.from({ length: nfree }, () => new Map());
  const b = new Float64Array(nfree);
  const cu = new Float64Array(6), cv = new Float64Array(6), vars = new Int32Array(6);
  for (let f = 0; f < nf; f++) {
    // row1 coefficients over (u0,v0,u1,v1,u2,v2), row2 likewise
    for (let k = 0; k < 3; k++) {
      const i = T[3 * f + k], gx = G[6 * f + 2 * k], gy = G[6 * f + 2 * k + 1];
      vars[2 * k] = 2 * i; vars[2 * k + 1] = 2 * i + 1;
      cu[2 * k] = gx; cu[2 * k + 1] = -gy;   // row 1
      cv[2 * k] = gy; cv[2 * k + 1] = gx;    // row 2
    }
    let r1 = 0, r2 = 0; // pinned contribution
    for (let a = 0; a < 6; a++) if (fi[vars[a]] < 0) { r1 += cu[a] * x[vars[a]]; r2 += cv[a] * x[vars[a]]; }
    for (let a = 0; a < 6; a++) {
      const ia = fi[vars[a]]; if (ia < 0) continue;
      b[ia] -= cu[a] * r1 + cv[a] * r2;
      const row = rows[ia];
      for (let c = 0; c < 6; c++) {
        const ic = fi[vars[c]]; if (ic < 0 || ic > ia) continue; // lower triangle incl. diagonal
        row.set(ic, (row.get(ic) || 0) + cu[a] * cu[c] + cv[a] * cv[c]);
      }
    }
  }
  // CSR of the lower triangle (sorted columns, diagonal last)
  const rp = new Int32Array(nfree + 1);
  for (let i = 0; i < nfree; i++) rp[i + 1] = rp[i] + rows[i].size;
  const ci = new Int32Array(rp[nfree]), cval = new Float64Array(rp[nfree]);
  for (let i = 0; i < nfree; i++) {
    const ent = [...rows[i].entries()].sort((p, q) => p[0] - q[0]);
    for (let j = 0; j < ent.length; j++) { ci[rp[i] + j] = ent[j][0]; cval[rp[i] + j] = ent[j][1]; }
  }
  const diagM = new Float64Array(nfree);
  for (let i = 0; i < nfree; i++) diagM[i] = cval[rp[i + 1] - 1] || 1e-30;
  const matvec = (v, out) => {
    out.fill(0);
    for (let i = 0; i < nfree; i++) {
      let s = 0;
      for (let p = rp[i]; p < rp[i + 1] - 1; p++) { const j = ci[p], a = cval[p]; s += a * v[j]; out[j] += a * v[i]; }
      out[i] += s + cval[rp[i + 1] - 1] * v[i];
    }
  };
  const Lv = incompleteCholesky(rp, ci, cval, nfree);
  const precond = (r, z) => {
    if (!Lv) { for (let i = 0; i < nfree; i++) z[i] = r[i] / diagM[i]; return; }
    for (let i = 0; i < nfree; i++) { // forward L y = r
      let s = r[i];
      for (let p = rp[i]; p < rp[i + 1] - 1; p++) s -= Lv[p] * z[ci[p]];
      z[i] = s / Lv[rp[i + 1] - 1];
    }
    for (let i = nfree - 1; i >= 0; i--) { // backward Lᵀ z = y
      z[i] /= Lv[rp[i + 1] - 1];
      const zi = z[i];
      for (let p = rp[i]; p < rp[i + 1] - 1; p++) z[ci[p]] -= Lv[p] * zi;
    }
  };

  const y = new Float64Array(nfree);
  for (let i = 0; i < nv; i++) if (fi[2 * i] >= 0) { y[fi[2 * i]] = x[2 * i]; y[fi[2 * i + 1]] = x[2 * i + 1]; }
  const r = new Float64Array(nfree), z = new Float64Array(nfree), p = new Float64Array(nfree), q = new Float64Array(nfree);
  matvec(y, q);
  let bnorm = 0;
  for (let i = 0; i < nfree; i++) { r[i] = b[i] - q[i]; bnorm += b[i] * b[i]; }
  bnorm = Math.sqrt(bnorm) || 1;
  precond(r, z); p.set(z);
  let rz = 0; for (let i = 0; i < nfree; i++) rz += r[i] * z[i];
  let it = 0;
  for (; it < maxIter; it++) {
    let rn = 0; for (let i = 0; i < nfree; i++) rn += r[i] * r[i];
    if (Math.sqrt(rn) <= tol * bnorm) break;
    matvec(p, q);
    let pq = 0; for (let i = 0; i < nfree; i++) pq += p[i] * q[i];
    if (pq <= 0) break;
    const alpha = rz / pq;
    for (let i = 0; i < nfree; i++) { y[i] += alpha * p[i]; r[i] -= alpha * q[i]; }
    precond(r, z);
    let rz2 = 0; for (let i = 0; i < nfree; i++) rz2 += r[i] * z[i];
    const beta = rz2 / rz; rz = rz2;
    for (let i = 0; i < nfree; i++) p[i] = z[i] + beta * p[i];
  }
  for (let i = 0; i < nv; i++) if (fi[2 * i] >= 0) { x[2 * i] = y[fi[2 * i]]; x[2 * i + 1] = y[fi[2 * i + 1]]; }
  return { uv: x, iterations: it };
}

/** IC(0) of a lower-triangular CSR SPD matrix (diagonal stored last in each row). Retries with a diagonal shift. */
function incompleteCholesky(rp, ci, val, n) {
  for (let shift = 0; shift < 6; shift++) {
    const alpha = shift === 0 ? 0 : 1e-3 * Math.pow(10, shift - 1);
    const L = Float64Array.from(val);
    for (let i = 0; i < n; i++) L[rp[i + 1] - 1] *= 1 + alpha;
    let ok = true;
    for (let i = 0; i < n && ok; i++) {
      for (let pp = rp[i]; pp < rp[i + 1]; pp++) {
        const k = ci[pp];
        // dot(row i, row k) over columns < k
        let s = L[pp];
        let a = rp[i], bq = rp[k];
        const aEnd = pp, bEnd = rp[k + 1] - 1;
        while (a < aEnd && bq < bEnd) {
          const ca = ci[a], cb = ci[bq];
          if (ca === cb) { s -= L[a] * L[bq]; a++; bq++; } else if (ca < cb) a++; else bq++;
        }
        if (k === i) {
          if (!(s > 1e-300)) { ok = false; break; }
          L[pp] = Math.sqrt(s);
        } else L[pp] = s / L[rp[k + 1] - 1];
      }
    }
    if (ok) return L;
  }
  return null;
}

function initialGuess(P, T, pins) {
  const nv = P.length / 3, nf = T.length / 3;
  // area-weighted average normal
  let nx = 0, ny = 0, nz = 0;
  for (let f = 0; f < nf; f++) {
    const a = T[3 * f], b = T[3 * f + 1], c = T[3 * f + 2];
    const ux = P[3 * b] - P[3 * a], uy = P[3 * b + 1] - P[3 * a + 1], uz = P[3 * b + 2] - P[3 * a + 2];
    const vx = P[3 * c] - P[3 * a], vy = P[3 * c + 1] - P[3 * a + 1], vz = P[3 * c + 2] - P[3 * a + 2];
    nx += uy * vz - uz * vy; ny += uz * vx - ux * vz; nz += ux * vy - uy * vx;
  }
  let l = Math.hypot(nx, ny, nz);
  if (l < 1e-20) { nx = 0; ny = 0; nz = 1; l = 1; }
  nx /= l; ny /= l; nz /= l;
  // tangent basis
  let tx, ty, tz;
  if (Math.abs(nx) < 0.9) { tx = 0; ty = nz; tz = -ny; } else { tx = -nz; ty = 0; tz = nx; }
  l = Math.hypot(tx, ty, tz); tx /= l; ty /= l; tz /= l;
  const bx = ny * tz - nz * ty, by = nz * tx - nx * tz, bz = nx * ty - ny * tx;
  const uv = new Float64Array(2 * nv);
  for (let i = 0; i < nv; i++) {
    uv[2 * i] = P[3 * i] * tx + P[3 * i + 1] * ty + P[3 * i + 2] * tz;
    uv[2 * i + 1] = P[3 * i] * bx + P[3 * i + 1] * by + P[3 * i + 2] * bz;
  }
  // similarity so that pin0 -> (0,0) and pin1 -> (d,0), d = 3D distance
  const [p0, p1] = pins;
  const d = Math.hypot(P[3 * p1] - P[3 * p0], P[3 * p1 + 1] - P[3 * p0 + 1], P[3 * p1 + 2] - P[3 * p0 + 2]) || 1;
  const ax = uv[2 * p1] - uv[2 * p0], ay = uv[2 * p1 + 1] - uv[2 * p0 + 1];
  const al = Math.hypot(ax, ay);
  let c = 1, s = 0, sc = 1;
  if (al > 1e-20) { c = ax / al; s = ay / al; sc = d / al; }
  const ox = uv[2 * p0], oy = uv[2 * p0 + 1];
  for (let i = 0; i < nv; i++) {
    const x = uv[2 * i] - ox, y = uv[2 * i + 1] - oy;
    uv[2 * i] = sc * (c * x + s * y);
    uv[2 * i + 1] = sc * (-s * x + c * y);
  }
  uv[2 * p0] = 0; uv[2 * p0 + 1] = 0; uv[2 * p1] = d; uv[2 * p1 + 1] = 0;
  return uv;
}

/** Signed area of each face in UV and 3D area; returns distortion stats. */
export function chartDistortion(P, T, uv) {
  const nf = T.length / 3;
  let A2 = 0, A3 = 0;
  const a2 = new Float64Array(nf), a3 = new Float64Array(nf);
  for (let f = 0; f < nf; f++) {
    const i = T[3 * f], j = T[3 * f + 1], k = T[3 * f + 2];
    a2[f] = 0.5 * ((uv[2 * j] - uv[2 * i]) * (uv[2 * k + 1] - uv[2 * i + 1]) - (uv[2 * k] - uv[2 * i]) * (uv[2 * j + 1] - uv[2 * i + 1]));
    const ux = P[3 * j] - P[3 * i], uy = P[3 * j + 1] - P[3 * i + 1], uz = P[3 * j + 2] - P[3 * i + 2];
    const vx = P[3 * k] - P[3 * i], vy = P[3 * k + 1] - P[3 * i + 1], vz = P[3 * k + 2] - P[3 * i + 2];
    a3[f] = 0.5 * Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx);
    A2 += a2[f]; A3 += a3[f];
  }
  const flipSign = A2 < 0 ? -1 : 1;
  A2 = Math.abs(A2);
  let flips = 0, flipArea = 0, dist = 0;
  for (let f = 0; f < nf; f++) {
    const s2 = flipSign * a2[f];
    if (s2 <= 0) { flips++; flipArea += a3[f]; continue; }
    const r = (s2 / A2) / (a3[f] / A3 || 1e-30);
    dist += a3[f] * Math.min(10, 0.5 * (r + 1 / r)); // cap so sliver triangles can't dominate
  }
  // flipped faces count as heavily distorted
  dist = (dist + flipArea * 10) / (A3 || 1);
  return { distortion: dist, flips, flipFrac: flipArea / (A3 || 1), mirrored: flipSign < 0, area2d: A2, area3d: A3 };
}
