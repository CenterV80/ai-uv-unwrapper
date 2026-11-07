// Chart packing: equal texel density, minimum-area rotation, skyline bin packing.

function convexHull(pts) {
  const n = pts.length / 2;
  const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => pts[2 * a] - pts[2 * b] || pts[2 * a + 1] - pts[2 * b + 1]);
  const cross = (o, a, b) => (pts[2 * a] - pts[2 * o]) * (pts[2 * b + 1] - pts[2 * o + 1]) - (pts[2 * a + 1] - pts[2 * o + 1]) * (pts[2 * b] - pts[2 * o]);
  const lower = [], upper = [];
  for (const i of idx) { while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], i) <= 0) lower.pop(); lower.push(i); }
  for (let j = idx.length - 1; j >= 0; j--) { const i = idx[j]; while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], i) <= 0) upper.pop(); upper.push(i); }
  upper.pop(); lower.pop();
  return lower.concat(upper);
}

/** Rotate uv in place to its minimum-area bounding rectangle; returns {w,h,minx,miny}. */
function orientChart(uv) {
  const hull = convexHull(uv);
  let bestA = Infinity, bestAng = 0;
  const tryAngle = ang => {
    const c = Math.cos(ang), s = Math.sin(ang);
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const i of hull) {
      const x = c * uv[2 * i] - s * uv[2 * i + 1], y = s * uv[2 * i] + c * uv[2 * i + 1];
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
    }
    const A = (x1 - x0) * (y1 - y0);
    if (A < bestA - 1e-12) { bestA = A; bestAng = ang; }
  };
  for (let k = 0; k < hull.length; k++) {
    const a = hull[k], b = hull[(k + 1) % hull.length];
    tryAngle(-Math.atan2(uv[2 * b + 1] - uv[2 * a + 1], uv[2 * b] - uv[2 * a]));
  }
  if (!hull.length) tryAngle(0);
  const c = Math.cos(bestAng), s = Math.sin(bestAng);
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (let i = 0; i < uv.length; i += 2) {
    const x = c * uv[i] - s * uv[i + 1], y = s * uv[i] + c * uv[i + 1];
    uv[i] = x; uv[i + 1] = y;
    if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
  }
  // landscape orientation keeps skyline shelves low
  if (y1 - y0 > x1 - x0) {
    for (let i = 0; i < uv.length; i += 2) { const x = uv[i]; uv[i] = uv[i + 1]; uv[i + 1] = -x; }
    [x0, x1, y0, y1] = [y0, y1, -x1, -x0];
  }
  for (let i = 0; i < uv.length; i += 2) { uv[i] -= x0; uv[i + 1] -= y0; }
  return { w: x1 - x0, h: y1 - y0 };
}

function skylinePack(rects, binW) {
  // rects: [{w,h}] (already padded); returns positions + used height, allowing 90° rotation
  let sky = [{ x: 0, y: 0, w: binW }];
  const out = new Array(rects.length);
  const order = rects.map((r, i) => i).sort((a, b) => Math.max(rects[b].h, rects[b].w) - Math.max(rects[a].h, rects[a].w) || rects[b].h - rects[a].h);
  let H = 0;
  const fit = (idx, w, h) => {
    let x = sky[idx].x; if (x + w > binW + 1e-9) return -1;
    let y = 0, rem = w, i = idx;
    while (rem > 1e-12) {
      if (i >= sky.length) return -1;
      y = Math.max(y, sky[i].y);
      rem -= sky[i].w; i++;
    }
    return y;
  };
  for (const ri of order) {
    const r = rects[ri];
    let best = null;
    for (const [w, h, rot] of [[r.w, r.h, false], [r.h, r.w, true]]) {
      if (w > binW + 1e-9) continue;
      for (let i = 0; i < sky.length; i++) {
        const y = fit(i, w, h);
        if (y < 0) continue;
        const top = y + h;
        if (!best || top < best.top - 1e-12 || (Math.abs(top - best.top) < 1e-12 && sky[i].x < best.x)) best = { i, x: sky[i].x, y, w, h, rot, top };
      }
    }
    if (!best) { // wider than bin (shouldn't happen when binW >= max side)
      best = { i: 0, x: 0, y: H, w: r.w, h: r.h, rot: false, top: H + r.h };
    }
    out[ri] = best;
    H = Math.max(H, best.top);
    // update skyline
    const nx = best.x, nw = best.w, ny = best.top;
    const next = [];
    for (const s of sky) {
      const s0 = s.x, s1 = s.x + s.w;
      if (s1 <= nx + 1e-12 || s0 >= nx + nw - 1e-12) { next.push(s); continue; }
      if (s0 < nx) next.push({ x: s0, y: s.y, w: nx - s0 });
      if (s1 > nx + nw) next.push({ x: nx + nw, y: s.y, w: s1 - nx - nw });
    }
    next.push({ x: nx, y: ny, w: nw });
    next.sort((a, b) => a.x - b.x);
    sky = [];
    for (const s of next) {
      const last = sky[sky.length - 1];
      if (last && Math.abs(last.y - s.y) < 1e-12) last.w += s.w; else sky.push({ ...s });
    }
  }
  return { pos: out, H };
}

function rotK(x, y, k) { // rotate by k*90° counter-clockwise
  switch (k & 3) { case 0: return [x, y]; case 1: return [-y, x]; case 2: return [-x, -y]; default: return [y, -x]; }
}

/** Does triangle (ax..cy) overlap the axis-aligned square [x0,x0+s]x[y0,y0+s]? (SAT) */
function triOverlapsCell(ax, ay, bx, by, cx, cy, x0, y0, s) {
  const xs = [x0, x0 + s, x0 + s, x0], ys = [y0, y0, y0 + s, y0 + s];
  const tx = [ax, bx, cx], ty = [ay, by, cy];
  for (let i = 0; i < 3; i++) {
    const j = (i + 1) % 3, k = (i + 2) % 3;
    const nx = ty[j] - ty[i], ny = tx[i] - tx[j];
    const tri = nx * tx[k] + ny * ty[k], base = nx * tx[i] + ny * ty[i];
    let lo = Infinity, hi = -Infinity;
    for (let q = 0; q < 4; q++) { const d = nx * xs[q] + ny * ys[q]; if (d < lo) lo = d; if (d > hi) hi = d; }
    if (tri >= base ? hi < base : lo > base) return false;
  }
  return true;
}

/** Rasterise a chart (uv + triangles) at cell size c with rotation k and dilation dil. */
function rasterChart(uv, T, c, k, dil) {
  const n = uv.length / 2;
  const P = new Float64Array(uv.length);
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < n; i++) {
    const [x, y] = rotK(uv[2 * i], uv[2 * i + 1], k);
    P[2 * i] = x; P[2 * i + 1] = y;
    if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
  }
  for (let i = 0; i < n; i++) { P[2 * i] -= x0; P[2 * i + 1] -= y0; }
  const w = Math.max(1, Math.ceil((x1 - x0) / c)) + 2 * dil, h = Math.max(1, Math.ceil((y1 - y0) / c)) + 2 * dil;
  const core = new Uint8Array(w * h);
  for (let t = 0; t < T.length; t += 3) {
    const a = T[t], b = T[t + 1], d = T[t + 2];
    const ax = P[2 * a], ay = P[2 * a + 1], bx = P[2 * b], by = P[2 * b + 1], cx = P[2 * d], cy = P[2 * d + 1];
    const i0 = Math.max(0, Math.floor(Math.min(ax, bx, cx) / c)), i1 = Math.min(w - 2 * dil - 1, Math.floor(Math.max(ax, bx, cx) / c));
    const j0 = Math.max(0, Math.floor(Math.min(ay, by, cy) / c)), j1 = Math.min(h - 2 * dil - 1, Math.floor(Math.max(ay, by, cy) / c));
    for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) {
      const idx = (j + dil) * w + i + dil;
      if (core[idx]) continue;
      if (i1 === i0 || j1 === j0 || triOverlapsCell(ax, ay, bx, by, cx, cy, i * c, j * c, c)) core[idx] = 1;
    }
  }
  // dilate (padding)
  let mask = core;
  if (dil > 0) {
    mask = new Uint8Array(w * h);
    for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
      if (!core[j * w + i]) continue;
      for (let dj = -dil; dj <= dil; dj++) for (let di = -dil; di <= dil; di++) {
        const jj = j + dj, ii = i + di;
        if (jj >= 0 && jj < h && ii >= 0 && ii < w) mask[jj * w + ii] = 1;
      }
    }
  }
  // cell list, bottom rows first (most likely to collide in bottom-left placement)
  const cells = [];
  for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) if (mask[j * w + i]) cells.push(i, j);
  return { P, w, h, cells: Int32Array.from(cells) };
}

/**
 * Raster packing: charts are placed by their real shape on an occupancy grid, so concave
 * pieces nest into each other's gaps. Tries 4 rotations per chart.
 */
function rasterPack(uvs, charts, total, params, G) {
  let S = Math.sqrt(total / 0.62);
  let best = null;
  for (let attempt = 0; attempt < 5; attempt++) {
    const c = S / G;
    const dil = Math.max(1, Math.ceil(params.padding * G / params.resolution / 2));
    const order = uvs.map((_, i) => i).sort((a, b) => charts[b].area3d - charts[a].area3d);
    let H = G * 3;
    let occ = new Uint8Array(G * H);
    let usedH = 0;
    const place = new Array(uvs.length);
    let failed = false;
    for (const ci of order) {
      let bestP = null;
      for (let k = 0; k < 4; k++) {
        const R = rasterChart(uvs[ci], charts[ci].T, c, k, dil);
        if (R.w > G) continue;
        const cells = R.cells, nc = cells.length;
        let found = false;
        const yMax = bestP ? bestP.y + bestP.h - R.h : Infinity;
        for (let y = 0; !found && y <= yMax; y++) {
          if ((y + R.h) > H) { // grow grid
            const nH = Math.max(H * 2, y + R.h + G);
            const n = new Uint8Array(G * nH); n.set(occ); occ = n; H = nH;
          }
          for (let x = 0; x + R.w <= G; x++) {
            let ok = true;
            for (let q = 0; q < nc; q += 2) if (occ[(y + cells[q + 1]) * G + x + cells[q]]) { ok = false; break; }
            if (ok) {
              const top = y + R.h;
              if (!bestP || top < bestP.y + bestP.h || (top === bestP.y + bestP.h && x < bestP.x)) bestP = { x, y, k, R, h: R.h };
              found = true; break;
            }
          }
        }
      }
      if (!bestP) { failed = true; break; }
      const { x, y, R } = bestP;
      for (let q = 0; q < R.cells.length; q += 2) occ[(y + R.cells[q + 1]) * G + x + R.cells[q]] = 1;
      usedH = Math.max(usedH, y + R.h);
      place[ci] = { x, y, k: bestP.k, P: R.P, dil };
    }
    if (failed) { S *= 1.15; continue; }
    const sz = Math.max(G, usedH) * c;
    if (!best || sz < best.sz) best = { sz, c, place };
    // aim for a square result
    const ratio = usedH / G;
    if (Math.abs(ratio - 1) < 0.04) break;
    S *= Math.sqrt(Math.max(0.6, Math.min(1.6, ratio))) * (ratio > 1 ? 1.01 : 0.995);
  }
  if (!best) return null;
  const out = uvs.map((uv, i) => {
    const p = best.place[i], o = new Float64Array(uv.length);
    for (let j = 0; j < uv.length; j += 2) {
      o[j] = ((p.x + p.dil) * best.c + p.P[j]) / best.sz;
      o[j + 1] = ((p.y + p.dil) * best.c + p.P[j + 1]) / best.sz;
    }
    return o;
  });
  return { uvs: out, efficiency: total / (best.sz * best.sz) };
}

/** @returns {{uvs: Float64Array[], efficiency: number}} uvs in [0,1] per chart */

export function packCharts(charts, params) {
  const uvs = charts.map(c => {
    const uv = Float64Array.from(c.uv);
    const s = c.area2d > 1e-30 ? Math.sqrt(c.area3d / c.area2d) : 1;
    for (let i = 0; i < uv.length; i++) uv[i] *= s;
    return uv;
  });
  const boxes = uvs.map(orientChart);
  let total = 0, maxSide = 0;
  for (let i = 0; i < boxes.length; i++) { total += charts[i].area3d; maxSide = Math.max(maxSide, boxes[i].w, boxes[i].h); }
  const boxArea = boxes.reduce((s, b) => s + b.w * b.h, 0);
  let bestRes = null;
  // padding in uv units depends on the final scale -> iterate a couple of times
  let side = Math.sqrt(boxArea) * 1.15;
  for (let iter = 0; iter < 3; iter++) {
    const pad = (params.padding / params.resolution) * side;
    const rects = boxes.map(b => ({ w: b.w + 2 * pad, h: b.h + 2 * pad }));
    const maxR = Math.max(...rects.map(r => Math.max(r.w, r.h)), 1e-12);
    const area = rects.reduce((s, r) => s + r.w * r.h, 0);
    let best = null;
    for (const f of [0.9, 1.0, 1.08, 1.16, 1.25, 1.35, 1.5]) {
      const W = Math.max(maxR, Math.sqrt(area) * f);
      const res = skylinePack(rects, W);
      const sz = Math.max(W, res.H);
      if (!best || sz < best.sz) best = { ...res, W, sz, pad };
    }
    bestRes = best;
    side = best.sz;
  }
  const { pos, sz, pad } = bestRes;
  for (let i = 0; i < uvs.length; i++) {
    const uv = uvs[i], p = pos[i];
    for (let j = 0; j < uv.length; j += 2) {
      let x = uv[j], y = uv[j + 1];
      if (p.rot) { const t = x; x = boxes[i].h - y; y = t; }
      uv[j] = (x + p.x + pad) / sz;
      uv[j + 1] = (y + p.y + pad) / sz;
    }
  }
  const shelf = { uvs, efficiency: total / (sz * sz) };
  if (charts.length > 3000) return shelf;
  const G = charts.length > 400 ? 384 : 256;
  const raster = rasterPack(uvs.map((uv, i) => { // shelf result is in [0,1]; rescale back to 3D units for rasterising
    const o = Float64Array.from(uv); for (let j = 0; j < o.length; j++) o[j] *= sz; return o;
  }), charts, total, params, G);
  return raster && raster.efficiency > shelf.efficiency ? raster : shelf;
}
