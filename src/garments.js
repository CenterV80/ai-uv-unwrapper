// Simple procedural garments built the way clothes are sewn: a front and a back panel cut from
// the same 2D pattern, "inflated" apart, and joined along the sewn edges (side seams, shoulders,
// inseams). Openings (neck, cuffs, hem, waist, ankles) stay open. Pure JS: returns positions/indices.

const all = () => true;
const smooth = t => { t = Math.max(0, Math.min(1, t)); return t * t * (3 - 2 * t); };

function distToPolylines(x, y, lines) {
  let best = Infinity;
  for (const L of lines) for (let i = 0; i + 1 < L.length; i++) {
    const [ax, ay] = L[i], [bx, by] = L[i + 1];
    const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy || 1e-12;
    const t = Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / l2));
    const d = Math.hypot(x - ax - t * dx, y - ay - t * dy);
    if (d < best) best = d;
  }
  return best;
}

/** Sample a curve x = f(y) as a polyline from y0 to y1. */
const curve = (f, y0, y1, n = 48) => Array.from({ length: n + 1 }, (_, i) => { const y = y0 + (y1 - y0) * i / n; return [f(y), y]; });

/**
 * @param parts  [{y0, y1, nu, nv, range: y => [xl, xr], seams}] pattern pieces as row intervals;
 *               seams = {left, right: y => bool, top, bottom: x => bool} marks the sewn edges
 * @param closed polylines (2D) along which front and back are sewn together
 * @param depth  how far the panels bulge apart, @param reach distance over which the bulge builds up
 * @param bend   curvature of the cross-section (wraps the garment around the body)
 */
export function sewPanels({ parts, closed, depth, reach, bend = 0.3 }) {
  const P = [], I = [];
  for (const side of [1, -1]) {
    for (const part of parts) {
      const base = P.length / 3;
      for (let j = 0; j <= part.nv; j++) {
        const y = part.y0 + (part.y1 - part.y0) * j / part.nv;
        const [xl, xr] = part.range(y);
        for (let i = 0; i <= part.nu; i++) {
          const x = xl + (xr - xl) * i / part.nu;
          const sm = part.seams || {};
          const sewn = (i === 0 && sm.left?.(y)) || (i === part.nu && sm.right?.(y)) ||
            (j === 0 && sm.bottom?.(x)) || (j === part.nv && sm.top?.(x));
          // quadratic ease-out: the panels leave the seam at an angle, so the seam is a crisp fold
          const u = Math.min(1, distToPolylines(x, y, closed) / reach);
          const t = sewn ? 0 : depth * (1 - (1 - u) * (1 - u));
          P.push(x, y, side * t - bend * x * x);
        }
      }
      const row = part.nu + 1;
      for (let j = 0; j < part.nv; j++) for (let i = 0; i < part.nu; i++) {
        const a = base + j * row + i, b = a + 1, c = a + row, d = c + 1;
        if (side > 0) I.push(a, b, d, a, d, c); else I.push(a, d, b, a, c, d);
      }
    }
  }
  return { positions: Float32Array.from(P), indices: Uint32Array.from(I) };
}

export function tshirt() {
  const H = 1.5, torso = 0.45, sleeve = 0.98, yA = 0.92, yB = 1.16, neck = 0.17;
  const w = y => torso + (sleeve - torso) * smooth((y - yA) / (yB - yA));
  return sewPanels({
    parts: [{ y0: 0, y1: H, nu: 44, nv: 44, range: y => [-w(y), w(y)],
      seams: { left: y => y <= yB + 1e-9, right: y => y <= yB + 1e-9, top: x => Math.abs(x) >= neck - 1e-9 } }],
    closed: [
      curve(y => -w(y), 0, yB), curve(y => w(y), 0, yB),          // side + underarm seams
      [[-sleeve, H], [-neck, H]], [[neck, H], [sleeve, H]],        // shoulder seams
    ],
    depth: 0.16, reach: 0.3, bend: 0.25,
  });
}

export function pants() {
  const H = 1.6, yC = 0.95, n = 18;
  const outer = y => y > yC ? 0.42 - 0.02 * (y - yC) / (H - yC) : 0.36 + 0.06 * smooth(y / yC);
  const gap = y => y >= yC ? 0 : 0.11 * Math.pow(1 - y / yC, 0.6) + 0.02 * (1 - y / yC);
  return sewPanels({
    parts: [
      { y0: yC, y1: H, nu: 2 * n, nv: 14, range: y => [-outer(y), outer(y)], seams: { left: all, right: all, bottom: x => Math.abs(x) < 1e-9 } },
      { y0: 0, y1: yC, nu: n, nv: 30, range: y => [-outer(y), -gap(y)], seams: { left: all, right: all } },
      { y0: 0, y1: yC, nu: n, nv: 30, range: y => [gap(y), outer(y)], seams: { left: all, right: all } },
    ],
    closed: [
      curve(y => -outer(y), 0, H), curve(y => outer(y), 0, H),    // outer side seams
      curve(y => -gap(y), 0, yC), curve(y => gap(y), 0, yC),      // inseams (meet at the crotch)
    ],
    depth: 0.13, reach: 0.18, bend: 0.2,
  });
}

export function skirt() {
  const H = 0.8, hem = 0.78, waist = 0.4;
  const w = y => hem + (waist - hem) * Math.pow(y / H, 0.8);
  return sewPanels({
    parts: [{ y0: 0, y1: H, nu: 48, nv: 22, range: y => [-w(y), w(y)], seams: { left: all, right: all } }],
    closed: [curve(y => -w(y), 0, H), curve(y => w(y), 0, H)],    // side seams
    depth: 0.32, reach: 0.45, bend: 0.15,
  });
}

/** Knit cap: a dome open at the bottom with a turned-up brim, as one surface. */
export function beanie() {
  const P = [], I = [], seg = 48;
  // profile from the brim's lower edge, up the outside, over the crown
  const prof = [];
  for (let i = 0; i <= 6; i++) prof.push([0.53, 0.0 + 0.16 * i / 6]);                 // brim
  for (let i = 1; i <= 3; i++) prof.push([0.53 - 0.03 * i / 3, 0.16]);                // brim fold
  for (let i = 1; i <= 24; i++) { const a = (i / 24) * Math.PI / 2; prof.push([i === 24 ? 0 : 0.5 * Math.cos(a), 0.16 + 0.48 * Math.sin(a)]); }
  for (let j = 0; j < prof.length; j++) for (let s = 0; s < seg; s++) {
    const a = 2 * Math.PI * s / seg, [r, y] = prof[j];
    P.push(r * Math.cos(a), y, r * Math.sin(a));
  }
  for (let j = 0; j + 1 < prof.length; j++) for (let s = 0; s < seg; s++) {
    const a = j * seg + s, b = j * seg + (s + 1) % seg, c = a + seg, d = b + seg;
    I.push(a, c, d, a, d, b);
  }
  return { positions: Float32Array.from(P), indices: Uint32Array.from(I) };
}

export const GARMENTS = { tshirt, pants, skirt, beanie };
