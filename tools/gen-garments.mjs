// Generates garments with ground-truth seams by construction: every face belongs to a pattern
// piece (front/back x panel), and the reference seams are the borders between pieces — the lines a
// tailor would sew. Output: labels-only OBJs readable by tools/bench.mjs, plus manifest.json.
//
//   node tools/gen-garments.mjs [--n 50] [--seed 1] [--out data/garments] [--types tee,pants,skirt,dress]
//
// Rules (documented in docs/research-plan.md, V8):
//  - front and back are separate pieces; they are joined along side seams, shoulders, inseams
//  - sleeves are separate pieces (set-in: border at the shoulder point; raglan: diagonal from the neck)
//  - optional bands (neck band, hem band, cuffs, waistband) are separate pieces
//  - pants: left and right halves are separate pieces (centre front / centre back seam)
//  - skirts: gores (2/4/6 per garment) split front and back vertically
//  - dress: bodice and skirt joined by a waist seam
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { sewPanels } from '../src/garments.js';

const argv = process.argv.slice(2);
const opt = (n, d) => { const i = argv.indexOf('--' + n); return i >= 0 ? argv[i + 1] : d; };
const N = +opt('n', 50), SEED = +opt('seed', 1), OUT = opt('out', 'data/garments');
const TYPES = opt('types', 'tee,pants,skirt,dress').split(',');

function rng(seed) { let s = seed >>> 0; return () => { s = (s + 0x6D2B79F5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const smooth = t => { t = Math.max(0, Math.min(1, t)); return t * t * (3 - 2 * t); };
const curve = (f, y0, y1, n = 48) => Array.from({ length: n + 1 }, (_, i) => { const y = y0 + (y1 - y0) * i / n; return [f(y), y]; });
const all = () => true;

// ------------------------------------------------------------------ garment types
function tee(r) {
  const p = {
    H: 1.3 + r() * 0.5, W: 0.38 + r() * 0.17, neck: 0.13 + r() * 0.07,
    sleeve: ['setin', 'raglan', 'none'][Math.floor(r() * 3)], long: r() < 0.4,
    cuff: r() < 0.5, neckBand: r() < 0.5, hemBand: r() < 0.4,
  };
  const { H, W, neck } = p;
  const yA = H * (0.58 + r() * 0.08);
  if (p.sleeve === 'none') {
    // tank top: armholes are openings, narrow shoulder straps
    const strap = neck + 0.08 + r() * 0.08;
    const w = y => W * (y < yA ? 1 : 1 - 0.55 * smooth((y - yA) / (H - yA)));
    const dims = w(H);
    return {
      p: { ...p, cuff: false }, depth: 0.15, reach: 0.28,
      parts: [{ y0: 0, y1: H, nu: 44, nv: 52, range: y => [-w(y), w(y)], seams: { left: y => y <= yA + 1e-9, right: y => y <= yA + 1e-9, top: x => Math.abs(x) >= neck - 1e-9 } }],
      closed: [curve(y => -w(y), 0, yA), curve(y => w(y), 0, yA), [[-dims, H], [-neck, H]], [[neck, H], [dims, H]]],
      panel: (x, y) => (p.neckBand && y > H - 0.06 && Math.abs(x) < neck + 0.06 ? 'neckband' : p.hemBand && y < 0.08 ? 'hem' : 'body'),
    };
  }
  const S = W + (p.long ? 0.55 + r() * 0.3 : 0.22 + r() * 0.18);
  const yB = yA + 0.16 + r() * 0.1;
  const w = y => W + (S - W) * smooth((y - yA) / (yB - yA));
  const sleeveOf = (x, y) => {
    const ax = Math.abs(x);
    if (p.sleeve === 'setin') return ax > W + 0.01;
    // raglan: border from the neck point (neck, H) down to the underarm (W, yA)
    const t = (H - y) / (H - yA); return t <= 1 && ax > neck + (W - neck) * t;
  };
  return {
    p, depth: 0.15, reach: 0.28,
    parts: [{ y0: 0, y1: H, nu: 56, nv: 52, range: y => [-w(y), w(y)], seams: { left: y => y <= yB + 1e-9, right: y => y <= yB + 1e-9, top: x => Math.abs(x) >= neck - 1e-9 } }],
    closed: [curve(y => -w(y), 0, yB), curve(y => w(y), 0, yB), [[-S, H], [-neck, H]], [[neck, H], [S, H]]],
    panel: (x, y) => {
      if (sleeveOf(x, y)) return (p.cuff && Math.abs(x) > S - 0.07 ? 'cuff' : 'sleeve') + (x < 0 ? 'L' : 'R');
      if (p.neckBand && y > H - 0.06 && Math.abs(x) < neck + 0.06) return 'neckband';
      if (p.hemBand && y < 0.08) return 'hem';
      return 'body';
    },
  };
}

function pants(r) {
  const shorts = r() < 0.35;
  const H = 1.5 + r() * 0.3, yC = H * (shorts ? 0.45 + r() * 0.08 : 0.6 + r() * 0.05), n = 18;
  const hip = 0.38 + r() * 0.08, hem = shorts ? hip * (0.9 + r() * 0.1) : 0.3 + r() * 0.1;
  const y0 = shorts ? yC - (0.25 + r() * 0.15) : 0;
  const outer = y => (y > yC ? hip - 0.02 * (y - yC) / (H - yC) : hem + (hip - hem) * smooth((y - y0) / (yC - y0)));
  const gap = y => (y >= yC ? 0 : (hip * 0.3) * Math.pow(1 - (y - y0) / (yC - y0), 0.6) + 0.01);
  const band = r() < 0.6 ? 0.06 + r() * 0.03 : 0;
  return {
    p: { shorts, band: band > 0 }, depth: 0.12, reach: 0.17,
    parts: [
      { y0: yC, y1: H, nu: 2 * n, nv: 16, range: y => [-outer(y), outer(y)], seams: { left: all, right: all, bottom: x => Math.abs(x) < 1e-9 } },
      { y0, y1: yC, nu: n, nv: 30, range: y => [-outer(y), -gap(y)], seams: { left: all, right: all } },
      { y0, y1: yC, nu: n, nv: 30, range: y => [gap(y), outer(y)], seams: { left: all, right: all } },
    ],
    closed: [curve(y => -outer(y), y0, H), curve(y => outer(y), y0, H), curve(y => -gap(y), y0, yC), curve(y => gap(y), y0, yC)],
    panel: (x, y) => (band && y > H - band ? 'waistband' : x < 0 ? 'legL' : 'legR'),
  };
}

function skirt(r) {
  const H = 0.6 + r() * 0.4, waist = 0.36 + r() * 0.06, hemW = waist + 0.1 + r() * 0.45;
  const gores = [2, 4, 6][Math.floor(r() * 3)], band = r() < 0.6 ? 0.06 : 0;
  const flare = 0.6 + r() * 0.6; // shape of the flare (fixed per garment)
  const w = y => hemW + (waist - hemW) * Math.pow(y / H, flare);
  return {
    p: { gores, band: band > 0 }, depth: 0.3, reach: 0.45,
    parts: [{ y0: 0, y1: H, nu: 48, nv: 26, range: y => [-w(y), w(y)], seams: { left: all, right: all } }],
    closed: [curve(y => -w(y), 0, H), curve(y => w(y), 0, H)],
    panel: (x, y) => {
      if (band && y > H - band) return 'waistband';
      const per = gores / 2, u = (x / w(y) + 1) / 2; // 0..1 across the front (or back) piece
      return 'gore' + Math.min(per - 1, Math.floor(u * per));
    },
  };
}

function dress(r) {
  const H = 1.6 + r() * 0.5, yW = H * (0.55 + r() * 0.1), W = 0.36 + r() * 0.08, neck = 0.14 + r() * 0.06;
  const hemW = W + 0.15 + r() * 0.45, yA = yW + (H - yW) * (0.45 + r() * 0.15);
  const strap = neck + 0.08 + r() * 0.06;
  const w = y => (y >= yW ? (y < yA ? W : W - (W - strap) * smooth((y - yA) / (H - yA))) : hemW + (W - hemW) * smooth(y / yW));
  return {
    p: { waistSeam: true }, depth: 0.2, reach: 0.35,
    parts: [{ y0: 0, y1: H, nu: 48, nv: 64, range: y => [-w(y), w(y)], seams: { left: y => y <= yA + 1e-9, right: y => y <= yA + 1e-9, top: x => Math.abs(x) >= neck - 1e-9 } }],
    closed: [curve(y => -w(y), 0, yA, 64), curve(y => w(y), 0, yA, 64), [[-strap, H], [-neck, H]], [[neck, H], [strap, H]]],
    panel: (x, y) => (y >= yW ? 'bodice' : 'skirt'),
  };
}

const GEN = { tee, pants, skirt, dress };

// ------------------------------------------------------------------ main
mkdirSync(OUT, { recursive: true });
const manifest = [];
for (const type of TYPES) {
  for (let i = 0; i < N; i++) {
    const seed = SEED * 100003 + i * 7919 + type.length * 131;
    const g = GEN[type](rng(seed));
    const { positions, indices } = sewPanels({ parts: g.parts, closed: g.closed, depth: g.depth, reach: g.reach, bend: 0.2 + 0.1 * rng(seed + 1)() });
    // label every face with its piece: side (front/back) x panel, from the face centroid in pattern space
    const nf = indices.length / 3, half = nf / 2;
    const ids = new Map(), chart = new Int32Array(nf);
    for (let f = 0; f < nf; f++) {
      let cx = 0, cy = 0;
      for (let k = 0; k < 3; k++) { cx += positions[3 * indices[3 * f + k]]; cy += positions[3 * indices[3 * f + k] + 1]; }
      const key = (f < half ? 'front:' : 'back:') + g.panel(cx / 3, cy / 3);
      if (!ids.has(key)) ids.set(key, ids.size);
      chart[f] = ids.get(key);
    }
    const name = `gen_${type}_s${SEED}_${String(i).padStart(4, '0')}`;
    const lines = ['# seamline: labels-only', `# generated ${type} seed ${seed} pieces ${[...ids.keys()].join(' ')}`];
    for (let v = 0; v < positions.length; v += 3) lines.push(`v ${+positions[v].toFixed(5)} ${+positions[v + 1].toFixed(5)} ${+positions[v + 2].toFixed(5)}`);
    for (let c = 0; c < ids.size; c++) lines.push(`vt ${c} 0.5`); // one texture coordinate per piece
    for (let f = 0; f < nf; f++) lines.push(`f ${indices[3 * f] + 1}/${chart[f] + 1} ${indices[3 * f + 1] + 1}/${chart[f] + 1} ${indices[3 * f + 2] + 1}/${chart[f] + 1}`);
    writeFileSync(join(OUT, name + '.obj'), lines.join('\n') + '\n');
    manifest.push({ file: name + '.obj', type, seed, faces: nf, pieces: [...ids.keys()], params: g.p });
  }
  console.log(`${type}: ${N} garments`);
}
writeFileSync(join(OUT, 'manifest.json'), JSON.stringify({ seed: SEED, n: N, items: manifest }, null, 1));
console.log(`wrote ${manifest.length} garments to ${OUT}`);
