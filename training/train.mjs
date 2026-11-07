// Train the base seam predictor on artist-made UV layouts from glTF sample models.
// Usage: node training/train.mjs [--epochs 30]
import { readdirSync, writeFileSync } from 'node:fs';
import { readGLB } from './glb.mjs';
import { buildMesh, seamsFromUV, UnionFind } from '../src/core/mesh.js';
import { computeEdgeFeatures, contextFeatures, NF, NF2, FEATURE_NAMES } from '../src/core/features.js';
import { MLP, SeamPredictor } from '../src/core/model.js';
import { sampleRows, concatRows, evaluate } from '../src/core/predict.js';
import { loadLicenses, checkModel, writeNotice } from './licenses.mjs';

const args = process.argv.slice(2);
const EPOCHS = +(args[args.indexOf('--epochs') + 1] || 30) || 30;
const NOWRITE = args.includes('--dry');
const dir = new URL('./models/', import.meta.url).pathname;
const HOLDOUT = new Set(['BoomBox', 'Corset', 'SheenChair', 'CesiumMilkTruck']);
const licenses = loadLicenses();
const excluded = [];
const MIN_FACES_PER_CHART = 60;   // filters out auto-generated / photogrammetry atlases
const MAX_ROWS_POS = 15000;
const MASK = (process.env.MASK ?? 'normalY,height,symmetryX,nearBoundary,valence,logLen,logAreaRatio').split(',').filter(Boolean);
const maskIdx = MASK.map(n => FEATURE_NAMES.indexOf(n)).filter(i => i >= 0);
if (MASK.length) console.log('masked features:', MASK.join(','));

let seed = 1234;
const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };

const items = [];
for (const file of readdirSync(dir).filter(f => f.endsWith('.glb')).sort()) {
  const model = file.replace('.glb', '');
  let meshes;
  const lic = checkModel(licenses, model);
  if (!lic.ok) { excluded.push(`${model} (${lic.reason})`); continue; }
  try { meshes = readGLB(dir + file); } catch (e) { console.log('skip', model, e.message); continue; }
  for (const m of meshes) {
    const mesh = buildMesh(m.positions, m.indices, m.cornerUV);
    if (mesh.nF < 60) continue;
    const labels = seamsFromUV(mesh);
    const nm = mesh.nonManifold.reduce((a, b) => a + b, 0);
    const uf = new UnionFind(mesh.nF);
    let pos = 0;
    for (let e = 0; e < mesh.nE; e++) { if (labels[e] === 0) uf.union(mesh.edgeF[2 * e], mesh.edgeF[2 * e + 1]); if (labels[e] === 1) pos++; }
    const roots = new Set(); for (let i = 0; i < mesh.nF; i++) roots.add(uf.find(i));
    const fpc = mesh.nF / roots.size;
    if (fpc < MIN_FACES_PER_CHART || nm > 0.01 * mesh.nE || pos < 5) continue;
    const feats = computeEdgeFeatures(mesh);
    for (let e = 0; e < mesh.nE; e++) for (const j of maskIdx) feats.X[e * NF + j] = 0;
    items.push({ model, name: m.name, mesh, labels, feats, holdout: HOLDOUT.has(model) });
  }
  process.stdout.write(`loaded ${model}\n`);
}
if (excluded.length) console.log('ライセンスにより除外:\n  ' + excluded.join('\n  '));
const train = items.filter(i => !i.holdout), test = items.filter(i => i.holdout);
console.log(`train meshes: ${train.length} (${new Set(train.map(i => i.model)).size} models), test meshes: ${test.length}`);

function rowsFor(list, featFn, nf) {
  return list.map(it => {
    const r = sampleRows(featFn(it), it.labels, nf, 4, rnd);
    if (r.nPos > MAX_ROWS_POS) { // cap huge meshes
      const keep = (MAX_ROWS_POS * 5) / r.y.length;
      const idx = []; for (let i = 0; i < r.y.length; i++) if (rnd() < keep) idx.push(i);
      const X = new Float32Array(idx.length * nf), y = new Uint8Array(idx.length);
      idx.forEach((s, i) => { X.set(r.X.subarray(s * nf, s * nf + nf), i * nf); y[i] = r.y[s]; });
      return { X, y, nPos: y.reduce((a, b) => a + b, 0), nNeg: y.length - y.reduce((a, b) => a + b, 0) };
    }
    return r;
  });
}

async function trainMLP(sizes, data, nf, label, sd) {
  const m = new MLP(sizes, sd);
  m.setNormalization(data.X, data.y.length);
  const t0 = Date.now();
  await m.train(data.X, data.y, data.w, {
    epochs: EPOCHS, lr: 2e-3, batch: 256, seed: sd,
    onEpoch: async (ep, loss) => { if (ep % 5 === 4 || ep === EPOCHS - 1) console.log(`  ${label} epoch ${ep + 1}/${EPOCHS} loss ${loss.toFixed(4)} (${((Date.now() - t0) / 1000).toFixed(0)}s)`); },
  });
  return m;
}

// ---- stage 1 (2-fold out-of-fold predictions feed stage 2)
const H1 = (process.env.H1 ?? '').split(',').filter(Boolean).map(Number);
const H2 = (process.env.H2 ?? '').split(',').filter(Boolean).map(Number);
const S1 = [NF, ...H1, 1], S2 = [NF2, ...H2, 1];
const models = [...new Set(train.map(i => i.model))];
const foldOf = it => models.indexOf(it.model) % 2;
const oof = new Map();
for (let k = 0; k < 2; k++) {
  console.log(`stage1 fold ${k}`);
  const tr = train.filter(i => foldOf(i) !== k);
  const d = concatRows(rowsFor(tr, it => it.feats.X, NF), NF);
  const m = await trainMLP(S1, d, NF, `fold${k}`, 21 + k);
  for (const it of train.filter(i => foldOf(i) === k)) oof.set(it, m.predict(it.feats.X, it.mesh.nE));
}
console.log('stage1 full');
const d1 = concatRows(rowsFor(train, it => it.feats.X, NF), NF);
const stage1 = await trainMLP(S1, d1, NF, 'stage1', 31);

console.log('stage2');
const d2 = concatRows(rowsFor(train, it => contextFeatures(it.mesh, it.feats.X, oof.get(it)), NF2), NF2);
const stage2 = await trainMLP(S2, d2, NF2, 'stage2', 41);
const predictor = new SeamPredictor(stage1, stage2);

// ---- evaluation on held-out models
const report = [];
const agg = { base: [[], []], s1: [[], []], s2: [[], []] };
for (const it of test) {
  const p1 = stage1.predict(it.feats.X, it.mesh.nE);
  const p2 = stage2.predict(contextFeatures(it.mesh, it.feats.X, p1), it.mesh.nE);
  const base = Array.from(it.feats.dihedral);
  const r = { model: it.model, mesh: it.name, edges: it.mesh.nE, dihedralAUC: evaluate(base, it.labels).auc, stage1: evaluate(p1, it.labels), stage2: evaluate(p2, it.labels) };
  report.push(r);
  console.log(`${it.model}/${it.name}: AUC dihedral-only ${r.dihedralAUC.toFixed(3)} | stage1 ${r.stage1.auc.toFixed(3)} | stage2 ${r.stage2.auc.toFixed(3)} (F1 ${r.stage2.f1.toFixed(3)})`);
  for (const [k, p] of [['base', base], ['s1', p1], ['s2', p2]]) for (let e = 0; e < it.labels.length; e++) if (it.labels[e] !== 255) { agg[k][0].push(p[e]); agg[k][1].push(it.labels[e]); }
}
const summary = {};
for (const k of Object.keys(agg)) summary[k] = evaluate(agg[k][0], Uint8Array.from(agg[k][1]));
console.log('pooled holdout:', JSON.stringify(summary, (k, v) => typeof v === 'number' ? +v.toFixed(4) : v));

// ---- calibrate the default threshold so the predicted seam rate matches artists' seam rate
const allP = [], rates = [];
for (const it of train) {
  const p1 = stage1.predict(it.feats.X, it.mesh.nE);
  const p2 = stage2.predict(contextFeatures(it.mesh, it.feats.X, p1), it.mesh.nE);
  let n = 0, pos = 0;
  const sub = Math.max(1, Math.floor(it.mesh.nE / 4000));
  for (let e = 0; e < it.mesh.nE; e++) if (it.labels[e] !== 255) { n++; pos += it.labels[e]; if (e % sub === 0) allP.push(p2[e]); }
  rates.push(pos / Math.max(1, n));
}
rates.sort((a, b) => a - b);
const targetRate = rates[Math.floor(rates.length / 2)] * 1.5; // a little above the median artist rate: LSCM prefers more cuts
allP.sort((a, b) => b - a);
const defaultThreshold = +allP[Math.min(allP.length - 1, Math.floor(allP.length * targetRate))].toFixed(2);
const atThr = evaluate(Array.from(agg.s2[0]), Uint8Array.from(agg.s2[1]), defaultThreshold);
console.log(`default threshold ${defaultThreshold} (target seam rate ${(targetRate * 100).toFixed(1)}%) -> holdout P ${atThr.precision.toFixed(3)} R ${atThr.recall.toFixed(3)} F1 ${atThr.f1.toFixed(3)}`);

const json = predictor.toJSON();
json.meta = { licensePolicy: 'CC0-1.0, CC-BY-4.0 only (see NOTICE.md)', defaultThreshold, holdoutAtDefault: atThr, masked: MASK, trainedOn: models, holdout: [...HOLDOUT], epochs: EPOCHS, holdoutSummary: summary, date: new Date().toISOString().slice(0, 10) };
if (!NOWRITE) writeNotice(licenses, models, new URL('../NOTICE.md', import.meta.url).pathname);
if (!NOWRITE) writeFileSync(new URL('../src/weights.js', import.meta.url), '// Generated by training/train.mjs — base seam predictor weights.\nexport default ' + JSON.stringify(json) + ';\n');
writeFileSync(new URL('./report.json', import.meta.url), JSON.stringify({ summary, report }, null, 1));
console.log('wrote src/weights.js');
