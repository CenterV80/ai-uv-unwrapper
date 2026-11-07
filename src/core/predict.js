// Inference and training helpers shared by the browser app and the Node training script.
import { computeEdgeFeatures, contextFeatures, NF, NF2 } from './features.js';
import { MLP, SeamPredictor } from './model.js';

export function predictSeams(mesh, predictor, feats = null) {
  const f = feats || computeEdgeFeatures(mesh);
  const p1 = predictor.stage1.predict(f.X, mesh.nE);
  const X2 = contextFeatures(mesh, f.X, p1);
  const p2 = predictor.stage2.predict(X2, mesh.nE);
  for (let e = 0; e < mesh.nE; e++) if (mesh.boundary[e]) p2[e] = 1;
  return { prob: p2, stage1: p1, feats: f };
}

/**
 * Collect training rows from a labelled mesh (labels: 0/1, 255 = ignore).
 * Keeps all positives and a random subset of negatives.
 */
export function sampleRows(X, labels, nf, negRatio = 4, rnd = Math.random) {
  const pos = [], neg = [];
  for (let e = 0; e < labels.length; e++) {
    if (labels[e] === 1) pos.push(e); else if (labels[e] === 0) neg.push(e);
  }
  const keepNeg = Math.min(neg.length, Math.max(200, pos.length * negRatio));
  for (let i = neg.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [neg[i], neg[j]] = [neg[j], neg[i]]; }
  const rows = pos.concat(neg.slice(0, keepNeg));
  const out = new Float32Array(rows.length * nf), y = new Uint8Array(rows.length);
  rows.forEach((e, i) => { out.set(X.subarray(e * nf, e * nf + nf), i * nf); y[i] = labels[e]; });
  return { X: out, y, nPos: pos.length, nNeg: keepNeg, negScale: neg.length / Math.max(1, keepNeg) };
}

export function concatRows(sets, nf) {
  const n = sets.reduce((s, r) => s + r.y.length, 0);
  const X = new Float32Array(n * nf), y = new Uint8Array(n), w = new Float32Array(n);
  let o = 0;
  for (const s of sets) {
    X.set(s.X, o * nf); y.set(s.y, o);
    // balance each mesh's contribution and the classes inside it
    // per-mesh weight grows with sqrt(size) so tiny meshes don't dominate
    const np = Math.max(1, s.nPos), nn = Math.max(1, s.nNeg);
    const mw = Math.sqrt(np + nn);
    for (let i = 0; i < s.y.length; i++) w[o + i] = mw * (s.y[i] ? 1.0 / np : 1 / nn);
    o += s.y.length;
  }
  const mean = n / w.reduce((a, b) => a + b, 0);
  for (let i = 0; i < n; i++) w[i] *= mean;
  return { X, y, w };
}

export function newPredictor() {
  return new SeamPredictor(new MLP([NF, 48, 24, 1], 11), new MLP([NF2, 32, 16, 1], 13));
}

/**
 * Fine-tune an existing predictor on new labelled meshes ([{mesh, labels, feats?}]).
 * Used in the browser: "learn from this model's UVs" / "learn from my seam edits".
 */
export async function fineTune(predictor, items, { epochs = 6, lr = 1e-3, onProgress = null } = {}) {
  const s1 = [], s2 = [];
  for (const it of items) {
    const f = it.feats || computeEdgeFeatures(it.mesh);
    s1.push(sampleRows(f.X, it.labels, NF));
  }
  const d1 = concatRows(s1, NF);
  await predictor.stage1.train(d1.X, d1.y, d1.w, {
    epochs, lr, onEpoch: async (ep, loss) => onProgress && onProgress('stage1', (ep + 1) / epochs, loss),
  });
  for (const it of items) {
    const f = it.feats || computeEdgeFeatures(it.mesh);
    const p1 = predictor.stage1.predict(f.X, it.mesh.nE);
    s2.push(sampleRows(contextFeatures(it.mesh, f.X, p1), it.labels, NF2));
  }
  const d2 = concatRows(s2, NF2);
  await predictor.stage2.train(d2.X, d2.y, d2.w, {
    epochs, lr, onEpoch: async (ep, loss) => onProgress && onProgress('stage2', (ep + 1) / epochs, loss),
  });
  return predictor;
}

/** Precision / recall / F1 / AUC of predictions vs labels (ignoring 255). */
export function evaluate(prob, labels, thr = 0.5) {
  let tp = 0, fp = 0, fn = 0, tn = 0;
  const posS = [], negS = [];
  for (let e = 0; e < labels.length; e++) {
    if (labels[e] === 255) continue;
    const pr = prob[e] >= thr;
    if (labels[e]) { posS.push(prob[e]); pr ? tp++ : fn++; } else { negS.push(prob[e]); pr ? fp++ : tn++; }
  }
  // AUC via rank statistic
  const all = posS.map(p => [p, 1]).concat(negS.map(p => [p, 0])).sort((a, b) => a[0] - b[0]);
  let rankSum = 0;
  for (let i = 0; i < all.length;) {
    let j = i; while (j < all.length && all[j][0] === all[i][0]) j++;
    const r = (i + j + 1) / 2;
    for (let k = i; k < j; k++) if (all[k][1]) rankSum += r;
    i = j;
  }
  const np = posS.length, nn = negS.length;
  const auc = np && nn ? (rankSum - np * (np + 1) / 2) / (np * nn) : NaN;
  const precision = tp / Math.max(1, tp + fp), recall = tp / Math.max(1, tp + fn);
  return { precision, recall, f1: 2 * precision * recall / Math.max(1e-9, precision + recall), auc, pos: np, neg: nn };
}
