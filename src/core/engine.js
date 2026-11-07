// Stateful engine used by the app (inside a Web Worker, or on the main thread as fallback).
import { buildMesh, seamsFromUV } from './mesh.js';
import { computeEdgeFeatures } from './features.js';
import { SeamPredictor } from './model.js';
import { predictSeams, fineTune, evaluate } from './predict.js';
import { unwrap } from './unwrap.js';
import baseWeights from '../weights.js';

export class Engine {
  constructor() {
    this.predictor = SeamPredictor.fromJSON(baseWeights);
    this.mesh = null; this.feats = null; this.prob = null; this.result = null;
    this.uvLabels = null;
  }

  setWeights(json) { this.predictor = json ? SeamPredictor.fromJSON(json) : SeamPredictor.fromJSON(baseWeights); return true; }
  baseMeta() { return baseWeights.meta || null; }

  load({ positions, indices, cornerUV }) {
    const mesh = buildMesh(positions, indices, cornerUV || null);
    this.mesh = mesh;
    this.feats = computeEdgeFeatures(mesh);
    this.uvLabels = cornerUV ? seamsFromUV(mesh) : null;
    this.prob = null; this.result = null;
    let uvSeams = 0;
    if (this.uvLabels) for (let e = 0; e < mesh.nE; e++) if (this.uvLabels[e] === 1) uvSeams++;
    return {
      nV: mesh.nV, nF: mesh.nF, nE: mesh.nE,
      pos: Float32Array.from(mesh.pos), faces: mesh.faces.slice(), edgeV: mesh.edgeV.slice(), edgeF: mesh.edgeF.slice(),
      faceEdges: mesh.faceEdges.slice(), boundary: mesh.boundary.slice(), diag: mesh.diag,
      hasUV: !!cornerUV, uvSeams,
      uvLabels: this.uvLabels ? this.uvLabels.slice() : null,
    };
  }

  origUV() { return this.mesh && this.mesh.cornerUV ? this.mesh.cornerUV.slice() : null; }

  predict() {
    const t0 = performance.now();
    const { prob } = predictSeams(this.mesh, this.predictor, this.feats);
    this.prob = prob;
    const out = { prob: Float32Array.from(prob), ms: performance.now() - t0 };
    if (this.uvLabels) out.eval = evaluate(prob, this.uvLabels);
    return out;
  }

  unwrap({ params, userSeam }, progress) {
    if (!this.prob) this.predict();
    const t0 = performance.now();
    const r = unwrap(this.mesh, this.prob, this.feats.dihedral, params, userSeam || null, progress);
    this.result = r;
    return { cornerUV: r.cornerUV, faceChart: r.faceChart, seam: r.seam, stats: { ...r.stats, ms: performance.now() - t0 } };
  }

  /** source: 'uv' -> learn from the model's own UV seams; 'seams' -> learn from given seam array */
  async train({ source, seams, epochs = 8 }, progress) {
    let labels;
    if (source === 'uv') {
      if (!this.uvLabels) throw new Error('このモデルには元のUVがありません');
      labels = this.uvLabels;
    } else {
      labels = new Uint8Array(this.mesh.nE);
      for (let e = 0; e < this.mesh.nE; e++) labels[e] = this.mesh.boundary[e] || this.mesh.nonManifold[e] ? 255 : (seams[e] ? 1 : 0);
    }
    const before = this.prob ? evaluate(this.prob, labels) : null;
    await fineTune(this.predictor, [{ mesh: this.mesh, labels, feats: this.feats }], {
      epochs, lr: 5e-3, onProgress: (stage, frac, loss) => progress && progress(stage, frac, loss),
    });
    const after = this.predict();
    // re-calibrate the threshold: predicted seam rate ≈ 1.5x the labelled rate on this model
    let n = 0, pos = 0; const ps = [];
    for (let e = 0; e < labels.length; e++) if (labels[e] !== 255) { n++; pos += labels[e]; ps.push(after.prob[e]); }
    ps.sort((a, b) => b - a);
    const k = Math.min(ps.length - 1, Math.max(0, Math.floor(ps.length * Math.min(0.3, 1.5 * pos / Math.max(1, n)))));
    const threshold = Math.max(0.05, Math.min(0.95, +ps[k].toFixed(2)));
    return { weights: this.predictor.toJSON(), before, after: evaluate(after.prob, labels, threshold), prob: after.prob, threshold };
  }
}
