// Small MLP (ReLU hidden layers, sigmoid output) with Adam, written from scratch so it can be
// trained in Node for the base model and fine-tuned in the browser.

function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class MLP {
  constructor(sizes, seed = 1) {
    this.sizes = sizes;
    const rnd = mulberry32(seed);
    this.W = []; this.b = [];
    for (let l = 0; l < sizes.length - 1; l++) {
      const nin = sizes[l], nout = sizes[l + 1];
      const sc = Math.sqrt(2 / nin);
      const W = new Float64Array(nin * nout);
      for (let i = 0; i < W.length; i++) {
        // Box-Muller
        const u = Math.max(1e-12, rnd()), v = rnd();
        W[i] = sc * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      }
      this.W.push(W); this.b.push(new Float64Array(nout));
    }
    this.mean = new Float64Array(sizes[0]);
    this.std = new Float64Array(sizes[0]).fill(1);
  }

  setNormalization(X, n) {
    const d = this.sizes[0];
    const m = new Float64Array(d), s = new Float64Array(d);
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) m[j] += X[i * d + j];
    for (let j = 0; j < d; j++) m[j] /= n;
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) { const t = X[i * d + j] - m[j]; s[j] += t * t; }
    for (let j = 0; j < d; j++) s[j] = Math.sqrt(s[j] / n) || 1;
    this.mean = m; this.std = s;
  }

  /** Forward pass for one normalised input; fills acts (array of Float64Array). */
  _forward(x, acts) {
    acts[0] = x;
    const L = this.W.length;
    for (let l = 0; l < L; l++) {
      const nin = this.sizes[l], nout = this.sizes[l + 1];
      const W = this.W[l], b = this.b[l], inp = acts[l], out = acts[l + 1];
      for (let o = 0; o < nout; o++) {
        let s = b[o];
        const off = o * nin;
        for (let i = 0; i < nin; i++) s += W[off + i] * inp[i];
        out[o] = l === L - 1 ? s : (s > 0 ? s : 0.01 * s);
      }
    }
    return acts[L][0];
  }

  _allocActs() { return this.sizes.map(n => new Float64Array(n)); }

  predict(X, n) {
    const d = this.sizes[0];
    const acts = this._allocActs();
    const x = new Float64Array(d);
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < d; j++) x[j] = (X[i * d + j] - this.mean[j]) / this.std[j];
      const z = this._forward(x, acts);
      out[i] = 1 / (1 + Math.exp(-z));
    }
    return out;
  }

  /**
   * Weighted binary cross-entropy training with Adam.
   * @param {Float32Array} X rows of features, @param {Uint8Array} y labels, @param {Float32Array} w sample weights
   */
  async train(X, y, w, { epochs = 10, lr = 3e-3, batch = 256, l2 = 1e-5, seed = 7, onEpoch = null } = {}) {
    const n = y.length, d = this.sizes[0], L = this.W.length;
    const rnd = mulberry32(seed);
    const order = new Int32Array(n); for (let i = 0; i < n; i++) order[i] = i;
    const gW = this.W.map(W => new Float64Array(W.length)), gb = this.b.map(b => new Float64Array(b.length));
    if (!this._adam) this._adam = {
      mW: this.W.map(W => new Float64Array(W.length)), vW: this.W.map(W => new Float64Array(W.length)),
      mb: this.b.map(b => new Float64Array(b.length)), vb: this.b.map(b => new Float64Array(b.length)), t: 0,
    };
    const A = this._adam;
    const acts = this._allocActs();
    const deltas = this.sizes.map(k => new Float64Array(k));
    const x = new Float64Array(d);
    const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    for (let ep = 0; ep < epochs; ep++) {
      for (let i = n - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); const t = order[i]; order[i] = order[j]; order[j] = t; }
      let lossSum = 0, wSum = 0;
      for (let start = 0; start < n; start += batch) {
        const end = Math.min(n, start + batch);
        for (const g of gW) g.fill(0);
        for (const g of gb) g.fill(0);
        let bw = 0;
        for (let k = start; k < end; k++) {
          const i = order[k];
          for (let j = 0; j < d; j++) x[j] = (X[i * d + j] - this.mean[j]) / this.std[j];
          const z = this._forward(x, acts);
          const p = 1 / (1 + Math.exp(-z));
          const wi = w ? w[i] : 1;
          const yi = y[i];
          lossSum -= wi * (yi ? Math.log(p + 1e-9) : Math.log(1 - p + 1e-9));
          wSum += wi; bw += wi;
          deltas[L][0] = wi * (p - yi);
          for (let l = L - 1; l >= 0; l--) {
            const nin = this.sizes[l], nout = this.sizes[l + 1];
            const W = this.W[l], inp = acts[l], dout = deltas[l + 1], din = deltas[l];
            if (l > 0) din.fill(0);
            for (let o = 0; o < nout; o++) {
              const g = dout[o];
              if (g === 0) continue;
              const off = o * nin;
              gb[l][o] += g;
              const gw = gW[l];
              for (let ii = 0; ii < nin; ii++) {
                gw[off + ii] += g * inp[ii];
                if (l > 0) din[ii] += g * W[off + ii];
              }
            }
            if (l > 0) for (let ii = 0; ii < nin; ii++) if (inp[ii] <= 0) din[ii] *= 0.01;
          }
        }
        A.t++;
        const c1 = 1 - Math.pow(b1, A.t), c2 = 1 - Math.pow(b2, A.t);
        const inv = 1 / Math.max(bw, 1e-9);
        for (let l = 0; l < L; l++) {
          const W = this.W[l], b = this.b[l];
          for (let i = 0; i < W.length; i++) {
            const g = gW[l][i] * inv + l2 * W[i];
            A.mW[l][i] = b1 * A.mW[l][i] + (1 - b1) * g;
            A.vW[l][i] = b2 * A.vW[l][i] + (1 - b2) * g * g;
            W[i] -= lr * (A.mW[l][i] / c1) / (Math.sqrt(A.vW[l][i] / c2) + eps);
          }
          for (let i = 0; i < b.length; i++) {
            const g = gb[l][i] * inv;
            A.mb[l][i] = b1 * A.mb[l][i] + (1 - b1) * g;
            A.vb[l][i] = b2 * A.vb[l][i] + (1 - b2) * g * g;
            b[i] -= lr * (A.mb[l][i] / c1) / (Math.sqrt(A.vb[l][i] / c2) + eps);
          }
        }
      }
      if (onEpoch) await onEpoch(ep, lossSum / Math.max(wSum, 1e-9));
    }
  }

  toJSON() {
    const r = a => Array.from(a, v => +v.toPrecision(6));
    return { sizes: this.sizes, W: this.W.map(r), b: this.b.map(r), mean: r(this.mean), std: r(this.std) };
  }

  static fromJSON(j) {
    const m = new MLP(j.sizes);
    m.W = j.W.map(a => Float64Array.from(a));
    m.b = j.b.map(a => Float64Array.from(a));
    m.mean = Float64Array.from(j.mean);
    m.std = Float64Array.from(j.std);
    return m;
  }
}

/** Two-stage seam predictor: stage 2 sees stage-1 predictions of neighbouring edges. */
export class SeamPredictor {
  constructor(stage1, stage2) { this.stage1 = stage1; this.stage2 = stage2; }
  toJSON() { return { version: 1, stage1: this.stage1.toJSON(), stage2: this.stage2.toJSON() }; }
  static fromJSON(j) { return new SeamPredictor(MLP.fromJSON(j.stage1), MLP.fromJSON(j.stage2)); }
}
