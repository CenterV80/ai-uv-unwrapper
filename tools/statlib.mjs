// Small statistics helpers for tools/stats.mjs (Wilcoxon, McNemar, cluster bootstrap, Holm).
export function median(a) { const s = a.filter(Number.isFinite).sort((x, y) => x - y); if (!s.length) return NaN; const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; }

/** Wilcoxon signed-rank test, two-sided. Exact (with tied ranks) for n <= 60, normal approximation above. */
export function wilcoxon(d) {
  const x = d.filter(v => Number.isFinite(v) && Math.abs(v) > 1e-12);
  const n = x.length;
  if (n === 0) return { n, p: 1, W: 0 };
  const idx = x.map((v, i) => i).sort((a, b) => Math.abs(x[a]) - Math.abs(x[b]));
  const rank = new Array(n);
  for (let i = 0; i < n;) { let j = i; while (j + 1 < n && Math.abs(Math.abs(x[idx[j + 1]]) - Math.abs(x[idx[i]])) < 1e-12) j++; const r = (i + j + 2) / 2; for (let k = i; k <= j; k++) rank[idx[k]] = r; i = j + 1; }
  let W = 0; for (let i = 0; i < n; i++) if (x[i] > 0) W += rank[i];
  if (n <= 60) {
    // distribution of the sum of a random subset of (doubled, integer) ranks
    const r2 = rank.map(r => Math.round(r * 2)), total = r2.reduce((a, b) => a + b, 0);
    let dist = new Float64Array(total + 1); dist[0] = 1;
    for (const r of r2) { const nd = new Float64Array(total + 1); for (let s = 0; s <= total; s++) if (dist[s]) { nd[s] += dist[s] / 2; nd[s + r] += dist[s] / 2; } dist = nd; }
    const w2 = Math.round(W * 2);
    let lo = 0, hi = 0; for (let s = 0; s <= total; s++) { if (s <= w2) lo += dist[s]; if (s >= w2) hi += dist[s]; }
    return { n, W, p: Math.min(1, 2 * Math.min(lo, hi)) };
  }
  const mu = n * (n + 1) / 4;
  let tie = 0; const seen = new Map(); for (const r of rank) seen.set(r, (seen.get(r) || 0) + 1); for (const t of seen.values()) tie += t ** 3 - t;
  const sd = Math.sqrt(n * (n + 1) * (2 * n + 1) / 24 - tie / 48);
  const z = (Math.abs(W - mu) - 0.5) / sd;
  return { n, W, p: Math.min(1, 2 * (1 - normCdf(z))) };
}
export function normCdf(z) { const t = 1 / (1 + 0.2316419 * Math.abs(z)); const d = 0.3989423 * Math.exp(-z * z / 2); const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274)))); return z > 0 ? 1 - p : p; }

/** Exact McNemar test on discordant pairs. */
export function mcnemar(a, b) {
  let ab = 0, ba = 0; for (let i = 0; i < a.length; i++) { if (a[i] && !b[i]) ab++; if (!a[i] && b[i]) ba++; }
  const n = ab + ba; if (!n) return { p: 1, ab, ba };
  const k = Math.min(ab, ba); let c = 0, lc = 0;
  for (let i = 0; i <= k; i++) { c += Math.exp(lc - n * Math.LN2); lc += Math.log(n - i) - Math.log(i + 1); }
  return { p: Math.min(1, 2 * c), ab, ba };
}

/** Cluster bootstrap CI for the median paired difference (resampling source models). */
export function clusterBootCI(diffs, clusters, B, seed = 7) {
  const byC = new Map(); diffs.forEach((d, i) => { if (!Number.isFinite(d)) return; if (!byC.has(clusters[i])) byC.set(clusters[i], []); byC.get(clusters[i]).push(d); });
  const keys = [...byC.keys()]; if (keys.length < 2) return [NaN, NaN];
  let s = seed; const rnd = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  const meds = [];
  for (let b = 0; b < B; b++) { const sample = []; for (let i = 0; i < keys.length; i++) sample.push(...byC.get(keys[Math.floor(rnd() * keys.length)])); meds.push(median(sample)); }
  meds.sort((x, y) => x - y);
  return [meds[Math.floor(0.025 * B)], meds[Math.floor(0.975 * B)]];
}

export function holm(tests) {
  const order = tests.map((t, i) => i).sort((a, b) => tests[a].p - tests[b].p);
  let run = 0;
  order.forEach((i, k) => { run = Math.max(run, Math.min(1, tests[i].p * (tests.length - k))); tests[i].pHolm = run; });
}

