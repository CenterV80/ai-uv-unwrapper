// V2: paired statistical comparison of methods from a bench results.csv.
//
//   node tools/stats.mjs --results results/v2-original/results.csv [--out stats.md] [--boot 4000]
//
// Unit of analysis: one source mesh (voxel variants of the same source are averaged).
// Tests: Wilcoxon signed-rank (exact up to n = 60) for continuous metrics, exact McNemar for validity.
// Confidence intervals: cluster bootstrap over source models (meshes of one model are not independent).
// Multiple comparisons: Holm correction within each split.
import { readFileSync, writeFileSync } from 'node:fs';
import { median, wilcoxon, mcnemar, clusterBootCI, holm } from './statlib.mjs';

const argv = process.argv.slice(2);
const opt = (n, d) => { const i = argv.indexOf('--' + n); return i >= 0 ? argv[i + 1] : d; };
const FILE = opt('results'), OUT = opt('out', null), BOOT = +opt('boot', 4000);

// ------------------------------------------------------------------ csv
function parseCSV(text) {
  const rows = []; let row = [], cell = '', q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) { if (c === '"' && text[i + 1] === '"') { cell += '"'; i++; } else if (c === '"') q = false; else cell += c; }
    else if (c === '"') q = true;
    else if (c === ',') { row.push(cell); cell = ''; }
    else if (c === '\n') { row.push(cell); rows.push(row); row = []; cell = ''; }
    else cell += c;
  }
  if (cell || row.length) { row.push(cell); rows.push(row); }
  const [h, ...rest] = rows;
  return rest.filter(r => r.length === h.length).map(r => Object.fromEntries(h.map((k, i) => [k, r[i]])));
}
const num = v => (v === '' || v === undefined ? NaN : +v);

// ------------------------------------------------------------------ data
const rows = parseCSV(readFileSync(FILE, 'utf8')).filter(r => r.reference === 'usable' && r.status !== 'error' && r.status !== 'skipped');
const unitOf = r => (r.model.match(/__v\d+$/) ? r.model.replace(/__v\d+$/, '') : r.model + '/' + r.mesh);

const METRICS = [
  { key: 'seamF1Tol', label: 'シームF1（許容1%）', better: 'high' },
  { key: 'seamF1Tol3', label: 'シームF1（許容3%）', better: 'high' },
  { key: 'foundLen', label: '見つけた割合（長さ）', better: 'high' },
  { key: 'extraLen', label: '余計な割合（長さ）', better: 'low' },
  { key: 'chartsGap', label: '枚数のずれ |log(枚数÷アーティスト)|', better: 'low', derive: r => Math.abs(Math.log(num(r.chartsVsArtist))) },
  { key: 'angleDistP95', label: '角度歪みP95', better: 'low' },
  { key: 'areaDistP95', label: '面積歪みP95', better: 'low' },
  { key: 'valid', label: 'UV有効', better: 'high', binary: true, derive: r => (r.status === 'ok' ? 1 : 0) },
];
const valueOf = (r, m) => (m.derive ? m.derive(r) : num(r[m.key]));

// unit -> method -> metric -> mean over variants
const units = new Map();
for (const r of rows) {
  const u = unitOf(r);
  if (!units.has(u)) units.set(u, { cluster: r.source || r.model, split: r.split, methods: {} });
  const U = units.get(u);
  (U.methods[r.method] ||= []).push(r);
  // chance level pseudo-method, from the ours row
  if (r.method === 'ours') (U.methods.chance ||= []).push({ ...r, seamF1Tol: r.seamF1TolChance, seamF1Tol3: r.seamF1Tol3Chance, method: 'chance' });
}
const agg = (list, m) => { const v = list.map(r => valueOf(r, m)).filter(Number.isFinite); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN; };

const PAIRS = [['ours', 'dihedral'], ['ours', 'xatlas'], ['dihedral', 'xatlas'], ['ours', 'chance'], ['dihedral', 'chance'], ['xatlas', 'chance']];
const NAME = { ours: '本手法', dihedral: '二面角だけ', xatlas: 'xatlas', chance: '偶然（ランダム）', artistSeams: '正解シーム' };

const report = { file: FILE, units: units.size, splits: {} };
const md = [`# 統計的比較（V2）`, '', `入力: \`${FILE}\` · 比較の単位: 元メッシュ ${units.size}件（ボクセル版は3解像度の平均）· ブートストラップ ${BOOT}回（モデル単位）`, ''];
for (const split of ['unseen', 'all']) {
  const us = [...units.values()].filter(U => split === 'all' || U.split === split);
  const tests = [];
  for (const [A, B] of PAIRS) {
    for (const m of METRICS) {
      if (B === 'chance' && !['seamF1Tol', 'seamF1Tol3'].includes(m.key)) continue;
      const pairs = us.filter(U => U.methods[A] && U.methods[B]).map(U => ({ a: agg(U.methods[A], m), b: agg(U.methods[B], m), c: U.cluster })).filter(p => Number.isFinite(p.a) && Number.isFinite(p.b));
      if (pairs.length < 3) continue;
      const sign = m.better === 'high' ? 1 : -1;
      const d = pairs.map(p => p.a - p.b);
      const t = { A, B, metric: m.key, label: m.label, n: pairs.length, medA: median(pairs.map(p => p.a)), medB: median(pairs.map(p => p.b)), medDiff: median(d),
        aBetter: pairs.filter(p => sign * (p.a - p.b) > 1e-12).length, bBetter: pairs.filter(p => sign * (p.b - p.a) > 1e-12).length };
      if (m.binary) { const r = mcnemar(pairs.map(p => p.a >= 0.5), pairs.map(p => p.b >= 0.5)); t.p = r.p; t.test = 'McNemar'; }
      else { t.p = wilcoxon(d).p; t.test = 'Wilcoxon'; t.ci = clusterBootCI(d, pairs.map(p => p.c), BOOT); }
      t.clusters = new Set(pairs.map(p => p.c)).size;
      tests.push(t);
    }
  }
  holm(tests);
  report.splits[split] = tests;
  md.push(`## ${split === 'unseen' ? 'unseen（シーム予測の学習に未使用の元モデルのみ）' : 'all（学習に使った元モデルも含む。ours には有利）'}`, '',
    '| 比較 A 対 B | 指標 | n（モデル数） | A 中央値 | B 中央値 | 差の中央値 [95%CI] | A が良い / B が良い | p（Holm補正後） |', '|---|---|---|---|---|---|---|---|');
  const f = v => (Number.isFinite(v) ? (Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(3)) : '–');
  for (const t of tests) {
    const sig = t.pHolm < 0.01 ? ' **' : t.pHolm < 0.05 ? ' *' : '';
    md.push(`| ${NAME[t.A]} 対 ${NAME[t.B]} | ${t.label} | ${t.n}（${t.clusters}） | ${f(t.medA)} | ${f(t.medB)} | ${t.ci ? `${f(t.medDiff)} [${f(t.ci[0])}, ${f(t.ci[1])}]` : '–'} | ${t.aBetter} / ${t.bBetter} | ${t.pHolm.toFixed(3)}${sig} |`);
  }
  md.push('');
}
md.push('`*` p < 0.05、`**` p < 0.01（Holm補正後）。差 = A − B。歪み・余計な割合・枚数のずれは小さいほど良い。');
const text = md.join('\n');
if (OUT) { writeFileSync(OUT, text + '\n'); writeFileSync(OUT.replace(/\.md$/, '.json'), JSON.stringify(report, null, 1)); }
console.log(text);
