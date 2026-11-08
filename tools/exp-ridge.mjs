// V8b: how strong must a seam's geometric trace be before each method finds it?
// Generates the same garments with seam ridges of increasing height and measures, per method,
// how much of the interior seams (pieces on the same side) and of the front/back folds is found.
//
//   node tools/exp-ridge.mjs [--n 30] [--levels 0,0.003,0.006,0.012,0.024] [--out results/v8-ridge]
import { mkdirSync, readdirSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { join } from 'node:path';
import { readOBJ } from './load.mjs';
import { buildMesh, seamsFromUV, cornerOf } from '../src/core/mesh.js';
import { computeMetrics, seamAgreement, uvSeamsAndCharts } from '../src/core/metrics.js';
import { METHODS } from './methods.mjs';
import { median, wilcoxon, holm } from './statlib.mjs';

const argv = process.argv.slice(2);
const opt = (n, d) => { const i = argv.indexOf('--' + n); return i >= 0 ? argv[i + 1] : d; };
const N = +opt('n', 30), LEVELS = opt('levels', '0,0.003,0.006,0.012,0.024').split(',').map(Number);
const OUT = opt('out', 'results/v8-ridge'), DATA = opt('data', 'data/garments-ridge');
const METHOD_IDS = opt('methods', 'ours,dihedral,xatlas').split(',');
const NAME = { ours: '本手法', oursAdaptive: '本手法＋相対的な切り方', oursV1: '旧モデル', oursV1Adaptive: '旧モデル＋相対的な切り方', dihedral: '二面角だけ', xatlas: 'xatlas' };

mkdirSync(OUT, { recursive: true });
const rows = [];
for (const h of LEVELS) {
  const dir = join(DATA, `r${String(Math.round(h * 1000)).padStart(3, '0')}`);
  execFileSync(process.execPath, ['tools/gen-garments.mjs', '--n', String(N), '--seed', '1', '--ridge', String(h), '--out', dir], { stdio: 'ignore' });
  for (const file of readdirSync(dir).filter(f => f.endsWith('.obj')).sort()) {
    const m = readOBJ(join(dir, file))[0];
    const pieces = (/pieces (.*)$/m.exec(m.header || '') || [])[1];
    const mesh = buildMesh(m.positions, m.indices, m.cornerUV);
    const labels = seamsFromUV(mesh);
    // reference seam types: fold (front/back join) or interior (two pieces on the same side)
    const sideOfChart = id => (pieces ? pieces.split(' ')[id].split(':')[0] : '?');
    const chartOf = f => Math.round(mesh.cornerUV[6 * f]);
    const interior = new Uint8Array(mesh.nE);
    let lenInterior = 0, lenFold = 0;
    for (let e = 0; e < mesh.nE; e++) {
      if (labels[e] !== 1) continue;
      const same = sideOfChart(chartOf(mesh.edgeF[2 * e])) === sideOfChart(chartOf(mesh.edgeF[2 * e + 1]));
      interior[e] = same ? 1 : 0;
      if (same) lenInterior += mesh.edgeLen[e]; else lenFold += mesh.edgeLen[e];
    }
    const refCharts = uvSeamsAndCharts(mesh, mesh.cornerUV).charts;
    for (const method of METHOD_IDS) {
      const res = await METHODS[method](mesh, { labels }, { params: {} });
      const { seam } = uvSeamsAndCharts(mesh, res.cornerUV);
      const ag = seamAgreement(mesh, seam, labels, 0.01, true);
      let missedI = 0, missedF = 0;
      for (let e = 0; e < mesh.nE; e++) if (ag.classes[e] === 3) { if (interior[e]) missedI += mesh.edgeLen[e]; else missedF += mesh.edgeLen[e]; }
      const met = computeMetrics(mesh, res.cornerUV);
      rows.push({
        ridge: h, garment: file.replace(/_r\d+\.obj$/, ''), type: file.split('_')[1], method,
        interiorFound: lenInterior ? 1 - missedI / lenInterior : null,
        foldFound: lenFold ? 1 - missedF / lenFold : null,
        extra: ag.lengths.pred ? ag.lengths.extra / ag.lengths.pred : null,
        f1: ag.seamF1Tol, chartsVsArtist: met.charts / refCharts, valid: met.valid ? 1 : 0,
      });
    }
  }
  console.log(`ridge ${h}: done`);
}

// ------------------------------------------------------------------ summary
const cols = Object.keys(rows[0]);
writeFileSync(join(OUT, 'results.csv'), [cols.join(','), ...rows.map(r => cols.map(c => r[c]).join(','))].join('\n') + '\n');
const pct = v => (Number.isFinite(v) ? `${Math.round(v * 100)}%` : '–');
const md = ['# 縫い目のふくらみと検出率（V8b）', '', `服 ${N * 4}着（4種類×${N}）× ふくらみ ${LEVELS.length}段階。割合は切れ目の長さで計算、許容1%、中央値。`, ''];
md.push('| ふくらみの高さ | 手法 | 内側の縫い目を見つけた | 前後の境目を見つけた | 余計な切れ目 | シームF1 | 枚数（正解比） | 有効率 |', '|---|---|---|---|---|---|---|---|');
for (const h of LEVELS) for (const m of METHOD_IDS) {
  const rs = rows.filter(r => r.ridge === h && r.method === m);
  md.push(`| ${h} | ${NAME[m] || m} | ${pct(median(rs.map(r => r.interiorFound)))} | ${pct(median(rs.map(r => r.foldFound)))} | ${pct(median(rs.map(r => r.extra)))} | ${median(rs.map(r => r.f1)).toFixed(3)} | ${median(rs.map(r => r.chartsVsArtist)).toFixed(2)} | ${pct(rs.reduce((s, r) => s + r.valid, 0) / rs.length)} |`);
}
// paired tests per level: ours vs each other method on interior seams found
const tests = [];
for (const h of LEVELS) for (const other of METHOD_IDS.filter(m => m !== 'ours')) {
  const by = new Map(); for (const r of rows.filter(r => r.ridge === h)) (by.get(r.garment) || by.set(r.garment, {}).get(r.garment))[r.method] = r.interiorFound;
  const d = [...by.values()].filter(v => Number.isFinite(v.ours) && Number.isFinite(v[other])).map(v => v.ours - v[other]);
  tests.push({ h, other, n: d.length, med: median(d), wins: d.filter(x => x > 1e-9).length, losses: d.filter(x => x < -1e-9).length, p: wilcoxon(d).p });
}
holm(tests);
md.push('', '## 本手法と他手法の差（内側の縫い目を見つけた割合、同じ服どうしの比較）', '', '| ふくらみ | 相手 | n | 差の中央値 | 本手法が良い / 悪い | p（Holm補正後） |', '|---|---|---|---|---|---|');
for (const t of tests) md.push(`| ${t.h} | ${NAME[t.other]} | ${t.n} | ${(t.med * 100).toFixed(1)}ポイント | ${t.wins} / ${t.losses} | ${t.pHolm.toFixed(3)}${t.pHolm < 0.01 ? ' **' : t.pHolm < 0.05 ? ' *' : ''} |`);
writeFileSync(join(OUT, 'summary.md'), md.join('\n') + '\n');
console.log(md.join('\n'));
