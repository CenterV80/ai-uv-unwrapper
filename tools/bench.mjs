// Batch evaluation of UV methods on a folder of meshes. Writes per-mesh CSV + summary JSON/Markdown.
//
//   node tools/bench.mjs --data training/models --methods ours,dihedral,artistSeams,artistUV,xatlas
//
// Options: --data DIR (GLB/OBJ files)   --methods LIST   --out DIR (default results/<timestamp>)
//          --filter REGEX (file names)   --max-faces N (skip larger meshes, default 150000)
//          --param key=value (pipeline parameter, repeatable, e.g. --param rejectOverlap=false)
//          --overlap-res N (raster resolution for the overlap metric, default 1024)
import { readdirSync, readFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { execSync } from 'node:child_process';
import { join, basename, extname } from 'node:path';
import { readGLB } from '../training/glb.mjs';
import { readOBJ } from './load.mjs';
import { buildMesh, seamsFromUV, UnionFind } from '../src/core/mesh.js';
import { computeMetrics, seamAgreement, uvSeamsAndCharts } from '../src/core/metrics.js';
import { evaluate } from '../src/core/predict.js';
import weights from '../src/weights.js';
import { METHODS } from './methods.mjs';

// ------------------------------------------------------------------ args
const argv = process.argv.slice(2);
const opt = (name, def) => { const i = argv.indexOf('--' + name); return i >= 0 ? argv[i + 1] : def; };
const params = {};
argv.forEach((a, i) => {
  if (a !== '--param') return;
  const [k, v] = argv[i + 1].split('=');
  params[k] = v === 'true' ? true : v === 'false' ? false : isNaN(+v) ? v : +v;
});
const dataDir = opt('data', 'training/models');
const methods = opt('methods', 'ours,dihedral,artistSeams,artistUV,xatlas').split(',');
const filter = new RegExp(opt('filter', '.'));
const maxFaces = +opt('max-faces', 150000);
const overlapRes = +opt('overlap-res', 1024);
const MIN_REF_FACES_PER_CHART = 60; // same rule as the training-data filter
const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
const outDir = opt('out', join('results', stamp));
for (const m of methods) if (!METHODS[m]) { console.error(`unknown method "${m}". available: ${Object.keys(METHODS).join(', ')}`); process.exit(2); }

// ------------------------------------------------------------------ loading
function meshStats(mesh) {
  const uf = new UnionFind(mesh.nF);
  for (let e = 0; e < mesh.nE; e++) if (mesh.edgeF[2 * e + 1] >= 0) uf.union(mesh.edgeF[2 * e], mesh.edgeF[2 * e + 1]);
  const comps = new Set(); for (let f = 0; f < mesh.nF; f++) comps.add(uf.find(f));
  const vuf = new UnionFind(mesh.nV); const bv = new Set();
  for (let e = 0; e < mesh.nE; e++) if (mesh.boundary[e]) { vuf.union(mesh.edgeV[2 * e], mesh.edgeV[2 * e + 1]); bv.add(mesh.edgeV[2 * e]); bv.add(mesh.edgeV[2 * e + 1]); }
  const loops = new Set([...bv].map(v => vuf.find(v))).size;
  const chi = mesh.nV - mesh.nE + mesh.nF;
  return {
    vertices: mesh.nV, components: comps.size, boundaryLoops: loops,
    genus: Math.max(0, (2 * comps.size - chi - loops) / 2), // total genus over components (approx. for non-manifold input)
    nonManifoldEdges: mesh.nonManifold.reduce((a, b) => a + b, 0),
  };
}

// ------------------------------------------------------------------ run
const trained = new Set(weights.meta.trainedOn || []);
const files = readdirSync(dataDir).filter(f => /\.(glb|obj)$/i.test(f) && filter.test(f)).sort();
const rows = [];
let gitCommit = 'unknown';
try { gitCommit = execSync('git rev-parse --short HEAD', { stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim() + (execSync('git status --porcelain', { stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim() ? '-dirty' : ''); } catch { /* not a git checkout */ }

for (const file of files) {
  let inputs;
  try { inputs = extname(file).toLowerCase() === '.obj' ? readOBJ(join(dataDir, file)) : readGLB(join(dataDir, file)); }
  catch (e) { console.log(`skip ${file}: ${e.message}`); continue; }
  const model = file.replace(/\.[^.]+$/, '');
  for (const inp of inputs) {
    const mesh = buildMesh(inp.positions, inp.indices, inp.cornerUV);
    if (mesh.nF < 20 || mesh.nF > maxFaces) continue;
    const labels = inp.cornerUV ? seamsFromUV(mesh) : null;
    const ctx = { labels };
    // derived meshes are named <sourceModel>__<mesh>__<variant>; the split follows the source model
    const srcModel = model.split('__')[0];
    if (inp.labelsOnly) mesh.labelsOnly = true;
    const base = { model, mesh: inp.name, source: srcModel, split: trained.has(srcModel) ? 'train' : 'unseen', faces: mesh.nF, ...meshStats(mesh) };
    // Reference quality: seam metrics are only meaningful against a hand-made layout with real seams.
    // Auto-generated atlases (tiny charts) and single-island layouts are recorded but not scored.
    if (labels) {
      const ref = uvSeamsAndCharts(mesh, mesh.cornerUV);
      let refSeams = 0; for (let e = 0; e < mesh.nE; e++) if (labels[e] === 1) refSeams++;
      base.refCharts = ref.charts; base.refSeams = refSeams;
      base.refFacesPerChart = +(mesh.nF / ref.charts).toFixed(1);
      base.reference = refSeams === 0 ? 'no-seams' : base.refFacesPerChart < MIN_REF_FACES_PER_CHART ? 'auto-atlas' : 'usable';
    } else base.reference = 'none';
    const scoreSeams = base.reference === 'usable';
    for (const method of methods) {
      const row = { ...base, method };
      const t0 = performance.now();
      try {
        const res = await METHODS[method](mesh, ctx, { params });
        row.timeMs = +(performance.now() - t0).toFixed(1);
        if (res.skip) { row.status = 'skipped'; row.error = res.skip; rows.push(row); continue; }
        if (res.error) throw new Error(res.error);
        Object.assign(row, computeMetrics(mesh, res.cornerUV, { overlapRes }), res.extra || {});
        row.chartsPer1kFaces = row.charts / mesh.nF * 1000;
        if (scoreSeams) {
          row.chartsVsArtist = row.charts / base.refCharts;
          const { seam } = uvSeamsAndCharts(mesh, res.cornerUV);
          Object.assign(row, seamAgreement(mesh, seam, labels));
          if (res.prob) row.seamAUC = evaluate(res.prob, labels).auc;
        }
        row.status = row.valid ? 'ok' : 'invalid';
      } catch (e) {
        row.timeMs = +(performance.now() - t0).toFixed(1);
        row.status = 'error'; row.error = String(e.message || e);
      }
      rows.push(row);
      console.log(`${row.status.padEnd(7)} ${method.padEnd(11)} ${model}/${inp.name} F=${mesh.nF} charts=${row.charts ?? '-'} overlap=${row.overlapRatio?.toFixed(4) ?? '-'} angleP95=${row.angleDistP95?.toFixed(3) ?? '-'} ${row.seamF1Tol !== undefined ? 'seamF1tol=' + row.seamF1Tol.toFixed(3) : ''} ${row.error || ''}`);
    }
  }
}

// ------------------------------------------------------------------ outputs
mkdirSync(outDir, { recursive: true });
const cols = [...new Set(rows.flatMap(r => Object.keys(r)))];
const esc = v => (v === undefined || v === null ? '' : typeof v === 'number' ? (Number.isInteger(v) ? v : +v.toPrecision(6)) : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : v);
writeFileSync(join(outDir, 'results.csv'), [cols.join(','), ...rows.map(r => cols.map(c => esc(r[c])).join(','))].join('\n') + '\n');

const median = a => { const s = a.filter(x => isFinite(x)).sort((x, y) => x - y); return s.length ? s[s.length >> 1] : NaN; };
const mean = a => { const s = a.filter(x => isFinite(x)); return s.length ? s.reduce((x, y) => x + y, 0) / s.length : NaN; };
const KEYS = ['charts', 'chartsPer1kFaces', 'chartsVsArtist', 'seamLengthNorm', 'areaDistMean', 'areaDistP95', 'angleDistMean', 'angleDistP95', 'overlapRatio', 'flips', 'utilization', 'seamF1', 'seamF1Tol', 'seamAUC', 'timeMs'];
const summary = { date: new Date().toISOString(), gitCommit, node: process.version, dataDir, methods, params, maxFaces, overlapRes, meshes: new Set(rows.map(r => r.model + '/' + r.mesh)).size, byMethod: {} };
for (const method of methods) {
  for (const split of ['all', 'unseen', 'train']) {
    const rs = rows.filter(r => r.method === method && r.status !== 'skipped' && (split === 'all' || r.split === split));
    if (!rs.length) continue;
    const s = { n: rs.length, nScored: rs.filter(r => r.reference === 'usable').length, validRate: rs.filter(r => r.status === 'ok').length / rs.length, errors: rs.filter(r => r.status === 'error').length };
    for (const k of KEYS) { const vals = rs.map(r => r[k]).filter(v => v !== undefined); if (vals.length) s[k] = { median: median(vals), mean: mean(vals) }; }
    (summary.byMethod[method] ||= {})[split] = s;
  }
}
writeFileSync(join(outDir, 'summary.json'), JSON.stringify(summary, null, 1));

const fmt = v => (v === undefined || !isFinite(v) ? '–' : Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(3));
const md = [`# UV benchmark ${summary.date}`, '', `commit \`${gitCommit}\` · data \`${dataDir}\` · ${summary.meshes} meshes · params ${JSON.stringify(params)}`, '',
  'Medians over meshes (unseen = not in the seam predictor\'s training set). valid = flipped area ≤ 0.01%, no out-of-range UVs, overlap ≤ 0.1%.',
  'Seam metrics and charts-vs-artist use only meshes whose artist layout is hand-made (reference = usable, n scored). artistUV often fails "valid" by design (tiling, mirrored overlaps).', ''];
for (const split of ['unseen', 'all']) {
  md.push(`## ${split}`, '', '| method | n | valid | charts / 1k faces | charts ÷ artist | seam len | area dist P95 | angle dist P95 | overlap | n scored | seam F1 (tol 1%) | seam AUC | time ms |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|');
  for (const method of methods) {
    const s = summary.byMethod[method]?.[split]; if (!s) continue;
    md.push(`| ${method} | ${s.n} | ${(s.validRate * 100).toFixed(0)}% | ${fmt(s.chartsPer1kFaces?.median)} | ${fmt(s.chartsVsArtist?.median)} | ${fmt(s.seamLengthNorm?.median)} | ${fmt(s.areaDistP95?.median)} | ${fmt(s.angleDistP95?.median)} | ${fmt(s.overlapRatio?.median)} | ${s.nScored} | ${fmt(s.seamF1Tol?.median)} | ${fmt(s.seamAUC?.median)} | ${fmt(s.timeMs?.median)} |`);
  }
  md.push('');
}
writeFileSync(join(outDir, 'summary.md'), md.join('\n'));
console.log(`\nwrote ${outDir}/results.csv, summary.json, summary.md`);
console.log(md.join('\n'));
