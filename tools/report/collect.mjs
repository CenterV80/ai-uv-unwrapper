// Collects everything the visual evaluation report needs for a few representative meshes:
// geometry, reference seams, and for every method its seams classified per edge
// (match / extra / missed), its UV layout, and plain numbers.
//
//   node tools/report/collect.mjs --out <dir>
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { readGLB } from '../../training/glb.mjs';
import { readOBJ } from '../load.mjs';
import { buildMesh, seamsFromUV } from '../../src/core/mesh.js';
import { computeMetrics, seamAgreement, uvSeamsAndCharts } from '../../src/core/metrics.js';
import { METHODS } from '../methods.mjs';

const argv = process.argv.slice(2);
const OUT = argv[argv.indexOf('--out') + 1] || 'results/report';
const VOXEL = 'data/voxel', MODELS = 'training/models';

// representative meshes: unseen by the seam predictor, with a hand-made artist layout
export const CASES = [
  { id: 'duck', title: 'アヒル（Duck）', model: 'Duck', mesh: 'LOD3spShape' },
  { id: 'boombox', title: 'ラジカセ（BoomBox）', model: 'BoomBox', mesh: 'BoomBox' },
  { id: 'truck', title: '牛乳トラックの車体（CesiumMilkTruck）', model: 'CesiumMilkTruck', mesh: 'Cesium_Milk_Truck' },
  { id: 'corset', title: 'コルセット（Corset）', model: 'Corset', mesh: 'pCube49' },
  { id: 'chair', title: '椅子の布部分（SheenChair）', model: 'SheenChair', mesh: 'SheenChair_fabric' },
];
const METHOD_LIST = [
  { id: 'ours', name: '本手法（学習したシーム予測）' },
  { id: 'dihedral', name: '二面角だけ（30°以上の折れ目で切る）' },
  { id: 'xatlas', name: 'xatlas（既存の自動展開ライブラリ）' },
];

function load(c, domain) {
  if (domain === 'original') {
    const m = readGLB(join(MODELS, c.model + '.glb')).find(x => x.name === c.mesh);
    return buildMesh(m.positions, m.indices, m.cornerUV);
  }
  const m = readOBJ(join(VOXEL, `${c.model}__${c.mesh}__v48.obj`))[0];
  const mesh = buildMesh(m.positions, m.indices, m.cornerUV);
  mesh.labelsOnly = true;
  return mesh;
}

const listOf = (cls, k) => { const o = []; for (let e = 0; e < cls.length; e++) if (cls[e] === k) o.push(e); return o; };
const r3 = x => +x.toFixed(3);

mkdirSync(OUT, { recursive: true });
const index = [];
for (const c of CASES) {
  for (const domain of ['original', 'voxel']) {
    const mesh = load(c, domain);
    const labels = seamsFromUV(mesh);
    const ref = uvSeamsAndCharts(mesh, mesh.cornerUV);
    const refEdges = []; for (let e = 0; e < mesh.nE; e++) if (labels[e] === 1) refEdges.push(e);
    const ctx = { labels };
    const entry = {
      case: c.id, title: c.title, domain, faces: mesh.nF, diag: mesh.diag,
      reference: { charts: ref.charts, seamEdges: refEdges.length },
      methods: [],
    };
    const geo = {
      pos: Array.from(mesh.pos, r3), faces: Array.from(mesh.faces), edgeV: Array.from(mesh.edgeV),
      refSeams: refEdges, methods: {},
    };
    // the artist's own layout is shown only for the original meshes (voxel meshes carry seams, not a layout)
    if (domain === 'original') { geo.refUV = Array.from(mesh.cornerUV, r3); geo.refFaceChart = Array.from(ref.faceChart); }
    for (const m of METHOD_LIST) {
      const res = await METHODS[m.id](mesh, ctx, { params: {} });
      const met = computeMetrics(mesh, res.cornerUV);
      const { seam, faceChart } = uvSeamsAndCharts(mesh, res.cornerUV);
      const ag = seamAgreement(mesh, seam, labels, 0.01, true);
      const L = ag.lengths;
      entry.methods.push({
        id: m.id, name: m.name,
        charts: met.charts, valid: met.valid, overlapPct: met.overlapRatio * 100,
        areaDistP95: met.areaDistP95, angleDistP95: met.angleDistP95,
        // plain shares, by seam length
        foundPct: L.ref ? (1 - L.missed / L.ref) * 100 : null,      // of the artist's seams, how much was found
        extraPct: L.pred ? (L.extra / L.pred) * 100 : null,          // of this method's seams, how much is off the artist's
        seamLenVsArtist: L.ref ? L.pred / L.ref : null,               // total cut length relative to the artist
        f1: ag.seamF1Tol,
        // same shares with a 3x wider tolerance (cuts that run parallel just beside the artist's line)
        ...(() => { const w = seamAgreement(mesh, seam, labels, 0.03, true).lengths; return {
          foundPctWide: w.ref ? (1 - w.missed / w.ref) * 100 : null, extraPctWide: w.pred ? (w.extra / w.pred) * 100 : null }; })(),
      });
      geo.methods[m.id] = {
        match: listOf(ag.classes, 1), extra: listOf(ag.classes, 2), missed: listOf(ag.classes, 3),
        uv: Array.from(res.cornerUV, r3), faceChart: Array.from(faceChart),
      };
      console.log(`${c.id}/${domain} ${m.id}: charts ${met.charts} (artist ${ref.charts}), found ${entry.methods.at(-1).foundPct?.toFixed(0)}%, extra ${entry.methods.at(-1).extraPct?.toFixed(0)}%`);
    }
    writeFileSync(join(OUT, `${c.id}-${domain}.json`), JSON.stringify(geo));
    index.push(entry);
  }
}
writeFileSync(join(OUT, 'index.json'), JSON.stringify(index, null, 1));
console.log('wrote', OUT);
