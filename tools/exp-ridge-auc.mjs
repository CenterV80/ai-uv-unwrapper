// V8b companion: can each predictor *rank* interior seam edges above non-seam edges
// (AUC, threshold-free)? Separates "does not recognise the seam" from "threshold too high".
//   node tools/exp-ridge-auc.mjs [--data data/garments-ridge]
import { readdirSync } from 'node:fs';
import { join } from 'node:path';
import { readOBJ } from './load.mjs';
import { buildMesh, seamsFromUV } from '../src/core/mesh.js';
import { computeEdgeFeatures } from '../src/core/features.js';
import { SeamPredictor } from '../src/core/model.js';
import { predictSeams, evaluate } from '../src/core/predict.js';
import { median } from './statlib.mjs';
import weights from '../src/weights.js';

const DATA = process.argv.includes('--data') ? process.argv[process.argv.indexOf('--data') + 1] : 'data/garments-ridge';
const predictor = SeamPredictor.fromJSON(weights);
const thr = weights.meta.defaultThreshold;
console.log('| ふくらみ | 本手法AUC（内側の縫い目 対 縫い目でない） | 二面角AUC | 本手法の値（内側の縫い目, 中央値） | 本手法の値（縫い目でない, 中央値） | しきい値以上の内側の縫い目 |');
console.log('|---|---|---|---|---|---|');
for (const dir of readdirSync(DATA).filter(d => /^r\d+$/.test(d)).sort()) {
  const aucO = [], aucD = [], pI = [], pN = [], above = [];
  for (const file of readdirSync(join(DATA, dir)).filter(f => f.endsWith('.obj'))) {
    const m = readOBJ(join(DATA, dir, file))[0];
    const pieces = /pieces (.*)$/m.exec(m.header)[1].split(' ').map(x => x.split(':')[0]);
    const mesh = buildMesh(m.positions, m.indices, m.cornerUV);
    const lab = seamsFromUV(mesh), f = computeEdgeFeatures(mesh), { prob } = predictSeams(mesh, predictor, f);
    const chartOf = fc => Math.round(mesh.cornerUV[6 * fc]);
    // labels restricted to interior seams (1) versus non-seams (0); front/back folds ignored
    const l = new Uint8Array(mesh.nE).fill(255);
    let nI = 0, nA = 0; const pi = [], pn = [];
    for (let e = 0; e < mesh.nE; e++) {
      if (lab[e] === 0) { l[e] = 0; pn.push(prob[e]); }
      else if (lab[e] === 1 && pieces[chartOf(mesh.edgeF[2 * e])] === pieces[chartOf(mesh.edgeF[2 * e + 1])]) { l[e] = 1; pi.push(prob[e]); nI++; if (prob[e] >= thr) nA++; }
    }
    if (!nI) continue;
    aucO.push(evaluate(prob, l).auc); aucD.push(evaluate(f.dihedral, l).auc);
    pI.push(median(pi)); pN.push(median(pn)); above.push(nA / nI);
  }
  const h = (+dir.slice(1) / 1000).toString();
  console.log(`| ${h} | ${median(aucO).toFixed(3)} | ${median(aucD).toFixed(3)} | ${median(pI).toFixed(2)} | ${median(pN).toFixed(2)} | ${Math.round(median(above) * 100)}% |`);
}
console.log(`\nしきい値 = ${thr}（アーティストデータで切れ目の量を合わせて決めた値）`);
