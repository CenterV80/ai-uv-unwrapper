import { readdirSync } from 'node:fs';
import { readGLB } from './glb.mjs';
import { buildMesh, seamsFromUV } from '../src/core/mesh.js';
import { computeEdgeFeatures, FEATURE_NAMES, NF } from '../src/core/features.js';
import { evaluate } from '../src/core/predict.js';
const dir = new URL('./models/', import.meta.url).pathname;
const per = {}; // model -> per-feature auc
for (const file of readdirSync(dir).filter(f => f.endsWith('.glb')).sort()) {
  let meshes; try { meshes = readGLB(dir + file); } catch { continue; }
  const cols = Array.from({length: NF}, () => []), labs = [];
  for (const m of meshes) {
    const mesh = buildMesh(m.positions, m.indices, m.cornerUV); if (mesh.nF < 60) continue;
    const lab = seamsFromUV(mesh); const { X } = computeEdgeFeatures(mesh);
    for (let e = 0; e < mesh.nE; e++) { if (lab[e] === 255) continue; labs.push(lab[e]); for (let j = 0; j < NF; j++) cols[j].push(X[e*NF+j]); }
  }
  if (!labs.length) continue;
  const L = Uint8Array.from(labs);
  per[file] = cols.map(c => evaluate(c, L).auc);
}
console.log('feature'.padEnd(14), Object.keys(per).map(k => k.slice(0,6)).join(' '));
FEATURE_NAMES.forEach((n, j) => console.log(n.padEnd(14), Object.values(per).map(a => a[j].toFixed(2).padStart(6)).join(' ')));
