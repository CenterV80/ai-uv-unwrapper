import { readGLB } from '../training/glb.mjs';
import { buildMesh } from '../src/core/mesh.js';
import { computeEdgeFeatures } from '../src/core/features.js';
import { unwrap } from '../src/core/unwrap.js';
import { SeamPredictor } from '../src/core/model.js';
import { predictSeams } from '../src/core/predict.js';
import weights from '../src/weights.js';
const predictor = SeamPredictor.fromJSON(weights);
const dir = new URL('../training/models/', import.meta.url).pathname;
for (const name of process.argv.slice(2)) {
  for (const m of readGLB(dir + name + '.glb').slice(0, 2)) {
    const mesh = buildMesh(m.positions, m.indices);
    const f = computeEdgeFeatures(mesh);
    const t0 = Date.now();
    const { prob } = predictSeams(mesh, predictor, f);
    const r = unwrap(mesh, prob, f.dihedral, { threshold: weights.meta.defaultThreshold });
    let bad = 0; for (const x of r.cornerUV) if (!(x >= -1e-6 && x <= 1 + 1e-6)) bad++;
    if (bad || !r.stats.charts) process.exitCode = 1;
    console.log(name, m.name, 'F', mesh.nF, JSON.stringify(r.stats, (k, v) => typeof v === 'number' ? +v.toFixed(3) : v), 'outOfRange', bad, (Date.now() - t0) + 'ms');
  }
}
