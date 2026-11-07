// Checks the procedural garments: openings are where they should be and they unwrap cleanly.
import { GARMENTS } from '../src/garments.js';
import { buildMesh, UnionFind } from '../src/core/mesh.js';
import { computeEdgeFeatures } from '../src/core/features.js';
import { unwrap } from '../src/core/unwrap.js';
import { SeamPredictor } from '../src/core/model.js';
import { predictSeams } from '../src/core/predict.js';
import weights from '../src/weights.js';

const EXPECTED_OPENINGS = { tshirt: 4, pants: 3, skirt: 2, beanie: 1 }; // neck+hem+2 cuffs, waist+2 ankles, ...
const predictor = SeamPredictor.fromJSON(weights);
let failed = false;
for (const [name, make] of Object.entries(GARMENTS)) {
  const { positions, indices } = make();
  const mesh = buildMesh(positions, indices);
  // boundary loops = openings
  const uf = new UnionFind(mesh.nV); const onB = new Set();
  for (let e = 0; e < mesh.nE; e++) if (mesh.boundary[e]) { uf.union(mesh.edgeV[2 * e], mesh.edgeV[2 * e + 1]); onB.add(mesh.edgeV[2 * e]); }
  const loops = new Set([...onB].map(v => uf.find(v))).size;
  const comps = new UnionFind(mesh.nF);
  for (let e = 0; e < mesh.nE; e++) if (mesh.edgeF[2 * e + 1] >= 0) comps.union(mesh.edgeF[2 * e], mesh.edgeF[2 * e + 1]);
  const pieces = new Set(Array.from({ length: mesh.nF }, (_, f) => comps.find(f))).size;
  const nonManifold = mesh.nonManifold.reduce((a, b) => a + b, 0);
  const f = computeEdgeFeatures(mesh);
  const { prob } = predictSeams(mesh, predictor, f);
  const r = unwrap(mesh, prob, f.dihedral, { threshold: weights.meta.defaultThreshold });
  let bad = 0; for (const x of r.cornerUV) if (!(x >= -1e-6 && x <= 1 + 1e-6)) bad++;
  const ok = loops === EXPECTED_OPENINGS[name] && pieces === 1 && nonManifold === 0 && bad === 0 && r.stats.flips === 0;
  if (!ok) failed = true;
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${name.padEnd(7)} faces ${String(mesh.nF).padStart(5)} openings ${loops}/${EXPECTED_OPENINGS[name]} pieces ${pieces} nonManifold ${nonManifold} | charts ${r.stats.charts} distortion ${r.stats.distortion.toFixed(3)} flips ${r.stats.flips} packing ${(r.stats.packingEfficiency * 100).toFixed(0)}%`);
}
process.exitCode = failed ? 1 : 0;
