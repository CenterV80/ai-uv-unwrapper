import { readdirSync } from 'node:fs';
import { readGLB } from './glb.mjs';
import { buildMesh, seamsFromUV, UnionFind } from '../src/core/mesh.js';
const dir = new URL('./models/', import.meta.url).pathname;
for (const f of readdirSync(dir).filter(f => f.endsWith('.glb')).sort()) {
  let meshes;
  try { meshes = readGLB(dir + f); } catch (e) { console.log(f, 'ERR', e.message); continue; }
  let F = 0, E = 0, S = 0, charts = 0, nm = 0;
  for (const m of meshes) {
    const mesh = buildMesh(m.positions, m.indices, m.cornerUV);
    const lab = seamsFromUV(mesh);
    const uf = new UnionFind(mesh.nF);
    for (let e = 0; e < mesh.nE; e++) { if (lab[e] === 0) uf.union(mesh.edgeF[2*e], mesh.edgeF[2*e+1]); if (lab[e]===1) S++; if (lab[e]!==255) E++; }
    const roots = new Set(); for (let i = 0; i < mesh.nF; i++) roots.add(uf.find(i));
    charts += roots.size; F += mesh.nF; nm += mesh.nonManifold.reduce((a,b)=>a+b,0);
  }
  console.log(f.padEnd(34), 'meshes', String(meshes.length).padStart(3), 'faces', String(F).padStart(8), 'seam%', (100*S/Math.max(1,E)).toFixed(1).padStart(5), 'charts', String(charts).padStart(6), 'faces/chart', (F/Math.max(1,charts)).toFixed(0).padStart(6), 'nonman', nm);
}
