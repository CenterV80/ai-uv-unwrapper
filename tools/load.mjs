// Mesh file readers shared by the benchmark and the report tools.
import { readFileSync } from 'node:fs';
import { basename } from 'node:path';

export function readOBJ(path) {
  const V = [], VT = [], idx = [], cuv = [];
  let hasUV = true;
  for (const line of readFileSync(path, 'utf8').split('\n')) {
    const p = line.trim().split(/\s+/);
    if (p[0] === 'v') V.push(+p[1], +p[2], +p[3]);
    else if (p[0] === 'vt') VT.push(+p[1], +p[2]);
    else if (p[0] === 'f') {
      const c = p.slice(1).map(s => s.split('/').map(x => (x ? parseInt(x, 10) : NaN)));
      const vi = c.map(x => (x[0] < 0 ? V.length / 3 + x[0] : x[0] - 1));
      const ti = c.map(x => (x[1] < 0 ? VT.length / 2 + x[1] : x[1] - 1));
      for (let k = 1; k + 1 < c.length; k++) for (const j of [0, k, k + 1]) {
        idx.push(vi[j]);
        if (isNaN(ti[j])) hasUV = false; else cuv.push(VT[2 * ti[j]], VT[2 * ti[j] + 1]);
      }
    }
  }
  // "labels-only" OBJs (tools/voxelize.mjs) carry reference seams in their texture coordinates, not a real layout
  const labelsOnly = /^# seamline: labels-only/m.test(readFileSync(path, 'utf8').slice(0, 200));
  return [{ name: basename(path), positions: Float32Array.from(V), indices: Uint32Array.from(idx), cornerUV: hasUV && cuv.length ? Float32Array.from(cuv) : null, labelsOnly }];
}
