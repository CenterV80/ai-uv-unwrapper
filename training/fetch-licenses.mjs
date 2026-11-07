// Records the license of every model in training/models/ (from glTF-Sample-Assets metadata.json)
// into training/model-licenses.json. train.mjs only uses models whose licenses allow commercial use.
// Usage: node training/fetch-licenses.mjs
import { readdirSync, writeFileSync } from 'node:fs';

const dir = new URL('./models/', import.meta.url).pathname;
const base = 'https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models';
const out = {};
for (const file of readdirSync(dir).filter(f => f.endsWith('.glb')).sort()) {
  const model = file.replace(/\.glb$/, '');
  try {
    const res = await fetch(`${base}/${encodeURIComponent(model)}/metadata.json`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const meta = await res.json();
    out[model] = {
      name: meta.name || model,
      source: `https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/${model}`,
      legal: (meta.legal || []).map(l => ({
        license: l.license, what: l.what, artist: l.artist, owner: l.owner, year: l.year, licenseUrl: l.licenseUrl,
      })),
    };
    console.log(model.padEnd(30), [...new Set(out[model].legal.map(l => l.license))].join('; '));
  } catch (e) {
    out[model] = { name: model, legal: [], error: String(e.message || e) };
    console.log(model.padEnd(30), 'NO METADATA -', e.message);
  }
}
writeFileSync(new URL('./model-licenses.json', import.meta.url), JSON.stringify(out, null, 1) + '\n');
console.log('wrote training/model-licenses.json');
