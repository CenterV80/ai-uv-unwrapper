// Renders the report images from tools/report/collect.mjs output with headless Chromium.
//   node tools/report/render.mjs --in <dir>   (needs `node serve.mjs` running on PORT, default 5173)
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const { chromium } = require('playwright');

const argv = process.argv.slice(2);
const IN = argv[argv.indexOf('--in') + 1];
const PORT = process.env.PORT || 5173;
const index = JSON.parse(readFileSync(join(IN, 'index.json'), 'utf8'));
mkdirSync(join(IN, 'img'), { recursive: true });

const browser = await chromium.launch({ args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage();
await page.goto(`http://localhost:${PORT}/tools/report/render.html`);
await page.waitForFunction(() => window.ready === true);
const save = (name, url) => writeFileSync(join(IN, 'img', name), Buffer.from(url.split(',')[1], 'base64'));

for (const entry of index) {
  const tag = `${entry.case}-${entry.domain}`;
  const geo = JSON.parse(readFileSync(join(IN, tag + '.json'), 'utf8'));
  await page.evaluate(g => { window.G = g; }, geo);
  save(`${tag}-ref.webp`, await page.evaluate(() => window.render3D(window.G, [{ edges: window.G.refSeams, kind: 'ref' }])));
  if (geo.refUV) save(`${tag}-ref-uv.webp`, await page.evaluate(() => window.renderUV(window.G.refUV, window.G.refFaceChart)));
  for (const m of entry.methods) {
    save(`${tag}-${m.id}.webp`, await page.evaluate(id => {
      const r = window.G.methods[id];
      return window.render3D(window.G, [{ edges: r.missed, kind: 'missed' }, { edges: r.extra, kind: 'extra' }, { edges: r.match, kind: 'match' }]);
    }, m.id));
    save(`${tag}-${m.id}-uv.webp`, await page.evaluate(id => window.renderUV(window.G.methods[id].uv, window.G.methods[id].faceChart), m.id));
  }
  console.log('rendered', tag);
}
await browser.close();
