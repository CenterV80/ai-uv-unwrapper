// Builds a static site into dist/: index.html wrapped in a full document + src/ as-is.
import { cpSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
const root = new URL('.', import.meta.url).pathname;
const dist = root + 'dist/';
rmSync(dist, { recursive: true, force: true });
mkdirSync(dist);
const page = readFileSync(root + 'index.html', 'utf8');
// everything before the first <div> (title, fonts, styles, import map) belongs in <head>
const cut = page.indexOf('<div');
const head = page.slice(0, cut), body = page.slice(cut);
writeFileSync(dist + 'index.html', `<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">${head}</head><body>${body}</body></html>\n`);
cpSync(root + 'src', dist + 'src', { recursive: true });
cpSync(root + 'NOTICE.md', dist + 'NOTICE.md');
writeFileSync(dist + '.nojekyll', '');
console.log('built dist/');
