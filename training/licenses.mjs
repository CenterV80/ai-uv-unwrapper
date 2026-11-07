// License policy for training data: only licenses that permit commercial use of a trained model.
// Anything else (non-commercial, custom EULAs, unknown, missing metadata) is excluded — fail closed.
import { existsSync, readFileSync, writeFileSync } from 'node:fs';

/** Copyright licenses accepted for training. */
export const ALLOWED = new Set(['CC0-1.0', 'CC-BY-4.0']);
/** Licenses that require attribution in NOTICE.md. */
export const NEEDS_ATTRIBUTION = new Set(['CC-BY-4.0']);
/** Trademark notices ("LegalMark") restrict use of logos, not training on geometry. */
const isTrademarkNotice = id => /^LicenseRef-LegalMark-/.test(id);

export function loadLicenses() {
  const p = new URL('./model-licenses.json', import.meta.url);
  if (!existsSync(p)) throw new Error('training/model-licenses.json がありません。先に node training/fetch-licenses.mjs を実行してください');
  return JSON.parse(readFileSync(p, 'utf8'));
}

/** @returns {{ok: boolean, reason: string}} */
export function checkModel(licenses, model) {
  const entry = licenses[model];
  if (!entry || !entry.legal || !entry.legal.length) return { ok: false, reason: 'ライセンス情報なし' };
  const bad = entry.legal.map(l => l.license).filter(id => !ALLOWED.has(id) && !isTrademarkNotice(id));
  if (bad.length) return { ok: false, reason: `許可外のライセンス: ${[...new Set(bad)].join(', ')}` };
  return { ok: true, reason: '' };
}

/** Write NOTICE.md crediting every training model (CC-BY requires it; CC0 listed for transparency). */
export function writeNotice(licenses, models, path) {
  const lines = [
    '# NOTICE — 学習データのクレジット',
    '',
    '`src/weights.js` のベースモデルは、以下の3Dモデルのジオメトリと UV シームを学習データとして作成しました。',
    'モデルのファイル自体はこのソフトウェアに含まれていません。',
    '出典: [Khronos glTF-Sample-Assets](https://github.com/KhronosGroup/glTF-Sample-Assets)',
    '',
    '| モデル | 対象 | 作者 / 権利者 | 年 | ライセンス |',
    '|---|---|---|---|---|',
  ];
  for (const m of models) {
    const e = licenses[m];
    for (const l of e.legal) {
      if (isTrademarkNotice(l.license)) continue;
      const who = [l.artist, l.owner].filter((x, i, a) => x && a.indexOf(x) === i).join(' / ');
      const url = /^https?:/.test(l.licenseUrl || '') ? `[${l.license}](${l.licenseUrl})` : l.license;
      lines.push(`| [${e.name}](${e.source}) | ${l.what || ''} | ${who} | ${l.year || ''} | ${url} |`);
    }
  }
  lines.push('', '各モデルの商標・ロゴ（LegalMark）は学習に使った形状とは別の権利で、このソフトウェアはそれらを含みません。', '');
  writeFileSync(path, lines.join('\n'));
}
