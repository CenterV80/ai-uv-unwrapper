// Builds the visual evaluation report page (report.html + img/) from collect.mjs / render.mjs output.
//   node tools/report/build-html.mjs --in <dir>
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const argv = process.argv.slice(2);
const IN = argv[argv.indexOf('--in') + 1];
const index = JSON.parse(readFileSync(join(IN, 'index.json'), 'utf8'));
const meta = JSON.parse(readFileSync(join(IN, 'meta.json'), 'utf8'));

const DOMAIN = { original: '元の形状（アーティストのメッシュ）', voxel: 'ボクセル版（ボクセル化して作り直した形状・解像度48）' };
const SHORT = { ours: '本手法', dihedral: '二面角だけ', xatlas: 'xatlas' };
const pct = x => (x === null || x === undefined ? '–' : `${Math.round(x)}%`);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

function sentence(m, artistCharts) {
  const pieces = m.charts > artistCharts * 1.5 ? `アーティストより細かく${m.charts}枚に切った`
    : m.charts < artistCharts / 1.5 ? `アーティストより粗く${m.charts}枚にしか切らなかった` : `アーティストと近い${m.charts}枚に切った`;
  return `${pieces}（アーティスト${artistCharts}枚）。アーティストの切れ目の<b>${pct(m.foundPct)}</b>を見つけた`
    + `（許容を3倍にすると${pct(m.foundPctWide)}）。この手法の切れ目のうち<b>${pct(m.extraPct)}</b>はアーティストが切っていない場所`
    + `（同${pct(m.extraPctWide)}）。切れ目の総延長はアーティストの${m.seamLenVsArtist.toFixed(1)}倍。`;
}

const cases = [...new Set(index.map(e => e.case))];
const rowsSummary = index.map(e => {
  const cells = e.methods.map(m => `<td class="num">${m.charts}</td><td class="num">${pct(m.foundPct)}<span class="sub"> / ${pct(m.foundPctWide)}</span></td><td class="num">${pct(m.extraPct)}<span class="sub"> / ${pct(m.extraPctWide)}</span></td>`).join('');
  return `<tr><th scope="row">${esc(e.title)}<span class="dom ${e.domain}">${e.domain === 'original' ? '元の形状' : 'ボクセル版'}</span></th><td class="num">${e.reference.charts}</td>${cells}</tr>`;
}).join('\n');

const sections = cases.map(cid => {
  const es = index.filter(e => e.case === cid);
  const blocks = es.map(e => {
    const tag = `${e.case}-${e.domain}`;
    const ref = `<figure class="card ref">
        <figcaption><span class="who">アーティスト（正解）</span><span class="pill">${e.reference.charts}枚</span></figcaption>
        <img src="img/${tag}-ref.webp" alt="${esc(e.title)} ${e.domain === 'original' ? '元の形状' : 'ボクセル版'}：アーティストの切れ目" loading="lazy" width="960" height="480">
        ${e.domain === 'original' ? `<img class="uv" src="img/${tag}-ref-uv.webp" alt="アーティストのUV配置" loading="lazy" width="420" height="420">` : '<p class="note">ボクセル版には正解の切れ目だけを写しており、UV配置はありません。</p>'}
      </figure>`;
    const ms = e.methods.map(m => `<figure class="card">
        <figcaption><span class="who">${SHORT[m.id]}</span><span class="pill">${m.charts}枚</span><span class="pill ${m.valid ? 'ok' : 'bad'}">${m.valid ? 'UV有効' : `UV重なり ${m.overlapPct.toFixed(2)}%`}</span></figcaption>
        <img src="img/${tag}-${m.id}.webp" alt="${esc(e.title)}：${SHORT[m.id]}の切れ目" loading="lazy" width="960" height="480">
        <p class="say">${sentence(m, e.reference.charts)}</p>
        <img class="uv" src="img/${tag}-${m.id}-uv.webp" alt="${SHORT[m.id]}のUV配置" loading="lazy" width="420" height="420">
      </figure>`).join('\n');
    return `<div class="domain"><h3>${DOMAIN[e.domain]}<span class="faces">${e.faces.toLocaleString()}面</span></h3><div class="grid">${ref}${ms}</div></div>`;
  }).join('\n');
  return `<section class="case" id="${cid}"><h2>${esc(es[0].title)}</h2>${blocks}</section>`;
}).join('\n');

const html = `<title>シーム評価レポート</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+JP:wght@400;500;600&family=Zen+Kaku+Gothic+New:wght@700&display=swap">
<style>
/* Layout: a lab report — findings first, then one plate per model: the artist's cut beside each method's cut, colour-coded by agreement. */
:root {
  --bg: #f3f4f7; --paper: #ffffff; --ink: #1b2130; --muted: #5b6577; --line: #d8dce4;
  --match: #2563eb; --missed: #dc2626; --extra: #d97706; --ok: #15803d; --bad: #b42318; --accent: #3f4a8a;
  --f-display: "Zen Kaku Gothic New", "Hiragino Sans", "Yu Gothic", sans-serif;
  --f-body: "IBM Plex Sans JP", "Hiragino Sans", "Yu Gothic", "Noto Sans JP", system-ui, sans-serif;
  --f-mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #12151c; --paper: #1a1f29; --ink: #e5e9f0; --muted: #9aa4b5; --line: #2c3340;
  --match: #6b9bff; --missed: #ff6b6b; --extra: #f5b041; --ok: #4cc483; --bad: #ff8a80; --accent: #a8b3ff; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #12151c; --paper: #1a1f29; --ink: #e5e9f0; --muted: #9aa4b5; --line: #2c3340;
  --match: #6b9bff; --missed: #ff6b6b; --extra: #f5b041; --ok: #4cc483; --bad: #ff8a80; --accent: #a8b3ff; color-scheme: dark; }
[hidden] { display: none !important; }
body { margin: 0; background: var(--bg); color: var(--ink); font-family: var(--f-body); font-size: 15px; line-height: 1.7; }
.wrap { max-width: 1240px; margin: 0 auto; padding-block: 36px 64px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); display: flex; flex-direction: column; gap: 40px; }
header h1 { font-family: var(--f-display); font-size: clamp(26px, 4vw, 36px); margin: 0 0 6px; text-wrap: balance; }
header p { margin: 0; color: var(--muted); max-width: 70ch; }
.meta { font-family: var(--f-mono); font-size: 12px; color: var(--muted); margin-top: 10px; }
h2 { font-family: var(--f-display); font-size: 24px; margin: 0 0 14px; text-wrap: balance; }
h3 { font-size: 15px; font-weight: 600; margin: 18px 0 10px; display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; }
h3 .faces { font-family: var(--f-mono); font-size: 12px; color: var(--muted); font-weight: 400; }
.panel { background: var(--paper); border: 1px solid var(--line); border-radius: 8px; padding: 20px 22px; }
.legend { display: flex; flex-wrap: wrap; gap: 10px 26px; margin: 12px 0 4px; }
.legend span { display: inline-flex; align-items: center; gap: 8px; }
.sw { width: 26px; height: 4px; border-radius: 2px; display: inline-block; }
.sw.match { background: var(--match); } .sw.missed { background: var(--missed); } .sw.extra { background: var(--extra); } .sw.ref { background: var(--ink); }
.findings { margin: 0; padding-left: 1.2em; display: flex; flex-direction: column; gap: 10px; max-width: 80ch; }
.findings b { font-weight: 600; }
.tablewrap { overflow-x: auto; }
table { border-collapse: collapse; font-size: 13px; min-width: 860px; width: 100%; }
th, td { padding: 7px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
thead th { font-weight: 600; color: var(--muted); font-size: 12px; }
thead tr.top th { color: var(--ink); font-size: 13px; border-bottom: 0; }
td.num { font-family: var(--f-mono); font-variant-numeric: tabular-nums; text-align: right; white-space: nowrap; }
.sub { color: var(--muted); }
.dom { display: inline-block; margin-left: 8px; font-size: 11px; padding: 0 7px; border-radius: 99px; border: 1px solid var(--line); color: var(--muted); font-weight: 400; }
.dom.voxel { border-color: var(--accent); color: var(--accent); }
.case { display: flex; flex-direction: column; }
.grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
.card { margin: 0; background: var(--paper); border: 1px solid var(--line); border-radius: 8px; padding: 10px; display: flex; flex-direction: column; gap: 8px; min-width: 0; }
.card.ref { border-color: var(--ink); }
.card img { width: 100%; height: auto; display: block; border-radius: 4px; background: #fff; }
.card img.uv { width: 62%; align-self: center; border: 1px solid var(--line); }
figcaption { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
.who { font-weight: 600; margin-right: auto; }
.pill { font-family: var(--f-mono); font-size: 11px; padding: 0 7px; border-radius: 99px; border: 1px solid var(--line); color: var(--muted); white-space: nowrap; }
.pill.ok { color: var(--ok); border-color: var(--ok); } .pill.bad { color: var(--bad); border-color: var(--bad); }
.say { margin: 0; font-size: 13px; line-height: 1.65; }
.say b { font-family: var(--f-mono); font-weight: 500; }
.note { margin: 0; font-size: 12.5px; color: var(--muted); }
dl.terms { display: grid; grid-template-columns: minmax(8em, max-content) 1fr; gap: 6px 18px; margin: 0; }
dl.terms dt { font-weight: 600; } dl.terms dd { margin: 0; color: var(--ink); max-width: 75ch; }
code { font-family: var(--f-mono); font-size: 13px; }
pre { font-family: var(--f-mono); font-size: 12.5px; background: var(--bg); border: 1px solid var(--line); border-radius: 6px; padding: 10px 12px; overflow-x: auto; }
nav.toc { display: flex; flex-wrap: wrap; gap: 8px; }
nav.toc a { color: var(--accent); text-decoration: none; border: 1px solid var(--line); border-radius: 99px; padding: 2px 12px; font-size: 13px; background: var(--paper); }
nav.toc a:focus-visible, nav.toc a:hover { border-color: var(--accent); }
@media (max-width: 980px) { .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 560px) { .grid { grid-template-columns: minmax(0, 1fr); } dl.terms { grid-template-columns: 1fr; } }
</style>
<div class="wrap">
<header>
  <h1>UV展開の切れ目を、アーティストの切り方と比べる</h1>
  <p>学習に使っていない5つのモデルについて、アーティストが手作業で入れた切れ目（正解）と、3つの自動手法の切れ目を同じ画面で重ねて表示しました。
  同じモデルをボクセル化して作り直した版（画像から3D化したメッシュに近い、角の丸いでこぼこした形）でも比べています。</p>
  <p class="meta">コミット ${esc(meta.commit)} · ${esc(meta.date)} · 許容 = モデル全体の大きさ（対角線）の1%（括弧内は3%）</p>
</header>

<section class="panel" aria-labelledby="h-read">
  <h2 id="h-read">図の読み方</h2>
  <p>各図は同じモデルを前後2方向から見たものです。線の色は、その手法の切れ目がアーティストの切れ目と合っているかを表します。</p>
  <div class="legend">
    <span><i class="sw ref"></i>アーティストの切れ目（正解の図のみ）</span>
    <span><i class="sw match"></i>一致：アーティストと同じ場所を切った</span>
    <span><i class="sw missed"></i>見逃し：アーティストは切ったが、この手法は切らなかった</span>
    <span><i class="sw extra"></i>余計：アーティストが切っていない場所を切った</span>
  </div>
  <p class="note">「同じ場所」とは、切れ目同士の距離がモデル全体の大きさの1%以内であることです。割合はすべて切れ目の<b>長さ</b>で数えています（以前の一括評価のF1はエッジの本数で数えていたため、値が異なります）。下の小さな図は、各手法が作ったUV配置（色ごとに1枚の布）です。</p>
</section>

<section class="panel" aria-labelledby="h-find">
  <h2 id="h-find">分かったこと</h2>
  <ol class="findings">
    <li><b>元の形状では、どの手法もアーティストの切れ目の多くを見つけるが、切りすぎる。</b>
      トラック・コルセット・椅子では、3手法ともアーティストの切れ目の8割以上を見つけています。一方で、切れ目の総延長はアーティストの1.6〜6.7倍あり、布の枚数も多くなります（例：椅子の布部分はアーティスト30枚に対し、本手法66枚、二面角だけ329枚、xatlas 770枚）。</li>
    <li><b>本手法は、他の2手法より切る量が少なく、枚数がアーティストに近い。</b>
      ただし、余計な切れ目の割合は他と大差なく、モデルによって上下します。アヒルでは枚数がアーティストと同じ10枚でしたが、切った場所は大きく違います。羽と尾の切れ目（赤）をほとんど見逃し、代わりに胴体を別の場所（オレンジ）で切っています。</li>
    <li><b>ボクセル版では、アーティストが切った場所の多くを見逃す。</b>
      角や溝がなまって形の手がかりが減るためです（赤が増える）。トラック・コルセット・椅子では、3手法とも切れ目の総延長がアーティストの0.4〜0.9倍に減り、切り足りなくなりました。アヒルとラジカセでは逆に、総延長は同等以上なのに位置が合っていません。</li>
    <li><b>ただし、ボクセル版の「ずれ」の多くは、正しい場所のすぐ横を切っているもの。</b>
      トラックのボクセル版では、本手法の切れ目がアーティストの切れ目のすぐ脇を平行に走っています（図のオレンジと赤の並走）。許容を1%から3%に広げると、本手法が見つけた割合は7%から49%に、余計な割合は85%から18%に変わります。
      前回の報告で「偶然と同じ水準」と書いたのは、判定が厳しすぎたことによる言い過ぎでした。評価の許容幅は、形状の作り方に合わせて決める必要があります。</li>
    <li><b>手法間の差は、モデル間の差より小さい。</b>
      どのモデルが難しいかで結果が大きく変わり、手法の優劣はモデルごとに入れ替わります。5モデルでは、どの手法が良いとは言えません。</li>
  </ol>
</section>

<section class="panel" aria-labelledby="h-table">
  <h2 id="h-table">一覧</h2>
  <p class="note">枚数 = UV上の布の枚数。見つけた = アーティストの切れ目のうち見つけた割合。余計 = その手法の切れ目のうち、アーティストが切っていない場所の割合。「/」の右は許容3%の場合。</p>
  <div class="tablewrap"><table>
    <thead>
      <tr class="top"><th></th><th>アーティスト</th><th colspan="3">本手法</th><th colspan="3">二面角だけ</th><th colspan="3">xatlas</th></tr>
      <tr><th>モデル</th><th>枚数</th>${'<th>枚数</th><th>見つけた</th><th>余計</th>'.repeat(3)}</tr>
    </thead>
    <tbody>
${rowsSummary}
    </tbody>
  </table></div>
</section>

<nav class="toc" aria-label="モデル">${cases.map(c => `<a href="#${c}">${esc(index.find(e => e.case === c).title)}</a>`).join('')}</nav>

${sections}

<section class="panel" aria-labelledby="h-terms">
  <h2 id="h-terms">用語と作り方</h2>
  <dl class="terms">
    <dt>本手法</dt><dd>エッジごとの形の特徴（折れ角、曲率など）から、学習したモデルで切る場所を予測し、LSCMで平らに展開する方法。</dd>
    <dt>二面角だけ</dt><dd>学習を使わず、面同士が30°以上折れている場所だけを切る方法。比較の基準。</dd>
    <dt>xatlas</dt><dd>広く使われている自動UV展開ライブラリ（MITライセンス）を既定の設定で実行したもの。</dd>
    <dt>ボクセル版</dt><dd>アーティストのメッシュを最長辺48分割のボクセルにし、表面を作り直して平滑化したもの。正解の切れ目は、元のメッシュのUVの区切りを最も近い面から写し取った。面単位で写すため、正解の線は少しギザギザになる。</dd>
    <dt>UV有効 / 重なり</dt><dd>展開したUVが自分自身や他の布と重なっていないか。重なりがUV面積の0.1%を超えると無効と表示。</dd>
    <dt>使ったモデル</dt><dd>Khronos glTF-Sample-Assets の Duck（SCEA Shared Source License）、BoomBox・Corset・SheenChair（CC0）、CesiumMilkTruck（CC-BY 4.0、Cesium）。いずれも本手法の学習には使っていない。</dd>
  </dl>
  <p class="note">再現手順（リポジトリ <code>research-eval</code> ブランチ）：</p>
<pre>node tools/voxelize.mjs
node tools/report/collect.mjs --out results/report
node serve.mjs &amp;   # 別のターミナルでも可
node tools/report/render.mjs --in results/report
node tools/report/build-html.mjs --in results/report</pre>
</section>
</div>
`;
writeFileSync(join(IN, 'report.html'), html);
console.log('wrote', join(IN, 'report.html'));
