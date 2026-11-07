# Seamline UV — AIシーム予測つきUV展開ツール

ブラウザで動くUV展開ツールです。学習済みモデルが「どこに切れ目（シーム）を入れるべきか」を予測し、
LSCMで展開、形状を考慮したパッキングでテクスチャ空間に詰め込みます。手作業でUV展開済みのモデルを読み込めば、
その切り方を追加学習できます。

研究の課題・評価プロトコル・検証項目は [docs/research-plan.md](docs/research-plan.md) にまとめています。

## 使い方

```bash
npm install
```

```bash
npm run serve
```

`http://localhost:5173/` を開き、OBJ / GLB / glTF(埋め込み) をドロップ（またはサンプルを選択）。

1. **AIシーム予測** — しきい値スライダーで切れ目の量を調整（「AI確率」表示でヒートマップ確認）
2. **展開とパッキング** — 許容面積歪み・パディング・解像度を指定して「UV展開を実行」
3. **シームを手で編集** — 3Dビューのエッジをクリックで「切る／つなぐ」を切り替え（自動で再展開）
4. **学習** — 「このモデルの元UVから学習」または「現在のシームを正解として学習」で、
   自分のスタイルにモデルを適応。重みはブラウザに保存され、JSONで保存・読込も可能
5. **書き出し** — OBJ（UV付き）、GLB（チェッカーテクスチャ付き）、UVレイアウトPNG

## GitHub Pages で公開

`main` に push すると `.github/workflows/pages.yml` が `node build.mjs` で `dist/` を作り、GitHub Pages に公開します。
初回のみ、リポジトリの Settings → Pages → Source を「GitHub Actions」にしてください。

## 仕組み

```
メッシュ → 位相構築(溶接/エッジ/隣接) → エッジ特徴量(22次元) → 2段階シーム予測
       → シーム抽出(しきい値+短い断片の除去) → チャート化
       → [円板でない] 境界ループ間を「シームらしい」最短路でカット / 閉曲面はウォーターシェッド分割
       → LSCM(IC0前処理CG) → [歪み超過] 再分割
       → 等テクセル密度化 → 最小面積回転 → ラスタ形状パッキング(4回転)
```

| ファイル | 役割 |
|---|---|
| `src/core/mesh.js` | 頂点溶接、エッジ/面隣接、元UVからの正解シーム抽出 |
| `src/core/features.js` | 二面角・凸凹・ガウス曲率・多重スケール折れ目・稜線連続性などの特徴量 |
| `src/core/model.js` | Adam付きの小さなニューラルネット（学習・推論、外部ライブラリなし） |
| `src/core/predict.js` | 2段階予測（第2段は近傍エッジの予測値を文脈として使う）、微調整、評価 |
| `src/core/unwrap.js` | シーム→チャート、位相修正カット、分割、全体パイプライン |
| `src/core/lscm.js` | LSCM（不完全コレスキー前処理つき共役勾配法）、歪み評価 |
| `src/core/pack.js` | 形状ラスタライズによるパッキング（棚詰めとの良い方を採用） |
| `src/app.js` | three.js ビューア、UVビュー、編集、学習UI、書き出し |
| `training/` | ベースモデルの学習スクリプト |

コアはthree.js非依存の純JSなので、Node・Web Worker・ブラウザのどこでも動きます。

## ベースモデルの学習

```bash
node training/fetch-licenses.mjs
```

```bash
npm run train
```

学習データは [Khronos glTF-Sample-Assets](https://github.com/KhronosGroup/glTF-Sample-Assets) の
アーティストがUV展開したモデル（`training/models/` に配置。リポジトリには含めていません）。
UVの不連続箇所を正解シームとして抽出し、自動生成アトラス（フォトグラメトリ等）は1チャートあたりの面数で除外しています。

### 学習データのライセンス方針（商用利用のため）

- `fetch-licenses.mjs` が各モデルの `metadata.json` からライセンスを取得し、`training/model-licenses.json` に記録します
- 学習・評価に使うのは **CC0-1.0 と CC-BY-4.0 のモデルだけ**です（`training/licenses.mjs` の `ALLOWED`）。
  非商用（CC-BY-NC）、独自EULA、ライセンス不明のモデルは自動的に除外されます（情報がなければ除外する安全側の設計）
- 学習のたびに、学習に使ったモデルのクレジット一覧 `NOTICE.md` が生成されます。配布時は同梱してください
- 現在の除外: BrainStem（Poser EULA）、DamagedHelmet（CC-BY-NC）、DragonAttenuation（Stanford独自）、Duck（SCEA）

### 評価（学習に使っていない4モデル、9.9万エッジ）

| 予測器 | AUC |
|---|---|
| 二面角だけ（ベースライン） | 0.860 |
| 学習モデル（ベース） | 0.868 |

- 23モデル程度のデータでは、MLPは過学習して二面角単体にすら負けました。そのため**ロジスティック回帰（線形）2段構成**を採用し、
  向きに依存する特徴量（高さ・法線Y・対称面）はベース学習では無効化しています。これらは微調整時に使われ、
  「裏側に縫い目を隠す」などの個人スタイルを学習できます。
- ベースモデル単体の改善幅は小さく、効いているのは「予測確率を経路コストとして使う位相修正カット・分割」と
  「ユーザーデータでの微調整」の組み合わせです。

## 別のPCへの移築

Node.js 18以上（推奨22）があれば Windows / macOS / Linux で動きます。GPUは不要です。

```bash
git clone -b research-eval https://github.com/CenterV80/ai-uv-unwrapper.git
```

```bash
cd ai-uv-unwrapper
```

```bash
npm install
```

```bash
npm run setup:data
```

```bash
npm test
```

- `setup:data` は学習・評価用の glTF サンプル（約3億バイト）と、そのライセンス情報を取得します（raw.githubusercontent.com への接続が必要）
- 画像→3D などで作った自分のメッシュは `data/` フォルダ（Git管理外）に OBJ / GLB で置き、
  `node tools/bench.mjs --data data/<フォルダ名> --methods ours,dihedral,xatlas` で評価します
  （元のUVがないメッシュでは、シーム一致度以外の指標が出ます）
- `results/` と `data/` は `.gitignore` 済みです。**このリポジトリは公開なので、社内データや評価結果はコミットしないでください**

## 研究用の評価（一括ベンチマーク）

```bash
node tools/bench.mjs --data training/models --methods ours,dihedral,artistSeams,artistUV,xatlas
```

`results/<日時>/` に `results.csv`（メッシュ×手法ごと）、`summary.json`（設定・コミットID込み）、`summary.md` を出力します。

| 手法 | 内容 |
|---|---|
| `ours` | 学習シーム予測＋本ツールの分割・展開・パッキング |
| `dihedral` | 同じパイプラインで、シームを二面角30°以上で決める（学習なしの基準） |
| `artistSeams` | 同じパイプラインにアーティストの正解シームを入力（シーム予測の良し悪しだけを切り分ける） |
| `artistUV` | アーティストのUVをそのまま測定（参照値） |
| `xatlas` | xatlas（MIT、WebAssembly版 xatlasjs）既定設定 |

指標（`src/core/metrics.js`）: チャート数、正規化シーム長（÷√表面積）、面積歪み・角度歪み（面積加重の平均と95パーセンタイル）、
反転面（3Dの面の向きをそろえてから判定）、UV重なり率（1024²ラスタ）、範囲外UV、UV使用率、
アーティストシームとの一致（完全一致と、対角線1%以内の測地距離を許容するF1）、シーム予測のAUC。
シーム一致度は、アーティストのUVが手作業の展開と判断できるメッシュ（1チャートあたり60面以上、シームあり）だけで計算します。
自動生成アトラスやシームのないメッシュは `reference` 列に記録し、集計から除外します。
チャート数はサイズの影響を除くため、1000面あたりの数とアーティスト比（÷ artist）でも集計します。
`--param key=value` で展開パラメータを変えて比較実験できます（例: `--param rejectOverlap=false`）。

## 既知の制限

- 大きなメッシュ（数十万面）では展開に数秒〜数十秒かかります（Web Worker内で実行するのでUIは固まりません）
- チャート内のUV自己重なりは検出して分割しますが、許容量（チャート面積の0.5%）以下の重なりは残ります（指標には記録されます）
- Draco / Meshopt 圧縮のglTFは未対応
- 部品単位の分割AI（PartUV が使う PartField など）は、商用利用できないライセンスのものが多いため採用していません
- claude.ai のArtifactとして開いた場合、OBJ/GLBはZIPに包んで保存されます（ビューアの保存形式制限のため）

## テスト

```bash
npm test
```

（`training/models/` に BoomBox.glb などが必要）
