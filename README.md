# SN2-Parts-2D — 波の干渉と回折で 2 次元を処理する部品

画像のような 2 次元の入力を光の場にして、**模様のある透明な板と回折だけ**で処理する部品（Python の関数）を作っています。重み行列（ニューロンのマップ）は持ちません。

> **開発中です（版 0.7.0）。** 合成した図形の分類（M1）、別々に学んだ技能の融合（M2）、手書き数字 MNIST（M3 の始め）まで。

```mermaid
flowchart LR
    I["2 次元の入力<br/>（画像など）"] --> F["光の場<br/>Ly × Lx"]
    F --> P1["Sin の板<br/>①②③"]
    P1 --> D1(("回折"))
    D1 --> P2["Sin の板"]
    P2 --> D2(("… × 16"))
    D2 --> L["レンズ<br/>（2 次元フーリエ変換）"]
    L --> W["スクリーンの窓<br/>一番明るい窓 = 答え"]
```

## これは何？

[Sin-CNN2](https://github.com/devaf09/Sin-CNN2) では、文字列を 1 列の光にして、模様のある板を何枚も通し、スクリーンのどの窓が明るいかで答える学習器 **SinWave** を作りました。学習するのは板の模様だけです。別々に学んだ技能を「足すだけ」で合わせられ、外せば完全に元に戻る、という特徴があります。

このプロジェクトでは、それを **2 次元（縦 × 横の場）** に広げます。板の模様は 3 つの形を用意して比べます。

| 板 | 模様 | ひとこと |
|---|---|---|
| ① 交点で掛け算 | 横の波 + 縦の波 | 軽い。条件が合えば 2 次元の計算が横 1 列 + 縦 1 列に分かれる。代わりに横と縦が混ざらない |
| ② 斜めの縞 | 向きのある縞 | 2 次元の計算が要る |
| ③ 波紋 | 池の波紋・楕円の輪 | ② を含む広い形。光をどこに集めるかを学べる |
| ④ 減衰する波紋（持ち主の案） | 中心が深く外ほど浅い波紋を重ねる | 波紋 1 つは深さ・幅・中心だけ。位置がずれても崩れにくい |
| ⑤ 楕円の減衰する波紋（持ち主の案） | ④ に楕円と届く範囲を足す | |

作った部品は、音の研究 [SN2-Acoustics](https://github.com/devaf09/SN2-Acoustics)（スペクトログラムなどの 2 次元の入力）でも使う予定です。

## いちばん簡単な使い方

必要なもの: Python 3.10 以上、PyTorch 2.0 以上。`sn2_parts_2d.py` を 1 つコピーすれば使えます。

```python
import torch
import sn2_parts_2d as p2

Ly = Lx = 64
y, x = p2.grid(Ly, Lx)
pid = p2.patch_index(Ly, Lx, 4, 4)                     # 4 × 4 の区画
A, phi = torch.rand(16), 6.28 * torch.rand(16)         # 区画ごとの振幅と初期位相（学習する数）
wx, wy = torch.rand(16), torch.rand(16)                # 縞の細かさと向き（固定）
H = p2.transfer(Ly, Lx, z=64.0)                        # 板の間隔 64 画素の回折

u = torch.ones(1, Ly, Lx, dtype=torch.complex64)       # 入力の光の場
u = p2.normalize(p2.propagate(p2.apply_plate(u, p2.plate_stripes(A, wx, wy, phi, pid, y, x)), H))
I = p2.intensity(p2.lens(u))                           # レンズの後のスクリーンの明るさ (1, 64, 64)
```

## 部品の一覧

| 関数 | 何をするか | 確かめたところ |
|---|---|---|
| `image_to_field` | 画像を光の場にする（振幅として・位相として） | 実験01・02（振幅が良い） |
| `Net2D`, `train`, `evaluate` | 1 本の光路（板の形を 1 枚ごとに選べる）、学習、正解率 | 実験01〜 |
| `plate_xy`, `plate_xy_parts` | ① 交点で掛け算の板 | 実験01（いちばん良い）・MNIST 94.4% |
| `plate_stripes` | ② 斜めの縞の板 | 実験01 |
| `plate_ripple`, `ripple_radius` | ③ 波紋の板（等間隔の輪・ゾーンプレート、楕円） | 実験01・03 |
| `plate_sinc` | ④ 減衰する波紋の板（深さ・幅・中心） | 実験11・13 |
| `plate_sinc_ellipse` | ⑤ 楕円で大きさを変えられる減衰する波紋 | 実験14 |
| `illumination`, `Net2D(illum=...)` | 2 次元の照明（区画どうしの干渉を消す） | 実験04（形 71 → 80%） |
| `merge`, `derive`, `combine` | 融合（光路を並べて足す）・取消・足し算 | 実験07・09 |
| `generic_images`, `train(reject=...)` | 「担当なし」の絵を見せる学習（融合の取り違えを防ぐ） | 実験10・12（融合 100%） |
| `grid`, `patch_index`, `band_index` | 場の座標、区画・帯の番号 | テスト |
| `transfer`, `transfer_sep`, `propagate` | 2 次元の回折（角スペクトル法・近軸） | テスト |
| `normalize`, `lens`, `intensity` | 明るさを平均 1 に、レンズ、明るさ | テスト |
| `propagate_sep`, `normalize_sep`, `lens_sep`, `outer` | ① の軽い計算（横 1 列 + 縦 1 列） | テスト・実験00 |
| `window_grid`, `read_windows` | スクリーンの窓を並べ、明るさを読む | テスト |

板の模様（学習前）:

![板の模様](docs/images/plates.png)

学習器を作って学習する例（64 × 64 の画像 `X`（0〜1）とラベル `y`）:

```python
net = p2.Net2D(64, n_classes=8, plates=["xy"] * 16)          # 板 ① を 16 枚
to_field = lambda v: p2.image_to_field(v, "amp")
p2.train(net, X, y, to_input=to_field)
print(p2.evaluate(net, X_test, y_test, to_input=to_field))  # 正解率（%）
```

これから作る部品と順番は [docs/design_m1.md](docs/design_m1.md) にあります。

## ここまでの結果

**本物の画像**（テスト正解率 %、3 seed の平均、実験15）

| 方式 | 学習する数 | 手書き数字 MNIST | 服の絵 Fashion-MNIST |
|---|---|---|---|
| ロジスティック回帰 | 40970 | 92.5 | 84.4 |
| 小さな CNN | 2882 | **97.8** | **86.1** |
| **SinWave 板 ①** | 2059 | **94.4** | 84.4 |

**別々に学んだ技能を、明るさを足すだけで融合**（向き × 位置の合成課題、実験07・10・12）

| 方式 | 2 技能 | 4 技能 |
|---|---|---|
| 足すだけ | 80.8 | 35.1 |
| **「担当なし」の絵も見せて学習 → 足すだけ** | **100.0** | **100.0** |
| 全部を 1 本で同時に学習 | 99.4 | 99.4 |

相手の技能のデータは一切使いません。融合から外せば、残りは元と完全に同じ答えに戻ります（取消）。

**4 つの合成課題**（64 × 64、テスト正解率 %、3 seed の平均、実験01・04・13）

![合成課題の例](docs/images/tasks.png)

| 方式 | 学習する数 | 形 | 縞の向き | 位置（3×3 のマス） | 線の向き × 左右 |
|---|---|---|---|---|---|
| 偶然 | – | 12.5 | 12.5 | 11.1 | 12.5 |
| ロジスティック回帰（画素） | 32776 | 37.2 | 13.8 | 100 | 64.9 |
| 小さな CNN | 2784 | **98.6** | 100 | 99.0 | 100 |
| SinWave 板 ① | 2057 | 70.6 | 99.9 | **99.8** | 99.4 |
| SinWave 板 ① + 2 次元の照明 16 通り | 2057 | 79.9 | | | 99.8 |
| SinWave ④ 波紋 128 個・浅め | 8201 | 76.3 | | | |

**できないこと（いまのところ）**
- 位置が少しずれても同じ形と答えること（形の課題 70〜80%、CNN は 98.6%）
- 小さな CNN に並ぶこと（MNIST で 3.4 点届かない）。計算も重く、学習は CNN の約 16 倍の時間
- 学習し直さずに場の大きさを変えること（実験06）
- アナログで作ったとき、板が 1 画素ずれるとほぼ当てずっぽうになる（実験05。模様の精度は 16 段階で足りる）

**計算量**（実験00、RTX 3060、板 16 枚・学習 1 ステップの 1 件あたり）

| 場の大きさ | 2 次元の場（②③） | ① 横と縦に分けた計算 | メモリ（256 件、2 次元の場） |
|---|---|---|---|
| 64 × 64 | 0.15 ms | 0.11 ms | 0.4 GB |
| 128 × 128 | 0.50 ms | 0.11 ms | 1.5 GB |
| 256 × 256 | 1.96 ms | 0.14 ms | 6.1 GB |
| 参考: 1 次元の SinWave（1024 画素） | 0.06 ms | | 0.1 GB |

① の軽い計算は、入力が「横の模様 × 縦の模様」の形のときだけ使えます（画像は一般にこの形ではありません）。

目標への到達度は [docs/scoreboard.md](docs/scoreboard.md)（今 67.3 / 100 点）。

## しくみ

光（波）は、模様のある板を通ると、模様に合わせて少しずつ向きを変えます。板を何枚も通し、板と板の間で光を広がらせる（回折）と、入力ごとに違う場所に光が集まるようになります。最後にレンズで「同じ向きに進む光」を 1 点に集め、どの窓が一番明るいかで答えます。学習するのは板の模様だけです。

くわしくは、実験の歩み [docs/experiments_guide.md](docs/experiments_guide.md)、研究ノート [NOTES.md](NOTES.md)、1 次元版の解説 [Sin-CNN2 の introduction.md](https://github.com/devaf09/Sin-CNN2/blob/main/docs/introduction.md) を見てください。

## フォルダの構成

| 場所 | 中身 |
|---|---|
| `sn2_parts_2d.py` | 部品（1 ファイル、PyTorch のみ）。先頭に変更履歴と `__version__` |
| `tests/` | 部品の確認 |
| `experiments/expNN_名前.py` | 実験のプログラム（NN = 実験番号）。条件は `experiments/sweeps/sweepNN.sh` |
| `results/expNN.txt` | 結果（1 回の実行 = 1 行の JSON） |
| `experiments/tasks2d.py`, `baselines2d.py` | 合成課題、比較相手（回帰・MLP・小さな CNN） |
| `tools/` | 集計（`summarize.py`）、ポイント表（`scoreboard.py`）、図（`fig_tasks.py`） |
| `docs/` | 説明資料、ポイント表、設計の案 |
| `vendor/` | Sin-CNN2 から写した 1 次元の SinWave（MIT、出典は [vendor/README.md](vendor/README.md)） |
| `NOTES.md` | 実験ノート（数字の正本） |

テストの実行（CPU で数秒）:

```bash
python -m pytest tests -q
```

実験のやり直し（GPU。実験01 は RTX 3060 で約 2 時間）と集計:

```bash
bash experiments/sweeps/sweep01.sh
```

```bash
python tools/summarize.py results/exp01.txt --rows method --cols task
```

## ライセンスとデータの出典

[MIT License](LICENSE)（Copyright (c) 2026 devaf09）。`vendor/` のコードは Sin-CNN2（MIT、同じ著作権者）からの写しです。

データ:
- 合成課題（実験00〜14）はすべてプログラムの中で作った入力です。
- **MNIST**（手書き数字、実験15〜）: Yann LeCun, Corinna Cortes。ライセンス CC BY-SA 3.0。torchvision の置き場から `python tools/fetch_mnist.py mnist` で `data/` に取得します。
- **Fashion-MNIST**（服の絵、実験15〜）: Zalando Research（[github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)）。ライセンス MIT。`python tools/fetch_mnist.py fashion` で取得します。
- データ自体はリポジトリに入れていません（`data/` は .gitignore）。
