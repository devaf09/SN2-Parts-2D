# experiments/text/ — 2 次元で言語を扱う実験（実験31〜65）

画像認識（`experiments/image/`、実験00〜30）の区切りのあと、2 次元の場で次の文字・単語の予測を試した実験。
実験番号は画像と通し。数字の正本は ../../NOTES.md、一番良かった組み合わせは ../../docs/best_configs.md、目的と実験の一覧は ../../docs/plan_text.md。
実行はリポジトリ直下から（例: `python experiments/text/exp58_split.py --task char --m 4 --G 16 --layers 8 --max_train 600000`）。条件は `sweeps/sweepNN.sh`、結果は `results/expNN.txt`（1 回 = 1 行の JSON）。

## データ
| ファイル | 内容 |
|---|---|
| textdata.py | 文字の課題（Sin-CNN2 の exp06 と同じ作り方）。CC-100 日本語 約 60 万字、直前 n 文字 → 次の文字、64 クラス |
| worddata.py | 単語の課題。MeCab（fugashi + unidic-lite）で単語に切る。直前 n 語 → 次の語、1024 クラス。語を意味の近さで並べる embed_layout（実験60） |

## 実験のプログラム
| ファイル | 実験 | 内容 |
|---|---|---|
| exp31_text2d.py | 31〜33・35〜39 | 文字を帯（縦 = 位置、横 = 文字の向きの平面波）に入れる光路、4f 系、GRU・回帰・n-gram |
| exp34_memory2d.py | 34 | 光の記憶の 2 次元版 |
| exp40_inputs.py | 40〜44・46・47・49 | 入れ方 6 通り（EncNet。以後の光路の土台）、符号読み、場の大きさ |
| exp45_modulate.py | 45 | 文字を板として掛け算で通す |
| exp48_strip.py | 48 | 短冊形（持ち主の案） |
| exp50_bounce.py | 50 | 鏡の往復（持ち主の案） |
| exp51_untried.py | 51 | tilt・アダマール・重み付き照明、非コヒーレント光、参照波 |
| exp52_words.py | 52・55〜57・59・60・64 | 単語の光路 1 本（--L 場、--z 板の間隔、--layout 語の並べ方、--fixed_codes 符号の固定） |
| exp53_grains.py | 53 | 2DT グレイン（k 語先）を並べて足す（持ち主の案） |
| exp54_related.py | 54 | 関連度グレイン（持ち主の案） |
| **exp58_split.py** | 58・62・64 | **分担の 2DTG を足すだけで融合**（--task word/char、--m 見せる単位、--G 個数、--fixed_codes） |
| exp61_tree.py | 61 | 2DTG の木（葉・枝・根） |
| exp63_amp_phase.py | 63 | 板の種類（位相・振幅・両方）× 読み方（明るさ・位相差も） |
| exp65_reverse.py | 65 | 光路を逆向きに通す |

## 数えた表での見積もり（光路を使わない、案の上限を手早く測る）
| ファイル | 内容 |
|---|---|
| est_grains_counts.py | 2DTG を足す方法（1 語だけ見る・2 語を見る） |
| est_relatedness_counts.py | 関連度（共起 → PPMI → 64 次元）で確率を掛け直す |
| est_analogy_counts.py | 差分（男 → 女 = 父 → ?） |
| est_product_counts.py | 2DTG の出力の掛け算（専門家の積） |
