#!/bin/bash
# 実験83: 足し合わせの相手を増やす（seed 0）。担当 4・直前 4 文字、担当 2・直前 8 文字、光路 1 本を出口の面で読むもの。
#   実験81 の 3 つ（光路 1 本・担当 4 直前 8・担当 8 直前 4）と合わせて足し合わせ、テスト全部で測る（exp81_fuse.py）
cd "$(dirname "$0")/../../.."
export PYTHONIOENCODING=utf-8 FUSE_OUT=results/exp83_fuse.txt
R="python experiments/text/exp80_split4f.py --device cuda --out results/exp83_split.txt"
F="python experiments/text/exp81_fuse.py"
D=results/logits
{
$R --n 4 --G 4 --stages 4 --steps 9000 --seed 0 --save_logits $D/g4_n4.pt
$R --n 8 --G 2 --stages 4 --steps 9000 --seed 0 --save_logits $D/g2_n8.pt
python experiments/text/exp78_text4f.py --enc spread --stages 6 --width 8 --steps 9000 --read plane --seed 0 --device cuda --out results/exp83_single.txt --save_logits $D/single_6x8_plane.pt
for f in g4_n4 g2_n8 single_6x8_plane; do $F $D/single_6x8.pt $D/$f.pt; done
$F $D/single_6x8.pt $D/g4_n8.pt $D/g8_n4.pt $D/g4_n4.pt
$F $D/single_6x8.pt $D/single_6x8_plane.pt $D/g4_n8.pt $D/g8_n4.pt
} >> results/sweep83.log 2>&1
