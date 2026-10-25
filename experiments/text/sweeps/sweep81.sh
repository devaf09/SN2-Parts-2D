#!/bin/bash
# 実験81: 分担の 2DTG（4f 系）の覚えすぎを抑える形を比べ、光路 1 本（4f 系）と確率を足し合わせる。seed 0
cd "$(dirname "$0")/../../.."
export PYTHONIOENCODING=utf-8
R="python experiments/text/exp80_split4f.py --device cuda --out results/exp81_split.txt"
D=results/logits
{
python experiments/text/exp78_text4f.py --enc spread --stages 6 --width 8 --steps 9000 --seed 0 --device cuda --out results/exp81_single.txt --save_logits $D/single_6x8.pt
$R --n 8 --G 8 --stages 4 --steps 6000 --seed 0 --save_logits $D/g8_n8.pt
$R --n 4 --G 16 --stages 4 --seed 0 --save_logits $D/g16_n4.pt
$R --n 8 --G 4 --stages 4 --steps 9000 --seed 0 --save_logits $D/g4_n8.pt
$R --n 4 --G 16 --stages 3 --steps 1500 --seed 0 --save_logits $D/g16_n4_small.pt
$R --n 4 --G 8 --stages 4 --steps 6000 --seed 0 --save_logits $D/g8_n4.pt
for f in g8_n8 g16_n4 g4_n8 g16_n4_small g8_n4; do python experiments/text/exp81_fuse.py $D/single_6x8.pt $D/$f.pt; done
python experiments/text/exp81_fuse.py $D/single_6x8.pt $D/g8_n8.pt $D/g16_n4.pt
} >> results/sweep81.log 2>&1
