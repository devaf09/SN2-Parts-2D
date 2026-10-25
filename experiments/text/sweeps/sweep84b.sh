#!/bin/bash
# 実験84b: 単語の課題、分担の 2DTG（中身は拡散 + 4f 系 + 平方根、担当は直前の語の番号の余りで決め打ち、seed 0）。
#   最後に 84a の光路 1 本と確率を足し合わせる（84a の対数確率ができるまで待つ）
cd "$(dirname "$0")/../../.."
export PYTHONIOENCODING=utf-8 FUSE_OUT=results/exp84_fuse.txt
R="python experiments/text/exp84_word4f.py --device cuda --seed 0 --stages 4 --width 4"
F="python experiments/text/exp81_fuse.py"
D=results/logits
{
$R --G 4 --n 4 --steps 9000 --save_logits $D/w_g4_n4.pt
$R --G 8 --n 2 --steps 6000 --save_logits $D/w_g8_n2.pt
$R --G 16 --n 2 --steps 3000 --save_logits $D/w_g16_n2.pt
until [ -f $D/w_single_6x8.pt ]; do sleep 60; done
for f in w_g4_n4 w_g8_n2 w_g16_n2; do $F $D/w_single_6x8.pt $D/$f.pt; done
$F $D/w_single_6x8.pt $D/w_g4_n4.pt $D/w_g8_n2.pt
} >> results/sweep84.log 2>&1
