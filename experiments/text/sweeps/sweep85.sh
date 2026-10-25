#!/bin/bash
# 実験85: 実験84（単語、4f 系の光路 1 本 + 分担の 2DTG の足し合わせ）を別の seed で確かめる。使い方: bash sweep85.sh <seed>
cd "$(dirname "$0")/../../.."
s=${1:?seed}
export PYTHONIOENCODING=utf-8 FUSE_OUT=results/exp85_fuse.txt
R="python experiments/text/exp84_word4f.py --device cuda --seed $s --out results/exp85.txt"
F="python experiments/text/exp81_fuse.py"
D=results/logits
{
$R --G 1 --stages 6 --width 8 --steps 9000 --save_logits $D/w_single_6x8_s$s.pt
$R --G 16 --n 2 --stages 4 --width 4 --steps 3000 --save_logits $D/w_g16_n2_s$s.pt
$R --G 8 --n 2 --stages 4 --width 4 --steps 6000 --save_logits $D/w_g8_n2_s$s.pt
$F $D/w_single_6x8_s$s.pt $D/w_g16_n2_s$s.pt
$F $D/w_single_6x8_s$s.pt $D/w_g16_n2_s$s.pt $D/w_g8_n2_s$s.pt
} >> results/sweep85.log 2>&1
