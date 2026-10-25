#!/bin/bash
# 実験82: 実験81（分担の 2DTG（4f 系）+ 光路 1 本の足し合わせ）を seed 1・2 で確かめる。足し合わせはテストを 2 つに分けて入れ替え、テスト全部で測る
cd "$(dirname "$0")/../../.."
export PYTHONIOENCODING=utf-8 FUSE_OUT=results/exp82_fuse.txt
R="python experiments/text/exp80_split4f.py --device cuda --out results/exp82_split.txt"
F="python experiments/text/exp81_fuse.py"
D=results/logits
{
for s in 1 2; do
  python experiments/text/exp78_text4f.py --enc spread --stages 6 --width 8 --steps 9000 --seed $s --device cuda --out results/exp82_single.txt --save_logits $D/single_6x8_s$s.pt
  $R --n 8 --G 4 --stages 4 --steps 9000 --seed $s --save_logits $D/g4_n8_s$s.pt
  $R --n 4 --G 8 --stages 4 --steps 6000 --seed $s --save_logits $D/g8_n4_s$s.pt
  $F $D/single_6x8_s$s.pt $D/g4_n8_s$s.pt
  $F $D/single_6x8_s$s.pt $D/g8_n4_s$s.pt
  $F $D/single_6x8_s$s.pt $D/g4_n8_s$s.pt $D/g8_n4_s$s.pt
done
} >> results/sweep82.log 2>&1
