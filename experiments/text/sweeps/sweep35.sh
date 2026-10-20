#!/bin/bash
# 実験35: 容量を増やすと伸びるか（2 次元の場 + 板 ① + 位置ごとの明るさ）。区画 16（学習する数 約 8 千）・板 32 枚、3 seed
cd "$(dirname "$0")/../../.."
for seed in 0 1 2; do
  python experiments/text/exp31_text2d.py --method plates --recency --patches 16 --seed $seed --device cuda
  python experiments/text/exp31_text2d.py --method plates --recency --patches 16 --layers 32 --seed $seed --device cuda
done > results/sweep35.log 2>&1
