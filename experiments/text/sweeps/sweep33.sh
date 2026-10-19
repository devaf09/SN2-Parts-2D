#!/bin/bash
# 実験33: 直前の文字ほど明るく（位置ごとの明るさを学習）。板 ① と 4f 系 3 段 + 非線形、3 seed
cd "$(dirname "$0")/../../.."
for seed in 0 1 2; do
  for m in corrNL3 plates; do
    python experiments/text/exp31_text2d.py --method $m --recency --seed $seed --device cuda
  done
done > results/sweep33.log 2>&1
