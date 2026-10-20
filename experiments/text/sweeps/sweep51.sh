#!/bin/bash
# 実験51: 未着手の工夫（tilt・アダマール・重み付き照明、非コヒーレント光、参照波）。帯、板 16 枚 × 区画 8、1500 ステップ、3 seed
cd "$(dirname "$0")/../../.."
for seed in 0 1 2; do
  for md in none ref weighted tilt incoh hadamard; do
    python experiments/text/exp51_untried.py --mode $md --seed $seed --device cuda
  done
done > results/sweep51.log 2>&1
