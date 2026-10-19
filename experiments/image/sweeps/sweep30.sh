#!/bin/bash
# 実験30: 10 種類の数字の位置を同時に出す（4f 系 2 段 + 非線形を数字ごとに 1 本）。3 seed
cd "$(dirname "$0")/../../.."
for seed in 0 1 2; do
  python experiments/image/exp30_ten.py --seed $seed --device cuda
done > results/sweep30.log 2>&1
