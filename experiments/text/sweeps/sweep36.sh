#!/bin/bash
# 実験36: 板を段に分け、間に光の非線形（明るさで光を出し直す）。位置ごとの明るさつき、区画 8・16、3 seed。実験35 の後に始める
cd "$(dirname "$0")/../../.."
until [ -f results/exp35.txt ] && [ $(wc -l < results/exp35.txt) -ge 6 ]; do sleep 30; done
for seed in 0 1 2; do
  for m in platesNL2 platesNL3; do
    for P in 8 16; do
      python experiments/text/exp31_text2d.py --method $m --recency --patches $P --seed $seed --device cuda
    done
  done
done > results/sweep36.log 2>&1
