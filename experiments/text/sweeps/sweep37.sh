#!/bin/bash
# 実験37: さらに大きく（板 64 枚 × 区画 16、板 32 枚 × 区画 32）。位置ごとの明るさつき、3 seed。実験36 の後に始める
cd "$(dirname "$0")/../../.."
until [ -f results/exp36.txt ] && [ $(wc -l < results/exp36.txt) -ge 12 ]; do sleep 30; done
for seed in 0 1 2; do
  python experiments/text/exp31_text2d.py --method plates --recency --layers 64 --patches 16 --seed $seed --device cuda --out results/exp37.txt
  python experiments/text/exp31_text2d.py --method plates --recency --layers 32 --patches 32 --seed $seed --device cuda --out results/exp37.txt
done > results/sweep37.log 2>&1
