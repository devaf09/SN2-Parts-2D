#!/bin/bash
# 実験38: 板 128 枚 × 区画 16（さらに深く）。実験39: 位置ごとに色を分ける（板 32 枚 × 区画 16）。どちらも位置ごとの明るさつき、3 seed
cd "$(dirname "$0")/../../.."
for seed in 0 1 2; do
  python experiments/text/exp31_text2d.py --method plates --recency --layers 128 --patches 16 --seed $seed --device cuda --out results/exp38.txt
done > results/sweep38.log 2>&1
for seed in 0 1 2; do
  python experiments/text/exp31_text2d.py --method plates --recency --color --layers 32 --patches 16 --seed $seed --device cuda --out results/exp39.txt
done > results/sweep39.log 2>&1
