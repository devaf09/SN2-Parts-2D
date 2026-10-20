#!/bin/bash
# 実験47: 間引き・等価時間標本化（場を 64 → 32 に小さくし、文字の向きは互いに素な歩幅で並べる）。板 16 枚 × 区画 8、1500 ステップ、3 seed
cd "$(dirname "$0")/../../.."
for seed in 0 1 2; do
  for Lf in 64 32; do
    for n in 8 2; do
      python experiments/text/exp40_inputs.py --enc bands --L $Lf --n $n --layers 16 --patches 8 --steps 1500 --seed $seed --device cuda --out results/exp47.txt
    done
  done
  python experiments/text/exp40_inputs.py --enc bands --L 32 --n 8 --layers 32 --patches 8 --steps 1500 --seed $seed --device cuda --out results/exp47.txt
done > results/sweep47.log 2>&1
