#!/bin/bash
# 実験27b: 実験27 の 4f 系で、目標の画素の重み（pos_weight）を変える。見分けの強さを AUC でも測る。3 seed
cd "$(dirname "$0")/../../.."
for seed in 0 1 2; do
  for pw in 20 5 1; do
    python experiments/image/exp27_locate.py --method fourier --pos_weight $pw --seed $seed --device cuda --out results/exp27b.txt
  done
done
