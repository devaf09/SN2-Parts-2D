#!/bin/bash
# 実験29: 共通する特徴を引く・相手と比べる（差分で読む）。4f 系 1 段・2 段 + 非線形、3 seed
cd "$(dirname "$0")/../.."
for seed in 0 1 2; do
  for m in fourier fourier2nl; do
    python experiments/exp29_diff.py --method $m --seed $seed --device cuda
  done
done > results/sweep29.log 2>&1
