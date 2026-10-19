#!/bin/bash
# 実験27: 光学相関器（4f 系）で数字の位置に光を集める。MNIST の 2 個並べ、3 と 7 の光路、3 seed
cd "$(dirname "$0")/../../.."
for seed in 0 1 2; do
  for m in fourier fourier4 plates; do
    python experiments/image/exp27_locate.py --method $m --seed $seed --device cuda
  done
done
