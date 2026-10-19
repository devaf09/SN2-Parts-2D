#!/bin/bash
# 実験06: 64 × 64 で学習した板を 128 × 128 へ移す（引き伸ばし・敷き詰め）。向き × 位置、板 ①②、3 seed
for seed in 0 1 2; do
  for m in xy stripes; do
    python experiments/image/exp06_scale.py --task conj --method $m --seed $seed --device cuda
  done
done
