#!/bin/bash
# 実験09: 共通の出発点（縞の向きで学習）から派生させた 2 技能の足し算。板 ①、2 課題 × 3 seed
for seed in 0 1 2; do
  for task in conj shape; do
    python experiments/image/exp09_arith.py --task $task --method xy --seed $seed --device cuda
  done
done
