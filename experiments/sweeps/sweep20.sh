#!/bin/bash
# 実験20: 親の光路から子の光路を作る木（Fashion-MNIST）。子のデータ 全部・1 クラス 300 枚 × 3 seed
for seed in 0 1 2; do
  for n in 0 300; do
    python experiments/exp20_tree.py --seed $seed --n_per_class $n --device cuda
  done
done
