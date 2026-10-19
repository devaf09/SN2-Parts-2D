#!/bin/bash
# 実験26: 部品ごとの窓で部品の有無を学習し、例を見せないクラスを部品で当てる。Fashion-MNIST、見せないクラス 4 通り × 3 seed
for seed in 0 1 2; do
  for h in 4 9 7 2; do
    python experiments/exp26_parts.py --holdout $h --seed $seed --device cuda
  done
done
