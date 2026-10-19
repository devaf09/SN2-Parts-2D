#!/bin/bash
# 実験24: 部品の符号の和だけで、例を見せないクラスを当てられるか（見せないクラスを変えて）。Fashion-MNIST、3 seed
for seed in 0 1 2; do
  for h in 9 7 2 4; do
    python experiments/image/exp22_codes.py --parts C --holdout $h --seed $seed --device cuda --out results/exp24.txt
  done
done
