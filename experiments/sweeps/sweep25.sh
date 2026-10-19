#!/bin/bash
# 実験25: 実験20（親から子の木）を、学習率を 0 まで下げ、子の学習を 4000 ステップにして測り直す。Fashion-MNIST、3 seed
for seed in 0 1 2; do
  for n in 0 300; do
    python experiments/exp20_tree.py --seed $seed --n_per_class $n --sched cosine --steps_child 4000 --device cuda --out results/exp25.txt
  done
done
