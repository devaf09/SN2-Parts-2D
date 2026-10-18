#!/bin/bash
# 実験21: 学習の谷の理由（細かい学習曲線・クラスごと・最良の時点・移動平均）。Fashion-MNIST、5000 ステップ、3 seed
for seed in 0 1 2; do
  for s in const cosine; do
    python experiments/exp21_dips.py --method xy --sched $s --seed $seed --device cuda
    python experiments/exp21_dips.py --method sinc --n_ripples 128 --sinc_depth 0.5 --sched $s --seed $seed --device cuda
  done
done
