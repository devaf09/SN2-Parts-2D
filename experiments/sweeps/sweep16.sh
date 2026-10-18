#!/bin/bash
# 実験16: 学習曲線（最大 10000 ステップ）。MNIST・Fashion-MNIST × (回帰・小さな CNN・板 ①・④ 波紋 128 個 深さ 0.5) × 3 seed
for seed in 0 1 2; do
  for data in mnist fashion; do
    python experiments/exp16_curve.py --data $data --method logreg --seed $seed --device cuda
    python experiments/exp16_curve.py --data $data --method cnn --seed $seed --device cuda
    python experiments/exp16_curve.py --data $data --method xy --seed $seed --device cuda
    python experiments/exp16_curve.py --data $data --method sinc --n_ripples 128 --sinc_depth 0.5 --seed $seed --device cuda
  done
done
