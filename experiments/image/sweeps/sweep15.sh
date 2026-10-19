#!/bin/bash
# 実験15: MNIST・Fashion-MNIST（64 × 64 の真ん中に 28 × 28）。比較相手と板 ①・④、3 seed。seed 0 の板 ①④ は保存して図にする
for seed in 0 1 2; do
  for data in mnist fashion; do
    for m in logreg cnn lens_logreg xy sinc; do
      sv=""; [ $seed = 0 ] && [ $m = xy -o $m = sinc ] && sv="--save results/models/exp15_${data}_${m}_s0.pt"
      python experiments/image/exp15_mnist.py --data $data --method $m --seed $seed --device cuda $sv
    done
  done
done
