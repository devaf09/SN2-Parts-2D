#!/bin/bash
# 実験04: 2 次元の tilt 照明（区画どうしの干渉を消す）で、形の課題の位置の雑音を減らせるか
# 照明なしは実験01。tilt:2 = 縦横 2 × 2 の区画で 4 通り（計算 4 倍）、tilt:4 = 4 × 4 で 16 通り（計算 16 倍）
for seed in 0 1 2; do
  for task in shape conj; do
    for m in xy stripes; do
      python experiments/exp01_basic.py --task $task --method $m --seed $seed --illum tilt:2 --device cuda --out results/exp04.txt
    done
  done
done
for seed in 0 1 2; do
  python experiments/exp01_basic.py --task shape --method xy --seed $seed --illum tilt:4 --device cuda --out results/exp04.txt
done
