#!/bin/bash
# 実験44: 深さ（板 64・128 枚）と符号で読む（窓 256 個）を合わせる。帯、区画 16、3 seed
cd "$(dirname "$0")/../../.."
for seed in 0 1 2; do
  python experiments/text/exp40_inputs.py --enc bands --K 256 --layers 64 --seed $seed --device cuda --out results/exp44.txt
done > results/sweep44.log 2>&1
for seed in 0 1 2; do
  python experiments/text/exp40_inputs.py --enc bands --K 256 --layers 128 --seed $seed --device cuda --out results/exp44.txt
done >> results/sweep44.log 2>&1
