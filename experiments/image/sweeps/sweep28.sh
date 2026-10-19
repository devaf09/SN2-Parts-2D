#!/bin/bash
# 実験28: 4f 系を 2 段、間に光の非線形（明るさで光を出し直す）を入れて、目標の数字とほかの数字の見分けを強くできるか。3 seed
# 実験25 が終わるのを待ってから始める（会話と切り離して起動する）
cd "$(dirname "$0")/../../.."
until [ -f results/exp25.txt ] && [ $(wc -l < results/exp25.txt) -ge 6 ]; do sleep 30; done
for seed in 0 1 2; do
  for m in fourier fourier2nl; do
    python experiments/image/exp27_locate.py --method $m --pos_weight 5 --seed $seed --device cuda --out results/exp28.txt
  done
done > results/sweep28.log 2>&1
