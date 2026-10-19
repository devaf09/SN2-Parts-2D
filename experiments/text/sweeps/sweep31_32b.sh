#!/bin/bash
# 実験31・32 の続き（相関器の読み方を直したあと）。seed 0 は相関器だけ、seed 1・2 は全部
cd "$(dirname "$0")/../../.."
for m in corr1 corrNL2 corrNL3; do
  python experiments/text/exp31_text2d.py --method $m --seed 0 --device cuda
done >> results/sweep31.log 2>&1
for seed in 1 2; do
  for m in gru logreg plates corr1 corrNL2 corrNL3; do
    python experiments/text/exp31_text2d.py --method $m --seed $seed --device cuda
  done
done >> results/sweep31.log 2>&1
