#!/bin/bash
# 実験31・32: 文章を 2 次元の場に載せて次の文字（CC-100 日本語、文脈 8 文字、64 クラス）。3 seed
cd "$(dirname "$0")/../../.."
python experiments/text/exp31_text2d.py --method ngram > results/sweep31.log 2>&1
for seed in 0 1 2; do
  for m in gru logreg plates corr1 corrNL2 corrNL3; do
    python experiments/text/exp31_text2d.py --method $m --seed $seed --device cuda
  done
done >> results/sweep31.log 2>&1
