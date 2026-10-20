#!/bin/bash
# 実験52: 単語の予測（直前 4 語 → 次の 1 語、1024 クラス）。実験51 が終わってから GPU を使う
cd "$(dirname "$0")/../../.."
until [ -f results/exp51.txt ] && [ $(wc -l < results/exp51.txt) -ge 18 ]; do sleep 30; done
export PYTHONIOENCODING=utf-8
for seed in 0 1 2; do
  for md in gru logreg plates; do
    python experiments/text/exp52_words.py --method $md --seed $seed --device cuda
  done
done > results/sweep52.log 2>&1
