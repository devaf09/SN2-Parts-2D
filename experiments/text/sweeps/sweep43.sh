#!/bin/bash
# 実験43: 正解の窓に光を集める損失を足す（重み 0.1・0.3・1）。帯、板 32 枚 × 区画 16、3 seed。実験42 の後に始める
cd "$(dirname "$0")/../../.."
until [ -f results/exp42.txt ] && [ $(wc -l < results/exp42.txt) -ge 6 ]; do sleep 30; done
for seed in 0 1 2; do
  for fw in 0.3 1 0.1; do
    python experiments/text/exp40_inputs.py --enc bands --focus $fw --seed $seed --device cuda --out results/exp43.txt
  done
done > results/sweep43.log 2>&1
