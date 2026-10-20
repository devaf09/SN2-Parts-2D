#!/bin/bash
# 実験41: 文脈を短く（n = 2・3・4、帯が太くなり文字の区別が細かくなる）。実験42: 窓を増やして符号で読む（K = 128・256）。板 32 枚 × 区画 16、3 seed
cd "$(dirname "$0")/../../.."
for seed in 0 1 2; do
  for n in 2 3 4; do
    python experiments/text/exp40_inputs.py --enc bands --n $n --seed $seed --device cuda --out results/exp41.txt
  done
done > results/sweep41.log 2>&1
for seed in 0 1 2; do
  for K in 128 256; do
    python experiments/text/exp40_inputs.py --enc bands --K $K --seed $seed --device cuda --out results/exp42.txt
  done
done > results/sweep42.log 2>&1
