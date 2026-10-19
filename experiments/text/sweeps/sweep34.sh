#!/bin/bash
# 実験34: 光の記憶の 2 次元版（α = 0.3・0.5、板 ① 8 枚、3 seed）。実験33 が終わってから始める
cd "$(dirname "$0")/../../.."
until [ -f results/exp33.txt ] && [ $(wc -l < results/exp33.txt) -ge 6 ]; do sleep 30; done
for seed in 0 1 2; do
  for al in 0.5 0.3; do
    python experiments/text/exp34_memory2d.py --alpha $al --seed $seed --device cuda
  done
done > results/sweep34.log 2>&1
