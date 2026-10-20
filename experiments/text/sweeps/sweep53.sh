#!/bin/bash
# 実験53: 2DT グレイン（持ち主の案）を 4 つ並べて次の単語。実験52 が終わってから GPU を使う
cd "$(dirname "$0")/../../.."
until [ -f results/exp52.txt ] && [ $(wc -l < results/exp52.txt) -ge 10 ]; do sleep 30; done
export PYTHONIOENCODING=utf-8
for seed in 0 1 2; do
  python experiments/text/exp53_grains.py --seed $seed --device cuda
done > results/sweep53.log 2>&1
