#!/bin/bash
# 実験54: 関連度グレイン（持ち主の案）。実験53 が終わってから GPU を使う
cd "$(dirname "$0")/../../.."
until [ -f results/exp53.txt ] && [ $(wc -l < results/exp53.txt) -ge 3 ]; do sleep 30; done
export PYTHONIOENCODING=utf-8
for seed in 0 1 2; do
  python experiments/text/exp54_related.py --seed $seed --device cuda
done > results/sweep54.log 2>&1
