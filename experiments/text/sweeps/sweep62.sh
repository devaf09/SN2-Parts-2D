#!/bin/bash
# 実験62: 分担して足す 2DTG を文字の課題へ、見せる文脈を長く。実験61 の後
cd "$(dirname "$0")/../../.."
until [ -f results/exp61.txt ] && [ $(wc -l < results/exp61.txt) -ge 2 ]; do sleep 60; done
export PYTHONIOENCODING=utf-8
{
for m in 2 3 4; do
  python experiments/text/exp58_split.py --task char --m $m --G 16 --layers 8 --max_train 600000 --seed 0 --device cuda --out results/exp62.txt
done
python experiments/text/exp58_split.py --task word --m 3 --G 16 --layers 8 --seed 0 --device cuda --out results/exp62.txt
python experiments/text/exp58_split.py --task word --m 4 --G 16 --layers 8 --seed 0 --device cuda --out results/exp62.txt
python experiments/text/exp58_split.py --task char --m 3 --G 64 --layers 2 --max_train 600000 --seed 0 --device cuda --out results/exp62.txt
} > results/sweep62.log 2>&1
