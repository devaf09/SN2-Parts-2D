#!/bin/bash
# 実験60: 使われ方の近い語を近い光の向きに並べる（持ち主の「差分・関連性の補完」の見立て）。実験59 の後
cd "$(dirname "$0")/../../.."
until [ -f results/exp59.txt ] && [ $(wc -l < results/exp59.txt) -ge 7 ]; do sleep 60; done
export PYTHONIOENCODING=utf-8
{
for mt in 20000 60000; do
  for lo in stride embed embed_rand; do
    python experiments/text/exp52_words.py --method plates --layout $lo --max_train $mt --seed 0 --device cuda --out results/exp60.txt
  done
done
} > results/sweep60.log 2>&1
