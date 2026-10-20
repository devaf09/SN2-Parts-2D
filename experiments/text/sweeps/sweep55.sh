#!/bin/bash
# 実験55: 1 つの光路が覚えられる対応表の大きさ（直前 1 語 → 次の語 = 2-gram の表）。板の枚数・区画・符号の窓を変える。実験54 の後
cd "$(dirname "$0")/../../.."
until [ -f results/exp54.txt ] && [ $(wc -l < results/exp54.txt) -ge 3 ]; do sleep 30; done
export PYTHONIOENCODING=utf-8
for cfg in "8 16 256" "16 16 256" "32 16 256" "64 16 256" "16 32 256" "16 16 1024" "16 8 256"; do
  set -- $cfg
  python experiments/text/exp52_words.py --method plates --n 1 --layers $1 --patches $2 --K $3 --seed 0 --device cuda --out results/exp55.txt
done > results/sweep55.log 2>&1
