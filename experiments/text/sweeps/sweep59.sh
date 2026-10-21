#!/bin/bash
# 実験59: 厚いホログラムに近づける（板の間隔 z と枚数）。直前 1 語 → 次の語（実験55 と同じ）。実験58 の後
cd "$(dirname "$0")/../../.."
until [ -f results/exp58.txt ] && [ $(wc -l < results/exp58.txt) -ge 4 ]; do sleep 60; done
export PYTHONIOENCODING=utf-8
{
for cfg in "32 4" "32 16" "32 256" "32 1024" "1 64" "128 16" "128 4"; do
  set -- $cfg
  python experiments/text/exp52_words.py --method plates --n 1 --layers $1 --z $2 --seed 0 --device cuda --out results/exp59.txt
done
} > results/sweep59.log 2>&1
