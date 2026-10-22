#!/bin/bash
# 実験71: 2DWIG（会話文から話者の一人称）。実験70 の後
cd "$(dirname "$0")/../../.."
until [ -f results/exp70.txt ] && [ $(wc -l < results/exp70.txt) -ge 8 ]; do sleep 60; done
export PYTHONIOENCODING=utf-8
{
for s in 0 1 2; do
  python experiments/text/exp71_wig.py --method plates --seed $s --device cuda
  python experiments/text/exp71_wig.py --method logreg --seed $s --device cuda
done
} > results/sweep71.log 2>&1
