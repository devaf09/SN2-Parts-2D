#!/bin/bash
# 実験72: 2DWIG を 1,000 作品（本物だけ / 話者を変えた水増しあり）で学習し直す
cd "$(dirname "$0")/../../.."
export PYTHONIOENCODING=utf-8
{
for s in 0 1 2; do
  for aug in 0 3; do
    python experiments/text/exp71_wig.py --method plates --aug $aug --seed $s --device cuda --out results/exp72.txt
    python experiments/text/exp71_wig.py --method logreg --aug $aug --seed $s --device cuda --out results/exp72.txt
  done
done
python experiments/text/exp71_wig.py --method nb --aug 0 --out results/exp72.txt
python experiments/text/exp71_wig.py --method nb --aug 3 --out results/exp72.txt
} > results/sweep72.log 2>&1
