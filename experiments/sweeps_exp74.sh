#!/bin/bash
# 実験74: 分担の 2DTG + 非線形（実験73 で良かったカー効果・平方根・折り返し）を seed 1・2 で確かめる。比べる: 非線形なし 1 段 × 16 枚
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
{
for s in 1 2; do
  python experiments/exp73_nl.py --task charsplit --kind none --stages 1 --per 16 --seed $s --device cuda --out results/exp74.txt
  for k in kerr sqrt fold; do
    python experiments/exp73_nl.py --task charsplit --kind $k --stages 2 --per 8 --seed $s --device cuda --out results/exp74.txt
  done
done
} >> results/sweep74.log 2>&1
