#!/bin/bash
# 実験65: 光路を逆向きに通す（持ち主の問い）。実験63 で保存した光路を使う（CPU で計算）
cd "$(dirname "$0")/../../.."
until [ -f results/exp63.txt ] && [ $(wc -l < results/exp63.txt) -ge 6 ]; do sleep 60; done
export PYTHONIOENCODING=utf-8
{
for pl in phase both amp; do
  python experiments/text/exp65_reverse.py --model results/models/exp63_${pl}_int_s0.pt --plate $pl --read int
done
} > results/sweep65.log 2>&1
