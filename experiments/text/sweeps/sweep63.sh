#!/bin/bash
# 実験63: 板の種類（位相・振幅・振幅 + 位相）× 読み方（明るさ・位相差も）。実験62 の後
cd "$(dirname "$0")/../../.."
until [ -f results/exp62.txt ] && [ $(wc -l < results/exp62.txt) -ge 6 ]; do sleep 60; done
export PYTHONIOENCODING=utf-8
{
for rd in int int_phase; do
  for pl in phase amp both; do
    python experiments/text/exp63_amp_phase.py --plate $pl --read $rd --seed 0 --device cuda
  done
done
} > results/sweep63.log 2>&1
