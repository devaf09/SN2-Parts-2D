#!/bin/bash
# 実験67 の準備: 位相差も読む光路（位相だけの板）を seed 1〜3 で学習（逆向きのぼんやりを固めるため）。実験66 の後
cd "$(dirname "$0")/../../.."
until [ -f results/exp66.txt ] && [ $(wc -l < results/exp66.txt) -ge 3 ]; do sleep 60; done
export PYTHONIOENCODING=utf-8
{
for s in 1 2 3; do
  python experiments/text/exp63_amp_phase.py --plate phase --read int_phase --seed $s --device cuda
done
} > results/sweep67.log 2>&1
