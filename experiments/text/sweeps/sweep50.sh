#!/bin/bash
# 実験50（ふるい分け: 1500 ステップ、seed 0）: 往復のたびにレイヤの違う場所を通す（持ち主の鏡の往復案）。shift = 0（毎回同じ場所）・16・32
cd "$(dirname "$0")/../../.."
for sh in 16 0 32; do
  python experiments/text/exp50_bounce.py --shift $sh --seed 0 --device cuda
done > results/sweep50.log 2>&1
