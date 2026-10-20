#!/bin/bash
# 実験48: 短冊形で、文字ごとの光が後ろの文字にだけ届く形（持ち主の案）。板 16 枚、1500 ステップ、3 seed。実験47 の後に始める
cd "$(dirname "$0")/../../.."
until [ -f results/exp47.txt ] && [ $(wc -l < results/exp47.txt) -ge 15 ]; do sleep 30; done
for seed in 0 1 2; do
  python experiments/text/exp48_strip.py --seed $seed --device cuda
  python experiments/text/exp48_strip.py --nomask --seed $seed --device cuda
  python experiments/text/exp48_strip.py --tilt 0 --seed $seed --device cuda
  python experiments/text/exp48_strip.py --tilt 12 --seed $seed --device cuda
  python experiments/text/exp48_strip.py --span 1.5 --seed $seed --device cuda
done > results/sweep48.log 2>&1
