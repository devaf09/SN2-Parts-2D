#!/bin/bash
# 実験56: 大きくして頭打ちを確かめる（持ち主の依頼 2026-10-20 夜）。場 64 → 128、板 32 → 128 枚、長く学習。実験55 の後
cd "$(dirname "$0")/../../.."
until [ -f results/exp55.txt ] && [ $(wc -l < results/exp55.txt) -ge 7 ]; do sleep 30; done
export PYTHONIOENCODING=utf-8
{
for cfg in "64 64 16 3000" "128 32 16 3000" "64 128 16 3000" "128 64 16 3000" "128 64 32 3000" "64 32 16 12000"; do
  set -- $cfg
  python experiments/text/exp52_words.py --method plates --L $1 --layers $2 --patches $3 --steps $4 --seed 0 --device cuda --out results/exp56.txt
done
python experiments/text/exp40_inputs.py --enc bands --K 256 --L 128 --layers 64 --patches 16 --seed 0 --device cuda --out results/exp56_char.txt
python experiments/text/exp52_words.py --method plates --L 128 --layers 64 --patches 16 --steps 9000 --seed 0 --device cuda --out results/exp56.txt
} > results/sweep56.log 2>&1
