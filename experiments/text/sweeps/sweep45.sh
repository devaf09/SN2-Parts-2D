#!/bin/bash
# 実験45（ふるい分け: seed 0、1500 ステップ、板 16 枚）: 文字を板として掛け算で通す。実験44 の 2 本目が終わってから始める
cd "$(dirname "$0")/../../.."
until [ -f results/exp44.txt ] && [ $(wc -l < results/exp44.txt) -ge 2 ]; do sleep 20; done
{
python experiments/text/exp45_modulate.py --mode add --seed 0 --device cuda
python experiments/text/exp45_modulate.py --mode mul --mask rand --seed 0 --device cuda
python experiments/text/exp45_modulate.py --mode mul --mask wave --seed 0 --device cuda
python experiments/text/exp45_modulate.py --mode mul --mask rand --depth 0.3 --seed 0 --device cuda
python experiments/text/exp45_modulate.py --mode mul --mask rand --K 256 --seed 0 --device cuda
python experiments/text/exp45_modulate.py --mode add --K 256 --seed 0 --device cuda
} > results/sweep45.log 2>&1
