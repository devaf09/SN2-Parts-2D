#!/bin/bash
# 実験58 の続き: 2DTG をさらに細かく分担（32×4 枚、64×2 枚）と、16×16 枚（計算 2 倍）。実験60 の後
cd "$(dirname "$0")/../../.."
until [ -f results/exp60.txt ] && [ $(wc -l < results/exp60.txt) -ge 6 ]; do sleep 60; done
export PYTHONIOENCODING=utf-8
{
python experiments/text/exp58_split.py --G 32 --layers 4 --seed 0 --device cuda
python experiments/text/exp58_split.py --G 64 --layers 2 --seed 0 --device cuda
python experiments/text/exp58_split.py --G 16 --layers 16 --seed 0 --device cuda
python experiments/text/exp58_split.py --G 16 --layers 8 --seed 1 --device cuda
} > results/sweep58b.log 2>&1
