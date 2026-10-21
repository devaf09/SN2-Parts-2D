#!/bin/bash
# 実験66: 公平な比べ方 — 光路 1 本に、分担の 2DTG と同じくらいのデジタルの符号を持たせる（窓を増やす）
cd "$(dirname "$0")/../../.."
export PYTHONIOENCODING=utf-8
{
python experiments/text/exp40_inputs.py --enc bands --K 256 --layers 128 --patches 16 --max_train 600000 --seed 0 --device cuda --out results/exp66.txt
python experiments/text/exp40_inputs.py --enc bands --K 4096 --layers 128 --patches 16 --max_train 600000 --seed 0 --device cuda --out results/exp66.txt
python experiments/text/exp52_words.py --method plates --K 4096 --layers 128 --max_train 400000 --seed 0 --device cuda --out results/exp66.txt
} > results/sweep66.log 2>&1
