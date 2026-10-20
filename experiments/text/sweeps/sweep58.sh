#!/bin/bash
# 実験58: 小さな 2DTG を直前の語で分担、バラバラに学習して足す（持ち主の方針 2026-10-21）。sweep56b が全部終わってから
cd "$(dirname "$0")/../../.."
until [ -f results/exp56.txt ] && [ $(wc -l < results/exp56.txt) -ge 7 ] && [ -f results/exp57.txt ] && [ $(wc -l < results/exp57.txt) -ge 8 ]; do sleep 60; done
export PYTHONIOENCODING=utf-8
{
python experiments/text/exp58_split.py --G 8 --layers 16 --seed 0 --device cuda
python experiments/text/exp58_split.py --G 16 --layers 8 --seed 0 --device cuda
python experiments/text/exp58_split.py --G 4 --layers 32 --seed 0 --device cuda
python experiments/text/exp58_split.py --G 8 --layers 16 --seed 1 --device cuda
} > results/sweep58.log 2>&1
