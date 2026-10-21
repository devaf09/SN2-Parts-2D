#!/bin/bash
# 実験61: 2DTG の木（葉 16・枝 4・根 1）。実験58 の続きの後
cd "$(dirname "$0")/../../.."
until [ -f results/exp58.txt ] && [ $(wc -l < results/exp58.txt) -ge 8 ]; do sleep 60; done
export PYTHONIOENCODING=utf-8
{
python experiments/text/exp61_tree.py --seed 0 --device cuda
python experiments/text/exp61_tree.py --leaves 64 --branches 16 --leaf_layers 2 --branch_layers 8 --root_layers 32 --seed 0 --device cuda
} > results/sweep61.log 2>&1
