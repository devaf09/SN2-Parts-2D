#!/bin/bash
# 実験64: 分担の効果は光路のおかげか、符号（デジタルの重み）が増えたおかげか。符号を全部同じでたらめに固定して比べる。実験63 の後
cd "$(dirname "$0")/../../.."
until [ -f results/exp63.txt ] && [ $(wc -l < results/exp63.txt) -ge 6 ]; do sleep 60; done
export PYTHONIOENCODING=utf-8
{
python experiments/text/exp52_words.py --method plates --fixed_codes --layers 128 --max_train 400000 --seed 0 --device cuda --out results/exp64.txt
python experiments/text/exp58_split.py --fixed_codes --G 16 --layers 8 --seed 0 --device cuda --out results/exp64.txt
python experiments/text/exp58_split.py --fixed_codes --G 8 --layers 16 --seed 0 --device cuda --out results/exp64.txt
python experiments/text/exp58_split.py --fixed_codes --task char --m 2 --G 16 --layers 8 --max_train 600000 --seed 0 --device cuda --out results/exp64.txt
} > results/sweep64.log 2>&1
