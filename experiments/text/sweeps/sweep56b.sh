#!/bin/bash
# 実験56 の残り（夜中 01:47 に sweep56 が止まったため続きから）と実験57
cd "$(dirname "$0")/../../.."
export PYTHONIOENCODING=utf-8
{
python experiments/text/exp52_words.py --method plates --L 128 --layers 64 --patches 32 --steps 3000 --seed 0 --device cuda --out results/exp56.txt
python experiments/text/exp52_words.py --method plates --L 64 --layers 32 --patches 16 --steps 12000 --seed 0 --device cuda --out results/exp56.txt
python experiments/text/exp40_inputs.py --enc bands --K 256 --L 128 --layers 64 --patches 16 --seed 0 --device cuda --out results/exp56_char.txt
} >> results/sweep56.log 2>&1
{
python experiments/text/exp52_words.py --method gru --max_train 400000 --steps 3000 --seed 0 --device cuda --out results/exp57.txt
python experiments/text/exp52_words.py --method gru --max_train 400000 --steps 20000 --seed 0 --device cuda --out results/exp57.txt
python experiments/text/exp52_words.py --method logreg --max_train 400000 --steps 20000 --seed 0 --device cuda --out results/exp57.txt
python experiments/text/exp52_words.py --method plates --max_train 400000 --steps 3000 --seed 0 --device cuda --out results/exp57.txt
python experiments/text/exp52_words.py --method plates --max_train 20000 --steps 3000 --seed 0 --device cuda --out results/exp57.txt
python experiments/text/exp52_words.py --method gru --max_train 20000 --steps 3000 --seed 0 --device cuda --out results/exp57.txt
python experiments/text/exp52_words.py --method plates --max_train 400000 --steps 12000 --seed 0 --device cuda --out results/exp57.txt
python experiments/text/exp52_words.py --method plates --L 64 --layers 128 --max_train 400000 --steps 3000 --seed 0 --device cuda --out results/exp57.txt
} > results/sweep57.log 2>&1
python experiments/text/exp52_words.py --method plates --L 128 --layers 64 --patches 16 --steps 9000 --seed 0 --device cuda --out results/exp56.txt >> results/sweep56.log 2>&1
