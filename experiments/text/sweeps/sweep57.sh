#!/bin/bash
# 実験57: データの疎らさの確認（持ち主の依頼 2026-10-20 夜）。単語の課題で訓練を 6 万件 → 全部（33 万件）に。実験56 の後
cd "$(dirname "$0")/../../.."
until [ -f results/exp56_char.txt ] && [ -f results/exp56.txt ] && [ $(wc -l < results/exp56.txt) -ge 7 ]; do sleep 60; done
export PYTHONIOENCODING=utf-8
{
python experiments/text/exp52_words.py --method gru --max_train 400000 --steps 3000 --seed 0 --device cuda --out results/exp57.txt
python experiments/text/exp52_words.py --method gru --max_train 400000 --steps 20000 --seed 0 --device cuda --out results/exp57.txt
python experiments/text/exp52_words.py --method logreg --max_train 400000 --steps 20000 --seed 0 --device cuda --out results/exp57.txt
python experiments/text/exp52_words.py --method plates --max_train 400000 --steps 3000 --seed 0 --device cuda --out results/exp57.txt
python experiments/text/exp52_words.py --method plates --max_train 400000 --steps 12000 --seed 0 --device cuda --out results/exp57.txt
python experiments/text/exp52_words.py --method plates --max_train 20000 --steps 3000 --seed 0 --device cuda --out results/exp57.txt
python experiments/text/exp52_words.py --method gru --max_train 20000 --steps 3000 --seed 0 --device cuda --out results/exp57.txt
} > results/sweep57.log 2>&1
