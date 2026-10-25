#!/bin/bash
# 実験84a: 単語の課題、拡散 + 4f 系 + 平方根の光路 1 本（seed 0）。同じテストで単語の n-gram も測る
cd "$(dirname "$0")/../../.."
export PYTHONIOENCODING=utf-8
R="python experiments/text/exp84_word4f.py --G 1 --device cuda --seed 0"
D=results/logits
{
$R --stages 4 --width 4 --steps 3000 --ngram
$R --stages 6 --width 8 --steps 9000 --save_logits $D/w_single_6x8.pt
} >> results/sweep84.log 2>&1
