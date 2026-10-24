#!/bin/bash
# 実験78: 言語の入力を面に広げて 4f 系の多段 + 平方根に通す。seed 0〜2
cd "$(dirname "$0")/../../.."
export PYTHONIOENCODING=utf-8
R="python experiments/text/exp78_text4f.py --device cuda"
{
for s in 0 1 2; do
  for e in bands smear rainbow spread rings; do
    for r in lens plane; do $R --enc $e --read $r --seed $s; done
  done
  $R --enc bands --kind none --seed $s
  $R --enc spread --kind none --seed $s
  $R --enc bands --kind kerr --fix_p 0.1 --seed $s
  $R --enc spread --kind kerr --fix_p 0.1 --seed $s
  $R --enc bands --stages 4 --seed $s
  $R --enc spread --stages 4 --seed $s
done
} >> results/sweep78.log 2>&1
