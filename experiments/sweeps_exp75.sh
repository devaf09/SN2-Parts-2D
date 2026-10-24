#!/bin/bash
# 実験75: 4f 系の多段 + 非線形で分類（光の CNN）。形の課題と MNIST
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
O="--seed 0 --device cuda"
{
for t in shape mnist; do
  python experiments/exp75_4f.py --task $t --stages 1 --width 4 --kind none $O
  python experiments/exp75_4f.py --task $t --stages 3 --width 4 --kind none $O
  for k in sqrt kerr fold ssat; do
    for s in 2 3; do
      python experiments/exp75_4f.py --task $t --stages $s --width 4 --kind $k $O --save
    done
  done
  python experiments/exp75_4f.py --task $t --stages 3 --width 4 --kind kerr --read plane $O
  python experiments/exp75_4f.py --task $t --stages 3 --width 8 --kind kerr $O
  python experiments/exp75_4f.py --task $t --stages 4 --width 4 --kind kerr $O
done
} >> results/sweep75.log 2>&1
