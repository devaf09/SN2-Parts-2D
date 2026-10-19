#!/bin/bash
# 実験10: 担当なしの一般の絵を見せて融合の取り違えを直す。板 ①、2 課題 × 重み 0・0.3・1 × 3 seed
for seed in 0 1 2; do
  for task in conj shape; do
    for w in 1 0.3 0; do
      python experiments/image/exp10_reject.py --task $task --method xy --w $w --seed $seed --device cuda
    done
  done
done
