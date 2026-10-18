#!/bin/bash
# 実験05: アナログで作ったときの強さ（作りの誤差を入れて評価）。板 ①②③、2 課題 × 3 seed
for seed in 0 1 2; do
  for task in conj shape; do
    for m in xy stripes ripple; do
      python experiments/exp05_analog.py --task $task --method $m --seed $seed --device cuda
    done
  done
done
