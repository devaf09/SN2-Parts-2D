#!/bin/bash
# 実験01: 4 課題 × (板の形 5 種 + でたらめな板 + レンズだけ + 比較相手 4 種) × 3 seed
# リポジトリ直下で実行する。結果は results/exp01.txt に 1 行ずつ足される
for seed in 0 1 2; do
  for task in shape orient position conj; do
    for m in logreg mlp cnn lens_logreg lens random xy stripes ripple ripple2 mix_xy_ripple; do
      python experiments/exp01_basic.py --task $task --method $m --seed $seed --device cuda
    done
  done
done
