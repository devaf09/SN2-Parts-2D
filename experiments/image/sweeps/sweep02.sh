#!/bin/bash
# 実験02: 入力を振幅として載せるか、位相として載せるか（板 ① と ②、4 課題 × 3 seed）
# 振幅の結果は実験01（results/exp01.txt）を使う。ここでは位相だけを回す
for seed in 0 1 2; do
  for task in shape orient position conj; do
    for m in xy stripes; do
      python experiments/image/exp01_basic.py --task $task --method $m --seed $seed --input phase --device cuda --out results/exp02.txt
    done
  done
done
