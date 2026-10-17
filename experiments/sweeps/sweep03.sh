#!/bin/bash
# 実験03: 区画の大きさ（板 1 枚の区画を縦横 2・4・16 個）と板の形。8 × 8 は実験01
# 区画が大きいと ③ 波紋の輪が見え、細かいと学習する数（容量）が増える。形と向き × 位置の 2 課題 × 3 seed
for seed in 0 1 2; do
  for task in shape conj; do
    for P in 2 4 16; do
      for m in xy stripes ripple ripple2; do
        python experiments/exp01_basic.py --task $task --method $m --seed $seed --patches $P --device cuda --out results/exp03.txt
      done
    done
  done
done
