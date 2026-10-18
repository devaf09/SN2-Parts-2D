#!/bin/bash
# 実験08: 融合の較正（各光路の温度・背景光だけを少量の共通データで合わせる）。板 ①、2 課題 × 3 seed
for seed in 0 1 2; do
  for task in conj shape; do
    python experiments/exp08_calib.py --task $task --method xy --seed $seed --device cuda
  done
done
