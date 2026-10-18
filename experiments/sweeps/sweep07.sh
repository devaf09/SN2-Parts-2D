#!/bin/bash
# 実験07: 別々に学習した 2 つの技能を足すだけで融合・外して取消（板 ①、2 課題 × 3 seed）
for seed in 0 1 2; do
  for task in conj shape; do
    python experiments/exp07_merge.py --task $task --method xy --seed $seed --device cuda
  done
done
