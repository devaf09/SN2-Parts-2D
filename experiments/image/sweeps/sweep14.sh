#!/bin/bash
# 実験14: ⑤ 楕円で大きさを変えられる減衰する波紋（持ち主の案）。4 課題 × 3 seed、波紋 18 個（学習する数 2025）
for seed in 0 1 2; do
  for task in shape orient position conj; do
    sv=""; [ $seed = 0 ] && sv="--save results/models/exp14_${task}_sinc_e_s0.pt"
    python experiments/image/exp01_basic.py --task $task --method sinc_e --seed $seed --device cuda --out results/exp14.txt $sv
  done
done
