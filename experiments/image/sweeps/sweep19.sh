#!/bin/bash
# 実験19: ④ 波紋の中心の初期配置（格子・格子 + ずらし。でたらめは実験17）。cosine、5000 ステップ、波紋 128 個・深さ 0.5、3 seed
for seed in 0 1 2; do
  for data in mnist fashion; do
    for c in grid jitter; do
      python experiments/image/exp16_curve.py --data $data --method sinc --n_ripples 128 --sinc_depth 0.5 --sched cosine --steps 5000 --sinc_centers $c --seed $seed --device cuda --out results/exp19.txt
    done
  done
done
