#!/bin/bash
# 実験18: ④ 波紋の数の種類ごとの学習率（中心を大きく・幅を小さく）。cosine、5000 ステップ、波紋 128 個・深さ 0.5、3 seed
for seed in 0 1 2; do
  for data in mnist fashion; do
    for lr in "10 0.1" "1 0.1" "10 1"; do
      set -- $lr
      python experiments/exp16_curve.py --data $data --method sinc --n_ripples 128 --sinc_depth 0.5 --sched cosine --steps 5000 --lr_center $1 --lr_width $2 --seed $seed --device cuda --out results/exp18.txt
    done
  done
done
