#!/bin/bash
# 実験12: 技能を 2〜4 個にしたときの足すだけの融合（担当なしの学習あり・なし）。板 ①、3 seed
for seed in 0 1 2; do
  python experiments/image/exp12_many.py --task conj --k 4 --seed $seed --device cuda
  python experiments/image/exp12_many.py --task position --k 3 --seed $seed --device cuda
done
