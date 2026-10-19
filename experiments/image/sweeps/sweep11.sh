#!/bin/bash
# 実験11: ④ 減衰する波紋の板（持ち主の案）。4 課題 × (波紋 32 個 = 学習する数 2057、128 個 = 8201、① 8 枚 + ④ 8 枚) × 3 seed
# seed 0 は学習した光路を results/models/ に保存し、tools/fig_sinc.py で板の図を作る
for seed in 0 1 2; do
  for task in shape orient position conj; do
    sv() { [ $seed = 0 ] && echo "--save results/models/exp11_${task}_$1_s0.pt"; }
    python experiments/image/exp01_basic.py --task $task --method sinc --seed $seed --device cuda --out results/exp11.txt $(sv sinc32)
    python experiments/image/exp01_basic.py --task $task --method sinc --n_ripples 128 --seed $seed --device cuda --out results/exp11.txt $(sv sinc128)
    python experiments/image/exp01_basic.py --task $task --method mix_xy_sinc --seed $seed --device cuda --out results/exp11.txt $(sv mix)
  done
done
