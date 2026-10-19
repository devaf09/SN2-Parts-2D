#!/bin/bash
# 実験13: ④ 減衰する波紋の板で形の課題を上げる。初期の深さ（波紋の数に合わせて浅く）と、2 次元の tilt 照明との組み合わせ。3 seed
for seed in 0 1 2; do
  python experiments/image/exp01_basic.py --task shape --method sinc --sinc_depth 0.5 --seed $seed --device cuda --out results/exp13.txt
  python experiments/image/exp01_basic.py --task shape --method sinc --n_ripples 128 --sinc_depth 0.5 --seed $seed --device cuda --out results/exp13.txt
  python experiments/image/exp01_basic.py --task shape --method sinc --n_ripples 128 --sinc_depth 0.25 --seed $seed --device cuda --out results/exp13.txt
  python experiments/image/exp01_basic.py --task shape --method sinc --illum tilt:2 --seed $seed --device cuda --out results/exp13.txt
  python experiments/image/exp01_basic.py --task shape --method sinc --illum tilt:4 --seed $seed --device cuda --out results/exp13.txt
done
