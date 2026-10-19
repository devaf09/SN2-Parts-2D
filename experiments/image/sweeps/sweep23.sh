#!/bin/bash
# 実験23: 例の平均の模様（典型）で全部のクラスの符号をそろえて、学習なしでクラスを足す。Fashion-MNIST、3 seed
for seed in 0 1 2; do
  python experiments/image/exp22_codes.py --parts B --seed $seed --device cuda --out results/exp23.txt
done
