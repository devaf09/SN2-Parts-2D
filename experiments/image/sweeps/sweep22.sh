#!/bin/bash
# 実験22: 符号で読む（窓の数を固定してクラスを表す・足す・部品で定義・融合）。Fashion-MNIST、3 seed
for seed in 0 1 2; do
  python experiments/image/exp22_codes.py --seed $seed --device cuda
done
