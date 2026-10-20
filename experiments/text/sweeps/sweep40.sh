#!/bin/bash
# 実験40: 文字列の入れ方（帯・にじませ・前後の平均・2 次元の虹・拡散符号・同心円）。板 ① 32 枚 × 区画 16、3 seed。実験39 の後に始める
cd "$(dirname "$0")/../../.."
until [ -f results/exp39.txt ] && [ $(wc -l < results/exp39.txt) -ge 3 ]; do sleep 30; done
for seed in 0 1 2; do
  for e in spread rainbow blur smear rings bands; do
    python experiments/text/exp40_inputs.py --enc $e --seed $seed --device cuda
  done
done > results/sweep40.log 2>&1
