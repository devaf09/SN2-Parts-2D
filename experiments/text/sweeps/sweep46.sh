#!/bin/bash
# 実験46（ふるい分け: 板 16 枚 × 区画 8、1500 ステップ、seed 0）: 直前 2 文字の組を入口で 1 本の光に（掛け算 / 組ごとの向き）。基準は帯（実験45 の add と同じ条件）
cd "$(dirname "$0")/../../.."
{
for e in pairmul pairhash; do
  python experiments/text/exp40_inputs.py --enc $e --w0 1.0 --layers 16 --patches 8 --steps 1500 --seed 0 --device cuda --out results/exp46.txt
  python experiments/text/exp40_inputs.py --enc $e --w0 1.0 --layers 16 --patches 8 --steps 1500 --K 256 --seed 0 --device cuda --out results/exp46.txt
done
python experiments/text/exp40_inputs.py --enc bands --layers 16 --patches 8 --steps 1500 --seed 0 --device cuda --out results/exp46.txt
} > results/sweep46.log 2>&1
