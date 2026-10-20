#!/bin/bash
# 実験49: 訓練データを全部（約 53 万件）に。光路（帯、板 32 枚 × 区画 16、窓 64・256）と GRU を 6 万件と比べる。3000 ステップ（1 回あたりの件数は同じ 256 件）、seed 0・1。
# n-gram は最初から全部の訓練文書で数えているので、6 万件の光路・GRU と比べると条件が違う（持ち主の「試験方法の絞り」の見立てを確かめる）
cd "$(dirname "$0")/../../.."
until [ -f results/exp48.txt ] && [ $(wc -l < results/exp48.txt) -ge 15 ]; do sleep 30; done
for seed in 0 1; do
  python experiments/text/exp40_inputs.py --enc bands --max_train 600000 --seed $seed --device cuda --out results/exp49.txt
  python experiments/text/exp40_inputs.py --enc bands --K 256 --max_train 600000 --seed $seed --device cuda --out results/exp49.txt
  python experiments/text/exp31_text2d.py --method gru --max_train 600000 --seed $seed --device cuda --out results/exp49.txt
  python experiments/text/exp31_text2d.py --method gru --max_train 600000 --steps 20000 --seed $seed --device cuda --out results/exp49.txt
  python experiments/text/exp40_inputs.py --enc bands --max_train 600000 --steps 12000 --seed $seed --device cuda --out results/exp49.txt
done > results/sweep49.log 2>&1
