#!/bin/bash
# 実験70: デジタルの重みを減らしても分担の 2DTG の精度を保てるか（頻度に合わせた分担・担当の決め打ち・小さい符号）。実験69 の後
cd "$(dirname "$0")/../../.."
until [ -f results/exp69.txt ] && [ $(wc -l < results/exp69.txt) -ge 8 ]; do sleep 60; done
export PYTHONIOENCODING=utf-8
O="--seed 0 --device cuda --out results/exp70.txt"
{
# 単語（直前 2 語、訓練は全部）
python experiments/text/exp58_split.py --G 16 --layers 8 --K 32 $O
python experiments/text/exp58_split.py --G 16 --layers 8 --K 32 --reject 0 $O
python experiments/text/exp58_split.py --assign freq --own 48 --G 64 --layers 2 --K 32 --reject 0 $O
python experiments/text/exp58_split.py --assign freq --own 48 --G 64 --layers 2 --K 32 $O
python experiments/text/exp58_split.py --assign freq --own 12 --G 16 --layers 8 --K 32 --reject 0 $O
# 文字（直前 4 文字、訓練は全部）
python experiments/text/exp58_split.py --task char --m 4 --G 16 --layers 8 --K 32 --max_train 600000 $O
python experiments/text/exp58_split.py --task char --m 4 --G 16 --layers 8 --K 32 --reject 0 --max_train 600000 $O
python experiments/text/exp58_split.py --task char --m 4 --assign freq --own 48 --G 64 --layers 2 --K 32 --reject 0 --max_train 600000 $O
} > results/sweep70.log 2>&1
