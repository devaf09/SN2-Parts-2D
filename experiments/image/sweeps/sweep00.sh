#!/bin/bash
# 実験00: 場の大きさ（64・128・256）と板の形（①②③）ごとの、学習 1 ステップの時間と最大メモリ
# リポジトリ直下で実行する。結果は results/exp00.txt に 1 行ずつ足される
python experiments/image/exp00_cost.py --device cuda --batch 64 --repeat 10
python experiments/image/exp00_cost.py --device cuda --batch 256 --repeat 5
