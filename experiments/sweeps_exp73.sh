#!/bin/bash
# 実験73: 光の非線形をもっと掘る（持ち主の依頼 2026-10-23）。持ち主の実行 OK を待って、results/queue.txt に足して回す
# 見込み: MNIST 12 本 × 5〜10 分、文字 11 本 × 5〜10 分、分担 5 本 × 20〜40 分。合計 6〜8 時間
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
{
# 1. MNIST: 非線形なし（16 枚・32 枚）と、2 段 × 16 枚で 9 通りの非線形。良い形は 3 段も
python experiments/exp73_nl.py --task mnist --kind none --stages 1 --per 16 --seed 0 --device cuda
python experiments/exp73_nl.py --task mnist --kind none --stages 1 --per 32 --seed 0 --device cuda
for k in sqrt square log power sat thresh kerr ssat; do
  python experiments/exp73_nl.py --task mnist --kind $k --stages 2 --per 16 --seed 0 --device cuda --save
done
python experiments/exp73_nl.py --task mnist --kind kerr --stages 3 --per 16 --seed 0 --device cuda
python experiments/exp73_nl.py --task mnist --kind sqrt --stages 3 --per 16 --seed 0 --device cuda
# 2. 文字（光路 1 本、訓練は全部）: 非線形なし 32 枚と、2 段 × 16 枚で 9 通り
python experiments/exp73_nl.py --task char --kind none --stages 1 --per 32 --seed 0 --device cuda
for k in sqrt square log power sat thresh kerr ssat; do
  python experiments/exp73_nl.py --task char --kind $k --stages 2 --per 16 --seed 0 --device cuda
done
# 3. 文字（分担の 2DTG 16 個、担当の決め打ち）: 1 段 × 8 枚と、2 段 × 8 枚で非線形
python experiments/exp73_nl.py --task charsplit --kind none --stages 1 --per 8 --seed 0 --device cuda
python experiments/exp73_nl.py --task charsplit --kind none --stages 1 --per 16 --seed 0 --device cuda
for k in sqrt kerr power; do
  python experiments/exp73_nl.py --task charsplit --kind $k --stages 2 --per 8 --seed 0 --device cuda
done
} >> results/sweep73.log 2>&1
