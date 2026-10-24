#!/bin/bash
# 実験79: 実験78 の最良（拡散 + 4f 系 + 平方根 + レンズの後で読む）を大きくする。まだ覚えきれていない（訓練 ≒ テスト）ため
cd "$(dirname "$0")/../../.."
export PYTHONIOENCODING=utf-8
R="python experiments/text/exp78_text4f.py --enc spread --device cuda --out results/exp79.txt"
{
for s in 0 1 2; do
  $R --stages 6 --seed $s
  $R --stages 4 --width 8 --seed $s
  $R --stages 4 --steps 9000 --seed $s
  $R --stages 6 --width 8 --steps 9000 --seed $s
done
} >> results/sweep79.log 2>&1
