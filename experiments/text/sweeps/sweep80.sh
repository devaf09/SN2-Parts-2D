#!/bin/bash
# 実験80: 分担の 2DTG（担当の決め打ち）の中身を拡散 + 4f 系 + 平方根に。seed 0 で形を比べ、主なものを seed 1・2 で確かめる
cd "$(dirname "$0")/../../.."
export PYTHONIOENCODING=utf-8
R="python experiments/text/exp80_split4f.py --device cuda"
{
$R --n 8 --stages 4 --seed 0
$R --n 4 --stages 4 --seed 0
$R --n 8 --stages 6 --seed 0
$R --n 8 --stages 4 --G 8 --steps 6000 --seed 0
for s in 1 2; do
  $R --n 8 --stages 4 --seed $s
  $R --n 8 --stages 6 --seed $s
done
} >> results/sweep80.log 2>&1
