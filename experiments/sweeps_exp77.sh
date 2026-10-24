#!/bin/bash
# 実験77: うまくいかない理由を探る（持ち主の依頼 2026-10-25「失敗に成功の理由があるかも」）
#   カー効果はなぜ不安定か（明るい点で位相が何周も回る？勾配が大きすぎる？）、平方根はなぜ効くか（= 位相を捨てること？）
#   どれも入力をずらす・小さな雑音を足すと出口がどれだけ変わるか（cos）と、勾配の大きさを記録する
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
R="python experiments/exp75_4f.py --stages 3 --width 4 --device cuda --out results/exp77.txt"
{
for s in 0 1 2; do
  for t in shape mnist; do
    for r in lens plane; do
      $R --task $t --seed $s --read $r --kind sqrt
      $R --task $t --seed $s --read $r --kind kerr
      $R --task $t --seed $s --read $r --kind kerr --clip 1
      $R --task $t --seed $s --read $r --kind kerrn
      $R --task $t --seed $s --read $r --kind phfrac
      for p in 0.03 0.1 0.3 1; do $R --task $t --seed $s --read $r --kind kerr --fix_p $p; done
    done
    for p in 0 0.25 0.5 1; do $R --task $t --seed $s --read lens --kind phfrac --fix_p $p; done
  done
done
} >> results/sweep77.log 2>&1
