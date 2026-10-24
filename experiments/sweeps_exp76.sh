#!/bin/bash
# 実験76: 実験75 の良いものを seed 0〜2 で確かめる・読み方（レンズの後 / 出口の面）を全部の非線形でそろえる
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
O="--width 4 --device cuda --out results/exp76.txt"
{
for s in 0 1 2; do
  for t in shape mnist; do
    for k in sqrt fold ssat kerr; do
      for r in lens plane; do
        python experiments/exp75_4f.py --task $t --stages 3 --kind $k --read $r --seed $s $O
      done
    done
    python experiments/exp75_4f.py --task $t --stages 4 --kind kerr --read plane --seed $s $O
    python experiments/exp75_4f.py --task $t --stages 4 --kind sqrt --read lens --seed $s $O
  done
done
} >> results/sweep76.log 2>&1
