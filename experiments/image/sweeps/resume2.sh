#!/bin/bash
# 実験25 の seed 1（300 枚）が終わるのを待ち、実験26 → （実験27 は別の待ち合わせで自動）→ 実験25 の seed 2 の順に回す
cd "$(dirname "$0")/../../.."
until [ -f results/exp25.txt ] && [ $(wc -l < results/exp25.txt) -ge 4 ]; do sleep 30; done
bash experiments/image/sweeps/sweep26.sh > results/sweep26.log 2>&1
until [ -f results/exp27.txt ] && [ $(wc -l < results/exp27.txt) -ge 9 ]; do sleep 30; done
for n in 0 300; do
  python experiments/image/exp20_tree.py --seed 2 --n_per_class $n --sched cosine --steps_child 4000 --device cuda --out results/exp25.txt
done >> results/sweep25.log 2>&1
