#!/bin/bash
# 実験25 の残り（PC 側で途中で止まった分）と実験26 を続けて回す。リポジトリ直下で実行する
cd "$(dirname "$0")/../.."
for args in "1 0" "1 300" "2 0" "2 300"; do
  set -- $args
  python experiments/exp20_tree.py --seed $1 --n_per_class $2 --sched cosine --steps_child 4000 --device cuda --out results/exp25.txt
done >> results/sweep25.log 2>&1
bash experiments/sweeps/sweep26.sh > results/sweep26.log 2>&1
