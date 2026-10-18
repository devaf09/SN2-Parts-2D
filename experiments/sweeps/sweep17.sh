#!/bin/bash
# 実験17: 学習率の変え方（一定 → 0 まで下げる cosine）と ④ の学習率。5000 ステップ、MNIST・Fashion-MNIST、3 seed
for seed in 0 1 2; do
  for data in mnist fashion; do
    python experiments/exp16_curve.py --data $data --method xy --sched cosine --steps 5000 --seed $seed --device cuda --out results/exp17.txt
    python experiments/exp16_curve.py --data $data --method sinc --n_ripples 128 --sinc_depth 0.5 --sched cosine --steps 5000 --seed $seed --device cuda --out results/exp17.txt
    python experiments/exp16_curve.py --data $data --method sinc --n_ripples 128 --sinc_depth 0.5 --sched cosine --lr 0.003 --steps 5000 --seed $seed --device cuda --out results/exp17.txt
  done
done
