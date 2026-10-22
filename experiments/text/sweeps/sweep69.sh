#!/bin/bash
# 実験69: サブワード（SentencePiece、その他なし）で GRU・光路 1 本・分担の 2DTG を比べる。持ち主の実行 OK を待って手で起動する
cd "$(dirname "$0")/../../.."
export PYTHONIOENCODING=utf-8
{
for V in 2048 4096; do
  python experiments/text/exp69_subword.py --method gru --V $V --device cuda
  python experiments/text/exp69_subword.py --method plates --layers 128 --K 128 --V $V --device cuda
  python experiments/text/exp58_split.py --task sp --V $V --K 128 --G 16 --layers 8 --m 4 --max_train 600000 --seed 0 --device cuda --out results/exp69.txt
done
} > results/sweep69.log 2>&1
