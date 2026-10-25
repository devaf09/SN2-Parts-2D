#!/bin/bash
# 実験86: 単語の課題で文脈を長くする（直前 2・4・8・16 語）。拡散 + 4f 系 6 段 × 8 + 平方根の光路 1 本、seed 0
#   持ち主の次の構想（長い文章の要所・一時記憶）の前に、光路 1 本がどこまで長い文脈を使えるかを測る。worddata は直前 n 語まで作れる
#   （exp84_word4f.py の --n。4 語より長いときは worddata をその長さで作る。n = 4 は実験84 の値）
cd "$(dirname "$0")/../../.."
export PYTHONIOENCODING=utf-8
{
for n in 2 8 16; do
  python experiments/text/exp84_word4f.py --G 1 --out results/exp86.txt --n $n --stages 6 --width 8 --steps 9000 --seed 0 --device cuda
done
} >> results/sweep86.log 2>&1
