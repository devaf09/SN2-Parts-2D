#!/bin/bash
# 実験68: 「その他」に光が集まる問題（案 1: 表示で除く目安、案 2: その他の中を受け持つ 2DTG）。持ち主の実行 OK を待って手で起動する
cd "$(dirname "$0")/../../.."
export PYTHONIOENCODING=utf-8
{
python experiments/text/exp68_other.py --direct --seed 0 --device cuda
python experiments/text/exp68_other.py --seed 1 --device cuda
} > results/sweep68.log 2>&1
