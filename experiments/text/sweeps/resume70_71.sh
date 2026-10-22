#!/bin/bash
# 実験70・71 の続き（止まっても、終わった分を飛ばして再開する。待ち行列から実行）
cd "$(dirname "$0")/../../.."
bash tools/run_missing.sh results/exp70.txt experiments/text/sweeps/lists/exp70.txt >> results/sweep70.log 2>&1
bash tools/run_missing.sh results/exp71.txt experiments/text/sweeps/lists/exp71.txt >> results/sweep71.log 2>&1
