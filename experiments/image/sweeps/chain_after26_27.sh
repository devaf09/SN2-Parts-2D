#!/bin/bash
# 実験26 が終わるのを待ってから実験27 を回す（会話と切り離して起動する）
cd "$(dirname "$0")/../../.."
until [ -f results/exp26.txt ] && [ $(wc -l < results/exp26.txt) -ge 12 ]; do sleep 30; done
bash experiments/image/sweeps/sweep27.sh > results/sweep27.log 2>&1
