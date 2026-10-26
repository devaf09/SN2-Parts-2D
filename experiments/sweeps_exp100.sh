#!/bin/bash
# 実験100: 段階的記憶（歯車）・新しい記憶が次々来る流れ・崩れた鍵（GPU）
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
python experiments/exp100_gears.py --device cuda >> results/sweep100.log 2>&1
