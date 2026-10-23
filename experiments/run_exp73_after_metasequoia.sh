#!/bin/bash
# 実験73 を、Metasequoia の GPU の実験（sweep13）が終わってから始める（Metasequoia の会話と相談済み、2026-10-23）。
# 目印: Project_Metasequoia の results/queue.log に "done sweep13" があり、logs/queue.lock が無いこと。
cd "$(dirname "$0")/.."
M=/d/AI_git/Project_Metasequoia
until grep -q "done sweep13" "$M/results/queue.log" 2>/dev/null && [ ! -f "$M/logs/queue.lock" ]; do sleep 120; done
echo "$(date '+%F %T') start exp73 (Metasequoia の sweep13 が終わった)" >> results/queue_runner.log
bash experiments/sweeps_exp73.sh
