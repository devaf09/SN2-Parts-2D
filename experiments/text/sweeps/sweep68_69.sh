#!/bin/bash
# 実験68 → 実験69 を続けて回す（持ち主の実行 OK、2026-10-22）
cd "$(dirname "$0")/../../.."
bash experiments/text/sweeps/sweep68.sh
bash experiments/text/sweeps/sweep69.sh
