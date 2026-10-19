#!/bin/bash
# 実験31 の seed 0（GRU・回帰・板）を、学習の経過を記録する版で測り直す（最初の 3 本は経過の記録の前に回していた）
cd "$(dirname "$0")/../../.."
until [ -f results/exp32.txt ] && [ $(wc -l < results/exp32.txt) -ge 9 ]; do sleep 30; done
python - <<'PY'
import json
p = "results/exp31.txt"
keep = [l for l in open(p, encoding="utf-8") if not (json.loads(l).get("seed") == 0 and json.loads(l)["method"] in ("gru", "logreg", "plates") and "history" not in json.loads(l))]
open(p, "w", encoding="utf-8").writelines(keep)
PY
for m in gru logreg plates; do
  python experiments/text/exp31_text2d.py --method $m --seed 0 --device cuda
done >> results/sweep31.log 2>&1
