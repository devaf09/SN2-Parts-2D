"""results/expNN.txt（1 行 = 1 回の JSON）を集計し、条件ごとの平均 ± 標準誤差の表を Markdown で出す。

    python tools/summarize.py results/exp01.txt --rows method --cols task --value test_acc
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict


def load(path, where=()):
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if all(str(r.get(k)) == v for k, v in where):
                    recs.append(r)
    return recs


def mean_se(v):
    m = sum(v) / len(v)
    if len(v) < 2:
        return m, None
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
    return m, sd / math.sqrt(len(v))


def table(recs, rows, cols, value, fmt="{:.1f}", order_rows=None, order_cols=None):
    cell = defaultdict(list)
    for r in recs:
        if r.get(value) is not None:
            cell[(r[rows], r[cols])].append(r[value])
    rk = order_rows or sorted({k[0] for k in cell}, key=str)
    ck = order_cols or sorted({k[1] for k in cell}, key=str)
    out = ["| " + rows + " | " + " | ".join(map(str, ck)) + " |", "|---" * (len(ck) + 1) + "|"]
    for a in rk:
        row = []
        for b in ck:
            v = cell.get((a, b))
            if not v:
                row.append("")
                continue
            m, se = mean_se(v)
            row.append(fmt.format(m) + ("" if se is None else " ± " + fmt.format(se)) + ("" if len(v) >= 3 else f" (n={len(v)})"))
        out.append(f"| {a} | " + " | ".join(row) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--rows", default="method")
    ap.add_argument("--cols", default="task")
    ap.add_argument("--value", default="test_acc")
    ap.add_argument("--fmt", default="{:.1f}")
    ap.add_argument("--where", nargs="*", default=[], help="key=value で絞り込む")
    a = ap.parse_args()
    recs = load(a.path, [w.split("=", 1) for w in a.where])
    sys.stdout.buffer.write((table(recs, a.rows, a.cols, a.value, a.fmt) + "\n").encode("utf-8"))
