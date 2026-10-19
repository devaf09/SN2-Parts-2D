"""学習の経過の図（持ち主の依頼、2026-10-20）。results/expNN.txt の各行の history（train(log_every=...) や実験スクリプトが記録）から、
条件ごとに seed を平均した学習曲線を描く。

    python tools/fig_history.py results/exp31.txt results/exp32.txt --group method --metrics test_acc train_acc test_ce --out docs/images/exp31_history.png

横軸はステップ（--x train_s なら学習の時間）。縦軸は --metrics の数ごとに 1 枚。実線 = 1 つ目の名前が test_ のもの、点線 = train_ のもの。
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
NAMES = {"test_acc": "テストの正解率（%）", "train_acc": "訓練の正解率（%）", "test_ce": "テストの確信度 CE（小さいほど良い）",
         "train_ce": "訓練の確信度 CE", "loss": "その回の損失", "lr": "学習率"}

ap = argparse.ArgumentParser()
ap.add_argument("paths", nargs="+")
ap.add_argument("--group", default="method")
ap.add_argument("--metrics", nargs="+", default=["test_acc", "test_ce"])
ap.add_argument("--x", default="step", choices=["step", "train_s"])
ap.add_argument("--logx", action="store_true")
ap.add_argument("--title", default="学習の経過（seed の平均）")
ap.add_argument("--out", default="docs/images/history.png")
a = ap.parse_args()

runs = defaultdict(list)
for p in a.paths:
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        if r.get("history"):
            runs[str(r.get(a.group))].append(r["history"])
groups = sorted(runs)
pairs = defaultdict(list)          # test_X と train_X を同じ図に
for m in a.metrics:
    pairs[m.replace("test_", "").replace("train_", "")].append(m)
fig, ax = plt.subplots(1, len(pairs), figsize=(6.2 * len(pairs), 4.4), squeeze=False)
cmap = plt.get_cmap("tab10")
for j, (key, ms) in enumerate(pairs.items()):
    for gi, g in enumerate(groups):
        hs = runs[g]
        n = min(len(h) for h in hs)
        xs = [sum(h[k][a.x] for h in hs) / len(hs) for k in range(n)]
        for m in ms:
            ys = [sum(h[k].get(m, float("nan")) for h in hs) / len(hs) for k in range(n)]
            ax[0, j].plot(xs, ys, "--" if m.startswith("train_") else "-", color=cmap(gi % 10),
                          label=f"{g}（{'訓練' if m.startswith('train_') else 'テスト' if m.startswith('test_') else m}、{len(hs)} 本）")
    ax[0, j].set_title(" / ".join(NAMES.get(m, m) for m in ms), fontsize=10)
    ax[0, j].set_xlabel("学習のステップ" if a.x == "step" else "学習の時間（秒）")
    if a.logx:
        ax[0, j].set_xscale("log")
    ax[0, j].grid(alpha=0.3)
ax[0, -1].legend(fontsize=7)
fig.suptitle(a.title)
fig.tight_layout()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
fig.savefig(a.out, dpi=100, bbox_inches="tight")
print(a.out)
