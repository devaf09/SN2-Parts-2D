"""実験17 の図: 学習率一定（実験16）と、最後に 0 まで下げる（cosine、実験17）の学習曲線（docs/images/exp17_cosine.png）。

    python tools/fig_exp17.py
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
load = lambda n: [json.loads(l) for l in open(os.path.join(ROOT, "results", n), encoding="utf-8")]
R16, R17 = load("exp16.txt"), load("exp17.txt")
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for i, data in enumerate(["mnist", "fashion"]):
    for m, lr, rs, col, ls, lab in [
        ("xy", 0.01, R16, "#1f77b4", ":", "板 ①、学習率一定"),
        ("xy", 0.01, R17, "#1f77b4", "-", "板 ①、0 まで下げる"),
        ("sinc", 0.01, R16, "#2ca02c", ":", "④ 波紋 128 個、学習率一定"),
        ("sinc", 0.01, R17, "#2ca02c", "-", "④ 波紋 128 個、0 まで下げる"),
        ("sinc", 0.003, R17, "#9467bd", "--", "④ 波紋 128 個、0 まで下げる・学習率 0.003")]:
        g = [r for r in rs if r["data"] == data and r["method"] == m and r["lr"] == lr]
        if not g:
            continue
        n = min(len(r["curve"]) for r in g)
        st = [g[0]["curve"][k][0] for k in range(n) if g[0]["curve"][k][0] <= 5000]
        te = [sum(r["curve"][k][2] for r in g) / len(g) for k in range(len(st))]
        ax[i].plot(st, te, ls, marker="o", ms=3, color=col, label=lab)
    ax[i].set_xscale("log")
    ax[i].set_xlabel("学習のステップ（対数）")
    ax[i].set_ylabel("テスト正解率（%）")
    ax[i].set_ylim(70 if data == "mnist" else 50, 100)
    ax[i].grid(alpha=0.3)
    ax[i].set_title({"mnist": "MNIST", "fashion": "Fashion-MNIST"}[data])
ax[1].legend(fontsize=8, loc="lower right")
fig.suptitle("学習率を最後に 0 まで下げると、谷が消えて最後が一番良くなる（3 seed の平均）")
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "docs", "images", "exp17_cosine.png"), dpi=110, bbox_inches="tight")
print("docs/images/exp17_cosine.png")
