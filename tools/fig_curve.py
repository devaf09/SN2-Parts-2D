"""実験16 の学習曲線の図（docs/images/exp16_curve.png）と、「十分に学習した」ステップの表を出す。

    python tools/fig_curve.py

「十分に学習した」= テストの正解率が、その後の最高値から 0.5 点以内に入った最初のステップ（seed ごとに求めて平均）。
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
R = [json.loads(l) for l in open(os.path.join(ROOT, "results", "exp16.txt"), encoding="utf-8")]
NAME = {"logreg": "ロジスティック回帰", "cnn": "小さな CNN", "xy": "SinWave 板 ①", "sinc": "SinWave ④ 波紋 128 個"}
COL = {"logreg": "#888888", "cnn": "#d62728", "xy": "#1f77b4", "sinc": "#2ca02c"}
groups = defaultdict(list)
for r in R:
    groups[(r["data"], r["method"])].append(r)


def enough(curve, tol=0.5):
    best = max(c[2] for c in curve)
    for c in curve:
        if c[2] >= best - tol:
            return c
    return curve[-1]


lines = ["| データ | 方式 | 最高のテスト | 十分に学習したステップ | そこまでの時間（秒） | そのときの訓練 − テスト | 1 万ステップでの訓練 − テスト |",
         "|---|---|---|---|---|---|---|"]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for i, data in enumerate(["mnist", "fashion"]):
    for m in ["logreg", "cnn", "xy", "sinc"]:
        rs = groups.get((data, m))
        if not rs:
            continue
        steps = [c[0] for c in rs[0]["curve"]]
        te = [sum(r["curve"][k][2] for r in rs) / len(rs) for k in range(len(steps))]
        tr = [sum(r["curve"][k][3] for r in rs) / len(rs) for k in range(len(steps))]
        ax[i].plot(steps, te, "-o", ms=3, color=COL[m], label=f"{NAME[m]}（テスト）")
        ax[i].plot(steps, tr, "--", color=COL[m], alpha=0.6, label=f"{NAME[m]}（訓練）")
        e = [enough(r["curve"]) for r in rs]
        best = sum(max(c[2] for c in r["curve"]) for r in rs) / len(rs)
        st = sum(c[0] for c in e) / len(e)
        sec = sum(c[1] for c in e) / len(e)
        gap = sum(c[3] - c[2] for c in e) / len(e)
        gap_end = sum(r["curve"][-1][3] - r["curve"][-1][2] for r in rs) / len(rs)
        ax[i].axvline(st, color=COL[m], lw=0.8, alpha=0.4)
        lines.append(f"| {data} | {NAME[m]} | {best:.1f} | {st:.0f} | {sec:.0f} | {gap:+.1f} | {gap_end:+.1f} |")
    ax[i].set_xscale("log")
    ax[i].set_xlabel("学習のステップ（1 ステップ 256 枚、対数）")
    ax[i].set_ylabel("正解率（%）")
    ax[i].set_title({"mnist": "MNIST（手書き数字）", "fashion": "Fashion-MNIST（服の絵）"}[data])
    ax[i].set_ylim(60 if data == "mnist" else 50, 100)
    ax[i].grid(alpha=0.3)
ax[1].legend(fontsize=7, loc="lower right")
fig.suptitle("学習曲線（3 seed の平均。実線 = テスト、点線 = 訓練、縦線 = 十分に学習したステップ）")
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "docs", "images", "exp16_curve.png"), dpi=110, bbox_inches="tight")
sys.stdout.buffer.write(("\n".join(lines) + "\n").encode("utf-8"))
