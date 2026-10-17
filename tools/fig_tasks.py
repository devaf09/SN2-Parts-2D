"""合成課題の例と、板 ①②③ の模様の図を docs/images/ に書き出す。

    python tools/fig_tasks.py
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import sn2_parts_2d as p2  # noqa: E402
import tasks2d  # noqa: E402

OUT = os.path.join(ROOT, "docs", "images")
os.makedirs(OUT, exist_ok=True)
plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]

# 1. 課題の例（課題ごとに 1 行、クラスごとに 1 枚）
names = {"shape": "A 形", "orient": "B 縞の向き", "position": "C 位置", "conj": "D 向き × 位置"}
fig, ax = plt.subplots(4, 9, figsize=(11, 5.2))
for i, t in enumerate(tasks2d.TASKS):
    X, y = tasks2d.make(t, 400, seed=7)
    for c in range(9):
        a = ax[i, c]
        a.axis("off")
        if c < tasks2d.TASKS[t]:
            a.imshow(X[(y == c).nonzero()[0, 0]], cmap="gray", vmin=0, vmax=1)
            a.set_title(f"{c}", fontsize=8)
    ax[i, 0].text(-0.3, 0.5, names[t], transform=ax[i, 0].transAxes, ha="right", va="center", fontsize=10)
fig.suptitle("合成課題の例（64 × 64、クラスごとに 1 枚）")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "tasks.png"), dpi=110, bbox_inches="tight")

# 2. 板の模様（学習前、同じ seed）
fig, ax = plt.subplots(1, 4, figsize=(11, 3))
titles = {"xy": "① 交点で掛け算", "stripes": "② 斜めの縞", "ripple": "③ 波紋（等間隔の輪）", "ripple2": "③ 波紋（ゾーンプレート）"}
for a, k in zip(ax, titles):
    net = p2.Net2D(64, 8, [k], seed=3)
    with torch.no_grad():
        a.imshow(net.plate_phase(0), cmap="twilight")
    a.set_title(titles[k], fontsize=10)
    a.axis("off")
fig.suptitle("板の位相の遅れの模様（学習前、区画 8 × 8）")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "plates.png"), dpi=110, bbox_inches="tight")
print("docs/images/tasks.png, docs/images/plates.png")
