"""実験75 の図: 4f 系の多段 + 非線形で分類（形の課題・MNIST、seed 0、同じ条件は最後の値）。実行: python tools/fig_exp75.py"""
import json
import os

import numpy as np

from figio import ROOT, plt

D = {}
for l in open(os.path.join(ROOT, "results", "exp75.txt"), encoding="utf-8"):
    r = json.loads(l)
    D[(r["task"], r["stages"], r["width"], r["kind"], r["read"])] = r["test_acc"]
rows = [("none", 1, "lens", "線形 1 段"), ("none", 3, "lens", "線形 3 段"), ("sqrt", 2, "lens", "平方根 2 段"), ("sqrt", 3, "lens", "平方根 3 段"),
        ("fold", 2, "lens", "折り返し 2 段"), ("fold", 3, "lens", "折り返し 3 段"), ("ssat", 3, "lens", "位相を残す可飽和 3 段"),
        ("kerr", 3, "lens", "カー 3 段（レンズで読む）"), ("kerr", 3, "plane", "カー 3 段（出口の面で読む）")]
ref = {"shape": [("板 ① 16 枚（実験01）", 70.6), ("板 ① + 照明 16 通り（実験04）", 79.9), ("小さな CNN", 98.6)],
       "mnist": [("板 ① 16 枚（実験17）", 95.3), ("④ 波紋（実験18、これまでの最良）", 96.3), ("小さな CNN", 98.1)]}
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
for a, t in zip(ax, ("shape", "mnist")):
    vals = [D.get((t, s, 4, k, rd), np.nan) for k, s, rd, _ in rows]
    a.barh(range(len(rows))[::-1], vals, color=["#7f8c8d" if k == "none" else "#2e86c1" for k, _, _, _ in rows])
    for i, v in enumerate(vals):
        a.text(v + 0.5, len(rows) - 1 - i, f"{v:.1f}", va="center", fontsize=8)
    for name, v in ref[t]:
        a.axvline(v, ls="--", lw=1, color="#e67e22" if "CNN" in name else "#999999")
        a.text(v, len(rows) - 0.4, name, rotation=90, fontsize=7, va="top", ha="right")
    a.set_yticks(range(len(rows))[::-1])
    a.set_yticklabels([n for _, _, _, n in rows], fontsize=9)
    a.set_xlim(0, 105)
    a.set_xlabel("テストの正解率 %")
    a.set_title({"shape": "形の課題（位置がでたらめ、8 クラス）", "mnist": "MNIST（10 クラス）"}[t])
fig.suptitle("実験75: 4f 系（レンズ → 周波数の面の板 → レンズ）を 4 本ずつ並べて多段に、間に光の非線形（seed 0）")
fig.tight_layout()
path = os.path.join(ROOT, "docs", "images", "exp75_4f.png")
fig.savefig(path, dpi=110)
print(path)
