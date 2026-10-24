"""実験76 の図: 4f 系の多段 + 非線形を seed 0〜2 で（平均と各 seed の点）。実行: python tools/fig_exp76.py"""
import collections
import json
import os

import numpy as np

from figio import ROOT, plt

D = collections.defaultdict(list)
for l in open(os.path.join(ROOT, "results", "exp76.txt"), encoding="utf-8"):
    r = json.loads(l)
    D[(r["task"], r["stages"], r["kind"], r["read"])].append(r["test_acc"])
names = {"sqrt": "平方根", "fold": "折り返し", "ssat": "位相を残す可飽和", "kerr": "カー"}
rows = [(3, k, rd) for k in ("sqrt", "fold", "ssat", "kerr") for rd in ("lens", "plane")] + [(4, "sqrt", "lens"), (4, "kerr", "plane")]
label = lambda s, k, rd: f"{names[k]} {s} 段（{'レンズの後' if rd == 'lens' else '出口の面'}）"
ref = {"shape": [("板 ① 16 枚", 70.6), ("小さな CNN", 98.6)], "mnist": [("板の最良", 96.3), ("小さな CNN", 98.1)]}
fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
for a, t in zip(ax, ("shape", "mnist")):
    n = len(rows)
    for i, (s, k, rd) in enumerate(rows):
        v = np.array(D[(t, s, k, rd)])
        y = n - 1 - i
        a.barh(y, v.mean(), color="#2e86c1" if rd == "lens" else "#16a085", alpha=0.8)
        a.plot(v, [y] * len(v), "k.", ms=6)
        a.text(101, y, f"{v.mean():.1f} ± {v.std():.1f}", va="center", fontsize=8)
    for name, v in ref[t]:
        a.axvline(v, ls="--", lw=1, color="#e67e22" if "CNN" in name else "#999999")
        a.text(v, n - 0.4, name, rotation=90, fontsize=7, va="top", ha="right")
    a.set_yticks(range(n)[::-1])
    a.set_yticklabels([label(*r) for r in rows], fontsize=9)
    a.set_xlim(0, 115)
    a.set_xlabel("テストの正解率 %（棒 = 3 seed の平均、点 = 各 seed）")
    a.set_title({"shape": "形の課題（位置がでたらめ、8 クラス）", "mnist": "MNIST（10 クラス）"}[t])
fig.suptitle("実験76: 4f 系 × 4 本の多段 + 非線形、seed 0〜2（青 = レンズの後で読む、緑 = 出口の面で読む）")
fig.tight_layout()
path = os.path.join(ROOT, "docs", "images", "exp76_4f_seeds.png")
fig.savefig(path, dpi=110)
print(path)
