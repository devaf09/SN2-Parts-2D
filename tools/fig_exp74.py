"""実験73・74 の図: 分担の 2DTG + 非線形（3 seed の平均とばらつき）。実行: python tools/fig_exp74.py"""
import json
import os
import statistics as st

from figio import ROOT, plt

G = {}
for f in ("exp73.txt", "exp74.txt"):
    for l in open(os.path.join(ROOT, "results", f), encoding="utf-8"):
        r = json.loads(l)
        if r["task"] == "charsplit" and (r["kind"], r["stages"], r["per"]) in [("none", 1, 16), ("kerr", 2, 8), ("fold", 2, 8), ("sqrt", 2, 8)]:
            G.setdefault((r["kind"], r["stages"], r["per"]), []).append(r["test_acc"])
names = {("none", 1, 16): "非線形なし\n1 段 × 16 枚", ("kerr", 2, 8): "カー効果\n2 段 × 8 枚", ("fold", 2, 8): "折り返し\n2 段 × 8 枚", ("sqrt", 2, 8): "平方根\n2 段 × 8 枚"}
keys = list(names)
m = [st.mean(G[k]) for k in keys]
s = [st.stdev(G[k]) for k in keys]
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.bar(range(4), m, yerr=s, capsize=6, color=["#7f8c8d", "#2e86c1", "#2e86c1", "#2e86c1"])
for i, k in enumerate(keys):
    ax.scatter([i] * len(G[k]), G[k], color="k", s=12, zorder=3)
    ax.text(i, m[i] + s[i] + 0.1, f"{m[i]:.2f}", ha="center", fontsize=9)
ax.set_xticks(range(4))
ax.set_xticklabels([names[k] for k in keys])
ax.set_ylim(51, 54)
ax.set_ylabel("テストの正解率 %（文字、3 seed、点 = 各 seed）")
ax.set_title("実験73・74: 分担の 2DTG 16 個 + 段の間の非線形（板の合計は同じ 16 枚）")
fig.tight_layout()
path = os.path.join(ROOT, "docs", "images", "exp74_split_nl.png")
fig.savefig(path, dpi=110)
print(path)
