"""実験79 の図: 拡散 + 4f 系 + 平方根を大きくすると（3 seed の平均）。実行: python tools/fig_exp79.py"""
import collections
import json
import os

import numpy as np

from figio import ROOT, plt

D = collections.defaultdict(list)
for f, ex in (("exp78.txt", "78"), ("exp79.txt", "79")):
    for l in open(os.path.join(ROOT, "results", f), encoding="utf-8"):
        r = json.loads(l)
        if r["enc"] == "spread" and r["kind"] == "sqrt" and r["read"] == "lens":
            D[(r["stages"], r["width"], r["steps"])].append((r["test_acc"], r["params"], r["train_s"]))
rows = sorted(D, key=lambda k: np.mean([x[0] for x in D[k]]))
fig, ax = plt.subplots(figsize=(10, 5))
for i, k in enumerate(rows):
    a = [x[0] for x in D[k]]
    ax.barh(i, np.mean(a), xerr=np.std(a), color="#2e86c1", capsize=3)
    ax.text(np.mean(a) + 0.3, i, f"{np.mean(a):.1f}（重み {D[k][0][1] / 1e4:.0f} 万、{D[k][0][2]:.0f} 秒）", va="center", fontsize=8)
for y, nm, c in [(50.1, "光路 1 本のこれまでの最良", "#555555"), (53.7, "3-gram", "#e67e22"), (54.0, "分担の 2DTG 16 個", "#8e44ad")]:
    ax.axvline(y, ls="--", lw=1, color=c)
    ax.text(y, len(rows) - 0.5, nm, fontsize=7, rotation=90, va="top", ha="right", color=c)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([f"{s} 段 × 幅 {w}、{st} ステップ" for s, w, st in rows], fontsize=9)
ax.set_xlim(45, 58)
ax.set_xlabel("テストの正解率 %（3 seed の平均 ± ばらつき）")
ax.set_title("実験79: 文字・拡散 + 4f 系 + 平方根（レンズの後で読む）を大きくする（光路 1 本）")
fig.tight_layout()
path = os.path.join(ROOT, "docs", "images", "exp79_scale.png")
fig.savefig(path, dpi=110)
print(path)
