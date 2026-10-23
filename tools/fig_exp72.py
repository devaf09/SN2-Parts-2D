"""実験71・72（2DWIG）の図: データを増やす・水増しの効果。実行: python tools/fig_exp72.py"""
import json
import os

import numpy as np

from figio import ROOT, plt

R71 = [json.loads(l) for l in open(os.path.join(ROOT, "results", "exp71.txt"), encoding="utf-8")]
R72 = [json.loads(l) for l in open(os.path.join(ROOT, "results", "exp72.txt"), encoding="utf-8")]


def avg(R, m, aug=0, key="test_acc"):
    v = [r[key] for r in R if r["method"] == m and r.get("aug", 0) == aug and r.get(key) is not None]
    return sum(v) / len(v) if v else np.nan


fig, ax = plt.subplots(1, 2, figsize=(14, 5))
meths = [("plates", "光の 2DWIG"), ("logreg", "ロジスティック回帰"), ("nb", "数えた表（ナイーブベイズ）")]
x = np.arange(len(meths))
cols = [("150 作品（実験71）", [avg(R71, m) if m != "nb" else 51.9 for m, _ in meths], "#7f8c8d"),
        ("1,000 作品・本物だけ", [avg(R72, m, 0) for m, _ in meths], "#2e86c1"),
        ("1,000 作品・本物 + 水増し", [avg(R72, m, 3) for m, _ in meths], "#e67e22")]
for i, (lab, vals, c) in enumerate(cols):
    ax[0].bar(x + (i - 1) * 0.27, vals, 0.27, label=lab, color=c)
    for xi, v in zip(x, vals):
        ax[0].text(xi + (i - 1) * 0.27, v + 0.6, f"{v:.1f}", ha="center", fontsize=8)
ax[0].axhline(R72[0]["majority"], color="k", ls="--", lw=1, label="いつも一番多い類（1,000 作品）")
ax[0].set_xticks(x)
ax[0].set_xticklabels([n for _, n in meths])
ax[0].set_ylabel("テストの正解率 %（話者の一人称 12 類、本物だけ）")
ax[0].set_ylim(20, 60)
ax[0].legend(fontsize=8)
ax[0].set_title("データを 7.7 倍にすると光も当てずっぽうを超える。水増しは効かない")
tr = [("150 作品", avg(R71, "plates", 0, "train_acc"), avg(R71, "plates")), ("1,000 作品", avg(R72, "plates", 0, "train_acc"), avg(R72, "plates", 0)),
      ("1,000 作品 + 水増し", avg(R72, "plates", 3, "train_acc"), avg(R72, "plates", 3))]
xx = np.arange(3)
ax[1].bar(xx - 0.2, [t for _, t, _ in tr], 0.4, label="訓練", color="#95a5a6")
ax[1].bar(xx + 0.2, [v for _, _, v in tr], 0.4, label="テスト", color="#2e86c1")
ax[1].set_xticks(xx)
ax[1].set_xticklabels([n for n, _, _ in tr])
ax[1].set_ylabel("正解率 %")
ax[1].legend()
ax[1].set_title("光の 2DWIG の覚えすぎ（訓練とテストの差）が縮む")
fig.tight_layout()
path = os.path.join(ROOT, "docs", "images", "exp72_wig.png")
fig.savefig(path, dpi=110)
print(path)
