"""実験78 の図: 言語の入れ方 × 4f 系の多段 + 平方根（3 seed の平均）。実行: python tools/fig_exp78.py"""
import collections
import json
import os

import numpy as np

from figio import ROOT, plt

D = collections.defaultdict(list)
for l in open(os.path.join(ROOT, "results", "exp78.txt"), encoding="utf-8"):
    r = json.loads(l)
    D[(r["enc"], r["stages"], r["kind"], r["read"])].append(r["test_acc"])
encs = [("bands", "帯に並べる"), ("smear", "帯 + 前後の平均"), ("rainbow", "虹（向きをずらして重ねる）"), ("rings", "同心円の輪"), ("spread", "拡散（位置ごとの模様を掛けて重ねる）")]
fig, ax = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [3, 2]})
x = np.arange(len(encs))
for j, (rd, c, nm) in enumerate([("lens", "#2e86c1", "レンズの後で読む"), ("plane", "#16a085", "出口の面で読む")]):
    v = [D[(e, 3, "sqrt", rd)] for e, _ in encs]
    ax[0].bar(x + (j - 0.5) * 0.38, [np.mean(a) for a in v], 0.38, yerr=[np.std(a) for a in v], color=c, capsize=3, label=nm)
for y, nm, c in [(47.0, "板を重ねる光路 1 本（実験73）", "#999999"), (50.1, "光路 1 本のこれまでの最良", "#555555"), (53.7, "3-gram", "#e67e22"), (54.0, "分担の 2DTG 16 個（実験62）", "#8e44ad")]:
    ax[0].axhline(y, ls="--", lw=1, color=c)
    ax[1].axvline(y, ls="--", lw=1, color=c)
    ax[0].text(len(encs) - 0.5, y, nm, fontsize=7, va="bottom", ha="right", color=c)
ax[0].set_xticks(x)
ax[0].set_xticklabels([n for _, n in encs], fontsize=8)
ax[0].set_ylim(30, 56)
ax[0].set_ylabel("テストの正解率 %（3 seed の平均 ± ばらつき）")
ax[0].set_title("入れ方（4f 系 3 段 × 4 本 + 平方根）")
ax[0].legend(fontsize=8, loc="upper left")
rows = [("bands", 3, "none", "帯・線形 3 段"), ("bands", 3, "kerr", "帯・カー 0.1"), ("bands", 3, "sqrt", "帯・平方根 3 段"), ("bands", 4, "sqrt", "帯・平方根 4 段"),
        ("spread", 3, "none", "拡散・線形 3 段"), ("spread", 3, "kerr", "拡散・カー 0.1"), ("spread", 3, "sqrt", "拡散・平方根 3 段"), ("spread", 4, "sqrt", "拡散・平方根 4 段")]
v = [D[(e, s, k, "lens")] for e, s, k, _ in rows]
ax[1].barh(range(len(rows))[::-1], [np.mean(a) for a in v], xerr=[np.std(a) for a in v], color=["#7f8c8d" if r[2] == "none" else "#2e86c1" for r in rows], capsize=3)
for i, a in enumerate(v):
    ax[1].text(np.mean(a) + 0.4, len(rows) - 1 - i, f"{np.mean(a):.1f}", va="center", fontsize=8)
ax[1].set_yticks(range(len(rows))[::-1])
ax[1].set_yticklabels([r[3] for r in rows], fontsize=8)
ax[1].set_xlim(30, 56)
ax[1].set_title("非線形と段の数（レンズの後で読む）")
fig.suptitle("実験78: 文字（直前 8 文字 → 次の文字）を面に広げて 4f 系の多段 + 平方根に通す（光路 1 本、seed 0〜2）")
fig.tight_layout()
path = os.path.join(ROOT, "docs", "images", "exp78_text4f.png")
fig.savefig(path, dpi=110)
print(path)
