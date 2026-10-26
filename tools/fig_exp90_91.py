"""実験90・91 の図: 案 1 の手順・実際の言葉を値に・板を系列に（3 seed の平均）。実行: python tools/fig_exp90_91.py"""
import collections
import json
import os

import numpy as np

from figio import ROOT, plt

ld = collections.defaultdict(list)
for l in open(os.path.join(ROOT, "results", "exp90_logdiff.txt"), encoding="utf-8"):
    r = json.loads(l)
    ld[(r["keys"], r["method"])].append(r["acc"])
ex89 = collections.defaultdict(list)
for l in open(os.path.join(ROOT, "results", "exp89.txt"), encoding="utf-8"):
    r = json.loads(l)
    ex89[(r["keys"], r["method"])].append(r["acc"])
w = collections.defaultdict(list)
for l in open(os.path.join(ROOT, "results", "exp90.txt"), encoding="utf-8"):
    r = json.loads(l)
    if r["part"] == "once":
        w[(r["N"], r["mode"], r["values"])].append(r)
s = collections.defaultdict(list)
for l in open(os.path.join(ROOT, "results", "exp91.txt"), encoding="utf-8"):
    r = json.loads(l)
    s[(r["keys"], r["method"])].append(r)
fig, ax = plt.subplots(1, 3, figsize=(19, 5.2))
kinds = [("rand", "でたらめな鍵"), ("word", "単語の鍵"), ("sim0.5", "似せた鍵 0.5")]
meth = [("plain", "そのまま", ex89), ("diff", "差分", ex89), ("logdiff", "案 1: 対数の差分だけ書く", ld), ("logdiff_soft", "案 1: softmax 版", ld), ("gated", "差分 + 足りない分だけ", ex89)]
x = np.arange(len(kinds))
for j, (m, nm, src) in enumerate(meth):
    ax[0].bar(x + (j - 2) * 0.16, [np.mean(src[(k, m)]) for k, _ in kinds], 0.16, label=nm)
ax[0].set_xticks(x)
ax[0].set_xticklabels([n for _, n in kinds])
ax[0].set_ylabel("出てきた全部の組の取り出し %")
ax[0].set_title("1. 案 1 の手順（短期 → 長期、300 組 × 60 回）", fontsize=10)
ax[0].legend(fontsize=7)
Ns = [32, 128, 512]
for (mode, vals, nm, st) in [("plain", "rand", "値 = でたらめな窓", "--"), ("plain", "word", "値 = 次の語（そのまま）", "-"), ("diff", "word", "値 = 次の語（差分）", "-")]:
    ax[1].plot(Ns, [np.mean([r["acc"] for r in w[(N, mode, vals)]]) for N in Ns], st, marker="o", label=nm + "・全体")
    if vals == "word":
        ax[1].plot(Ns, [np.mean([r["acc_rare"] for r in w[(N, mode, vals)]]) for N in Ns], ":", marker="x", label=nm + "・まれな値の組")
ax[1].plot(Ns, [np.mean([r["majority"] for r in w[(N, "plain", "word")]]) for N in Ns], color="k", lw=1, label="いつも一番多い値を答える")
ax[1].set_xscale("log")
ax[1].set_xticks(Ns)
ax[1].set_xticklabels(Ns)
ax[1].minorticks_off()
ax[1].set_xlabel("1 枚に書いた組の数")
ax[1].set_title("4. 値を実際の言葉に（直前 4 語 → 次の語、1 回で書く）", fontsize=10)
ax[1].legend(fontsize=7)
cols = [("acc", "全体"), ("acc_5up", "5 回以上"), ("acc_important", "重要"), ("acc_recent", "最近"), ("changed_new", "変わった値")]
xx = np.arange(len(cols))
for j, (m, nm) in enumerate([("single_gated", "板 1 枚"), ("series", "短期 → 中期 → 長期の系列"), ("split3", "板 3 枚に振り分け")]):
    ax[2].bar(xx + (j - 1) * 0.27, [np.mean([r[c] for r in s[("word", m)] if r[c] is not None]) for c, _ in cols], 0.27, label=nm)
ax[2].set_xticks(xx)
ax[2].set_xticklabels([n for _, n in cols])
ax[2].set_title("5. 板を何枚かに（単語の鍵、確信度が一番高い板の答え）", fontsize=10)
ax[2].legend(fontsize=7)
fig.suptitle("実験90・91: 記憶の残りの実験（CPU、3 seed）")
fig.tight_layout()
path = os.path.join(ROOT, "docs", "images", "exp90_91_memory.png")
fig.savefig(path, dpi=110)
print(path)
