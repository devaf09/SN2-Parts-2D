"""実験96 の図: 段の選び方（確信度・なじみ・両方）と〈他〉を別々にした単語の鍵。実行: python tools/fig_exp96.py"""
import collections
import json
import os

import numpy as np

from figio import ROOT, plt

D = collections.defaultdict(list)
for l in open(os.path.join(ROOT, "results", "exp96.txt"), encoding="utf-8"):
    r = json.loads(l)
    D[(r["T"], r["keys"])].append(r)
sels = [("acc_conf1", "確信度で 1 段（実験94）"), ("acc_conf_sum", "確信度で重み（実験95）"), ("acc_fam", "なじみで重み"), ("acc_conf_fam", "確信度 × なじみで重み"), ("oracle", "選び方が完全なら")]
cases = [((60, "word"), "単語・〈他〉まとめる\n60 区切り"), ((60, "word_hash"), "単語・〈他〉を別々に\n60 区切り"),
         ((600, "word"), "単語・〈他〉まとめる\n600 区切り + 中期"), ((600, "word_hash"), "単語・〈他〉を別々に\n600 区切り + 中期"), ((60, "rand"), "でたらめ\n60 区切り")]
fig, ax = plt.subplots(figsize=(15, 5.5))
x = np.arange(len(cases))
for j, (k, nm) in enumerate(sels):
    v = [np.mean([r[k] for r in D[c]]) for c, _ in cases]
    ax.bar(x + (j - 2) * 0.16, v, 0.16, label=nm, color="#27ae60" if k == "oracle" else None)
    for xi, vi in zip(x, v):
        ax.text(xi + (j - 2) * 0.16, vi, f"{vi:.0f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels([n for _, n in cases], fontsize=9)
ax.set_ylabel("取り出し %（3 seed）")
ax.set_title("実験96: 段の選び方に「なじみ（関連度）」を足す・〈他〉を語ごとに別々にする（満ちたら送る、1.5 倍ずつ 10 段）")
ax.legend(fontsize=8)
fig.tight_layout()
path = os.path.join(ROOT, "docs", "images", "exp96_familiar.png")
fig.savefig(path, dpi=110)
print(path)
