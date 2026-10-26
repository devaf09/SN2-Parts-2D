"""実験98 の図: 環状の板の枚数と融合の比率（600 区切り、3 seed）。実行: python tools/fig_exp98.py"""
import collections
import json
import os

import numpy as np

from figio import ROOT, plt

D = collections.defaultdict(list)
for f in ("exp97", "exp98"):
    for l in open(os.path.join(ROOT, "results", f + ".txt"), encoding="utf-8"):
        r = json.loads(l)
        D[(r["keys"], r["method"])].append(r["acc"])
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
Ns = [8, 16, 32, 64]
for a, kind, nm in ((ax[0], "rand", "でたらめな鍵"), (ax[1], "word_hash", "単語の鍵（〈他〉を別々）")):
    for beta, lab, c in ((0.0, "融合しない（順に振り分けるだけ）", "#27ae60"), (0.5, "半分ずつ融合", "#2e86c1"), (1.0, "まとめて融合", "#c0392b")):
        xs = [n for n in Ns if D.get((kind, f"ring_N{n}_b{beta}"))]
        a.plot(xs, [np.mean(D[(kind, f"ring_N{n}_b{beta}")]) for n in xs], "-o", color=c, label=f"環状・{lab}")
    a.axhline(np.mean(D[(kind, "elab_q0.0")]), ls="--", color="k", lw=1, label="満ちたら送る + 中期（板 約 20 枚）")
    a.set_xscale("log", base=2)
    a.set_xticks(Ns)
    a.set_xticklabels(Ns)
    a.minorticks_off()
    a.set_xlabel("環状の板の枚数")
    a.set_title(nm, fontsize=10)
    a.set_ylim(40, 101)
ax[0].set_ylabel("出てきた全部の組の取り出し %")
ax[0].legend(fontsize=8)
fig.suptitle("実験98: 環状の板を 64 枚に（持ち主の想定）。300 組から 16 組 × 600 区切り")
fig.tight_layout()
path = os.path.join(ROOT, "docs", "images", "exp98_ring64.png")
fig.savefig(path, dpi=110)
print(path)
