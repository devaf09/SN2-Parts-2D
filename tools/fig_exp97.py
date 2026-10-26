"""実験97 の図: 環状の板・精緻化リハーサル（600 区切り、3 seed）。実行: python tools/fig_exp97.py"""
import collections
import json
import os

import numpy as np

from figio import ROOT, plt

D = collections.defaultdict(list)
for l in open(os.path.join(ROOT, "results", "exp97.txt"), encoding="utf-8"):
    r = json.loads(l)
    D[(r["keys"], r["method"])].append(r)
meths = [("elab_q0.0", "満ちたら送る + 中期（精緻化なし）"), ("elab_q0.2", "+ 精緻化（なじみ上位 20%）"), ("elab_q0.5", "+ 精緻化（上位 50%）"),
         ("ring_N8_b1.0", "環状 8 枚（まとめて融合）"), ("ring_N16_b0.5", "環状 16 枚（半分ずつ融合）")]
cols = [("acc", "全体"), ("acc_2to4", "2〜4 回"), ("acc_5up", "5 回以上"), ("acc_important", "重要"), ("acc_recent", "最近"), ("acc_old", "30 回以上前"), ("changed_new", "変わった値")]
fig, ax = plt.subplots(1, 2, figsize=(18, 5.2))
for a, kind, nm in ((ax[0], "rand", "でたらめな鍵"), (ax[1], "word_hash", "単語の鍵（〈他〉を別々）")):
    x = np.arange(len(cols))
    for j, (m, lab) in enumerate(meths):
        a.bar(x + (j - 2) * 0.16, [np.mean([r[c] for r in D[(kind, m)] if r.get(c) is not None] or [np.nan]) for c, _ in cols], 0.16, label=lab)
    a.set_xticks(x)
    a.set_xticklabels([n for _, n in cols], fontsize=9)
    a.set_title(nm, fontsize=10)
    a.set_ylim(0, 105)
ax[0].set_ylabel("取り出し %")
ax[0].legend(fontsize=7)
fig.suptitle("実験97: 環状の板と精緻化リハーサル（300 組から 16 組 × 600 区切り、読みは確信度 × なじみ）")
fig.tight_layout()
path = os.path.join(ROOT, "docs", "images", "exp97_ring_elab.png")
fig.savefig(path, dpi=110)
print(path)
