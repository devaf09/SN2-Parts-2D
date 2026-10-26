"""実験100 の図: 歯車・環状 64 枚・デジタルの辞書（新しい記憶が次々来る流れ、崩れた鍵）。実行: python tools/fig_exp100.py"""
import collections
import json
import os

import numpy as np

from figio import ROOT, plt

D = collections.defaultdict(list)
for l in open(os.path.join(ROOT, "results", "exp100.txt"), encoding="utf-8"):
    r = json.loads(l)
    D[(r["keys"], r["method"])].append(r)
meths = [("ring64_keep", "環状 64 枚・忘却なし"), ("ring64_decay", "環状 64 枚・ただ薄れる"), ("gears_last4", "歯車（ラスト 4 枚）"), ("gears_extract4", "歯車（抽出 4 枚）")]
cols = [("acc", "全体"), ("acc_fuzzy", "崩れた鍵"), ("acc_age0", "最近 16 区切り"), ("acc_age16", "16〜256 区切り前"), ("acc_age256", "256 区切り以上前"),
        ("acc_5up", "5 回以上来た"), ("acc_important", "重要")]
fig, ax = plt.subplots(1, 2, figsize=(18, 5.3))
for a, kind, nm in ((ax[0], "rand", "でたらめな鍵"), (ax[1], "word_hash", "単語の鍵（〈他〉を別々）")):
    x = np.arange(len(cols))
    for j, (m, lab) in enumerate(meths):
        a.bar(x + (j - 1.5) * 0.2, [np.mean([r[c] for r in D[(kind, m)]]) for c, _ in cols], 0.2, label=lab)
    a.axhline(np.mean([r["dict_fuzzy"] for r in D[(kind, "ring64_keep")]]), color="k", ls="--", lw=1, label="デジタルの辞書（崩れた鍵、容量無限）")
    a.set_xticks(x)
    a.set_xticklabels([n for _, n in cols], fontsize=8)
    a.set_ylim(0, 105)
    a.set_title(f"{nm}（出てきた記憶 約 {np.mean([r['n_seen'] for r in D[(kind, 'ring64_keep')]]):.0f} 組）", fontsize=10)
ax[0].set_ylabel("取り出し %（3 seed）")
ax[0].legend(fontsize=7)
fig.suptitle("実験100: 記憶の候補 3000 組から 16 組 × 2400 区切り（新しい記憶が次々来る）。板はどれも 64 枚")
fig.tight_layout()
path = os.path.join(ROOT, "docs", "images", "exp100_gears.png")
fig.savefig(path, dpi=110)
print(path)
