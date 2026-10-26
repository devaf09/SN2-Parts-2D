"""実験99 の図: 板 64 枚どうしの比べと、融合による忘却（3 seed）。実行: python tools/fig_exp99.py"""
import collections
import json
import os

import numpy as np

from figio import ROOT, plt

D = collections.defaultdict(list)
for f in ("exp99_ring", "exp99_flow"):
    for l in open(os.path.join(ROOT, "results", f + ".txt"), encoding="utf-8"):
        r = json.loads(l)
        D[(r["T"], r["keys"], r["method"])].append(r)
meths = [("flow64_r1.1", "流れ型 64 枚（1.1 倍ずつ）"), ("ring64_keep", "環状 64 枚・忘却なし"), ("ring64_decay", "環状・ただ薄れる（0.995 倍）"),
         ("ring64_lap", "環状・1 周ごとに半分"), ("ring64_consol", "環状・長期の環（16 枚）へ融合")]
fig, ax = plt.subplots(1, 2, figsize=(17, 5.2))
for a, kind, nm in ((ax[0], "rand", "でたらめな鍵"), (ax[1], "word_hash", "単語の鍵（〈他〉を別々）")):
    cases = [(600, "全体（600）"), (2400, "全体（2400）")]
    x = np.arange(len(cases) + 2)
    for j, (m, lab) in enumerate(meths):
        vals = [np.mean([r["acc"] for r in D[(T, kind, m)]]) for T, _ in cases]
        vals += [np.mean([r["acc_recent"] for r in D[(2400, kind, m)]]), np.mean([r["acc_old"] for r in D[(2400, kind, m)]])]
        a.bar(x + (j - 2) * 0.16, vals, 0.16, label=lab)
    a.set_xticks(x)
    a.set_xticklabels([n for _, n in cases] + ["最近 10 区切り（2400）", "30 区切り以上前（2400）"], fontsize=9)
    a.set_ylim(0, 105)
    a.set_title(nm, fontsize=10)
ax[0].set_ylabel("取り出し %")
ax[0].legend(fontsize=7)
fig.suptitle("実験99: 板 64 枚どうしで比べる・融合による忘却（記憶の候補は 300 組のまま、流れの長さ 600・2400 区切り）")
fig.tight_layout()
path = os.path.join(ROOT, "docs", "images", "exp99_plates64.png")
fig.savefig(path, dpi=110)
print(path)
