"""実験95 の図: 段の選び方と長い流れ（中期の系列）。実行: python tools/fig_exp95.py"""
import collections
import json
import os

import numpy as np

from figio import ROOT, plt

D = collections.defaultdict(list)
for l in open(os.path.join(ROOT, "results", "exp95.txt"), encoding="utf-8"):
    r = json.loads(l)
    D[(r["T"], r["keys"], r["method"])].append(r)
f = lambda rs, k: np.mean([r[k] for r in rs if r.get(k) is not None])
fig, ax = plt.subplots(1, 2, figsize=(17, 5.5))
sels = [("acc", "確信度で 1 段を選ぶ"), ("acc_sel_recent0.9", "新しい段を優先（0.9）"), ("acc_sel_sum", "全段の明るさを足す"), ("acc_sel_vote", "全段の割合を足す（多数決）"),
        ("acc_sel_conf_sum", "確信度で重みを付けて足す"), ("acc_oracle", "選び方が完全なら")]
cases = [((60, "word", "flush_r1.5"), "単語・60 区切り"), ((600, "word", "flush_r1.5_mid"), "単語・600 区切り + 中期"), ((600, "rand", "flush_r1.5_mid"), "でたらめ・600 区切り + 中期")]
x = np.arange(len(cases))
for j, (k, nm) in enumerate(sels):
    ax[0].bar(x + (j - 2.5) * 0.14, [f(D[c], k) for c, _ in cases], 0.14, label=nm, color="#27ae60" if k == "acc_oracle" else None)
ax[0].set_xticks(x)
ax[0].set_xticklabels([n for _, n in cases])
ax[0].set_ylabel("取り出し %（3 seed）")
ax[0].set_title("1. 段の選び方（満ちたら送る、1.5 倍ずつ 10 段）", fontsize=10)
ax[0].legend(fontsize=7)
meths = [("blocks", "時間で分けて倍々（実験92）"), ("flush_r1.5", "満ちたら送る（短期だけ）"), ("flush_r1.5_mid", "満ちたら送る + 中期の系列")]
cols = [("acc", "全体"), ("acc_recent", "最近"), ("acc_old", "30 回以上前"), ("acc_5up", "5 回以上"), ("acc_important", "重要"), ("changed_new", "変わった値")]
xx = np.arange(len(cols))
for j, (m, nm) in enumerate(meths):
    ax[1].bar(xx + (j - 1) * 0.27, [f(D[(600, "rand", m)], c) for c, _ in cols], 0.27, label=nm)
ax[1].set_xticks(xx)
ax[1].set_xticklabels([n for _, n in cols])
ax[1].set_title("2. 長い流れ（600 区切り、でたらめな鍵）: 中期の系列が効く", fontsize=10)
ax[1].legend(fontsize=8)
fig.suptitle("実験95: 実験94 のマイナスポイントへの手当て")
fig.tight_layout()
path = os.path.join(ROOT, "docs", "images", "exp95_select_long.png")
fig.savefig(path, dpi=110)
print(path)
