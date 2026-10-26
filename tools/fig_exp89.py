"""実験89 の図: 短期記憶から長期記憶へ移す方法の比べ（3 seed の平均）。実行: python tools/fig_exp89.py"""
import collections
import json
import os

import numpy as np

from figio import ROOT, plt

D = collections.defaultdict(list)
for l in open(os.path.join(ROOT, "results", "exp89.txt"), encoding="utf-8"):
    r = json.loads(l)
    D[(r["keys"], r["method"])].append(r)
f = lambda rs, k: np.mean([r[k] for r in rs if r[k] is not None]) if any(r[k] is not None for r in rs) else np.nan
meth = [("plain", "そのまま書く"), ("diff", "差分（共通部分を引く）"), ("repeat", "差分 + 繰り返しで 3 倍"), ("relate", "差分 + 関連で重く"),
        ("update", "差分 + 古い値を引く"), ("gated", "差分 + 足りない分だけ書く"), ("gated_upd", "足りない分 + 古い値を引く")]
kinds = [("rand", "でたらめな鍵", "#7f8c8d"), ("word", "単語の並びの鍵", "#2e86c1"), ("sim0.5", "似せた鍵（0.5）", "#c0392b")]
fig, ax = plt.subplots(1, 2, figsize=(17, 5.5), gridspec_kw={"width_ratios": [3, 2]})
x = np.arange(len(meth))
for j, (k, nm, c) in enumerate(kinds):
    ax[0].bar(x + (j - 1) * 0.27, [f(D[(k, m)], "acc") for m, _ in meth], 0.27, color=c, label=nm)
ax[0].set_xticks(x)
ax[0].set_xticklabels([n for _, n in meth], fontsize=8, rotation=15)
ax[0].set_ylabel("出てきた全部の組の取り出し %（最新の値が正解）")
ax[0].set_title("移し方の比べ（300 組から 16 組 × 60 回、長期は毎回 0.97 倍に薄れる）", fontsize=10)
ax[0].legend(fontsize=8)
cols = [("acc_once", "1 回だけ"), ("acc_2to4", "2〜4 回"), ("acc_5up", "5 回以上"), ("acc_important", "重要の印"), ("acc_recent", "最近 10 回以内"),
        ("acc_old", "30 回以上前"), ("changed_new", "値が変わった組:\n新しい値")]
xx = np.arange(len(cols))
for j, (m, nm, c) in enumerate([("diff", "差分", "#95a5a6"), ("gated", "差分 + 足りない分だけ", "#27ae60")]):
    ax[1].bar(xx + (j - 0.5) * 0.38, [f(D[("word", m)], k) for k, _ in cols], 0.38, color=c, label=nm)
ax[1].set_xticks(xx)
ax[1].set_xticklabels([n for _, n in cols], fontsize=8)
ax[1].set_title("単語の鍵: どんな記憶が残るか", fontsize=10)
ax[1].legend(fontsize=8)
fig.suptitle("実験89: 短期記憶 → 長期記憶（4f 系の板への重ね書き）。差分記憶法と「足りない分だけ書く」繰り返し記憶")
fig.tight_layout()
path = os.path.join(ROOT, "docs", "images", "exp89_consolidate.png")
fig.savefig(path, dpi=110)
print(path)
