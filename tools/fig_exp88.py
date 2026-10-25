"""実験88 の図: 似た鍵での記憶 Grain の融合と確信度（3 seed の平均）。実行: python tools/fig_exp88.py"""
import collections
import json
import os

import numpy as np

from figio import ROOT, plt

D = collections.defaultdict(list)
for l in open(os.path.join(ROOT, "results", "exp88.txt"), encoding="utf-8"):
    r = json.loads(l)
    D[(r["keys"], r["n"], r["alpha"])].append(r)
m = lambda rs, f: np.mean([f(r) for r in rs])
kinds = [("rand", "でたらめ"), ("word", "単語（拡散）"), ("sim0.25", "似せた"), ("sim0.5", "似せた"), ("sim0.75", "似せた")]
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
x = np.arange(len(kinds))
for j, (n, c) in enumerate([(8, "#27ae60"), (32, "#2e86c1"), (128, "#c0392b")]):
    g1 = [m(D[(k, n, 0.8)], lambda r: r["G1"]["acc"]) for k, _ in kinds]
    g2 = [m(D[(k, n, 0.8)], lambda r: r["G2"]["acc"]) for k, _ in kinds]
    ax[0].bar(x + (j - 1) * 0.27, g1, 0.27, color=c, alpha=0.9, label=f"{n} 組ずつ・G1（80%）")
    ax[0].bar(x + (j - 1) * 0.27, g2, 0.13, color="k", alpha=0.45, label="G2（20%、細い黒）" if j == 0 else None)
ax[0].set_xticks(x)
ax[0].set_xticklabels([f"{nm}\n似度 {m(D[(k, 32, 0.8)], lambda r: r['key_sim']):.2f}" for k, nm in kinds], fontsize=8)
ax[0].set_ylabel("取り出しの正解率 %")
ax[0].set_title("鍵の似ている度合いと取り出し（80 : 20）", fontsize=10)
ax[0].legend(fontsize=7)
al = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for n, c in [(32, "#2e86c1"), (128, "#c0392b")]:
    ax[1].plot(al, [m(D[("word", n, a)], lambda r: r["G1"]["margin"]) for a in al], "-o", color=c, ms=3, label=f"{n} 組・G1（短期）")
    ax[1].plot(al, [m(D[("word", n, a)], lambda r: r["G2"]["margin"]) for a in al], "--o", color=c, ms=3, label=f"{n} 組・G2")
ax[1].set_yscale("log")
ax[1].set_xlabel("G1 の比率 α")
ax[1].set_ylabel("確信度（一番明るい窓 / 二番目）")
ax[1].set_title("確信度（単語の鍵）: 比率の大きい記憶ほど確信度が高い", fontsize=10)
ax[1].legend(fontsize=7)
for n, c in [(32, "#2e86c1"), (128, "#c0392b")]:
    ax[2].plot(al, [m(D[("word", n, a)], lambda r: r["conflict"]["new"]) for a in al], "-o", color=c, ms=3, label=f"{n} 組・新しい値（G1）が出る")
    ax[2].plot(al, [m(D[("word", n, a)], lambda r: r["conflict"]["old"]) for a in al], "--o", color=c, ms=3, label=f"{n} 組・古い値（G2）が出る")
ax[2].set_xlabel("G1（短期）の比率 α")
ax[2].set_ylabel("%")
ax[2].set_title("食い違い（同じ鍵に新旧の値、単語の鍵）: 短期が勝つ", fontsize=10)
ax[2].legend(fontsize=7)
fig.suptitle("実験88: 記憶 Grain の重ね書き。似た鍵での取り出し・確信度・新旧の食い違い")
fig.tight_layout()
path = os.path.join(ROOT, "docs", "images", "exp88_memkeys.png")
fig.savefig(path, dpi=110)
print(path)
