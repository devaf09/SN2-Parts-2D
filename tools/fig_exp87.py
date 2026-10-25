"""実験87 の図: 2 つの記憶 Grain を比率 α : 1 − α で融合したときの取り出しの正解率とピークの明るさ（3 seed の平均）。実行: python tools/fig_exp87.py"""
import collections
import json
import os

import numpy as np

from figio import ROOT, plt

D = collections.defaultdict(list)
for l in open(os.path.join(ROOT, "results", "exp87.txt"), encoding="utf-8"):
    r = json.loads(l)
    if len(r["w"]) == 2:
        D[(r["mode"], r["n"], r["w"][0])].append(r)
al = sorted({k[2] for k in D})
mname = {"field": "1 枚に重ね書き（二重露光）", "phase": "重ね書きして位相だけ残す", "inten": "板は別々、明るさを足す（光路 2 本）"}
col = {8: "#27ae60", 32: "#2e86c1", 128: "#c0392b"}
fig, ax = plt.subplots(1, 4, figsize=(19, 4.8))
for a, mode in zip(ax[:3], ("field", "phase", "inten")):
    for n, c in col.items():
        for gi, ls in ((0, "-"), (1, "--")):
            v = [np.mean([r["grains"][gi]["acc"] for r in D[(mode, n, x)]]) for x in al]
            a.plot(al, v, ls, color=c, marker="o", ms=3, label=f"{n} 組ずつ・{'G1' if gi == 0 else 'G2'}")
    a.set_title(mname[mode], fontsize=10)
    a.set_xlabel("G1 の比率 α（G2 は 1 − α）")
    a.set_ylim(-3, 103)
ax[0].set_ylabel("取り出しの正解率 %（窓 256 個から正しい値）")
ax[0].legend(fontsize=7)
for n, c in col.items():
    v = [np.mean([r["grains"][0]["peak"] / (r["grains"][0]["peak"] + r["grains"][1]["peak"]) for r in D[("field", n, x)]]) for x in al]
    ax[3].plot(al, v, color=c, marker="o", ms=3, label=f"{n} 組ずつ")
ax[3].plot([0, 1], [0, 1], "k:", lw=1, label="比率どおり")
ax[3].set_title("ピークの明るさの比 G1 / (G1 + G2)（重ね書き）", fontsize=10)
ax[3].set_xlabel("G1 の比率 α")
ax[3].legend(fontsize=7)
fig.suptitle("実験87: 記憶 Grain 2 つ（鍵 → 値の組を書き込んだ 4f 系の板）を比率で融合する。実線 = G1、破線 = G2")
fig.tight_layout()
path = os.path.join(ROOT, "docs", "images", "exp87_memfuse.png")
fig.savefig(path, dpi=110)
print(path)
