"""実験77 の図: カー効果の強さと位相の縮め方で正解率がどう変わるか（3 seed の平均）。実行: python tools/fig_exp77.py"""
import collections
import json
import os

import numpy as np

from figio import ROOT, plt

D = collections.defaultdict(list)
for l in open(os.path.join(ROOT, "results", "exp77.txt"), encoding="utf-8"):
    r = json.loads(l)
    D[(r["task"], r["read"], r["kind"], r["fix_p"], r["clip"])].append(r["test_acc"])
m = lambda k: (np.mean(D[k]), np.std(D[k]))
col = {("shape", "lens"): "#2e86c1", ("shape", "plane"): "#85c1e9", ("mnist", "lens"): "#c0392b", ("mnist", "plane"): "#f1948a"}
nm = {"shape": "形の課題", "mnist": "MNIST", "lens": "レンズの後で読む", "plane": "出口の面で読む"}
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
ks = [0.03, 0.1, 0.3, 1.0]
for (t, rd), c in col.items():
    mu = np.array([m((t, rd, "kerr", k, 0.0)) for k in ks])
    ax[0].errorbar(ks, mu[:, 0], mu[:, 1], marker="o", color=c, capsize=3, label=f"{nm[t]}・{nm[rd]}")
    ax[0].scatter([2.0], [m((t, rd, "kerrn", None, 0.0))[0]], marker="*", s=150, color=c, edgecolor="k", zorder=5)
    ax[0].scatter([3.0], [m((t, rd, "sqrt", None, 0.0))[0]], marker="s", s=60, color=c, edgecolor="k", zorder=5)
ax[0].set_xscale("log")
ax[0].set_xticks(ks + [2.0, 3.0])
ax[0].set_xticklabels(["0.03", "0.1", "0.3", "1\n（今まで）", "★ 明るい点で\n抑えたカー", "■ 平方根"], fontsize=8)
ax[0].set_ylabel("テストの正解率 %（3 seed の平均 ± ばらつき）")
ax[0].set_title("カー効果の強さ κ（固定）: 弱いほど良い")
ax[0].legend(fontsize=8)
ax[0].set_ylim(0, 100)
ps = [0.0, 0.25, 0.5, 1.0]
for t in ("shape", "mnist"):
    mu = np.array([m((t, "lens", "phfrac", p, 0.0)) for p in ps])
    ax[1].errorbar(ps, mu[:, 0], mu[:, 1], marker="o", color=col[(t, "lens")], capsize=3, label=f"{nm[t]}・レンズの後で読む")
ax[1].set_xticks(ps)
ax[1].set_xticklabels(["0\n（位相を捨てる\n= 平方根）", "0.25", "0.5", "1\n（何もしない\n= 線形）"])
ax[1].set_xlabel("位相を何倍に縮めるか p（振幅はそのまま）")
ax[1].set_title("位相を少しでも崩せば効く（崩さないと線形と同じ）")
ax[1].legend(fontsize=8)
ax[1].set_ylim(0, 100)
fig.suptitle("実験77: 4f 系 3 段 × 4 本で、うまくいかない理由を探る（seed 0〜2）")
fig.tight_layout()
path = os.path.join(ROOT, "docs", "images", "exp77_why.png")
fig.savefig(path, dpi=110)
print(path)
