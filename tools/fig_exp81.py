"""実験80・81 の図: 分担の 2DTG（4f 系）と、光路 1 本との確率の足し合わせ（seed 0、足し合わせはテストの後半で測る）。実行: python tools/fig_exp81.py"""
import json
import os

from figio import ROOT, plt

R = [json.loads(l) for l in open(os.path.join(ROOT, "results", "exp81.txt"), encoding="utf-8")]
nm = {"g8_n8": "担当 8・直前 8 文字", "g16_n4": "担当 16・直前 4 文字", "g4_n8": "担当 4・直前 8 文字", "g16_n4_small": "担当 16・直前 4 文字・小さく", "g8_n4": "担当 8・直前 4 文字"}
fig, ax = plt.subplots(figsize=(11, 5.5))
labels, a1, a2, f = [], [], [], []
for r in R:
    ks = [os.path.basename(p)[:-3] for p in r["files"][1:]]
    labels.append(" + ".join(nm[k] for k in ks))
    a1.append(r["alone"][0][0])
    a2.append(max(x[0] for x in r["alone"][1:]))
    f.append(r["fused"][0])
y = range(len(R))
ax.barh([i + 0.27 for i in y], a1, 0.27, color="#95a5a6", label="光路 1 本（4f 系 6 段 × 8）だけ")
ax.barh(list(y), a2, 0.27, color="#2e86c1", label="分担の 2DTG（4f 系）だけ")
ax.barh([i - 0.27 for i in y], f, 0.27, color="#c0392b", label="確率を足し合わせ")
for i, v in enumerate(f):
    ax.text(v + 0.1, i - 0.27, f"{v:.1f}", va="center", fontsize=8)
for v, n, c in [(53.7, "3-gram", "#e67e22"), (54.0, "分担の 2DTG（板、実験62）", "#8e44ad"), (55.4, "4-gram", "#d35400")]:
    ax.axvline(v, ls="--", lw=1, color=c)
    ax.text(v, len(R) - 0.4, n, rotation=90, fontsize=7, va="top", ha="right", color=c)
ax.set_yticks(list(y))
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlim(50, 58)
ax.set_xlabel("テストの正解率 %（テストの後半 5000 件。重みは前半で選んだ）")
ax.set_title("実験81: 文字の課題、分担の 2DTG（4f 系）と光路 1 本（4f 系）の確率を足し合わせる（seed 0）")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
path = os.path.join(ROOT, "docs", "images", "exp81_fuse.png")
fig.savefig(path, dpi=110)
print(path)
