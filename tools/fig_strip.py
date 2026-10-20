"""実験48 の短冊形の入れ方の図（docs/images/strip_input.png）。文字ごとに照らす範囲（前側をマスクで切る）と、重ねた光。

    python tools/fig_strip.py
"""
from __future__ import annotations

import colorsys
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments", "text")]
import sn2_parts_2d as p2  # noqa: E402
import textdata  # noqa: E402
from exp48_strip import StripNet  # noqa: E402

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
(X, y), _, info = textdata.load(8, 64, 2000, 10, 0)
voc = info["in_vocab"]
x = X[5:6]
ctx = "".join(voc[c] if c < len(voc) else "？" for c in x[0].tolist())
m = StripNet(8, info["V_in"], 64)
with torch.no_grad():
    m.w.zero_()
    fig, ax = plt.subplots(1, 4, figsize=(13, 7.5), gridspec_kw={"width_ratios": [1.6, 1, 1, 1]})
    for k in range(8):
        r, g, b = colorsys.hsv_to_rgb(k / 8, 0.9, 1)
        ys = m.M[k].sum(1).nonzero()[:, 0]
        ax[0].add_patch(plt.Rectangle((k * 1.1, ys.min()), 1, len(ys), color=(r, g, b), alpha=0.8))
        ax[0].text(k * 1.1 + 0.5, ys.min() - 4, ctx[k], ha="center", fontsize=12)
    ax[0].set_xlim(-0.3, 9)
    ax[0].set_ylim(m.Ly, -12)
    ax[0].set_xticks([])
    ax[0].set_ylabel("短冊の縦の位置（画素）")
    ax[0].set_title("文字ごとに照らす範囲\n（0.5 ずつずらし、3 単位分だけ下を照らす）", fontsize=10)
    u = m.field(x)[0]
    ax[1].imshow(u.abs(), cmap="gray", aspect="auto")
    ax[1].set_title("重ねた光の強さ\n（重なりが多い所ほど明るい）", fontsize=10)
    ax[2].imshow(torch.angle(u), cmap="twilight", aspect="auto")
    ax[2].set_title("重ねた光の位相\n（文字の向き × 下向きのプリズム）", fontsize=10)
    ax[3].imshow(torch.log1p(p2.intensity(p2.lens(u / u.abs().pow(2).mean().sqrt()))), cmap="magma", aspect="auto")
    ax[3].set_title("レンズで見た明るさ（対数）", fontsize=10)
    for a in ax[1:]:
        a.axis("off")
fig.suptitle(f"実験48 の短冊形（{m.Ly} × {m.Lx} 画素、幅 1 単位 = {m.U} 画素）。文脈「{ctx}」", y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "docs", "images", "strip_input.png"), dpi=100, bbox_inches="tight")
print("docs/images/strip_input.png")
