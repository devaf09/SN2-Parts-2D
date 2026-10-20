"""文字列を 2 次元の場に載せる方法の図（docs/images/text_encode.png）。実験31〜37 の encode。

    python tools/fig_text_encode.py
"""
from __future__ import annotations

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
from exp31_text2d import encode  # noqa: E402

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
(X, y), _, info = textdata.load(8, 64, 2000, 10, 0)
voc = info["in_vocab"]
i = 5
s = "".join(voc[c] if c < len(voc) else "？" for c in X[i].tolist())
u = encode(X[i:i + 1], 8, info["V_in"])[0]
fig, ax = plt.subplots(1, 3, figsize=(14, 4.6))
ax[0].imshow(torch.angle(u), cmap="twilight")
for m, ch in enumerate(s):
    ax[0].text(-3, m * 8 + 4, ch, ha="right", va="center", fontsize=12)
ax[0].set_title(f"入力の場の位相（文脈「{s}」）\n縦 8 行ずつ = 1 文字。上が古く、下が直前の文字", fontsize=10)
ax[0].set_xticks([])
ax[0].set_yticks([])
I = p2.intensity(p2.lens(u))
ax[1].imshow(torch.log1p(I), cmap="magma")
ax[1].set_title("レンズで見た明るさ（対数）\n各文字が、文字ごとの向きの点（縦に少しにじむ）になる", fontsize=10)
ax[1].axis("off")
one = encode(X[i:i + 1, -1:].repeat(1, 1), 1, info["V_in"])[0]
ax[2].imshow(torch.angle(one), cmap="twilight")
ax[2].set_title(f"参考: 1 文字「{s[-1]}」だけを場全体に（光の記憶の実験34 の入れ方）", fontsize=10)
ax[2].axis("off")
fig.suptitle("文字列の渡し方: 位置ごとの帯に、文字ごとに決まった向きの平面波（明るさ 1、位相だけの模様）を置く")
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "docs", "images", "text_encode.png"), dpi=100, bbox_inches="tight")
print("docs/images/text_encode.png")
