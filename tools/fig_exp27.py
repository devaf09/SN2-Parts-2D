"""実験27 の図: 4f 系（光学相関器）の出力。3 の光路を赤、7 の光路を緑として、入力の絵に重ねる（docs/images/exp27_locate.png）。

    python tools/fig_exp27.py
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
import sn2_parts_2d as p2  # noqa: E402
from exp27_locate import Fourier4f, make_scenes  # noqa: E402

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
ms = {}
for d in (3, 7):
    ck = torch.load(os.path.join(ROOT, "results", "models", f"exp27_fourier_d{d}_s0.pt"), map_location="cpu", weights_only=False)
    m = Fourier4f(1)
    m.load_state_dict(ck["state"])
    ms[d] = m
X, objs = make_scenes(400, 200, False)
pick = [i for i, o in enumerate(objs) if len(o) == 2][:8]
fig, ax = plt.subplots(2, 8, figsize=(17, 4.8))
with torch.no_grad():
    for c, i in enumerate(pick):
        u = p2.image_to_field(X[i:i + 1], "amp")
        p3 = torch.sigmoid(ms[3](u))[0]
        p7 = torch.sigmoid(ms[7](u))[0]
        ax[0, c].imshow(X[i], cmap="gray")
        ax[0, c].set_title("入力: " + "・".join(str(d) for _, _, d in objs[i]), fontsize=9)
        rgb = torch.stack([p3, p7, torch.zeros_like(p3)], -1).clamp(0, 1) * 0.85 + X[i][..., None] * 0.15
        ax[1, c].imshow(rgb)
        ax[1, c].set_title(f"3: {float(p3.max()) * 100:.0f}%  7: {float(p7.max()) * 100:.0f}%", fontsize=9)
        for a in ax[:, c]:
            a.axis("off")
fig.suptitle("4f 系（光学相関器）の出力。赤 = 3 の光路、緑 = 7 の光路（2 本を別の色で重ねた融合）。数字は各光路の最大の確率")
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "docs", "images", "exp27_locate.png"), dpi=100, bbox_inches="tight")
print("docs/images/exp27_locate.png")
