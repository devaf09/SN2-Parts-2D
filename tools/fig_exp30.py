"""実験30 の図: 10 本の光路（数字ごと）を融合し、場所ごとに一番光った光路の数字を色と文字で示す（docs/images/exp30_ten.png）。

    python tools/fig_exp30.py
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments", "image")]
import sn2_parts_2d as p2  # noqa: E402
from exp27_locate import Fourier4fNL, make_scenes  # noqa: E402

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
paths = []
for d in range(10):
    m = Fourier4fNL(0)
    m.load_state_dict(torch.load(os.path.join(ROOT, "results", "models", f"exp30_d{d}_s0.pt"), map_location="cpu", weights_only=False)["state"])
    paths.append(m)
X, objs = make_scenes(200, 400, False, uniform=True)
pick = list(range(12))
cmap = plt.get_cmap("tab10")
fig, ax = plt.subplots(2, 12, figsize=(22, 4.4))
with torch.no_grad():
    for c, i in enumerate(pick):
        u = p2.image_to_field(X[i:i + 1], "amp")
        P = torch.stack([torch.sigmoid(m(u))[0] for m in paths])          # (10, L, L)
        best, who = P.max(0)
        rgb = torch.tensor(cmap(who.numpy())[..., :3]).float() * best[..., None].clamp(0, 1)
        ax[0, c].imshow(X[i], cmap="gray")
        ax[0, c].set_title("入力: " + "・".join(str(d) for _, _, d in objs[i]), fontsize=9)
        ax[1, c].imshow(rgb * 0.9 + X[i][..., None] * 0.1)
        for cy, cx, _ in objs[i]:
            win = P[:, max(cy - 4, 0):cy + 5, max(cx - 4, 0):cx + 5].flatten(1).max(1).values
            ax[1, c].text(cx, cy - 12, f"{int(win.argmax())}（{float(win.max()) * 100:.0f}%）", color="w", fontsize=8, ha="center")
        for a in ax[:, c]:
            a.axis("off")
fig.suptitle("10 本の光路（数字ごと）を融合。色 = その場所で一番光った光路の数字、明るさ = 確率。文字 = 数字の位置での答えと確率")
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "docs", "images", "exp30_ten.png"), dpi=90, bbox_inches="tight")
print("docs/images/exp30_ten.png")
