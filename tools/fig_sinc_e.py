"""⑤ 楕円で大きさを変えられる減衰する波紋（持ち主の案）の図を docs/images/sinc_ellipse.png に書き出す。

    python tools/fig_sinc_e.py
"""
from __future__ import annotations

import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
import sn2_parts_2d as p2  # noqa: E402

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
L = 64
y, x = p2.grid(L, L)
T = lambda v: torch.tensor([float(v)])
cases = [("④ と同じ（円・大きさ大）\nq=0, s=200", 0, 0, 200),
         ("大きさを小さく\ns=12", 0, 0, 12),
         ("大きさを中くらい\ns=24", 0, 0, 24),
         ("楕円（横長 2 倍）\nq=1.4", 1.4, 0, 200),
         ("楕円を 45° 傾ける\nq=1.4, θ=45°", 1.4, math.pi / 4, 200),
         ("楕円 + 小さく\nq=1.4, θ=45°, s=16", 1.4, math.pi / 4, 16)]
fig, ax = plt.subplots(1, 6, figsize=(14, 3))
for a, (t, q, th, s) in zip(ax, cases):
    ph = p2.plate_sinc_ellipse(T(2), T(0.5), T(32), T(32), T(q), T(th), T(math.log(s)), y, x)
    a.imshow(ph, cmap="RdBu_r", vmin=-2, vmax=2)
    a.set_title(t, fontsize=9)
    a.axis("off")
fig.suptitle("⑤ 楕円で大きさを変えられる減衰する波紋: A · sin(k·ρ)/(k·ρ) · exp(−(ρ/s)²)（ρ = 楕円の距離。A=2, k=0.5）", y=1.1)
fig.tight_layout()
os.makedirs(os.path.join(ROOT, "docs", "images"), exist_ok=True)
fig.savefig(os.path.join(ROOT, "docs", "images", "sinc_ellipse.png"), dpi=110, bbox_inches="tight")

fig, ax = plt.subplots(1, 3, figsize=(10, 3.3))
for a, (t, kind) in zip(ax, [("④ 波紋 32 個", "sinc"), ("⑤ 楕円・大きさつき 18 個", "sinc_e"), ("⑤（学習前、別の seed）", "sinc_e")]):
    net = p2.Net2D(L, 8, [kind], seed=3 if "別" not in t else 5)
    with torch.no_grad():
        a.imshow(net.plate_phase(0), cmap="twilight")
    a.set_title(t, fontsize=10)
    a.axis("off")
fig.suptitle("学習前の板（同じ学習する数: 約 2000 個 / 16 枚）")
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "docs", "images", "plates_sinc_ellipse.png"), dpi=110, bbox_inches="tight")
print("docs/images/sinc_ellipse.png, docs/images/plates_sinc_ellipse.png")
