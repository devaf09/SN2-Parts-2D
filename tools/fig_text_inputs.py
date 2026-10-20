"""実験39・40 の文字列の入れ方のイメージ（docs/images/text_inputs.png）。同じ文脈を 7 通りの入れ方で場に載せ、
上段 = 入力の場（明るさ × 位相の色。明るい所ほど光が強い）、下段 = レンズで見た明るさ（対数）。
位置ごとの明るさは、実験40 で学習したもの（直前ほど明るい）に近い値を使う。

    python tools/fig_text_inputs.py
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
from exp40_inputs import EncNet  # noqa: E402

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
(X, y), _, info = textdata.load(8, 64, 2000, 10, 0)
voc = info["in_vocab"]
k = 5
x = X[k:k + 1]
ctx = "".join(voc[c] if c < len(voc) else "？" for c in x[0].tolist())
learned = torch.log(torch.tensor([0.03, 0.03, 0.04, 0.06, 0.08, 0.16, 1.11, 3.6]))   # 実験40 で学習した明るさに近い値


def complex_rgb(u):
    """複素数の場 → 色（色相 = 位相、明るさ = 振幅）。"""
    a = u.abs()
    a = (a / (a.max() + 1e-9)).numpy()
    h = ((torch.angle(u) + torch.pi) / (2 * torch.pi)).numpy()
    rgb = [[colorsys.hsv_to_rgb(h[i, j], 1.0, a[i, j]) for j in range(u.shape[1])] for i in range(u.shape[0])]
    return rgb


names = {"bands": "帯（いままで）", "blur": "にじませ", "smear": "前後の平均", "rainbow": "2 次元の虹",
         "spread": "拡散符号（参照光）", "rings": "同心円"}
fig, ax = plt.subplots(2, 7, figsize=(24, 7.4))
with torch.no_grad():
    for j, enc in enumerate(names):
        m = EncNet(enc, 8, info["V_in"], 64, layers=1, patches=2, seed=0)
        m.w.copy_(learned)
        u = m.field(x)[0]
        ax[0, j].imshow(complex_rgb(u))
        ax[0, j].set_title(names[enc], fontsize=11)
        ax[1, j].imshow(torch.log1p(p2.intensity(p2.lens(u))), cmap="magma")
    # 位置ごとに色を分ける（実験39）: 帯ごとに別の色の光。明るさを足すだけ（干渉しない）
    m = EncNet("bands", 8, info["V_in"], 64, layers=1, patches=2, seed=0)
    m.w.copy_(learned)
    a = torch.exp(learned)
    a = a / a.pow(2).mean().sqrt()
    band = (torch.arange(64) * 8 // 64)
    img = torch.zeros(64, 64, 3)
    lens_sum = torch.zeros(64, 64)
    for pos in range(8):
        r, g_, b = colorsys.hsv_to_rgb(pos / 8, 1, 1)
        mask = (band == pos).float()[:, None].expand(64, 64)
        u = m.field(x)[0] * mask
        img += (u.abs() / u.abs().max().clamp(min=1e-9))[..., None] * torch.tensor([r, g_, b]) * float(a[pos] / a.max()) ** 0.5
        lens_sum += p2.intensity(p2.lens(u))
    ax[0, 6].imshow(img.clamp(0, 1).numpy())
    ax[0, 6].set_title("位置ごとに色を分ける（実験39）\n色 = 位置（波長）", fontsize=11)
    ax[1, 6].imshow(torch.log1p(lens_sum), cmap="magma")
for a_ in ax.flat:
    a_.axis("off")
ax[0, 0].text(-4, 32, "入力の場\n（色 = 位相、\n明るさ = 光の強さ）", ha="right", va="center", fontsize=10)
ax[1, 0].text(-4, 32, "レンズで\n見た明るさ\n（対数）", ha="right", va="center", fontsize=10)
fig.suptitle(f"文字列の入れ方 7 通り（文脈「{ctx}」、位置ごとの明るさは学習後に近い値: 直前の文字がいちばん明るい）", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "docs", "images", "text_inputs.png"), dpi=90, bbox_inches="tight")
print("docs/images/text_inputs.png")
