"""「入力 → 学習後の板 → 出力」の図を作るための共通の道具（fig_exp45.py・fig_exp48.py・fig_exp67.py が使う）。"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "text"))
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]


def train(m, X, y, steps=1500, lr=0.01, seed=0, dev="cuda"):
    """実験と同じ学習（Adam、学習率 lr → 0、256 件ずつ）。"""
    m = m.to(dev)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        i = torch.randint(len(y), (256,), generator=g)
        loss = F.cross_entropy(m(X[i].to(dev)), y[i].to(dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
    return m.eval()


def show_field(ax, u, title, kind="phase"):
    """光の場 u (Ly, Lx) を表示。phase = 位相を色・明るさを濃さで、amp = 明るさだけ。"""
    u = u.detach().cpu()
    if kind == "phase":
        a = u.abs()
        a = (a / (a.max() + 1e-12)).numpy()
        rgba = plt.get_cmap("twilight")((torch.angle(u).numpy() + 3.1416) / 6.2832)
        rgba[..., :3] *= a[..., None] ** 0.5
        ax.imshow(rgba)
    else:
        ax.imshow(u.abs().numpy() ** 2, cmap="magma")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def show_plate(ax, ph, title):
    ax.imshow(torch.remainder(ph.detach().cpu(), 6.2832).numpy(), cmap="twilight", vmin=0, vmax=6.2832)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def show_output(ax, I, det, title):
    """レンズの後の明るさ（対数）に、窓の場所を白い線で重ねる。"""
    I = I.detach().cpu()
    ax.imshow(torch.log1p(I / (I.mean() + 1e-12)).numpy(), cmap="inferno")
    ax.contour((det.detach().cpu().sum(0) > 0).float().numpy(), levels=[0.5], colors="white", linewidths=0.3)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def show_bars(ax, probs, labels, title, truth=None, k=5):
    v, i = probs.detach().cpu().topk(k)
    names = [labels[j] for j in i.tolist()]
    cols = ["#c0392b" if (truth is not None and j == truth) else "#2e86c1" for j in i.tolist()]
    ax.barh(range(k)[::-1], (100 * v).numpy(), color=cols)
    ax.set_yticks(range(k)[::-1])
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("%", fontsize=8)
    ax.set_title(title, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
