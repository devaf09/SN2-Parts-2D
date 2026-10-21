"""図: 光路の表現の力が足りない、とはどういうことか（持ち主の依頼、2026-10-21）。

(a) 入力の語 = 光の向き。板 1 枚の光路では、向きを少し変えた入力の出口の模様は、同じ模様がずれただけになる（ホログラムの「ずれるだけ」）。
    板を重ねると、ずれただけではない別の模様になっていく。
(b) (a) を数で: 隣の向きの入力どうしで、「ずらして重ねたときの模様の似かた」（1 = ずれただけ、0 = 無関係）を板の枚数ごとに。
(c) 覚えたい表（直前 1 語 → 次の語、2-gram）と、光路 1 本の正解率（実験55・59）、分担した 2DTG（実験58）。

実行: python tools/fig_capacity.py
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

plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
L = 64


def outputs(net, D, ks):
    """板 D 枚（net の先頭 D 枚）に、向き (kx, ky) の平面波を入れたときの、レンズの後の明るさ。"""
    y = torch.arange(L, dtype=torch.float32)
    res = []
    with torch.no_grad():
        for kx, ky in ks:
            u = torch.polar(torch.ones(L, L), 2 * math.pi * (kx * y[None, :] + ky * y[:, None]) / L)[None]
            for d in range(D):
                u = p2.normalize(p2.propagate(p2.apply_plate(u, net.plate_phase(d)), net.H))
            res.append(p2.intensity(p2.lens(u))[0])
    return res


def shift_sim(a, b, dx):
    """b を −dx だけ横にずらして a と重ねたときの相関（ずれただけなら 1）。"""
    b2 = torch.roll(b, -dx, dims=1)
    a, b2 = a - a.mean(), b2 - b2.mean()
    return float((a * b2).sum() / (a.norm() * b2.norm() + 1e-12))


def main():
    net = p2.Net2D(L, 64, ["xy"] * 32, patches=16, seed=0)
    with torch.no_grad():
        net.A.mul_(3.0)                       # 学習後の板と同じくらいの深さに（初期値は浅いので）
    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 1.15])
    ks = [(10, 20), (11, 20), (12, 20)]
    for row, D in enumerate((1, 32)):
        outs = outputs(net, D, ks)
        for j, (o, k) in enumerate(zip(outs, ks)):
            ax = fig.add_subplot(gs[row, j])
            ax.imshow(torch.log1p(o).numpy(), cmap="inferno")
            ax.set_title(f"板 {D} 枚・入力の向き kx={k[0]}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        ax = fig.add_subplot(gs[row, 3])
        diff = torch.log1p(outs[1]) - torch.roll(torch.log1p(outs[0]), 1, dims=1)
        ax.imshow(diff.numpy(), cmap="coolwarm", vmin=-2, vmax=2)
        ax.set_title(f"板 {D} 枚: 2 つ目 − 1 つ目をずらしたもの", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    # (b) 似かた
    Ds = [1, 2, 4, 8, 16, 32]
    sims = []
    g = torch.Generator().manual_seed(1)
    pairs = [(int(torch.randint(4, 60, (1,), generator=g)), int(torch.randint(4, 60, (1,), generator=g))) for _ in range(12)]
    for D in Ds:
        s = []
        for kx, ky in pairs:
            a, b = outputs(net, D, [(kx, ky), (kx + 1, ky)])
            s.append(shift_sim(a, b, 1))
        sims.append(sum(s) / len(s))
    ax = fig.add_subplot(gs[0:2, 4:6])
    ax.plot(Ds, sims, "o-", color="#c0392b")
    ax.set_xscale("log", base=2)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("板の枚数")
    ax.set_ylabel("隣の向きの入力どうしの出口の模様の似かた\n（1 = ずれただけで同じ、0 = 無関係）")
    ax.set_title("(b) 板が少ないと、隣の向きの語は\n「ずれただけの同じ模様」になる")
    ax.grid(alpha=0.3)
    # (c) 正解率
    ax = fig.add_subplot(gs[2, 0:3])
    plates = [1, 8, 16, 32, 64, 128]
    acc = [25.9, 27.8, 28.2, 29.0, 29.6, 30.5]
    ax.plot(plates, acc, "o-", label="光路 1 本（直前 1 語 → 次の語、実験55・59）")
    ax.axhline(33.6, color="k", ls="--", label="数えた表（2-gram）= 覚えたい表を全部覚えた場合")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("板の枚数")
    ax.set_ylabel("正解率 %")
    ax.set_title("(c) 板を倍にしても +0.5〜1 点。表まで届かない")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    ax = fig.add_subplot(gs[2, 3:6])
    names = ["光路 1 本\n板 128 枚", "2DTG 4 個\n× 32 枚", "2DTG 8 個\n× 16 枚", "2DTG 16 個\n× 8 枚", "GRU", "2 語の表の\n上限（見積もり）"]
    vals = [32.7, 33.7, 35.7, 36.0, 36.9, 37.1]
    cols = ["#7f8c8d", "#5dade2", "#2e86c1", "#1b4f72", "#e67e22", "#000000"]
    ax.bar(range(len(vals)), vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.2, f"{v:.1f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(28, 39)
    ax.set_ylabel("正解率 %（直前 4 語 → 次の語、全部のデータ）")
    ax.set_title("(d) 表を分担させると、同じ板の合計 128 枚で上限近くまで届く（実験58）")
    fig.suptitle("光路の表現の力: 入力の語 = 光の向き。板が少ないと、向きを変えても模様が「ずれる」だけで、語ごとに別の答えを出しにくい", fontsize=12)
    fig.tight_layout()
    out = os.path.join(ROOT, "docs", "images", "capacity.png")
    fig.savefig(out, dpi=110)
    print(out, [round(s, 3) for s in sims])


if __name__ == "__main__":
    main()
