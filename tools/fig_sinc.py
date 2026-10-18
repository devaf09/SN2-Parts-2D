"""④ 減衰する波紋の板（持ち主の案）の図を docs/images/ に書き出す。

    python tools/fig_sinc.py                     # 波紋 1 つの形と、学習前の板
    python tools/fig_sinc.py results/models/exp11_conj_sinc_s0.pt ...   # 学習した板（先頭 4 枚と、深さ・中心の分布）
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
import sn2_parts_2d as p2  # noqa: E402

OUT = os.path.join(ROOT, "docs", "images")
os.makedirs(OUT, exist_ok=True)
plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
L = 64
y, x = p2.grid(L, L)
T = lambda *v: torch.tensor(v, dtype=torch.float32)

if len(sys.argv) == 1:
    # 1. 波紋 1 つ: 深さ A・幅 k・中心 (cx, cy) を変えると
    cases = [("基本\nA=2, k=0.5, 中心 (32, 32)", 2, 0.5, 32, 32),
             ("幅を細かく\nk=1.2", 2, 1.2, 32, 32),
             ("幅を粗く\nk=0.2", 2, 0.2, 32, 32),
             ("深く\nA=4", 4, 0.5, 32, 32),
             ("中心をずらす\n(10, 50)", 2, 0.5, 10, 50),
             ("中心を場の外へ\n(−40, 32)：弧になる", 2, 0.5, -40, 32)]
    fig, ax = plt.subplots(2, 6, figsize=(14, 5), gridspec_kw={"height_ratios": [3, 1.3]})
    for i, (t, A, k, cx, cy) in enumerate(cases):
        ph = p2.plate_sinc(T(A), T(k), T(cx), T(cy), y, x)
        ax[0, i].imshow(ph, cmap="RdBu_r", vmin=-2, vmax=2)
        ax[0, i].set_title(t, fontsize=9)
        ax[0, i].axis("off")
        ax[1, i].plot(ph[min(max(cy, 0), L - 1)].numpy() if 0 <= cy < L else ph[32].numpy())
        ax[1, i].set_ylim(-2.5, 4.2)
        ax[1, i].set_xticks([])
        ax[1, i].axhline(0, color="gray", lw=0.5)
    ax[1, 0].set_ylabel("横 1 列の\n位相の遅れ", fontsize=8)
    fig.suptitle("④ 減衰する波紋 1 つ: 位相の遅れ = A · sin(k·r) / (k·r)（r = 中心からの距離）")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sinc_single.png"), dpi=110, bbox_inches="tight")

    # 2. 学習前の板（波紋 32 個・8 個・128 個）と、ほかの板との比較
    fig, ax = plt.subplots(1, 5, figsize=(14, 3.2))
    for a, (t, kind, M) in zip(ax, [("① 交点で掛け算", "xy", None), ("② 斜めの縞", "stripes", None),
                                    ("④ 波紋 8 個", "sinc", 8), ("④ 波紋 32 個", "sinc", 32), ("④ 波紋 128 個", "sinc", 128)]):
        net = p2.Net2D(L, 8, [kind], seed=3, n_ripples=M)
        with torch.no_grad():
            a.imshow(net.plate_phase(0), cmap="twilight")
        a.set_title(t, fontsize=10)
        a.axis("off")
    fig.suptitle("板の位相の遅れの模様（学習前）。④ は区画に分けず、波紋を重ねるだけ")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plates_sinc.png"), dpi=110, bbox_inches="tight")
    print("docs/images/sinc_single.png, docs/images/plates_sinc.png")
else:
    # 3. 学習した板（先頭 4 枚と、学習前の同じ板）。波紋の中心を点で重ねる
    sys.path.insert(0, os.path.join(ROOT, "experiments"))
    from exp01_basic import PLATES  # noqa: E402
    for path in sys.argv[1:]:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        a = ck["args"]
        kinds = PLATES[a["method"]]
        net = p2.Net2D(a.get("L", 64), ck["classes"], kinds, patches=a.get("patches", 8), z=a.get("z", 64.0), seed=a["seed"],
                        n_ripples=a.get("n_ripples"), sinc_depth=a.get("sinc_depth", 1.0))
        init = p2.Net2D(a.get("L", 64), ck["classes"], kinds, patches=a.get("patches", 8), z=a.get("z", 64.0), seed=a["seed"],
                        n_ripples=a.get("n_ripples"), sinc_depth=a.get("sinc_depth", 1.0))
        net.load_state_dict(ck["state"])
        show = [d for d in range(len(kinds)) if kinds[d] == "sinc"][:4] or list(range(4))
        fig, ax = plt.subplots(2, len(show), figsize=(3.2 * len(show), 6.4))
        for c, d in enumerate(show):
            for r, (m, t) in enumerate([(init, "学習前"), (net, "学習後")]):
                with torch.no_grad():
                    ax[r, c].imshow(m.plate_phase(d), cmap="twilight", extent=(-0.5, a.get("L", 64) - 0.5, a.get("L", 64) - 0.5, -0.5))
                    if kinds[d] == "sinc":
                        i = m.sinc_layers.index(d)
                        ax[r, c].scatter(m.scx[i].numpy(), m.scy[i].numpy(), s=6, c="k")
                ax[r, c].set_xlim(-0.5, a.get("L", 64) - 0.5)
                ax[r, c].set_ylim(a.get("L", 64) - 0.5, -0.5)
                ax[r, c].set_title(f"板 {d + 1}（{t}）", fontsize=9)
                ax[r, c].axis("off")
        name = os.path.splitext(os.path.basename(path))[0]
        fig.suptitle(f"学習した板: {a.get('task', a.get('data'))}・{a['method']}・seed {a['seed']}（黒い点 = 場の中にある波紋の中心）")
        fig.tight_layout()
        out = os.path.join(OUT, f"{name}_plates.png")
        fig.savefig(out, dpi=100, bbox_inches="tight")
        print(out)
