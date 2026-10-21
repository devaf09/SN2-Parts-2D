"""実験63 の図: 学習した板（位相・振幅）と、出口の光の明るさ・位相差（参照光との）の模様。

実行: python tools/fig_exp63.py --model results/models/exp63_both_int_phase_s0.pt --plate both --read int_phase --out docs/images/exp63_both.png
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "text"))
import sn2_parts_2d as p2  # noqa: E402
import worddata  # noqa: E402
from exp63_amp_phase import AmpPhaseNet  # noqa: E402

plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--plate", default="both")
    ap.add_argument("--read", default="int_phase")
    ap.add_argument("--words", default="ありがとう,東京,天気")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    _, _, info = worddata.load(1, 1024, 100, 100, 0, 4096)
    m = AmpPhaseNet(a.plate, a.read, 1, 4096, 1024)
    m.load_state_dict(torch.load(a.model))
    m.eval()
    words = [w for w in a.words.split(",") if w in info["in_idx"]]
    fig, ax = plt.subplots(2, 2 + len(words), figsize=(3 * (2 + len(words)), 6.2))
    with torch.no_grad():
        ph = m.net.plate_phase(0).numpy()
        am = torch.exp(0.5 * m.ampnet.plate_phase(0)).numpy()
        ax[0, 0].imshow(ph if a.plate != "amp" else 0 * ph, cmap="twilight")
        ax[0, 0].set_title("板 1 枚目: 位相の遅れ")
        ax[1, 0].imshow(am if a.plate != "phase" else 0 * am + 1, cmap="gray")
        ax[1, 0].set_title("板 1 枚目: 振幅（透過）")
        ph2 = m.net.plate_phase(len(m.net.kinds) - 1).numpy()
        ax[0, 1].imshow(ph2 if a.plate != "amp" else 0 * ph2, cmap="twilight")
        ax[0, 1].set_title("最後の板: 位相の遅れ")
        am2 = torch.exp(0.5 * m.ampnet.plate_phase(len(m.net.kinds) - 1)).numpy()
        ax[1, 1].imshow(am2 if a.plate != "phase" else 0 * am2 + 1, cmap="gray")
        ax[1, 1].set_title("最後の板: 振幅（透過）")
        for j, w in enumerate(words):
            u = m.field(torch.tensor([[info["in_idx"][w]]]))
            for d in range(len(m.net.kinds)):
                if a.plate in ("phase", "both"):
                    u = p2.apply_plate(u, m.net.plate_phase(d))
                if a.plate in ("amp", "both"):
                    u = u * torch.exp(0.5 * m.ampnet.plate_phase(d))
                u = p2.normalize(p2.propagate(u, m.net.H))
            U = p2.lens(u)[0]
            ax[0, 2 + j].imshow(torch.log1p(U.abs() ** 2).numpy(), cmap="inferno")
            ax[0, 2 + j].set_title(f"「{w}」出口の明るさ（対数）")
            ax[1, 2 + j].imshow(torch.angle(U).numpy(), cmap="twilight")
            ax[1, 2 + j].set_title(f"「{w}」参照光との位相差")
    for x in ax.flat:
        x.set_xticks([])
        x.set_yticks([])
    fig.suptitle(f"実験63: 板 = {a.plate}、読み方 = {a.read}")
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=110)
    print(a.out)


if __name__ == "__main__":
    main()
