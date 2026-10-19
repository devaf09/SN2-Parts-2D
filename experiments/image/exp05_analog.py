"""実験05: アナログな装置で作ったときの強さ（板の形ごと）

着想: 持ち主の依頼の評価の軸「アナログで作りやすいか」。1 次元の実験48 では、位相の精度は粗くてよい（4 ビット）が、
      板の位置合わせ（1 画素のずれで 93.5 → 26%）と間隔（5% で 33%）に極端に弱かった。2 次元の板 ①②③ で比べる。
方法: 理想の装置で学習した光路を、作りの誤差を入れて評価する（学習し直さない）。
  bits      : 板の位相を 2^bits 段階に丸める
  pix_noise : 画素ごとの位相の誤差（標準偏差、ラジアン）
  shift_x/y : 板を x / y 方向に 1・2 画素ずらす（板ごとに向きはでたらめ、ずれの大きさは同じ）
  gap       : 板の間隔を (1 ± 誤差) 倍にする（板ごとにでたらめ）
  det_noise : 検出器の雑音（窓の明るさに掛かる相対誤差）
比べるもの: 理想の装置（誤差なし）。1 次元の実験48 の値。
結果: results/exp05.txt（1 回 = 1 行の JSON。1 行に全部の誤差の結果を入れる）

実行: python experiments/image/exp05_analog.py --task conj --method xy --seed 0 --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import sn2_parts_2d as p2  # noqa: E402
import tasks2d  # noqa: E402
from exp01_basic import PLATES  # noqa: E402


@torch.no_grad()
def eval_perturbed(net, X, y, pert, seed=0, batch=500):
    """作りの誤差 pert を入れて正解率（%）を測る。誤差の乱数は seed で固定（装置 1 台分）。"""
    g = torch.Generator().manual_seed(seed)
    dev = net.det.device
    D, L = len(net.kinds), net.L
    phases = []
    for d in range(D):
        ph = net.plate_phase(d)
        if "bits" in pert:
            q = 2 * math.pi / 2 ** pert["bits"]
            ph = torch.round(ph / q) * q
        if "pix_noise" in pert:
            ph = ph + pert["pix_noise"] * torch.randn(L, L, generator=g).to(dev)
        s = pert.get("shift", 0)
        if s:
            ax = pert["axis"]
            sign = 1 if torch.rand(1, generator=g).item() < 0.5 else -1
            ph = torch.roll(ph, sign * s, dims=(-1 if ax == "x" else -2))
        phases.append(ph)
    Hs = []
    for d in range(D):
        if "gap" in pert:
            z = 64.0 * (1 + pert["gap"] * (2 * torch.rand(1, generator=g).item() - 1))
            Hs.append(p2.transfer(L, L, z, device=dev))
        else:
            Hs.append(net.H)
    ok = 0
    for i in range(0, len(X), batch):
        u = p2.image_to_field(X[i:i + batch].to(dev), "amp")
        for d in range(D):
            u = p2.normalize(p2.propagate(p2.apply_plate(u, phases[d]), Hs[d]))
        I = p2.intensity(p2.lens(u))
        I = I / (I.mean((-2, -1), keepdim=True) + 1e-12)
        w = p2.read_windows(I, net.det)
        if "det_noise" in pert:
            w = w * (1 + pert["det_noise"] * torch.randn(w.shape, generator=g).to(dev))
        ok += ((net.gain * w + net.bias).argmax(1).cpu() == y[i:i + batch]).sum().item()
    return 100.0 * ok / len(X)


PERTS = {
    "ideal": {},
    "bits2": {"bits": 2}, "bits3": {"bits": 3}, "bits4": {"bits": 4}, "bits6": {"bits": 6},
    "pix0.1": {"pix_noise": 0.1}, "pix0.3": {"pix_noise": 0.3},
    "shiftx1": {"shift": 1, "axis": "x"}, "shiftx2": {"shift": 2, "axis": "x"},
    "shifty1": {"shift": 1, "axis": "y"}, "shifty2": {"shift": 2, "axis": "y"},
    "gap1": {"gap": 0.01}, "gap5": {"gap": 0.05},
    "det0.05": {"det_noise": 0.05}, "det0.2": {"det_noise": 0.2},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="conj", choices=list(tasks2d.TASKS))
    ap.add_argument("--method", default="xy", choices=[k for k in PLATES if k not in ("random", "lens")])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--patches", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp05.txt")
    a = ap.parse_args()
    dev = torch.device(a.device)
    C = tasks2d.TASKS[a.task]
    Xtr, ytr = tasks2d.make(a.task, a.n_train, a.L, seed=1000 + a.seed)
    Xte, yte = tasks2d.make(a.task, a.n_test, a.L, seed=2000 + a.seed)
    torch.manual_seed(a.seed)
    net = p2.Net2D(a.L, C, PLATES[a.method], patches=a.patches, seed=a.seed).to(dev)
    f = lambda v: p2.image_to_field(v, "amp")
    t0 = time.perf_counter()
    p2.train(net, Xtr, ytr, steps=a.steps, seed=a.seed, to_input=f)
    t_train = time.perf_counter() - t0
    acc = {k: eval_perturbed(net, Xte, yte, v, seed=a.seed) for k, v in PERTS.items()}
    rec = dict(exp="05", **vars(a), params=net.n_params(), train_s=t_train, acc=acc,
               version=p2.__version__, torch=torch.__version__)
    print(json.dumps(rec, ensure_ascii=False))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
