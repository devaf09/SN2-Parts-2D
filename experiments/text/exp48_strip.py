"""実験48: 短冊形の場で、文字ごとの光が後ろの文字の範囲にだけ届く形（持ち主の案、2026-10-20）

着想: 持ち主の案。正方形ではなく 1:4 程度の短冊形にして、文字ごとに照明を当てる。各文字の光は、自分の位置から後ろ（下）へ
      一定の範囲だけを照らす（プリズムで下向きに傾け、前側をマスクで切る）。文字の位置は 0.5 ずつずらすので、照らす範囲が少しずつ重なる。
        1 文字目 → 1〜4 の範囲、2 文字目 → 1.5〜4.5 の範囲、…
      前の文字の光は後ろの文字の範囲にしか届かないので、光が時間の向きに一方通行で流れる。
方法: 幅 U 画素（既定 32）の短冊。文字 m（0 = 一番古い）の光は、縦の範囲 [0.5·m·U, (0.5·m + 3)·U) だけを照らす。
      光の模様 = 文字ごとの向きの平面波（実験31 と同じ並べ方。等価時間標本化の考え方で、横 U × 縦 U の向きを互いに素な歩幅で使う）×
      プリズム（縦方向に下向きの傾き tilt 周期）。位置ごとの明るさは学習（実験33）。全部の文字の光を重ねて、板 ① D 枚（回折しながら）→ レンズ → 窓 64 個。
      短冊の長さ = (0.5·(n−1) + 3)·U（n = 8、U = 32 なら 208 画素）。
比べる相手: 正方形の帯（実験47 の 32 × 32・64 × 64、同じ板の枚数・学習の長さ）。
結果: results/exp48.txt（ふるい分けは 1500 ステップ）

実行: python experiments/text/exp48_strip.py --seed 0 --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import sn2_parts_2d as p2  # noqa: E402
import textdata  # noqa: E402
from exp31_text2d import stride  # noqa: E402


class StripNet(torch.nn.Module):
    def __init__(self, n, V, C, U=32, span=3.0, step=0.5, tilt=4.0, layers=16, bands=16, seed=0, mask=True):
        super().__init__()
        self.n, self.U = n, U
        self.Ly = int(round((step * (n - 1) + span) * U))
        self.Lx = U
        Ly, Lx = self.Ly, self.Lx
        gs = stride(U * U)
        pos = (torch.arange(V) * gs) % (U * U)
        self.register_buffer("kx", (pos % U).float())
        self.register_buffer("ky", (pos // U).float() * Ly / U)      # 縦の向きは短冊の長さに合わせて同じ角度に
        y = torch.arange(Ly, dtype=torch.float32)[:, None].expand(Ly, Lx)
        x = torch.arange(Lx, dtype=torch.float32)[None, :].expand(Ly, Lx)
        self.register_buffer("yy", y.clone())
        self.register_buffer("xx", x.clone())
        M = torch.zeros(n, Ly, Lx)
        for m in range(n):
            a, b = int(step * m * U), int((step * m + span) * U)
            M[m, a:b] = 1.0
        if not mask:
            M[:] = 1.0
        self.register_buffer("M", M)                                   # 文字 m が照らす範囲（前側はマスクで切る）
        self.register_buffer("prism", 2 * math.pi * tilt * y / Ly)     # プリズム: 下向きに傾ける位相
        self.w = torch.nn.Parameter(0.3 * (torch.arange(n, dtype=torch.float32) - n + 1))
        g = torch.Generator().manual_seed(seed)
        r = lambda *s: torch.rand(*s, generator=g)
        lo, hi = math.log(2 * math.pi / max(Ly, Lx)), math.log(math.pi)
        self.A = torch.nn.Parameter(0.5 + 0.5 * r(layers, 2 * bands))
        self.phi = torch.nn.Parameter(2 * math.pi * r(layers, 2 * bands))
        self.register_buffer("wv", torch.exp(lo + (hi - lo) * r(layers, 2 * bands)))
        self.register_buffer("bx", p2.band_index(Lx, bands))
        self.register_buffer("by", p2.band_index(Ly, bands))
        self.register_buffer("x1", torch.arange(Lx, dtype=torch.float32))
        self.register_buffer("y1", torch.arange(Ly, dtype=torch.float32))
        self.register_buffer("H", p2.transfer(Ly, Lx, 64.0))
        self.register_buffer("det", p2.window_grid(Ly, Lx, C, 4))
        self.gain = torch.nn.Parameter(torch.tensor(5.0))
        self.bias = torch.nn.Parameter(torch.zeros(C))
        self.nb = bands

    def field(self, X):
        kx, ky = self.kx[X], self.ky[X]
        ph = 2 * math.pi * (kx[..., None, None] * self.xx / self.Lx + ky[..., None, None] * self.yy / self.Ly) + self.prism
        W = torch.polar(torch.ones_like(ph), ph) * self.M[None]
        a = torch.exp(self.w)
        a = a / a.pow(2).mean().sqrt()
        return (W * a[None, :, None, None]).sum(1)

    def forward(self, X):
        u = p2.normalize(self.field(X))
        nb = self.nb
        for d in range(self.A.shape[0]):
            ph = p2.plate_xy(self.A[d, :nb], self.wv[d, :nb], self.phi[d, :nb], self.bx,
                             self.A[d, nb:], self.wv[d, nb:], self.phi[d, nb:], self.by, self.x1, self.y1)
            u = p2.normalize(p2.propagate(p2.apply_plate(u, ph), self.H))
        I = p2.intensity(p2.lens(u))
        I = I / (I.mean((-2, -1), keepdim=True) + 1e-12)
        return self.gain * p2.read_windows(I, self.det) + self.bias


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--U", type=int, default=32)
    ap.add_argument("--span", type=float, default=3.0)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--tilt", type=float, default=4.0, help="プリズムの傾き（短冊の長さあたりの縞の数）")
    ap.add_argument("--nomask", action="store_true", help="前側のマスクなし（全部の文字が短冊全体を照らす、比較用）")
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--bands", type=int, default=16)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp48.txt")
    a = ap.parse_args()
    dev = torch.device(a.device)
    (X, y), (Xt, yt), info = textdata.load(a.n, 64, 60000, 10000, a.seed)
    torch.manual_seed(a.seed)
    m = StripNet(a.n, info["V_in"], info["C"], a.U, a.span, a.step, a.tilt, a.layers, a.bands, a.seed, not a.nomask).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    g = torch.Generator().manual_seed(a.seed)
    history, t_train, t_mark = [], 0.0, time.perf_counter()

    def peek():
        r = {}
        with torch.no_grad():
            for name, XX, yy in (("test", Xt[:2000], yt[:2000]), ("train", X[:2000], y[:2000])):
                S = torch.cat([m(XX[j:j + 250].to(dev)).cpu() for j in range(0, len(yy), 250)])
                r[f"{name}_acc"] = 100.0 * (S.argmax(1) == yy).float().mean().item()
                r[f"{name}_ce"] = F.cross_entropy(S, yy).item()
        return r
    for step in range(1, a.steps + 1):
        i = torch.randint(len(y), (256,), generator=g)
        loss = F.cross_entropy(m(X[i].to(dev)), y[i].to(dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
        if step % 250 == 0 or step == a.steps:
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t_train += time.perf_counter() - t_mark
            history.append(dict(step=step, train_s=round(t_train, 2), loss=float(loss.detach()), lr=opt.param_groups[0]["lr"], **peek()))
            t_mark = time.perf_counter()
    with torch.no_grad():
        S = torch.cat([m(Xt[j:j + 250].to(dev)).cpu() for j in range(0, len(yt), 250)])
    rec = dict(exp="48", **vars(a), Ly=m.Ly, Lx=m.Lx, test_acc=100.0 * (S.argmax(1) == yt).float().mean().item(),
               test_ce=F.cross_entropy(S, yt).item(), params=sum(p.numel() for p in m.parameters() if p.requires_grad),
               train_s=t_train, history=history, pos_brightness=torch.exp(m.w.detach()).cpu().tolist(),
               version=p2.__version__, torch=torch.__version__)
    print(json.dumps(rec, ensure_ascii=False))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
