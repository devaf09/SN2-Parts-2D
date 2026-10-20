"""実験45: 文字を「板」として光に通す（文字どうしを掛け算で組み合わせる）

着想: 実験40〜44 で、光路は直前 2 文字しか使わず、その 2 文字の組み合わせ（511 × 511 通り）も区別しきれていなかった。
      いままでは文字の光を「足し合わせて」入れていた（光の場に対して線形なので、組み合わせは明るさの 2 乗の交差項にしか出ない）。
      文字そのものを位相の板（空間光変調器）にして、光をその板に順に通せば、文字どうしは光の上で「掛け算」になる。
      板の間に学習する板と回折をはさめば、前の文字の模様が次の文字の板に当たる場所を変えるので、順番も表せる（光の RNN に近い）。
      持ち主の案（参照光で文字の前後に合わせて光に変化を与える）を、文字が次の文字の参照光になる形にしたもの。
      持ち主の方針（2026-10-20）: 板を少なめにして、根本的な解決を速く回して試す。
方法: 明るさ 1 の平らな光から始め、位置 m = 0（古い）〜 n−1（直前）の順に:
        光 ← 光 × exp(j·文字 c_m の模様)  →  学習する板 k 枚（回折しながら）
      最後にレンズ → 窓 64 個（1 クラス 1 窓）または窓 256 個の符号。
      文字の模様: wave = 文字ごとの向きの平面波の位相（実験31 と同じ）、rand = 文字ごとに決まったでたらめな位相の模様（0〜2π）。
条件: 板の合計は 16 枚（文字 1 つごとに 2 枚 × 8 文字）。区画 8。学習率 0.01 → 0。
      ふるい分け: 1500 ステップ・seed 0 だけ（1 本 約 1 分）。良いものを 3 seed・3000 ステップで確かめる。
比べる相手: 実験33（帯に足し合わせて入れる、板 16 枚、43.6%）を同じふるい分けの条件で測り直したもの（--mode add）。
結果: results/exp45.txt

実行: python experiments/text/exp45_modulate.py --mode mul --mask rand --seed 0 --device cuda
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
from exp40_inputs import EncNet  # noqa: E402

L = 64


class ModNet(torch.nn.Module):
    def __init__(self, n, V, C, per=2, patches=8, mask="rand", seed=0, K=0, depth=1.0):
        super().__init__()
        self.n, self.per = n, per
        self.net = p2.Net2D(L, K or C, ["xy"] * (n * per), patches=patches, seed=seed)
        if mask == "rand":
            g = torch.Generator().manual_seed(777)
            masks = 2 * math.pi * torch.rand(V, L, L, generator=g)
        else:
            gs = stride(L * L)
            pos = (torch.arange(V) * gs) % (L * L)
            kx, ky = (pos % L).float(), (pos // L).float()
            y = torch.arange(L, dtype=torch.float32)
            masks = 2 * math.pi * (kx[:, None, None] * y[None, None, :] + ky[:, None, None] * y[None, :, None]) / L
        self.register_buffer("masks", masks * depth)                  # (V, L, L) 文字ごとの位相の板
        self.codes = torch.nn.Parameter(torch.randn(C, K, generator=torch.Generator().manual_seed(seed)) if K else torch.zeros(0))
        self.temp = torch.nn.Parameter(torch.tensor(10.0))

    def forward(self, X):
        B = X.shape[0]
        u = torch.ones(B, L, L, dtype=torch.complex64, device=X.device)
        d = 0
        for m in range(self.n):
            ph = self.masks[X[:, m]]
            u = u * torch.polar(torch.ones_like(ph), ph)               # 文字の板を通す（掛け算）
            for _ in range(self.per):
                u = p2.normalize(p2.propagate(p2.apply_plate(u, self.net.plate_phase(d)), self.net.H))
                d += 1
        I = p2.intensity(p2.lens(u))
        I = I / (I.mean((-2, -1), keepdim=True) + 1e-12)
        w = p2.read_windows(I, self.net.det)
        if self.codes.numel() == 0:
            return self.net.gain * w + self.net.bias
        return p2.code_scores(w, self.codes, 1.0) * self.temp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mul", choices=["mul", "add"], help="mul = 文字を板として掛け算、add = 帯に足し合わせる（実験33 の形）")
    ap.add_argument("--mask", default="rand", choices=["rand", "wave"])
    ap.add_argument("--depth", type=float, default=1.0, help="文字の板の位相の深さの倍率")
    ap.add_argument("--per", type=int, default=2, help="文字 1 つごとの学習する板の枚数")
    ap.add_argument("--patches", type=int, default=8)
    ap.add_argument("--K", type=int, default=0)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp45.txt")
    a = ap.parse_args()
    dev = torch.device(a.device)
    (X, y), (Xt, yt), info = textdata.load(a.n, 64, 60000, 10000, a.seed)
    torch.manual_seed(a.seed)
    if a.mode == "mul":
        m = ModNet(a.n, info["V_in"], info["C"], a.per, a.patches, a.mask, a.seed, a.K, a.depth).to(dev)
    else:
        m = EncNet("bands", a.n, info["V_in"], info["C"], a.n * a.per, a.patches, a.seed, a.K).to(dev)
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
    rec = dict(exp="45", **vars(a), test_acc=100.0 * (S.argmax(1) == yt).float().mean().item(), test_ce=F.cross_entropy(S, yt).item(),
               params=sum(p.numel() for p in m.parameters() if p.requires_grad), train_s=t_train, history=history,
               version=p2.__version__, torch=torch.__version__)
    print(json.dumps(rec, ensure_ascii=False))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
