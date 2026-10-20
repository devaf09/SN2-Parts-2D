"""実験50: 光を往復させる（出力を入力に戻す）たびに、レイヤの違う場所を通す — 持ち主の「鏡の往復」案（2026-10-20）

着想: 持ち主の案。向かい合った鏡の間を、わずかに傾けたレーザーが往復し、毎回回折するレイヤを通る。往復のたびに光は少しずれて
      レイヤの違う場所を通るので、1 枚のレイヤが「1 文字目用・2 文字目用…」の板を兼ね、順番は「どこを通ったか」で区別される
      （「あい」と「いあ」は違う道筋になる）。光の状態がそれまでの履歴を持ち越す（1 次元の光の記憶と同じ形）。
      持ち主の補足: 鏡の厳密な向かい合わせは気にしなくてよい。出力を入力に戻す形でよい。
方法: 大きなレイヤ（縦 U × 横 U + shift·(n−1) 画素、板 ① を D 枚重ねたもの）を用意し、往復 t 回目はその横 [t·shift, t·shift + U) の部分を通る。
      往復 t: 入力 = (1−α)·文字 t の光（向きの平面波、U × U）+ α·前の往復の光 → レイヤの t 番目の部分 D 枚（回折しながら）
              → 光を次の往復へ持ち越す。同時にレンズで見た明るさを窓 64 個で読み、次の文字を当てる（全部の往復で学習）。
      shift = 0 なら毎回同じ場所（= 実験34 の光の記憶の 2 次元版）。shift = U なら毎回まったく別の場所（往復ごとに別の板）。
条件: U = 32（等価時間標本化の考え方で小さめの場）、D = 8、α = 0.3、文脈 8 文字、訓練 6 万件。学習率 0.01 → 0。
      ふるい分け: 1500 ステップ、seed 0。
比べる相手: 実験34（光の記憶、64 × 64、42.4%）、実験33（帯に重ねて入れる、43.6%）、GRU（47.6%）。
結果: results/exp50.txt

実行: python experiments/text/exp50_bounce.py --shift 16 --seed 0 --device cuda
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


class Bounce(torch.nn.Module):
    def __init__(self, n, V, C, U=32, shift=16, D=8, alpha=0.3, bands=16, seed=0):
        super().__init__()
        self.n, self.U, self.shift, self.alpha = n, U, shift, alpha
        Wbig = U + shift * (n - 1)
        g = torch.Generator().manual_seed(seed)
        r = lambda *s: torch.rand(*s, generator=g)
        lo, hi = math.log(2 * math.pi / U), math.log(math.pi)
        nbx = bands * Wbig // U                                   # 横の帯は大きなレイヤ全体にわたって同じ細かさで
        self.A = torch.nn.Parameter(0.5 + 0.5 * r(D, nbx + bands))
        self.phi = torch.nn.Parameter(2 * math.pi * r(D, nbx + bands))
        self.register_buffer("wv", torch.exp(lo + (hi - lo) * r(D, nbx + bands)))
        self.register_buffer("bx", p2.band_index(Wbig, nbx))
        self.register_buffer("by", p2.band_index(U, bands))
        self.register_buffer("x1", torch.arange(Wbig, dtype=torch.float32))
        self.register_buffer("y1", torch.arange(U, dtype=torch.float32))
        self.register_buffer("H", p2.transfer(U, U, 64.0))
        self.nbx = nbx
        gs = stride(U * U)
        pos = (torch.arange(V) * gs) % (U * U)
        self.register_buffer("kx", (pos % U).float())
        self.register_buffer("ky", (pos // U).float())
        yy = torch.arange(U, dtype=torch.float32)
        self.register_buffer("yy", yy[:, None].expand(U, U).clone())
        self.register_buffer("xx", yy[None, :].expand(U, U).clone())
        self.register_buffer("det", p2.window_grid(U, U, C, 2))
        self.gain = torch.nn.Parameter(torch.tensor(5.0))
        self.bias = torch.nn.Parameter(torch.zeros(C))

    def layer(self, d):
        nbx = self.nbx
        return p2.plate_xy(self.A[d, :nbx], self.wv[d, :nbx], self.phi[d, :nbx], self.bx,
                           self.A[d, nbx:], self.wv[d, nbx:], self.phi[d, nbx:], self.by, self.x1, self.y1)   # (U, Wbig)

    def forward(self, X):
        """X (B, n) → 各往復の点数 (n, B, C)。"""
        phases = [self.layer(d) for d in range(self.A.shape[0])]
        u, outs = None, []
        for t in range(self.n):
            ph = 2 * math.pi * (self.kx[X[:, t]][:, None, None] * self.xx + self.ky[X[:, t]][:, None, None] * self.yy) / self.U
            v = torch.polar(torch.ones_like(ph), ph)
            v = v if u is None else (1 - self.alpha) * v + self.alpha * u
            a = t * self.shift
            for P in phases:                                     # 往復 t は、レイヤの横 [a, a + U) の部分を通る
                v = p2.normalize(p2.propagate(p2.apply_plate(v, P[:, a:a + self.U]), self.H))
            u = v
            I = p2.intensity(p2.lens(v))
            I = I / (I.mean((-2, -1), keepdim=True) + 1e-12)
            outs.append(self.gain * p2.read_windows(I, self.det) + self.bias)
        return torch.stack(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift", type=int, default=16)
    ap.add_argument("--U", type=int, default=32)
    ap.add_argument("--D", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max_train", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp50.txt")
    a = ap.parse_args()
    dev = torch.device(a.device)
    (X, y), (Xt, yt), info = textdata.load(a.n, 64, a.max_train, 10000, a.seed)
    V, C = info["V_in"], info["C"]
    to_cls = lambda Z: torch.where(Z < C - 1, Z, torch.full_like(Z, C - 1))
    Y = torch.cat([to_cls(X[:, 1:]), y[:, None]], 1)
    torch.manual_seed(a.seed)
    m = Bounce(a.n, V, C, a.U, a.shift, a.D, a.alpha, seed=a.seed).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    g = torch.Generator().manual_seed(a.seed)
    history, t_train, t_mark = [], 0.0, time.perf_counter()

    def peek():
        r = {}
        with torch.no_grad():
            for name, XX, yy in (("test", Xt[:2000], yt[:2000]), ("train", X[:2000], y[:2000])):
                S = torch.cat([m(XX[j:j + 250].to(dev))[-1].cpu() for j in range(0, len(yy), 250)])
                r[f"{name}_acc"] = 100.0 * (S.argmax(1) == yy).float().mean().item()
                r[f"{name}_ce"] = F.cross_entropy(S, yy).item()
        return r
    for step in range(1, a.steps + 1):
        i = torch.randint(len(y), (256,), generator=g)
        out = m(X[i].to(dev))
        loss = F.cross_entropy(out.reshape(-1, C), Y[i].T.reshape(-1).to(dev))
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
        S = torch.cat([m(Xt[j:j + 250].to(dev)).cpu() for j in range(0, len(yt), 250)], 1)
    by_ctx = [100.0 * (S[t].argmax(1) == (to_cls(Xt[:, t + 1]) if t < a.n - 1 else yt)).float().mean().item() for t in range(a.n)]
    rec = dict(exp="50", **vars(a), test_acc=by_ctx[-1], test_ce=F.cross_entropy(S[-1], yt).item(), acc_by_context=by_ctx,
               params=sum(p.numel() for p in m.parameters() if p.requires_grad), train_s=t_train, history=history,
               version=p2.__version__, torch=torch.__version__)
    print(json.dumps(rec, ensure_ascii=False))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
