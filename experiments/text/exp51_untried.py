"""実験51: Sin-CNN2 で効いたが、2 次元の言語でまだ試していない工夫（持ち主の依頼、2026-10-20）

docs/sin_cnn2_checklist_text.md の未着手のうち、次の 4 つを同じ土台（帯、板 ① 16 枚 × 区画 8、位置ごとの明るさ、1500 ステップ）で試す。
  tilt     : tilt 照明（Sin-CNN2 実験10〜12）。照明 j（0〜n−1）で、位置 m の帯の光に exp(j·2π·j·m/n) を掛け、n 通りの明るさを平均する。
             違う位置の文字どうしの干渉がちょうど 0 になる。計算は n 倍
  hadamard : アダマール照明（同、位相 0 か π だけ）
  weighted : 照明の重み付き平均（実験25）。tilt の n 通りの明るさに、学習する重み（softmax）を掛けて平均する。
             重みで「どの位置どうしの干渉を残すか」を選べる（1 次元では、隣どうしだけ残すと順序の課題が 62 → 100% に戻った）
  incoh    : 非コヒーレント光源（実験08）。照明ごとに、位置ごとのでたらめな位相を掛けて平均する（8 通り）
  ref      : 参照波（実験28b の考え方）。レンズの直前に、強さを学習する平らな参照光を足す。明るさ |U + βR|² に、
             光の場 U そのもの（2 乗でない、位相つき）に比例する成分 2β·Re(U R*) が加わる（ホログラムと同じ読み方）
  none     : 比較（実験45 の add と同じ）
結果: results/exp51.txt

実行: python experiments/text/exp51_untried.py --mode weighted --seed 0 --device cuda
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
from exp40_inputs import EncNet  # noqa: E402

L = 64


class UntriedNet(EncNet):
    def __init__(self, mode, n, V, C, layers=16, patches=8, seed=0, P=8):
        super().__init__("bands", n, V, C, layers, patches, seed)
        self.mode = mode
        j = torch.arange(n, dtype=torch.float32)[:, None]
        m = torch.arange(n, dtype=torch.float32)[None, :]
        if mode in ("tilt", "weighted"):
            codes = torch.polar(torch.ones(n, n), 2 * math.pi * j * m / n)
        elif mode == "hadamard":
            Hm = torch.ones(1, 1)
            while Hm.shape[0] < n:
                Hm = torch.cat([torch.cat([Hm, Hm], 1), torch.cat([Hm, -Hm], 1)], 0)
            codes = Hm[:n, :n].to(torch.complex64)
        elif mode == "incoh":
            g = torch.Generator().manual_seed(4242)
            codes = torch.polar(torch.ones(P, n), 2 * math.pi * torch.rand(P, n, generator=g))
        else:
            codes = torch.ones(1, n, dtype=torch.complex64)
        self.register_buffer("icodes", codes)                        # (照明の数, n) 照明ごとの位置の符号
        self.pw = torch.nn.Parameter(torch.zeros(self.icodes.shape[0]), requires_grad=(mode == "weighted"))
        self.beta = torch.nn.Parameter(torch.tensor(1.0), requires_grad=(mode == "ref"))

    def comps(self, X):
        """位置ごとの光 (B, n, L, L)（帯の置き場所と位置ごとの明るさ込み）。"""
        kx, ky = self.kx[X], self.ky[X]
        ph = 2 * math.pi * (kx[..., None, None] * self.xx + ky[..., None, None] * self.yy) / L
        W = torch.polar(torch.ones_like(ph), ph)
        a = torch.exp(self.w)
        a = a / a.pow(2).mean().sqrt()
        return W * self.M[None] * a[None, :, None, None]

    def forward(self, X):
        comp = self.comps(X)                                          # (B, n, L, L)
        u = torch.einsum("pn,bnyx->bpyx", self.icodes, comp)           # 照明ごとの場 (B, P, L, L)
        B, P = u.shape[:2]
        u = u.reshape(B * P, L, L)
        net = self.net
        for d in range(len(net.kinds)):
            u = p2.normalize(p2.propagate(p2.apply_plate(u, net.plate_phase(d)), net.H))
        if self.mode == "ref":    # 参照光（平らな光 = レンズの後では中心の 1 点）を足す
            u = u + self.beta
        I = p2.intensity(p2.lens(u)).reshape(B, P, L, L)
        wts = torch.softmax(self.pw, 0)
        I = (I * wts[None, :, None, None]).sum(1)
        I = I / (I.mean((-2, -1), keepdim=True) + 1e-12)
        return net.gain * p2.read_windows(I, net.det) + net.bias


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="weighted", choices=["none", "tilt", "hadamard", "weighted", "incoh", "ref"])
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--patches", type=int, default=8)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp51.txt")
    a = ap.parse_args()
    dev = torch.device(a.device)
    (X, y), (Xt, yt), info = textdata.load(a.n, 64, 60000, 10000, a.seed)
    torch.manual_seed(a.seed)
    m = UntriedNet(a.mode, a.n, info["V_in"], info["C"], a.layers, a.patches, a.seed).to(dev)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    g = torch.Generator().manual_seed(a.seed)
    history, t_train, t_mark = [], 0.0, time.perf_counter()

    def peek():
        r = {}
        with torch.no_grad():
            for name, XX, yy in (("test", Xt[:2000], yt[:2000]), ("train", X[:2000], y[:2000])):
                S = torch.cat([m(XX[j:j + 125].to(dev)).cpu() for j in range(0, len(yy), 125)])
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
        S = torch.cat([m(Xt[j:j + 125].to(dev)).cpu() for j in range(0, len(yt), 125)])
    rec = dict(exp="51", **vars(a), test_acc=100.0 * (S.argmax(1) == yt).float().mean().item(), test_ce=F.cross_entropy(S, yt).item(),
               params=sum(p.numel() for p in m.parameters() if p.requires_grad), train_s=t_train, history=history,
               pos_brightness=torch.exp(m.w.detach()).cpu().tolist(), pattern_weights=torch.softmax(m.pw.detach(), 0).cpu().tolist(),
               beta=float(m.beta), version=p2.__version__, torch=torch.__version__)
    print(json.dumps(rec, ensure_ascii=False))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
