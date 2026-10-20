"""実験40: 文字列の入れ方の工夫（持ち主の案、2026-10-20）

着想: 持ち主の見立て。文字を上から帯に並べると、文字列によって反応する場所が変わりすぎるのでは。にじませ（拡散）が要るのでは。
      参照光で文字の前後に合わせて光に変化を与え、文字列を周波数拡散して重ね合わせ、レイヤ全体に入れるのが良いのでは。
      円の深さに文字列を対応させる（真ん中に 1 文字目、外のリングほど後の文字）のも工夫の一つ。
      1 次元では「前後の文字との平均」（滲み層、Sin-CNN2 実験18 で +10 点）と「各文字の光がレイヤ全体を照らし、位置ごとに周波数をずらす虹」
      （v1.1、実験30・33）が効いた。
方法: 文字 c の光 W_c = 場全体に広がる平面波（向きは実験31 と同じ）。位置 m の明るさ e^{w_m} は学習（実験33 と同じ）。入れ方:
  bands  : 実験33 と同じ。位置 m の帯（8 行）に W_c
  blur   : 帯をぼかす。位置 m の重み = 帯の中心からの縦の距離のガウス（幅 = 帯の高さ）。隣の帯と重なる（にじませ）
  smear  : 帯 m に、前後の文字との平均 (W_{m−1} + W_m + W_{m+1}) / 3 を置く（1 次元の滲み層）
  rainbow: 全部の文字を場全体に重ねる。位置 m の光の向きを縦に m·(64/n) だけずらす（2 次元の虹）
  spread : 全部の文字を場全体に重ねる。位置 m の光に、位置ごとに決まったでたらめな位相の模様（参照光）を掛ける（拡散符号）
  rings  : 同心円の輪。真ん中が 1 文字目（いちばん古い）、外の輪ほど後の文字
光路: 板 ① 32 枚 × 区画 16（実験35 で 46.9%）+ レンズ + 窓 64 個。学習率 0.01 → 0、3000 ステップ × 256 件、3 seed。学習の経過を記録。
結果: results/exp40.txt

実行: python experiments/text/exp40_inputs.py --enc spread --seed 0 --device cuda
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

L = 64


class EncNet(torch.nn.Module):
    """文字の番号 (B, n) → 光の場（入れ方 enc、位置ごとの明るさを学習）→ 板 → レンズ → 窓。"""

    def __init__(self, enc, n, V, C, layers=32, patches=16, seed=0, K=0):
        super().__init__()
        self.enc, self.n = enc, n
        # K > 0: 窓を K 個にして、クラスを窓の模様（符号、学習する）で読む（画像の実験22 の形。実験42）
        self.net = p2.Net2D(L, K or C, ["xy"] * layers, patches=patches, seed=seed)
        self.codes = torch.nn.Parameter(torch.randn(C, K, generator=torch.Generator().manual_seed(seed)) if K else torch.zeros(0))
        self.temp = torch.nn.Parameter(torch.tensor(10.0))
        self.w = torch.nn.Parameter(0.3 * (torch.arange(n, dtype=torch.float32) - n + 1))
        g = stride(L * L)
        pos = (torch.arange(V) * g) % (L * L)
        self.register_buffer("kx", (pos % L).float())
        self.register_buffer("ky", (pos // L).float())
        y = torch.arange(L, dtype=torch.float32)
        self.register_buffer("yy", y[:, None].expand(L, L).clone())
        self.register_buffer("xx", y[None, :].expand(L, L).clone())
        h = L / n
        centers = (torch.arange(n, dtype=torch.float32) + 0.5) * h
        if enc == "bands":
            M = (torch.arange(L)[None, :] * n // L == torch.arange(n)[:, None]).float()[:, :, None].expand(n, L, L)
        elif enc in ("blur", "smear"):
            M = (torch.arange(L)[None, :] * n // L == torch.arange(n)[:, None]).float()
            if enc == "blur":
                M = torch.exp(-(y[None, :] - centers[:, None]) ** 2 / (2 * h ** 2))
            M = M[:, :, None].expand(n, L, L)
        elif enc == "rings":
            r = torch.sqrt((self.yy - L / 2 + 0.5) ** 2 + (self.xx - L / 2 + 0.5) ** 2)
            ring = (r / (L / 2 * math.sqrt(2)) * n).floor().clamp(max=n - 1)
            M = (ring[None] == torch.arange(n)[:, None, None]).float()
        else:   # rainbow・spread は場全体
            M = torch.ones(n, L, L)
        self.register_buffer("M", M.contiguous())                      # (n, L, L) 位置ごとの置き場所
        gr = torch.Generator().manual_seed(12345)
        ref = 2 * math.pi * torch.rand(n, L, L, generator=gr) if enc == "spread" else torch.zeros(n, L, L)
        if enc == "rainbow":
            ref = 2 * math.pi * (torch.arange(n)[:, None, None] * (L // n)) * self.yy[None] / L
        self.register_buffer("ref", ref)                               # 位置ごとの参照光の位相

    def field(self, X):
        kx, ky = self.kx[X], self.ky[X]                                # (B, n)
        ph = 2 * math.pi * (kx[..., None, None] * self.xx + ky[..., None, None] * self.yy) / L + self.ref[None]
        W = torch.polar(torch.ones_like(ph), ph)                       # (B, n, L, L)
        if self.enc == "smear":
            pad = torch.cat([W[:, :1], W, W[:, -1:]], 1)
            W = (pad[:, :-2] + pad[:, 1:-1] + pad[:, 2:]) / 3
        a = torch.exp(self.w)
        a = a / a.pow(2).mean().sqrt()
        u = (W * self.M[None] * a[None, :, None, None]).sum(1)
        return u / (u.abs().pow(2).mean((-2, -1), keepdim=True) + 1e-12).sqrt()

    def forward(self, X):
        if self.codes.numel() == 0:
            return self.net(self.field(X))
        I = self.net.screen(self.field(X))
        return p2.code_scores(p2.read_windows(I, self.net.det), self.codes, 1.0) * self.temp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default="spread", choices=["bands", "blur", "smear", "rainbow", "spread", "rings"])
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--patches", type=int, default=16)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--K", type=int, default=0, help="窓の数（0 なら 1 クラス 1 窓）。符号で読む（実験42）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp40.txt")
    a = ap.parse_args()
    dev = torch.device(a.device)
    (X, y), (Xt, yt), info = textdata.load(a.n, 64, 60000, 10000, a.seed)
    torch.manual_seed(a.seed)
    m = EncNet(a.enc, a.n, info["V_in"], info["C"], a.layers, a.patches, a.seed, a.K).to(dev)
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
    rec = dict(exp="40", **vars(a), test_acc=100.0 * (S.argmax(1) == yt).float().mean().item(), test_ce=F.cross_entropy(S, yt).item(),
               params=sum(p.numel() for p in m.parameters() if p.requires_grad), train_s=t_train, history=history,
               pos_brightness=torch.exp(m.w.detach()).cpu().tolist(), version=p2.__version__, torch=torch.__version__)
    print(json.dumps(rec, ensure_ascii=False))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
