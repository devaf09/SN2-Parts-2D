"""実験31・32: 文章を 2 次元の場に載せて、次の文字を当てる（CC-100 日本語、1 次元の Sin-CNN2 と同じ課題）

着想: 持ち主の依頼（2026-10-20）。1 次元では、文脈を一度に入れる方式は 48% で頭打ちだった（Sin-CNN2 実験27・30・32）。
      2 次元なら、縦 = 位置、横 = 文字 に分けて持てる。画像の実験28 では、光学相関器の間に光の非線形を入れると見分けが大きく強くなった。
方法: 直前 n 文字（n = 8）を 64 × 64 の場に載せる。場を縦に n 本の帯に分け、位置 m の帯に文字 c の平面波を置く。
      平面波の向き（周波数）は文字ごとに決める: kx = (c·g) mod 64、ky = ⌊(c·g) / 64⌋ mod 64（g は 4096 と互いに素な歩幅）。
      光路:
        plates   : 実験31。板 ① 16 枚 + レンズ + 窓 64 個（1 クラス 1 窓）
        corr1    : 実験32。4f 系（周波数の面の位相の板）1 段 + レンズ + 窓
        corrNL2  : 実験32。4f 系 2 段、間に光の非線形（明るさを測って光を出し直す）+ レンズ + 窓
        corrNL3  : 実験32。同じく 3 段（非線形 2 回）
      比べる相手: n-gram（2・3・4）、GRU（埋め込み + 記憶 128、1 次元の実験47 の最良に近い大きさ）、
                  ロジスティック回帰（n 文字の one-hot → 64 クラス）。
      学習: 学習率 0.01 → 0（cosine）、3000 ステップ × 256 件（GRU・回帰も同じ）。訓練 6 万件・テスト 1 万件。
指標: テストの正解率、確信度（クロスエントロピー、nats）、学習する数、学習の時間。
結果: results/exp31.txt（plates と比べる相手）、results/exp32.txt（相関器）

実行: python experiments/text/exp31_text2d.py --method plates --seed 0 --device cuda
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

L = 64


def stride(N):
    g = int(N * 0.6180339887) | 1
    while math.gcd(g, N) != 1:
        g += 2
    return g


def encode(X, n, V):
    """文字の番号 (N, n) → 光の場 (N, L, L)。位置 m の帯（縦 L/n 画素）に、文字ごとの向きの平面波。"""
    g = stride(L * L)
    c = (X.long() * g) % (L * L)
    kx, ky = (c % L).float(), (c // L).float()                      # (N, n)
    y = torch.arange(L, dtype=torch.float32)
    x = torch.arange(L, dtype=torch.float32)
    band = (y * n // L).long()                                      # 各行がどの位置の帯か
    KX, KY = kx[:, band], ky[:, band]                               # (N, L) 行ごとの向き
    ph = 2 * math.pi * (KX[:, :, None] * x[None, None, :] + KY[:, :, None] * y[None, :, None]) / L
    return torch.polar(torch.ones_like(ph), ph) / math.sqrt(n)


class CorrNet(torch.nn.Module):
    """4f 系を stages 段（段の間で明るさを測って光を出し直す = 光の非線形）→ レンズ → 窓 C 個。"""

    def __init__(self, stages, C, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.phase = torch.nn.Parameter(0.1 * torch.randn(stages, L, L, generator=g))
        self.register_buffer("det", p2.window_grid(L, L, C, 4))
        self.gain = torch.nn.Parameter(torch.tensor(5.0))
        self.bias = torch.nn.Parameter(torch.zeros(C))

    def forward(self, u):
        for s in range(self.phase.shape[0]):
            if s > 0:   # 光の非線形: 明るさを測り、その明るさ（平方根を振幅）で光を出し直す
                I = p2.intensity(u)
                u = torch.sqrt(I / (I.mean((-2, -1), keepdim=True) + 1e-12) + 1e-12).to(torch.complex64)
            U = torch.fft.fft2(u, norm="ortho") * torch.polar(torch.ones_like(self.phase[s]), self.phase[s])
            u = p2.normalize(torch.fft.ifft2(U, norm="ortho"))
        I = p2.intensity(p2.lens(u))
        I = I / (I.mean((-2, -1), keepdim=True) + 1e-12)
        return self.gain * p2.read_windows(I, self.det) + self.bias


class GRU(torch.nn.Module):
    def __init__(self, V, C, hid=128):
        super().__init__()
        self.emb = torch.nn.Embedding(V, hid)
        self.rnn = torch.nn.GRU(hid, hid, batch_first=True)
        self.out = torch.nn.Linear(hid, C)

    def forward(self, X):
        h, _ = self.rnn(self.emb(X))
        return self.out(h[:, -1])


class LogReg(torch.nn.Module):
    def __init__(self, V, C, n):
        super().__init__()
        self.V = V
        self.fc = torch.nn.Linear(n * V, C)

    def forward(self, X):
        return self.fc(F.one_hot(X, self.V).float().flatten(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="plates", choices=["plates", "corr1", "corrNL2", "corrNL3", "gru", "logreg", "ngram"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--max_train", type=int, default=60000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or ("results/exp32.txt" if a.method.startswith("corr") else "results/exp31.txt")
    dev = torch.device(a.device)
    (X, y), (Xt, yt), info = textdata.load(a.n, 64, a.max_train, 10000, a.seed)
    V, C = info["V_in"], info["C"]
    t0 = time.perf_counter()
    if a.method == "ngram":
        res = {}
        for o in (2, 3, 4):
            acc, ce = textdata.ngram(info, o)
            res[f"{o}gram_acc"], res[f"{o}gram_ce"] = acc, ce
        rec = dict(exp="31", **vars(a), **res, total_s=time.perf_counter() - t0)
    else:
        torch.manual_seed(a.seed)
        if a.method == "plates":
            m = p2.Net2D(L, C, ["xy"] * 16, seed=a.seed)
            f = lambda v: encode(v, a.n, V)
        elif a.method.startswith("corr"):
            m = CorrNet({"corr1": 1, "corrNL2": 2, "corrNL3": 3}[a.method], C, a.seed)
            f = lambda v: encode(v, a.n, V)
        elif a.method == "gru":
            m, f = GRU(V, C), (lambda v: v)
        else:
            m, f = LogReg(V, C, a.n), (lambda v: v)
        m = m.to(dev)
        n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        lr = 0.003 if a.method == "logreg" else 0.01
        opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
        g = torch.Generator().manual_seed(a.seed)
        for _ in range(a.steps):
            i = torch.randint(len(y), (256,), generator=g)
            loss = F.cross_entropy(m(f(X[i]).to(dev) if a.method in ("gru", "logreg") else f(X[i]).to(dev)), y[i].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_train = time.perf_counter() - t0
        with torch.no_grad():
            S = torch.cat([m(f(Xt[j:j + 500]).to(dev)).cpu() for j in range(0, len(yt), 500)])
        acc = 100.0 * (S.argmax(1) == yt).float().mean().item()
        ce = F.cross_entropy(S, yt).item()
        rec = dict(exp="32" if a.method.startswith("corr") else "31", **vars(a), test_acc=acc, test_ce=ce, params=n_params,
                   train_s=t_train, version=p2.__version__, torch=torch.__version__)
    print(json.dumps(rec, ensure_ascii=False))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
