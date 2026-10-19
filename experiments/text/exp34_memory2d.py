"""実験34: 光の記憶の 2 次元版（1 文字ずつ入れ、前の光を混ぜて持ち越す）

着想: 1 次元の Sin-CNN2 で一番効いたのは「光の記憶」（実験41・47・50、最大 54.2%）。1 文字ずつ入れ、
      入力 = (1−α)·今の文字の光 + α·前の光、とすると、直前の文字は常に一番強く、古い文字は薄まりながら残る。
      実験31〜33 で、8 文字を同じ重さで重ねる方式は 36〜39% だった。2 次元でも光の記憶なら伸びるか。
方法: 文字 c は場全体を照らす 2 次元の平面波（向き = (c·g) mod 4096 を縦横に分けたもの）。レンズの後では、その向きの 1 点に集まる。
      各時刻: 入力の場 = (1−α)·文字の光 + α·前の時刻の光（板を通った後、レンズの前の光）→ 板 ① D 枚（回折しながら）→ 光を持ち越す。
      答えはレンズの後の明るさを窓で読む。窓はクラスの文字の光が集まる位置（出力の光が次の入力と同じ形になる、1 次元と同じ）。
      全時刻で「その時点までの文脈から次の文字」を学習し、文脈 8 文字の時点の正解率で比べる。
条件: 実験31 と同じ課題（64 クラス、訓練 6 万・テスト 1 万）。板 ① D = 8 枚（学習する数 1024 + 窓）。α = 0.3・0.5。学習率 0.01 → 0、3000 ステップ × 256 件。3 seed。
比べる相手: 実験31（2 次元の場に一度に入れる 38.7%）、1 次元の光の記憶（Sin-CNN2 実験41 の小さいモデル 45.4%、大きいモデル 51.1〜54.2%）、3-gram 53.7、GRU 47.6。
結果: results/exp34.txt（学習の経過つき）

実行: python experiments/text/exp34_memory2d.py --alpha 0.5 --seed 0 --device cuda
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


class Memory2D(torch.nn.Module):
    def __init__(self, V, C, D=8, alpha=0.5, seed=0, kind="xy"):
        super().__init__()
        self.net = p2.Net2D(L, C, [kind] * D, seed=seed)
        self.alpha, self.V, self.C = alpha, V, C
        g = stride(L * L)
        pos = (torch.arange(V) * g) % (L * L)                      # 文字ごとの向き（= レンズの後に集まる画素）
        self.register_buffer("kx", (pos % L).float())
        self.register_buffer("ky", (pos // L).float())
        det = torch.zeros(C, L, L)                                 # クラス k の窓 = その文字の光が集まる画素（fftshift 後）の周り 3 × 3
        for k in range(C):
            c = k if k < C - 1 else V - 1
            cy, cx = (int(self.ky[c]) + L // 2) % L, (int(self.kx[c]) + L // 2) % L
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    det[k, (cy + dy) % L, (cx + dx) % L] = 1 / 9
        self.register_buffer("det", det)
        y, x = torch.arange(L, dtype=torch.float32), torch.arange(L, dtype=torch.float32)
        self.register_buffer("yy", y[:, None])
        self.register_buffer("xx", x[None, :])

    def char_field(self, c):
        ph = 2 * math.pi * (self.kx[c][:, None, None] * self.xx + self.ky[c][:, None, None] * self.yy) / L
        return torch.polar(torch.ones_like(ph), ph)

    def forward(self, X):
        """X (B, n) → 各時刻の点数 (n, B, C)。"""
        u, outs = None, []
        for t in range(X.shape[1]):
            v = self.char_field(X[:, t])
            v = v if u is None else (1 - self.alpha) * v + self.alpha * u
            for d in range(len(self.net.kinds)):
                v = p2.normalize(p2.propagate(p2.apply_plate(v, self.net.plate_phase(d)), self.net.H))
            u = v
            I = p2.intensity(p2.lens(v))
            I = I / (I.mean((-2, -1), keepdim=True) + 1e-12)
            outs.append(self.net.gain * p2.read_windows(I, self.det) + self.net.bias)
        return torch.stack(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--D", type=int, default=8)
    ap.add_argument("--kind", default="xy")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp34.txt")
    a = ap.parse_args()
    dev = torch.device(a.device)
    (X, y), (Xt, yt), info = textdata.load(a.n, 64, 60000, 10000, a.seed)
    V, C = info["V_in"], info["C"]
    to_cls = lambda Z: torch.where(Z < C - 1, Z, torch.full_like(Z, C - 1))     # 入力の文字の番号 → 答えのクラス
    Y = torch.cat([to_cls(X[:, 1:]), y[:, None]], 1)                             # 各時刻の次の文字 (N, n)
    torch.manual_seed(a.seed)
    m = Memory2D(V, C, a.D, a.alpha, a.seed, a.kind).to(dev)
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
        loss = F.cross_entropy(out.reshape(-1, C), Y[i].T.reshape(-1).to(dev))     # 全時刻で次の文字
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
        S = torch.cat([m(Xt[j:j + 250].to(dev)).cpu() for j in range(0, len(yt), 250)], 1)     # (n, N, C)
    by_ctx = [100.0 * (S[t].argmax(1) == (to_cls(Xt[:, t + 1]) if t < a.n - 1 else yt)).float().mean().item() for t in range(a.n)]
    rec = dict(exp="34", **vars(a), test_acc=by_ctx[-1], test_ce=F.cross_entropy(S[-1], yt).item(), acc_by_context=by_ctx,
               params=sum(p.numel() for p in m.parameters() if p.requires_grad), train_s=t_train, history=history,
               version=p2.__version__, torch=torch.__version__)
    print(json.dumps(rec, ensure_ascii=False))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
