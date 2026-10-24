"""実験75: 4f 系を多段にして、間に光の非線形を入れて分類する（光の CNN、持ち主の依頼 2026-10-24）

着想: 4f 系（レンズ → 周波数の面の板 → レンズ）は、入力と模様の畳み込みを光が通るだけで計算する（CNN の畳み込みの層と同じ）。
      線形のままでは何段重ねても 1 段と同じだが、間に非線形を入れると見分けが大きく強くなった（実験28、位置の検出）。
      板を重ねる形では非線形はほぼ効かなかった（実験73・74）。分類にはまだ使っていない。
      今の弱点は平行移動への強さ（形の課題 70〜80%、CNN 98.6%。MNIST で CNN に 1.8 点届かない）で、畳み込み + 非線形がちょうど補える所。
方法: 段 s: u → [幅 W 本の 4f 系（周波数の面に、振幅と位相を学習する板）を並べ、出口の場を学習する複素数の重みで足す] → 非線形（nl_parts）。
      最後の段の後: 読み方 lens = レンズ（明るさは平行移動で変わらない）→ 窓、plane = 出口の面の窓（位置で変わる）。
      非線形: sqrt・kerr・fold・ssat（実験73 の部品）、none（比べる。線形なら何段でも 1 段と同じ）。
課題: mnist（10 クラス）、shape（合成の形 8 クラス、位置をでたらめにずらした図形。実験01 で板 ① 70.6%、CNN 98.6%）。
比べる: 板 ① 16 枚（mnist 95.3%、shape 70.6%）、小さな CNN（mnist 98.1%、shape 98.6%）。
条件: 3000 ステップ、学習率 0.01 → 0、seed 0。
結果: results/exp75.txt。--save で光路を保存（図のため）。
実行: python experiments/exp75_4f.py --task shape --stages 3 --width 4 --kind kerr --read lens --seed 0 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "image"))
import nl_parts as nl  # noqa: E402
import sn2_parts_2d as p2  # noqa: E402


class Multi4f(torch.nn.Module):
    def __init__(self, L, n_out, stages=3, width=4, kind="kerr", read="lens", seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.stages, self.width, self.read = stages, width, read
        self.phase = torch.nn.Parameter(0.1 * torch.randn(stages, width, L, L, generator=g))     # 周波数の面の板の位相
        self.logamp = torch.nn.Parameter(torch.zeros(stages, width, L, L))                        # 同じく振幅（対数）
        self.mix = torch.nn.Parameter(torch.randn(stages, width, 2, generator=g) / width ** 0.5)  # 幅の足し合わせの複素数の重み
        self.nls = torch.nn.ModuleList([nl.NL(kind) for _ in range(stages - 1)])
        self.register_buffer("det", p2.window_grid(L, L, n_out, 4))
        self.gain = torch.nn.Parameter(torch.tensor(5.0))
        self.bias = torch.nn.Parameter(torch.zeros(n_out))

    def stage(self, u, s):
        U = torch.fft.fft2(u, norm="ortho")[:, None]                                            # (B, 1, L, L)
        filt = torch.polar(torch.exp(self.logamp[s].clamp(max=3)), self.phase[s])[None]         # (1, W, L, L)
        v = torch.fft.ifft2(U * filt, norm="ortho")                                             # (B, W, L, L)
        w = torch.complex(self.mix[s, :, 0], self.mix[s, :, 1])
        return p2.normalize((v * w[None, :, None, None]).sum(1))

    def screen(self, u, keep=False):
        inter = []
        for s in range(self.stages):
            u = self.stage(u, s)
            if s < self.stages - 1:
                before = u
                u = self.nls[s](u)
                if keep:
                    inter.append((before, u))
        out = p2.lens(u) if self.read == "lens" else u
        I = p2.intensity(out)
        I = I / (I.mean((-2, -1), keepdim=True) + 1e-12)
        return (I, inter) if keep else I

    def forward(self, u):
        return self.gain * p2.read_windows(self.screen(u), self.det) + self.bias


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="shape", choices=["mnist", "shape"])
    ap.add_argument("--stages", type=int, default=3)
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--kind", default="kerr", choices=nl.KINDS)
    ap.add_argument("--read", default="lens", choices=["lens", "plane"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--fix_p", type=float, default=None, help="非線形の数を固定（学習しない、実験77）")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=0.0, help="勾配の大きさの上限（0 = なし）")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp75.txt")
    a = ap.parse_args()
    dev = torch.device(a.device)
    if a.task == "mnist":
        from exp15_mnist import load
        X, y, Xt, yt = load("mnist")
        C = 10
    else:
        import tasks2d
        X, y = tasks2d.make("shape", 4000, 64, seed=a.seed)
        Xt, yt = tasks2d.make("shape", 2000, 64, seed=a.seed + 1000)
        C = int(y.max()) + 1
    torch.manual_seed(a.seed)
    m = Multi4f(64, C, a.stages, a.width, a.kind, a.read, a.seed).to(dev)
    if a.fix_p is not None:
        for q in m.nls:
            q.p.data.fill_(a.fix_p)
            q.p.requires_grad_(False)
    to_in = lambda x: p2.image_to_field(x, "amp")
    opt = torch.optim.Adam([q for q in m.parameters() if q.requires_grad], lr=a.lr)
    gn = []
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    g = torch.Generator().manual_seed(a.seed)
    t0 = time.perf_counter()
    for _ in range(a.steps):
        i = torch.randint(len(y), (256,), generator=g)
        loss = F.cross_entropy(m(to_in(X[i].to(dev))), y[i].to(dev))
        opt.zero_grad()
        loss.backward()
        gn.append(float(torch.nn.utils.clip_grad_norm_(m.parameters(), a.clip if a.clip > 0 else 1e9)))
        opt.step()
        sch.step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    train_s = time.perf_counter() - t0

    def ev(XX, yy):
        with torch.no_grad():
            S = torch.cat([m(to_in(XX[j:j + 250].to(dev))).cpu() for j in range(0, len(yy), 250)])
        return 100.0 * (S.argmax(1) == yy).float().mean().item(), F.cross_entropy(S, yy).item()
    acc, ce = ev(Xt, yt)
    tr_acc, _ = ev(X[:5000], y[:5000])
    # 失敗の理由を探る（実験77）: 入力を 2 画素ずらしたときの正解率と、出口の明るさの変わり方（cos の近さ）
    acc_shift2, _ = ev(torch.roll(Xt, (2, 2), (-2, -1)), yt)
    with torch.no_grad():
        x0 = to_in(Xt[:250].to(dev))
        s0, s1 = m.screen(x0).flatten(1), m.screen(torch.roll(x0, (1, 1), (-2, -1))).flatten(1)
        cos_shift1 = float(F.cosine_similarity(s0, s1).mean())
        n0 = 1e-3 * torch.randn_like(x0.real)
        cos_noise = float(F.cosine_similarity(s0, m.screen(p2.normalize(x0 + n0)).flatten(1)).mean())
    rec = dict(exp="75", **vars(a), test_acc=acc, test_ce=ce, train_acc=tr_acc, train_s=train_s,
               acc_shift2=acc_shift2, cos_shift1=cos_shift1, cos_noise=cos_noise,
               grad_norm_med=float(torch.tensor(gn).median()), grad_norm_max=max(gn),
               params=sum(p.numel() for p in m.parameters() if p.requires_grad), nl_params=[float(q.p.detach()) for q in m.nls],
               version=p2.__version__, torch=torch.__version__)
    print(json.dumps(rec, ensure_ascii=False))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if a.save:
        os.makedirs("results/models", exist_ok=True)
        torch.save({k: v.cpu() for k, v in m.state_dict().items()}, f"results/models/exp75_{a.task}_{a.kind}_{a.stages}x{a.width}_{a.read}_s{a.seed}.pt")


if __name__ == "__main__":
    main()
