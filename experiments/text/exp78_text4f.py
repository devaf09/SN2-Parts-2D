"""実験78: 言語の入力を面全体に広げて、4f 系の多段 + 平方根に通す（持ち主と決めた方針 2026-10-25: 平方根 + レンズの後で読む）

着想: 4f 系 3〜4 段 + 平方根（= 位相を崩す）で、画像は小さな CNN 並みになった（実験76・77）。板を重ねる形では、
      言語（直前 8 文字 → 次の文字）の入れ方を変えても 46〜47% で差がなかった（実験40）。畳み込み + 非線形の形なら、
      面全体に広げた入れ方（spread・rainbow）が効くかもしれない（持ち主の問い「非線形とセットの広げる実験は必要か」）。
方法: 入れ方は実験40 の EncNet.field（bands・smear・rainbow・spread・rings、位置ごとの明るさを学習）→ exp75_4f.Multi4f（幅 4）→ 窓 64 個。
      非線形: sqrt（基本）、none（線形、比べる）、kerr（強さ 0.1 に固定、実機向け。実験77）。読み方: lens（基本）、plane（比べる）。
課題: 文字（直前 8 文字 → 次の文字 64 クラス、訓練は全部 = 実験73 と同じ）。比べる: 板 ① 32 枚の光路 1 本 47.0%（実験73）。
条件: 3000 ステップ、学習率 0.01 → 0、seed 0〜2。結果: results/exp78.txt
実行: python experiments/text/exp78_text4f.py --enc spread --stages 3 --kind sqrt --read lens --seed 0 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import sn2_parts_2d as p2  # noqa: E402
import textdata  # noqa: E402
from exp40_inputs import EncNet  # noqa: E402
from exp75_4f import Multi4f  # noqa: E402


class Text4f(torch.nn.Module):
    def __init__(self, enc, n, V, C, stages, width, kind, read, seed):
        super().__init__()
        self.enc = EncNet(enc, n, V, C, 1, 8, seed)      # 入れ方と位置ごとの明るさだけを使う（中の板は使わない）
        self.body = Multi4f(64, C, stages, width, kind, read, seed)

    def forward(self, X):
        return self.body(self.enc.field(X))

    def opt_params(self):
        return [self.enc.w] + [q for q in self.body.parameters() if q.requires_grad]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default="spread", choices=["bands", "smear", "rainbow", "spread", "rings", "blur"])
    ap.add_argument("--stages", type=int, default=3)
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--kind", default="sqrt")
    ap.add_argument("--fix_p", type=float, default=None)
    ap.add_argument("--read", default="lens", choices=["lens", "plane"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp78.txt")
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    (X, y), (Xt, yt), info = textdata.load(8, 64, 600000, 10000, a.seed)
    V, C = info["V_in"], info["C"]
    m = Text4f(a.enc, 8, V, C, a.stages, a.width, a.kind, a.read, a.seed).to(dev)
    if a.fix_p is not None:
        for q in m.body.nls:
            q.p.data.fill_(a.fix_p)
            q.p.requires_grad_(False)
    opt = torch.optim.Adam(m.opt_params(), lr=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    g = torch.Generator().manual_seed(a.seed)
    t0 = time.perf_counter()
    for _ in range(a.steps):
        i = torch.randint(len(y), (256,), generator=g)
        loss = F.cross_entropy(m(X[i].to(dev)), y[i].to(dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    train_s = time.perf_counter() - t0

    def ev(XX, yy):
        with torch.no_grad():
            S = torch.cat([m(XX[j:j + 250].to(dev)).cpu() for j in range(0, len(yy), 250)])
        return 100.0 * (S.argmax(1) == yy).float().mean().item(), F.cross_entropy(S, yy).item()
    acc, ce = ev(Xt, yt)
    tr_acc, _ = ev(X[:10000], y[:10000])
    rec = dict(exp="78", **vars(a), test_acc=acc, test_ce=ce, train_acc=tr_acc, train_s=train_s,
               params=sum(q.numel() for q in m.opt_params()), nl_params=[float(q.p.detach()) for q in m.body.nls],
               version=p2.__version__, torch=torch.__version__)
    print(json.dumps(rec, ensure_ascii=False))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
