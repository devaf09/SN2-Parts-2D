"""実験80: 分担の 2DTG の中身を 4f 系の多段 + 平方根に替える（持ち主の OK 2026-10-25）

着想: 光路 1 本でも、拡散 + 4f 系 + 平方根で 53.0% になった（実験79、6 段 × 8、9000 ステップ）。表を分担した 2DTG 16 個は
      54.0%（実験62、中身は板 ① 8 枚）。2 つを組み合わせる: 直前の文字の余りで担当を決め打ち（16 個）、各 2DTG の中身を 4f 系に。
方法: 各 2DTG = exp78_text4f.Text4f（拡散、レンズの後で読む、窓 64 個）を、自分の担当の例だけで学習。確率をそのまま並べる（担当は 1 つ）。
比べる: 分担の 2DTG 16 個（板） 52.3〜54.0%（実験62・74）、光路 1 本の 4f 系 53.0%（実験79）、3-gram 53.7%、4-gram 55.4%。
結果: results/exp80.txt
実行: python experiments/text/exp80_split4f.py --n 8 --stages 4 --width 4 --seed 0 --device cuda
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
from exp78_text4f import Text4f  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="直前の何文字を入れるか")
    ap.add_argument("--G", type=int, default=16)
    ap.add_argument("--enc", default="spread")
    ap.add_argument("--stages", type=int, default=4)
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--kind", default="sqrt")
    ap.add_argument("--read", default="lens", choices=["lens", "plane"])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp80.txt")
    a = ap.parse_args()
    dev = torch.device(a.device)
    (X, y), (Xt, yt), info = textdata.load(8, 64, 600000, 10000, a.seed)
    V, C = info["V_in"], info["C"]
    X, Xt = X[:, -a.n:], Xt[:, -a.n:]
    grp, grpt = X[:, -1] % a.G, Xt[:, -1] % a.G
    S = torch.zeros(len(yt), C)
    params, tr_hit, tr_n = 0, 0, 0
    t0 = time.perf_counter()
    for gi in range(a.G):
        own, ownt = (grp == gi).nonzero().squeeze(1), (grpt == gi).nonzero().squeeze(1)
        sd = a.seed * 100 + gi
        torch.manual_seed(sd)
        m = Text4f(a.enc, a.n, V, C, a.stages, a.width, a.kind, a.read, sd).to(dev)
        opt = torch.optim.Adam(m.opt_params(), lr=0.01)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
        g = torch.Generator().manual_seed(sd)
        for _ in range(a.steps):
            i = own[torch.randint(len(own), (256,), generator=g)]
            loss = F.cross_entropy(m(X[i].to(dev)), y[i].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
        with torch.no_grad():
            if len(ownt):
                S[ownt] = torch.cat([m(Xt[ownt][j:j + 250].to(dev)).cpu() for j in range(0, len(ownt), 250)])
            o = own[:1000]
            tr_hit += int((m(X[o].to(dev)).cpu().argmax(1) == y[o]).sum())
            tr_n += len(o)
        params += sum(q.numel() for q in m.opt_params())
        print(f"担当 {gi}: 訓練 {len(own)} 件、テスト {len(ownt)} 件", flush=True)
    train_s = time.perf_counter() - t0
    acc, ce = 100.0 * (S.argmax(1) == yt).float().mean().item(), F.cross_entropy(S, yt).item()
    rec = dict(exp="80", **vars(a), test_acc=acc, test_ce=ce, train_acc=100.0 * tr_hit / max(1, tr_n), params=params, train_s=train_s,
               version=p2.__version__, torch=torch.__version__)
    print(json.dumps(rec, ensure_ascii=False))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
