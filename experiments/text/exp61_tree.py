"""実験61: 2DTG の木 — 葉（細かい担当）を枝・根（広い担当）が補う（持ち主の案、2026-10-21）

着想: 持ち主の案。実験58 で、小さな 2DTG に担当を分けて足すだけで 1 本の光路より +3.3 点（16 個 × 8 枚で 36.0%）。
      融合を樹形図にすれば、細かい担当（葉）が落とした所を、広い担当（枝・根）が補える。n-gram の後退と同じ考え方。
方法: 担当は直前の語の番号（頻度順）の余りで決める。葉 = 余り 16 で 16 個、枝 = 余り 4 で 4 個（葉 g は枝 g mod 4 の下）、根 = 全部 1 個。
      それぞれ (2 語前, 直前) を入れ、葉・枝は担当外に「担当なし」を学ぶ（実験58 と同じ）。根は担当なしなし。全部バラバラに学習。
      融合:
        leaves   = 葉だけを足す（実験58）
        fixed    = 葉・枝・根を足したものを、重み (a, b, c) で混ぜる。重みは訓練から除いた 5000 件で選ぶ
        backoff  = 葉の自信（一番上の確率）が低いほど枝・根に寄せる: 重み = 葉 s、枝・根 (1 − s) を b : c で。s = 葉の一番上の確率 ^ γ（γ も 5000 件で選ぶ）
条件: 葉 16 × 板 8、枝 4 × 板 16、根 1 × 板 32（合計 224 枚）、区画 16、符号読み 256、訓練は全部（33 万件）、3000 ステップずつ。
結果: results/exp61.txt

実行: python experiments/text/exp61_tree.py --seed 0 --device cuda
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import sn2_parts_2d as p2  # noqa: E402
import worddata  # noqa: E402
from exp40_inputs import EncNet  # noqa: E402

pair = lambda Z: Z[:, -2:]


def train_grain(X, y, own, other, C, layers, patches, K, seed, steps, reject, dev):
    torch.manual_seed(seed)
    m = EncNet("bands", 2, 4096, C + 1, layers, patches, seed, K).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed)
    nr = int(256 * reject) if len(other) else 0
    for _ in range(steps):
        io = own[torch.randint(len(own), (256 - nr,), generator=g)]
        xb, yb = X[io], y[io]
        if nr:
            ir = other[torch.randint(len(other), (nr,), generator=g)]
            xb, yb = torch.cat([xb, X[ir]]), torch.cat([yb, torch.full((nr,), C)])
        loss = F.cross_entropy(m(pair(xb).to(dev)), yb.to(dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
    return m.eval()


def level_probs(grains, XX, C, dev):
    """1 つの段の 2DTG を足すだけで融合した確率 (N, C)。"""
    with torch.no_grad():
        P = torch.stack([torch.cat([torch.softmax(m(pair(XX[j:j + 250]).to(dev)), -1).cpu() for j in range(0, len(XX), 250)]) for m in grains])
    r = 1 - P[:, :, C]
    q = (P[:, :, :C] * r[..., None]).sum(0)
    return q / q.sum(1, keepdim=True)


def ev(p, yy):
    nll = -torch.log(p[torch.arange(len(yy)), yy] + 1e-12)
    return dict(acc=100.0 * (p.argmax(1) == yy).float().mean().item(), ce=nll.mean().item(),
                top5=100.0 * (p.topk(5, 1).indices == yy[:, None]).any(1).float().mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leaves", type=int, default=16)
    ap.add_argument("--branches", type=int, default=4)
    ap.add_argument("--leaf_layers", type=int, default=8)
    ap.add_argument("--branch_layers", type=int, default=16)
    ap.add_argument("--root_layers", type=int, default=32)
    ap.add_argument("--patches", type=int, default=16)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--C", type=int, default=1024)
    ap.add_argument("--reject", type=float, default=0.3)
    ap.add_argument("--max_train", type=int, default=400000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp61.txt")
    a = ap.parse_args()
    dev = torch.device(a.device)
    C = a.C
    (X, y), (Xt, yt), info = worddata.load(4, C, a.max_train, 10000, a.seed, 4096)
    Xv, yv = X[-5000:], y[-5000:]
    X, y = X[:-5000], y[:-5000]
    t0 = time.perf_counter()
    levels = {}
    for name, G, layers in (("leaf", a.leaves, a.leaf_layers), ("branch", a.branches, a.branch_layers), ("root", 1, a.root_layers)):
        grp = X[:, -1] % G
        levels[name] = [train_grain(X, y, (grp == gi).nonzero().squeeze(1), (grp != gi).nonzero().squeeze(1), C, layers, a.patches, a.K,
                                    a.seed * 1000 + G * 10 + gi, a.steps, a.reject, dev) for gi in range(G)]
    train_s = time.perf_counter() - t0
    Pv = {k: level_probs(v, Xv, C, dev) for k, v in levels.items()}
    Pt = {k: level_probs(v, Xt, C, dev) for k, v in levels.items()}
    res = {k: ev(Pt[k], yt) for k in Pt}
    grid = [i / 10 for i in range(11)]
    ce_v = lambda p: ev(p, yv)["ce"]
    best = min(((wa, wb) for wa, wb in itertools.product(grid, grid) if wa + wb <= 1),
               key=lambda w: ce_v(w[0] * Pv["leaf"] + w[1] * Pv["branch"] + (1 - w[0] - w[1]) * Pv["root"]))
    wa, wb = best
    res["fixed"] = dict(weights=[wa, wb, 1 - wa - wb], **ev(wa * Pt["leaf"] + wb * Pt["branch"] + (1 - wa - wb) * Pt["root"], yt))

    def backoff(P, gam, bfrac):
        s = P["leaf"].max(1, keepdim=True).values ** gam
        return s * P["leaf"] + (1 - s) * (bfrac * P["branch"] + (1 - bfrac) * P["root"])
    bg = min(((g_, b_) for g_ in (0.1, 0.25, 0.5, 1.0, 2.0) for b_ in grid), key=lambda w: ce_v(backoff(Pv, *w)))
    res["backoff"] = dict(gamma=bg[0], branch_frac=bg[1], **ev(backoff(Pt, *bg), yt))
    rec = dict(exp="61", **vars(a), results=res, test_acc=max(res["fixed"]["acc"], res["backoff"]["acc"]),
               params=sum(p.numel() for v in levels.values() for m in v for p in m.parameters() if p.requires_grad), train_s=train_s,
               version=p2.__version__, torch=torch.__version__)
    print(json.dumps(rec, ensure_ascii=False))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
