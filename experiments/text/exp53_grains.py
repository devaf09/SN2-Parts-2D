"""実験53: 2DT グレイン（2DTG）を並べて次の単語を当てる（持ち主の案、2026-10-20）

着想: 持ち主の案。「2D のトークンから確率を予想する学習器」を 2DT グレイン（2DTG）と呼ぶ。
      2DTG-k は k 語先の単語の確率を出す。欲しい単語 D の前が A B C なら、C を 2DTG-1、B を 2DTG-2、A を 2DTG-3 に通し、
      確率を割り当て（固定の重み）か確信度で重みをつけて合わせる。1 つの文章で全部を並列に学習する。
      数えた表での見積もり（2026-10-20）: 1 語だけ見せる 2DTG を足しても 33.6 → 34.2% とほとんど伸びない。
      各 2DTG に「直前の語 + k 語前の語」の 2 語を見せると 37.1%（3-gram 36.4）、CE 3.24（3-gram 3.96）。
方法: 2DTG-1 は直前の 1 語、2DTG-k（k ≥ 2）は (k 語前, 直前) の 2 語を帯に並べて入れる（実験52 と同じ入れ方・符号読み）。
      それぞれ別々に学習する（足すだけの融合の形。お互いのデータや重みを見ない）。
      合わせ方: avg = 同じ重み、fixed = 割り当て 0.4・0.3・0.2・0.1、conf = 一番上の確率で重み、
                learned = 訓練の一部（学習に使っていない 5000 件）で 4 つの重みだけを選ぶ（足すだけの融合で、重みの調整だけ）。
条件: 各 2DTG は板 ① 8 枚 × 区画 16（4 つで 32 枚 = 実験52 の 1 本の光路と同じ枚数）。3000 ステップ、学習率 0.01 → 0。
比べる相手: 実験52（1 本の光路で直前 4 語、板 32 枚）、単語の n-gram、GRU。
結果: results/exp53.txt

実行: python experiments/text/exp53_grains.py --seed 0 --device cuda
"""
from __future__ import annotations

import argparse
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
from exp52_words import PROMPTS, topk  # noqa: E402


def grain_input(X, k):
    """直前 n 語 X (N, n) → 2DTG-k の入力。k = 1 は直前の 1 語、k ≥ 2 は (k 語前, 直前)。"""
    n = X.shape[1]
    return X[:, n - 1:] if k == 1 else torch.stack([X[:, n - k], X[:, n - 1]], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grains", type=int, default=4)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--patches", type=int, default=16)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--C", type=int, default=1024)
    ap.add_argument("--V", type=int, default=4096)
    ap.add_argument("--max_train", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp53.txt")
    a = ap.parse_args()
    dev = torch.device(a.device)
    n = a.grains
    (X, y), (Xt, yt), info = worddata.load(n, a.C, a.max_train + 5000, 10000, a.seed, a.V)
    Xv, yv = X[a.max_train:], y[a.max_train:]          # 重みを選ぶ用（学習に使わない）
    X, y = X[:a.max_train], y[:a.max_train]
    grains, per, t_all = [], [], 0.0
    for k in range(1, n + 1):
        torch.manual_seed(a.seed * 100 + k)
        m = EncNet("bands", 1 if k == 1 else 2, a.V, a.C, a.layers, a.patches, a.seed * 100 + k, a.K).to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=0.01)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
        g = torch.Generator().manual_seed(a.seed * 100 + k)
        Xk = grain_input(X, k)
        t0 = time.perf_counter()
        for step in range(a.steps):
            i = torch.randint(len(y), (256,), generator=g)
            loss = F.cross_entropy(m(Xk[i].to(dev)), y[i].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_all += time.perf_counter() - t0
        grains.append(m.eval())

    def probs(XX):
        with torch.no_grad():
            return torch.stack([torch.cat([torch.softmax(m(grain_input(XX[j:j + 250], k + 1).to(dev)), -1).cpu()
                                           for j in range(0, len(XX), 250)]) for k, m in enumerate(grains)])   # (n, N, C)

    def ev(p, yy):
        return (100.0 * (p.argmax(1) == yy).float().mean().item(), -torch.log(p[torch.arange(len(yy)), yy] + 1e-12).mean().item(),
                100.0 * (p.topk(5, 1).indices == yy[:, None]).any(1).float().mean().item())

    Pt, Pv = probs(Xt), probs(Xv)
    for k in range(n):
        acc, ce, t5 = ev(Pt[k], yt)
        per.append(dict(k=k + 1, test_acc=acc, test_ce=ce, test_top5=t5))
    # 重みを選ぶ（確率の重み付き和。重みは 0.05 刻みの格子から、検証用 5000 件の CE が最小のもの）
    best, best_ce = None, 1e9
    import itertools
    grid = [i / 20 for i in range(21)]
    for wts in itertools.product(grid, repeat=n - 1):
        if sum(wts) > 1:
            continue
        w = torch.tensor([1 - sum(wts), *wts])
        ce = -torch.log((Pv * w[:, None, None]).sum(0)[torch.arange(len(yv)), yv] + 1e-12).mean().item()
        if ce < best_ce:
            best, best_ce = w, ce
    fixed = torch.tensor([0.4, 0.3, 0.2, 0.1][:n])
    fixed = fixed / fixed.sum()
    conf = Pt.max(2).values[..., None]
    combos = dict(avg=Pt.mean(0), fixed=(Pt * fixed[:, None, None]).sum(0), conf=(Pt * conf / conf.sum(0, keepdim=True)).sum(0),
                  learned=(Pt * best[:, None, None]).sum(0))
    res = {}
    for name, p in combos.items():
        acc, ce, t5 = ev(p, yt)
        res[name] = dict(test_acc=acc, test_ce=ce, test_top5=t5)
    demo = {}
    for ptxt in PROMPTS:
        xp, ws = worddata.encode_prompt(ptxt, info, n)
        p = (probs(xp) * best[:, None, None]).sum(0)[0]
        demo[ptxt] = dict(words=ws, top5=topk(torch.log(p), info))
    rec = dict(exp="53", **vars(a), per_grain=per, combos=res, learned_weights=best.tolist(), test_acc=res["learned"]["test_acc"],
               test_ce=res["learned"]["test_ce"], test_top5=res["learned"]["test_top5"],
               params=sum(p.numel() for m in grains for p in m.parameters() if p.requires_grad), train_s=t_all, demo=demo,
               version=p2.__version__, torch=torch.__version__)
    print(json.dumps(rec, ensure_ascii=False))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
