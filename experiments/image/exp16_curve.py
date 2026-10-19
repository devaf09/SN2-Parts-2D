"""実験16: 学習曲線 — 何ステップで「十分に学習した」と言えるか（MNIST・Fashion-MNIST）

着想: 持ち主の依頼（2026-10-19）。過学習と計算コストの目安にするため、学習の途中で何度も正解率を測り、
      どのくらいの学習で頭打ちになるか、訓練とテストの差（過学習）がいつ開くかを調べる。
方法: 実験15 と同じ条件（28 × 28 を 64 × 64 の真ん中に、振幅として）で、最大 10000 ステップ（1 ステップ 256 件）学習し、
      決めたステップごとにテスト（1 万枚全部）と訓練（先頭 1 万枚）の正解率を測る。測る時間は学習の時間に入れない。
      「十分に学習した」= テストの正解率が、その後の最高値から 0.5 点以内に入った最初のステップ（tools で集計）。
比べるもの: ロジスティック回帰、小さな CNN。
結果: results/exp16.txt（1 回の学習 = 1 行。曲線は curve に [ステップ, 経過秒, テスト, 訓練] の並び）

実行: python experiments/image/exp16_curve.py --data mnist --method xy --seed 0 --device cuda
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
from baselines2d import BASELINES  # noqa: E402
from exp01_basic import PLATES  # noqa: E402
from exp15_mnist import load  # noqa: E402

CHECK = [50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 7500, 10000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="mnist", choices=["mnist", "fashion"])
    ap.add_argument("--method", default="xy", choices=list(PLATES) + list(BASELINES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--n_ripples", type=int, default=None)
    ap.add_argument("--sinc_depth", type=float, default=1.0)
    ap.add_argument("--lr_center", type=float, default=1.0, help="④⑤ の波紋の中心の学習率の倍率（実験18）")
    ap.add_argument("--lr_width", type=float, default=1.0, help="④⑤ の波紋の幅 k の学習率の倍率（実験18）")
    ap.add_argument("--sinc_centers", default="random", choices=["random", "grid", "jitter"], help="波紋の中心の初期配置（実験19）")
    ap.add_argument("--sched", default="const", choices=["const", "cosine"], help="学習率の変え方（cosine は 0 まで下げる。実験17）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp16.txt")
    a = ap.parse_args()
    dev = torch.device(a.device)
    Xtr, ytr, Xte, yte = load(a.data, 64)
    torch.manual_seed(a.seed)
    if a.method in PLATES:
        net = p2.Net2D(64, 10, PLATES[a.method], seed=a.seed, n_ripples=a.n_ripples, sinc_depth=a.sinc_depth,
                        sinc_centers=a.sinc_centers).to(dev)
        f, n_params, lr = (lambda v: p2.image_to_field(v, "amp")), net.n_params(), a.lr
    else:
        net = BASELINES[a.method](64, 10).to(dev)
        f, n_params = (lambda v: v), sum(p.numel() for p in net.parameters())
        lr = 0.003 if a.method in ("logreg", "lens_logreg", "mlp") else a.lr
    groups = {"center": [], "width": [], "other": []}
    for name, p in net.named_parameters():
        if p.requires_grad:
            groups["center" if name in ("scx", "scy") else "width" if name == "sk" else "other"].append(p)
    mult = {"center": a.lr_center, "width": a.lr_width, "other": 1.0}
    opt = torch.optim.Adam([{"params": v, "lr": lr * mult[k]} for k, v in groups.items() if v], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps) if a.sched == "cosine" else None
    g = torch.Generator().manual_seed(a.seed)
    curve, t_train = [], 0.0
    for step in range(1, a.steps + 1):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        idx = torch.randint(len(Xtr), (a.batch,), generator=g)
        loss = F.cross_entropy(net(f(Xtr[idx].to(dev))), ytr[idx].to(dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_train += time.perf_counter() - t0
        if step in CHECK or step == a.steps:
            te = p2.evaluate(net, Xte, yte, to_input=f)
            tr = p2.evaluate(net, Xtr[:10000], ytr[:10000], to_input=f)
            curve.append([step, round(t_train, 2), te, tr])
            print(a.data, a.method, step, round(t_train, 1), te, tr, flush=True)
    rec = dict(exp="16", **vars(a), lr_used=lr, params=n_params, curve=curve, version=p2.__version__, torch=torch.__version__)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
