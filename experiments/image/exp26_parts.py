"""実験26: 部品ごとの窓で「部品があるか」を学習し、例を見せないクラスを部品の組み合わせで当てる（1 次元の語根の 2 次元版）

着想: 実験24 で、部品の符号の和を符号にしただけでは、例を見せないクラスは見せたクラスに吸い込まれた（0〜9%）。
      1 次元の Sin-CNN2 実験34・36 では、部品（語根）ごとに窓を持たせて「その部品が含まれるか」を学習し（複数の答えが同時に正しい学習）、
      新しいクラスを部品の組み合わせとして定義すると 89.7% で当たった（偏りの補正 + γ で全体からも 77.5%）。それを 2 次元で試す。
方法: Fashion-MNIST、板 ① 16 枚、学習率 0.01 → 0、3000 ステップ。窓 = 部品 9 個（上半身・長袖・前開き・下半身・全身・足・運動靴・丈が高い・持ち物）。
      学習: 各部品の窓の明るさ × 温度 + 背景光 を、その部品があるか（0/1）の確率として学習（二値のクロスエントロピー）。1 クラスは見せない。
      答え方: クラスの点数 = Σ（そのクラスにある部品の log σ(s)）+ Σ（ない部品の log σ(−s)）。
        prior: 部品の出やすさの偏りを差し引く（実験36）
        gamma: 見せたクラスの点数から一定の値 γ を引く（見せないクラスに引き寄せられる偏りを直す。実験36 の 2）
比べるもの: 実験24（部品の符号の和）。
結果: results/exp26.txt

実行: python experiments/image/exp26_parts.py --holdout 4 --seed 0 --device cuda
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
from exp15_mnist import load  # noqa: E402
from exp22_codes import CLASS_PARTS, PARTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp26.txt")
    a = ap.parse_args()
    dev = torch.device(a.device)
    Xtr, ytr, Xte, yte = load("fashion", 64)
    P = len(PARTS)
    M = torch.zeros(10, P)
    for c, ps in CLASS_PARTS.items():
        M[c, ps] = 1
    seen = [c for c in range(10) if c != a.holdout]
    mtr = torch.isin(ytr, torch.tensor(seen))
    X, Y = Xtr[mtr], M[ytr[mtr]]
    net = p2.Net2D(64, P, ["xy"] * 16, seed=a.seed).to(dev)
    f = lambda v: p2.image_to_field(v.to(dev), "amp")
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    g = torch.Generator().manual_seed(a.seed)
    t0 = time.perf_counter()
    for _ in range(a.steps):
        i = torch.randint(len(X), (256,), generator=g)
        loss = F.binary_cross_entropy_with_logits(net(f(X[i])), Y[i].to(dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
    with torch.no_grad():
        S = torch.cat([net(f(Xte[i:i + 1000])).cpu() for i in range(0, len(Xte), 1000)])        # 部品の点数 (N, P)
        Str = torch.cat([net(f(X[i:i + 1000])).cpu() for i in range(0, 20000, 1000)])
    part_acc = 100.0 * ((S > 0).float() == M[yte]).float().mean().item()
    res = {"part_acc": part_acc}
    prior = torch.log(Y[:20000].mean(0).clamp(1e-3, 1 - 1e-3) / (1 - Y[:20000].mean(0)).clamp(1e-3, 1))   # 部品の出やすさ（対数オッズ）
    for name, s in [("raw", S), ("prior", S - prior)]:
        cls = F.logsigmoid(s) @ M.T + F.logsigmoid(-s) @ (1 - M).T                                # (N, 10)
        for gam in (0.0, 1.0, 2.0, 4.0):
            c2 = cls.clone()
            c2[:, seen] -= gam
            pred = c2.argmax(1)
            un = yte == a.holdout
            acc_un = 100.0 * (pred[un] == a.holdout).float().mean().item()
            acc_seen = 100.0 * (pred[~un] == yte[~un]).float().mean().item()
            hm = 2 * acc_un * acc_seen / (acc_un + acc_seen + 1e-9)
            res[f"{name}_g{gam:g}"] = dict(unseen=acc_un, seen=acc_seen, harmonic=hm, all=100.0 * (pred == yte).float().mean().item())
    rec = dict(exp="26", **vars(a), **res, train_s=time.perf_counter() - t0, version=p2.__version__, torch=torch.__version__)
    print(json.dumps(rec, ensure_ascii=False))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
