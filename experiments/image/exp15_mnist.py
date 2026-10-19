"""実験15: 本物の画像（MNIST・Fashion-MNIST）の認識（M3）

着想: 持ち主の提案（2026-10-18）。合成課題（M1）と融合（M2）が成り立ったので、本物の画像で試す。
方法: 28 × 28 の画像を 64 × 64 の場の真ん中に置き（周りは暗い）、振幅として載せる。板 16 枚、学習 3000 ステップ × 256 件。
      訓練 6 万枚・テスト 1 万枚（公式の分け方）。
比べるもの: ロジスティック回帰（Sin-CNN2 のポイント表の目標 92.5%）、小さな CNN（同じくらいの学習する数）、レンズの後の明るさの回帰。
結果: results/exp15.txt

データは tools/fetch_mnist.py で data/ に取得する（リポジトリには入れない）。
実行: python experiments/image/exp15_mnist.py --data mnist --method xy --seed 0 --device cuda
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

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def load(name, L=64):
    """(X_train, y_train, X_test, y_test)。画像は 0〜1、L × L の真ん中に置く。"""
    from torchvision import datasets
    cls = {"mnist": datasets.MNIST, "fashion": datasets.FashionMNIST}[name]
    out = []
    for train in (True, False):
        d = cls(ROOT, train=train, download=False)
        X = d.data.float() / 255.0
        p = (L - X.shape[-1]) // 2
        X = F.pad(X, (p, L - X.shape[-1] - p, p, L - X.shape[-1] - p))
        out += [X, d.targets.clone()]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="mnist", choices=["mnist", "fashion"])
    ap.add_argument("--method", default="xy", choices=list(PLATES) + list(BASELINES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--patches", type=int, default=8)
    ap.add_argument("--n_ripples", type=int, default=None)
    ap.add_argument("--illum", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save", default=None)
    ap.add_argument("--out", default="results/exp15.txt")
    a = ap.parse_args()
    dev = torch.device(a.device)
    Xtr, ytr, Xte, yte = load(a.data, a.L)
    torch.manual_seed(a.seed)
    if a.method in PLATES:
        net = p2.Net2D(a.L, 10, PLATES[a.method], patches=a.patches, seed=a.seed, n_ripples=a.n_ripples,
                       freeze=(a.method == "random"),
                       illum=None if a.illum is None else (int(a.illum.split(":")[1]), a.illum.split(":")[0])).to(dev)
        to_input, n_params, lr = (lambda v: p2.image_to_field(v, "amp")), net.n_params(), a.lr
    else:
        net = BASELINES[a.method](a.L, 10).to(dev)
        to_input, n_params = None, sum(p.numel() for p in net.parameters())
        lr = 0.003 if a.method in ("logreg", "lens_logreg", "mlp") else a.lr
    t0 = time.perf_counter()
    p2.train(net, Xtr, ytr, steps=a.steps, lr=lr, batch_size=a.batch, seed=a.seed, to_input=to_input)
    t_train = time.perf_counter() - t0
    acc = p2.evaluate(net, Xte, yte, to_input=to_input)
    acc_tr = p2.evaluate(net, Xtr[:10000], ytr[:10000], to_input=to_input)
    if a.save:
        os.makedirs(os.path.dirname(a.save) or ".", exist_ok=True)
        torch.save({"state": net.state_dict(), "args": vars(a), "classes": 10}, a.save)
    rec = dict(exp="15", **vars(a), lr_used=lr, test_acc=acc, train_acc=acc_tr, params=n_params, train_s=t_train,
               version=p2.__version__, torch=torch.__version__)
    print(json.dumps(rec, ensure_ascii=False))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
