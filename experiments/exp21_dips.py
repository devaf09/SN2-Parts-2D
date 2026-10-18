"""実験21: 学習の途中で正解率が落ち込む（谷）理由を調べる。細かい学習曲線とクラスごとの正解率

着想: 持ち主の問い（2026-10-19）。学習の途中で正解率がバウンドする。学習が谷で終わるともったいない。
      (1) 学習の進め方の問題（歩幅が大きすぎる・1 回の 256 枚の偏り）なら、訓練とテストが一緒に落ち、クラス全体が揺れる。
      (2) 的外れな着眼点（持ち主の例: 服の影を服の特徴として覚える）なら、訓練は保ったままテストだけが落ちる、
          または特定のクラスだけが落ちる。
      どちらかを、細かい学習曲線（50 ステップごと）とクラスごとの正解率で見分ける。
      合わせて、谷で終わらないための 2 つの工夫を測る:
        - 確かめ用のデータ（訓練から 5000 枚取り分け、学習には使わない）で一番良かった時点の光路を使う（最良の時点）
        - 板の数の移動平均（EMA、直近の数百ステップの平均の板）で答える
方法: Fashion-MNIST、訓練 55000 枚・確かめ用 5000 枚・テスト 1 万枚。5000 ステップ、50 ステップごとに測る。
比べるもの: 学習率一定と cosine（0 まで下げる）。板 ① と ④（波紋 128 個・深さ 0.5）。
結果: results/exp21.txt（curve: [ステップ, 学習の時間, テスト, 訓練, 確かめ用, テスト(EMA), 1 回の損失, クラスごとのテスト ×10]）

実行: python experiments/exp21_dips.py --method xy --sched const --seed 0 --device cuda
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sn2_parts_2d as p2  # noqa: E402
from exp01_basic import PLATES  # noqa: E402
from exp15_mnist import load  # noqa: E402


@torch.no_grad()
def pred(net, X, f, batch=1000):
    dev = next(net.parameters()).device
    return torch.cat([net(f(X[i:i + batch].to(dev))).argmax(1).cpu() for i in range(0, len(X), batch)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="fashion", choices=["mnist", "fashion"])
    ap.add_argument("--method", default="xy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--sched", default="const", choices=["const", "cosine"])
    ap.add_argument("--ema", type=float, default=0.995, help="板の移動平均の係数（直近 約 1/(1−係数) ステップの平均）")
    ap.add_argument("--n_ripples", type=int, default=None)
    ap.add_argument("--sinc_depth", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp21.txt")
    a = ap.parse_args()
    dev = torch.device(a.device)
    Xtr, ytr, Xte, yte = load(a.data, 64)
    g = torch.Generator().manual_seed(1234)
    perm = torch.randperm(len(Xtr), generator=g)
    Xva, yva = Xtr[perm[:5000]], ytr[perm[:5000]]
    Xtr, ytr = Xtr[perm[5000:]], ytr[perm[5000:]]
    f = lambda v: p2.image_to_field(v, "amp")
    torch.manual_seed(a.seed)
    net = p2.Net2D(64, 10, PLATES[a.method], seed=a.seed, n_ripples=a.n_ripples, sinc_depth=a.sinc_depth).to(dev)
    ema = copy.deepcopy(net)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps) if a.sched == "cosine" else None
    gb = torch.Generator().manual_seed(a.seed)
    curve, t_train = [], 0.0
    best = (-1, None, None)   # (確かめ用の正解率, ステップ, テスト)
    for step in range(1, a.steps + 1):
        t0 = time.perf_counter()
        idx = torch.randint(len(Xtr), (256,), generator=gb)
        loss = F.cross_entropy(net(f(Xtr[idx].to(dev))), ytr[idx].to(dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.mul_(a.ema).add_(pn, alpha=1 - a.ema)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_train += time.perf_counter() - t0
        if step % a.every == 0 or step == a.steps:
            pt = pred(net, Xte, f)
            te = 100.0 * (pt == yte).float().mean().item()
            tr = 100.0 * (pred(net, Xtr[:5000], f) == ytr[:5000]).float().mean().item()
            va = 100.0 * (pred(net, Xva, f) == yva).float().mean().item()
            te_ema = 100.0 * (pred(ema, Xte, f) == yte).float().mean().item()
            per = [round(100.0 * (pt[yte == c] == c).float().mean().item(), 1) for c in range(10)]
            curve.append([step, round(t_train, 1), te, tr, va, te_ema, round(loss.item(), 4)] + per)
            if va > best[0]:
                best = (va, step, te)
            if step % 500 == 0:
                print(a.method, a.sched, step, round(t_train), te, tr, va, te_ema, flush=True)
    rec = dict(exp="21", **vars(a), params=net.n_params(), final_test=curve[-1][2], final_test_ema=curve[-1][5],
               best_val=best[0], best_step=best[1], best_test=best[2], curve=curve,
               version=p2.__version__, torch=torch.__version__)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
