"""実験10: 「担当なし」の一般の絵を見せて、融合の取り違えを直す（M2）

着想: 実験07〜09 で、融合の落ち込みの元は「各技能が自分の担当でない絵にも自分の窓を明るくする」ことと分かった。
      相手の技能のデータを使わずにこれを直すため、どの課題とも関係のない一般の絵（tasks2d.make_generic: ぼやけた塊・斑の模様）を
      「担当なし」として見せ、そのときはどの窓も同じ明るさになる（答えを出さない）ように学習させる。
方法: 技能ごとの学習の損失 = 自分のデータのクロスエントロピー + w × 一般の絵での「全部の窓で一様」へのクロスエントロピー。
      一般の絵は技能ごとに別の乱数で作る（技能どうしで共有しなくてよい）。融合は実験07 と同じ足すだけ。
比べるもの: 担当なしなし（実験07 と同じ条件をこのスクリプトで測り直す）、同時に学習。
結果: results/exp10.txt

実行: python experiments/exp10_reject.py --task conj --seed 0 --device cuda
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sn2_parts_2d as p2  # noqa: E402
import tasks2d  # noqa: E402
from exp01_basic import PLATES  # noqa: E402
from exp07_merge import SPLITS, acc, scores  # noqa: E402


def train_reject(net, X, y, G, w, steps=1500, lr=0.01, batch=256, seed=0):
    """自分のデータ (X, y) と担当なしの絵 G で学習する。w = 0 なら普通の学習。"""
    dev = next(net.parameters()).device
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr)
    g = torch.Generator().manual_seed(seed)
    f = lambda v: p2.image_to_field(v.to(dev), "amp")
    for _ in range(steps):
        i = torch.randint(len(X), (batch,), generator=g)
        loss = F.cross_entropy(net(f(X[i])), y[i].to(dev))
        if w > 0:
            j = torch.randint(len(G), (batch // 2,), generator=g)
            loss = loss + w * (-F.log_softmax(net(f(G[j])), 1).mean())   # 全部の窓で一様（答えを出さない）
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="conj", choices=list(SPLITS))
    ap.add_argument("--method", default="xy", choices=["xy", "stripes", "ripple"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--w", type=float, default=1.0, help="担当なしの損失の重み")
    ap.add_argument("--n_generic", type=int, default=4000)
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp10.txt")
    a = ap.parse_args()
    dev = torch.device(a.device)
    C = tasks2d.TASKS[a.task]
    X, y = tasks2d.make(a.task, a.n_train, 64, seed=1000 + a.seed)
    T, t = tasks2d.make(a.task, a.n_test, 64, seed=2000 + a.seed)
    ca, cb = SPLITS[a.task]
    ina, inb = torch.isin(y, torch.tensor(ca)), torch.isin(y, torch.tensor(cb))
    tna, tnb = torch.isin(t, torch.tensor(ca)), torch.isin(t, torch.tensor(cb))
    Ga = tasks2d.make_generic(a.n_generic, 64, seed=5000 + a.seed)
    Gb = tasks2d.make_generic(a.n_generic, 64, seed=6000 + a.seed)
    t0 = time.perf_counter()
    A = train_reject(p2.Net2D(64, C, PLATES[a.method], seed=a.seed).to(dev), X[ina], y[ina], Ga, a.w, a.steps, seed=a.seed)
    B = train_reject(p2.Net2D(64, C, PLATES[a.method], seed=a.seed + 100).to(dev), X[inb], y[inb], Gb, a.w, a.steps, seed=a.seed + 1)
    t_train = time.perf_counter() - t0
    own = lambda s, cls: s[:, cls]
    sA, sB, sM = scores(A, T), scores(B, T), scores(p2.merge([A, B]), T)
    res = dict(
        skill_a=acc(own(sA, ca)[tna], torch.tensor([ca.index(v) for v in t[tna].tolist()])),
        skill_b=acc(own(sB, cb)[tnb], torch.tensor([cb.index(v) for v in t[tnb].tolist()])),
        merged=acc(sM, t),
        merged_skill_pick=100.0 * ((sM[:, ca].max(1).values >= sM[:, cb].max(1).values) == tna).float().mean().item(),
        cancel_maxdiff=(scores(p2.merge([A]), T) - sA).abs().max().item(),
    )
    rec = dict(exp="10", **vars(a), **res, train_s=t_train, version=p2.__version__, torch=torch.__version__)
    print(json.dumps(rec, ensure_ascii=False))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
