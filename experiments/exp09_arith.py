"""実験09: 共通の出発点から派生させた技能の足し算（1 本の光路で融合、M2）

着想: 1 次元の実験23 では、共通の出発点 θ0 から派生させた技能どうしなら θ0 + ΔA + ΔB で 1 本の光路に融合でき、
      並列光路と同等だった（ゼロから別々に学んだものどうしでは壊れる）。実験07・08 で、別々の光路を足す融合は
      2 次元では技能を取り違えて落ちた（較正しても 86.8%）。共通の出発点なら直るか。
方法: 出発点 θ0 = 別の課題（縞の向き 8 クラス）で学習した光路。そこから技能 A・B をそれぞれ自分のデータだけで追加学習する
      （窓 8 個を共有、A は自分の 4 クラス、B は残りの 4 クラスを担当。実験07 と同じ分け方）。
  combine : θ0 + ΔA + ΔB の 1 本の光路（計算は 1 本分）
  merge   : 派生させた A と B を並列光路として足す（計算は 2 本分）
  scratch : 実験07 と同じ、ゼロから別々に学習した A と B を並列光路として足す
比べるもの: 同時に学習（1 本）。
結果: results/exp09.txt

実行: python experiments/exp09_arith.py --task conj --seed 0 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sn2_parts_2d as p2  # noqa: E402
import tasks2d  # noqa: E402
from exp01_basic import PLATES  # noqa: E402
from exp07_merge import SPLITS, acc, scores  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="conj", choices=list(SPLITS))
    ap.add_argument("--base_task", default="orient")
    ap.add_argument("--method", default="xy", choices=["xy", "stripes", "ripple"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp09.txt")
    a = ap.parse_args()
    dev = torch.device(a.device)
    C = tasks2d.TASKS[a.task]
    f = lambda v: p2.image_to_field(v, "amp")
    X, y = tasks2d.make(a.task, a.n_train, 64, seed=1000 + a.seed)
    T, t = tasks2d.make(a.task, a.n_test, 64, seed=2000 + a.seed)
    Xb, yb = tasks2d.make(a.base_task, a.n_train, 64, seed=4000 + a.seed)
    ca, cb = SPLITS[a.task]
    ina, inb = torch.isin(y, torch.tensor(ca)), torch.isin(y, torch.tensor(cb))
    tna, tnb = torch.isin(t, torch.tensor(ca)), torch.isin(t, torch.tensor(cb))
    t0 = time.perf_counter()
    base = p2.train(p2.Net2D(64, C, PLATES[a.method], seed=a.seed).to(dev), Xb, yb, steps=a.steps, seed=a.seed, to_input=f)
    A = p2.train(p2.derive(base), X[ina], y[ina], steps=a.steps, seed=a.seed, to_input=f)
    B = p2.train(p2.derive(base), X[inb], y[inb], steps=a.steps, seed=a.seed + 1, to_input=f)
    t_train = time.perf_counter() - t0
    own = lambda s, cls: s[:, cls]
    sA, sB = scores(A, T), scores(B, T)
    res = dict(
        skill_a=acc(own(sA, ca)[tna], torch.tensor([ca.index(v) for v in t[tna].tolist()])),
        skill_b=acc(own(sB, cb)[tnb], torch.tensor([cb.index(v) for v in t[tnb].tolist()])),
        combine=acc(scores(p2.combine(base, [(1, A), (1, B)]), T), t),
        merge_derived=acc(scores(p2.merge([A, B]), T), t),
        base_on_task=acc(scores(base, T), t),
    )
    for lam in (0.5, 0.75):
        res[f"combine_{lam}"] = acc(scores(p2.combine(base, [(lam, A), (lam, B)]), T), t)
    rec = dict(exp="09", **vars(a), **res, train_s=t_train, version=p2.__version__, torch=torch.__version__)
    print(json.dumps(rec, ensure_ascii=False))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
