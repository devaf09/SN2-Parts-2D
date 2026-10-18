"""実験12: 技能を 3〜4 個に増やしても、「担当なし」の学習で足すだけの融合が保てるか（M2）

着想: 実験10 で、「担当なし」の一般の絵を見せて学習すると、2 技能の足すだけの融合が同時学習を上回った（向き × 位置 100.0%）。
      1 次元では技能が増えるほど融合後の判別が落ちた（4 技能で同時学習より約 4 点低い、Sin-CNN2 実験21）。2 次元ではどうか。
方法: クラスを k 個の技能に等分し（向き × 位置 8 クラスなら k = 2・4、位置 9 クラスなら k = 3）、各技能は自分のクラスのデータと
      一般の絵（技能ごとに別の乱数）だけで学習する（部品 0.6.0 の train(reject=...)、重み 0.3）。融合は足すだけ。
比べるもの: 同じ課題を 1 本の光路で同時に学習。「担当なし」なしの融合。
結果: results/exp12.txt

実行: python experiments/exp12_many.py --task conj --k 4 --seed 0 --device cuda
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
from exp07_merge import acc, scores  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="conj", choices=["conj", "position", "shape", "orient"])
    ap.add_argument("--k", type=int, default=4, help="技能の数（クラスを等分する）")
    ap.add_argument("--method", default="xy")
    ap.add_argument("--w", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--joint", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp12.txt")
    a = ap.parse_args()
    dev = torch.device(a.device)
    C = tasks2d.TASKS[a.task]
    if C % a.k:
        raise ValueError("クラス数が k で割り切れない")
    f = lambda v: p2.image_to_field(v, "amp")
    X, y = tasks2d.make(a.task, a.n_train, 64, seed=1000 + a.seed)
    T, t = tasks2d.make(a.task, a.n_test, 64, seed=2000 + a.seed)
    # クラスの分け方: 向き × 位置は向きでまとめる（y % 4 が同じものを同じ技能に）。それ以外は番号順
    order = sorted(range(C), key=lambda c: (c % 4, c)) if a.task == "conj" else list(range(C))
    groups = [order[i * C // a.k:(i + 1) * C // a.k] for i in range(a.k)]
    t0 = time.perf_counter()
    nets, nets0 = [], []
    for i, cls in enumerate(groups):
        m = torch.isin(y, torch.tensor(cls))
        G = p2.generic_images(4000, 64, seed=5000 + 100 * i + a.seed)
        nets.append(p2.train(p2.Net2D(64, C, PLATES[a.method], seed=a.seed + 100 * i).to(dev), X[m], y[m],
                             steps=a.steps, seed=a.seed + i, to_input=f, reject=G, reject_weight=a.w))
        nets0.append(p2.train(p2.Net2D(64, C, PLATES[a.method], seed=a.seed + 100 * i).to(dev), X[m], y[m],
                              steps=a.steps, seed=a.seed + i, to_input=f))
    res = dict(merged=acc(scores(p2.merge(nets), T), t), merged_noreject=acc(scores(p2.merge(nets0), T), t))
    if a.joint:
        J = p2.train(p2.Net2D(64, C, PLATES[a.method], seed=a.seed).to(dev), X, y, steps=a.steps, seed=a.seed, to_input=f)
        res["joint_acc"] = acc(scores(J, T), t)
    rec = dict(exp="12", **vars(a), groups=groups, **res, train_s=time.perf_counter() - t0,
               version=p2.__version__, torch=torch.__version__)
    print(json.dumps(rec, ensure_ascii=False))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
