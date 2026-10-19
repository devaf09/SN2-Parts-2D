"""実験20: 親の光路から子の光路を作る「木」（Fashion-MNIST）

着想: 持ち主の構想（2026-10-19、Project Metasequoia）。服を学習した光路から、ジャケットやパジャマを学習する光路を作るとき、
      共通性の高い親を参考にできるはず。ばらばらに学習したものを融合するときも、学習初期の板が同じ方が有利な気がする。
方法: 親 = 大まかな 3 分類（上着類 {T シャツ・プルオーバー・ドレス・コート・シャツ}、靴 {サンダル・スニーカー・ブーツ}、その他 {ズボン・バッグ}）
      を学習した光路（板 ① 16 枚）。子 = 各グループの中の細かい分類（3 本）。子の作り方を 4 通り比べる:
  indep  : 子ごとに別のでたらめな板から
  same   : 全部の子が同じでたらめな板から（学習前の板が同じ）
  derive : 学習した親をそのまま出発点に、全部の板を追加学習
  front  : 親の前半 8 枚を共有して凍結し、子は後半 8 枚と窓だけ学習（SN2-Acoustics 実験04 の形）
      子はどれも「担当なし」の一般の絵でも学習する（実験10、重み 0.3）。融合 = 3 本の子のスコアを足すだけ（10 分類）。
      子のデータは、全部（グループの訓練データ全部）と、少ない（1 クラス 300 枚）の 2 通り。
指標: 子の正解率（自分のグループの中で）、融合の 10 分類の正解率。比べるもの: 10 分類を 1 本で同時に学習。
結果: results/exp20.txt

実行: python experiments/image/exp20_tree.py --seed 0 --n_per_class 0 --device cuda
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import sn2_parts_2d as p2  # noqa: E402
from exp07_merge import acc, scores  # noqa: E402
from exp15_mnist import load  # noqa: E402

GROUPS = [[0, 2, 3, 4, 6], [5, 7, 9], [1, 8]]       # 上着類・靴・その他
COARSE_WIN = [0, 5, 1]                              # 親が大まかな分類に使う窓（各グループの代表のクラスの窓）


def freeze_front(net, n_front):
    """板の先頭 n_front 枚の A・φ を学習しない（勾配を 0 にする）。"""
    mask = torch.ones_like(net.A)
    mask[:n_front] = 0
    net.A.register_hook(lambda g: g * mask)
    net.phi.register_hook(lambda g: g * mask)
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", default="xy")
    ap.add_argument("--steps_parent", type=int, default=3000)
    ap.add_argument("--steps_child", type=int, default=2000)
    ap.add_argument("--n_per_class", type=int, default=0, help="子の訓練データを 1 クラスこの枚数に（0 = 全部）")
    ap.add_argument("--w", type=float, default=0.3)
    ap.add_argument("--variants", nargs="+", default=["indep", "same", "derive", "front"])
    ap.add_argument("--sched", default="const", choices=["const", "cosine"], help="学習率の変え方（実験25 で cosine）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp20.txt")
    a = ap.parse_args()
    dev = torch.device(a.device)
    kinds = ["xy"] * 16 if a.method == "xy" else [a.method] * 16
    f = lambda v: p2.image_to_field(v, "amp")
    Xtr, ytr, Xte, yte = load("fashion", 64)
    t0 = time.perf_counter()
    # 親: 大まかな 3 分類
    coarse = torch.zeros_like(ytr)
    for gi, g in enumerate(GROUPS):
        coarse[torch.isin(ytr, torch.tensor(g))] = COARSE_WIN[gi]
    parent = p2.train(p2.Net2D(64, 10, kinds, seed=a.seed).to(dev), Xtr, coarse, steps=a.steps_parent, seed=a.seed, to_input=f, schedule=a.sched)
    ct = torch.zeros_like(yte)
    for gi, g in enumerate(GROUPS):
        ct[torch.isin(yte, torch.tensor(g))] = COARSE_WIN[gi]
    sp = scores(parent, Xte)
    res = {"parent_coarse": acc(sp[:, COARSE_WIN], torch.tensor([COARSE_WIN.index(v) for v in ct.tolist()]))}
    t_parent = time.perf_counter() - t0
    gen = torch.Generator().manual_seed(a.seed)
    for var in a.variants:
        children = []
        for gi, g in enumerate(GROUPS):
            m = torch.isin(ytr, torch.tensor(g)).nonzero()[:, 0]
            if a.n_per_class:
                m = torch.cat([(ytr == c).nonzero()[:, 0][torch.randperm(int((ytr == c).sum()), generator=gen)[:a.n_per_class]] for c in g])
            if var == "indep":
                net = p2.Net2D(64, 10, kinds, seed=a.seed + 100 * (gi + 1)).to(dev)
            elif var == "same":
                net = p2.Net2D(64, 10, kinds, seed=a.seed + 100).to(dev)
            elif var == "derive":
                net = copy.deepcopy(parent)
            else:
                net = freeze_front(copy.deepcopy(parent), 8)
            G = p2.generic_images(4000, 64, seed=5000 + 100 * gi + a.seed)
            p2.train(net, Xtr[m], ytr[m], steps=a.steps_child, seed=a.seed + gi, to_input=f, reject=G, reject_weight=a.w, schedule=a.sched)
            children.append(net)
            tm = torch.isin(yte, torch.tensor(g))
            s = scores(net, Xte[tm])
            res[f"{var}_child{gi}"] = acc(s[:, g], torch.tensor([g.index(v) for v in yte[tm].tolist()]))
        res[f"{var}_merged"] = acc(scores(p2.merge(children), Xte), yte)
    rec = dict(exp="20", **vars(a), **res, t_parent=t_parent, total_s=time.perf_counter() - t0,
               version=p2.__version__, torch=torch.__version__)
    print(json.dumps(rec, ensure_ascii=False))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
