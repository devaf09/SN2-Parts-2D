"""実験07: 2 次元でも、別々に学習した光路を足すだけで融合でき、外せば完全に元に戻るか（M2）

着想: Sin-CNN2 の一番の特徴（実験02・03・21）。互いに干渉しない光路を技能ごとに持たせ、スクリーンで明るさを足すだけで融合する。
方法: 8 クラスの課題を 4 クラスずつ 2 つの技能に分ける。技能 A・B は自分のデータ（自分の 4 クラス）だけで、別々に学習する。
      窓は 8 個を共有し、A は 0〜3 番目のクラス、B は 4〜7 番目のクラスの窓を担当する。
  conj : 線の向き 4 種 × 左右。A = 横・/（左右とも）、B = 縦・\（左右とも）。向きで分けるので、左右の位置の手がかりでは技能を選べない
  shape: 形 8 種。A = 円・正方形・三角・十字、B = L 字・横棒・リング・X
比べるもの: 全 8 クラスを 1 本の光路で同時に学習（同じ板、学習する数は融合の半分）。
指標: 技能単独（自分のデータ・自分の窓の中で）、融合（全データ・全窓）、取消（融合から B を外したものと A 単独のスコアの差）、
      融合の別の読み方「自信で選ぶ」（各光路の自分の窓の中での確信度が高い方の答え。SN2-Acoustics 実験02 で強かった）。
結果: results/exp07.txt

実行: python experiments/exp07_merge.py --task conj --method xy --seed 0 --device cuda
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

# 技能の分け方: 課題 → (A のクラス, B のクラス)。窓の番号はクラスの番号そのまま
SPLITS = {"conj": ([0, 1, 4, 5], [2, 3, 6, 7]), "shape": ([0, 1, 2, 3], [4, 5, 6, 7])}


@torch.no_grad()
def scores(net, X, batch=500):
    dev = next(net.parameters()).device
    return torch.cat([net(p2.image_to_field(X[i:i + batch].to(dev), "amp")).cpu() for i in range(0, len(X), batch)])


def acc(s, y):
    return 100.0 * (s.argmax(1) == y).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="conj", choices=list(SPLITS))
    ap.add_argument("--method", default="xy", choices=["xy", "stripes", "ripple"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp07.txt")
    a = ap.parse_args()
    dev = torch.device(a.device)
    C = tasks2d.TASKS[a.task]
    f = lambda v: p2.image_to_field(v, "amp")
    X, y = tasks2d.make(a.task, a.n_train, 64, seed=1000 + a.seed)
    T, t = tasks2d.make(a.task, a.n_test, 64, seed=2000 + a.seed)
    ca, cb = SPLITS[a.task]
    ina, inb = torch.isin(y, torch.tensor(ca)), torch.isin(y, torch.tensor(cb))
    tna, tnb = torch.isin(t, torch.tensor(ca)), torch.isin(t, torch.tensor(cb))
    t0 = time.perf_counter()
    # 技能 A・B（別々の板の乱数、別々のデータ）と、同時学習の比較
    A = p2.train(p2.Net2D(64, C, PLATES[a.method], seed=a.seed).to(dev), X[ina], y[ina], steps=a.steps, seed=a.seed, to_input=f)
    B = p2.train(p2.Net2D(64, C, PLATES[a.method], seed=a.seed + 100).to(dev), X[inb], y[inb], steps=a.steps, seed=a.seed + 1, to_input=f)
    J = p2.train(p2.Net2D(64, C, PLATES[a.method], seed=a.seed).to(dev), X, y, steps=a.steps, seed=a.seed, to_input=f)
    t_train = time.perf_counter() - t0
    sA, sB, sJ = scores(A, T), scores(B, T), scores(J, T)
    sM = scores(p2.merge([A, B]), T)
    sMA = scores(p2.merge([A]), T)
    # 技能単独: 自分のデータ・自分の窓の中で
    own = lambda s, cls: s[:, cls]
    a_alone = acc(own(sA, ca)[tna], torch.tensor([ca.index(v) for v in t[tna].tolist()]))
    b_alone = acc(own(sB, cb)[tnb], torch.tensor([cb.index(v) for v in t[tnb].tolist()]))
    # 自信で選ぶ: 各光路の自分の窓の中での softmax の最大値が大きい方
    pa, pb = own(sA, ca).softmax(1), own(sB, cb).softmax(1)
    pick_a = pa.max(1).values >= pb.max(1).values
    conf = torch.where(pick_a, torch.tensor(ca)[pa.argmax(1)], torch.tensor(cb)[pb.argmax(1)])
    res = dict(skill_a=a_alone, skill_b=b_alone, merged=acc(sM, t), merged_conf=100.0 * (conf == t).float().mean().item(),
               joint=acc(sJ, t), cancel_maxdiff=(sMA - sA).abs().max().item(),
               merged_skill_pick=100.0 * ((sM[:, ca].max(1).values >= sM[:, cb].max(1).values) == tna).float().mean().item())
    rec = dict(exp="07", **vars(a), **res, params_skill=A.n_params(), train_s=t_train, version=p2.__version__, torch=torch.__version__)
    print(json.dumps(rec, ensure_ascii=False))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
