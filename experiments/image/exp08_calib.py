"""実験08: 融合の取り違えを、少量の共通データでの較正で直す（M2）

着想: 実験07 で、足すだけの融合は技能の取り違えで大きく落ちた（向き × 位置 80.8%、同時学習 99.4%）。
      SN2-Acoustics の実験02 では、専門家ごとの温度と背景光を、生の点数のまま少量のデータで合わせる較正（jcal）が効いた。
方法: 実験07 と同じに 2 技能を別々に学習したあと、板は動かさず、各光路の温度（gain）と窓ごとの背景光（bias）だけを、
      全クラスから取った少量の較正データ（n_cal 件）で、融合したスコアのクロスエントロピーが小さくなるように合わせる。
      学習する数は 2 × (1 + 8) = 18 個。取消（片方を外す）は、外したい光路を除いて元の温度・背景光に戻せば完全に元どおり。
比べるもの: 較正なしの融合（実験07）、同時学習。
結果: results/exp08.txt

実行: python experiments/image/exp08_calib.py --task conj --seed 0 --device cuda
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
import tasks2d  # noqa: E402
from exp01_basic import PLATES  # noqa: E402
from exp07_merge import SPLITS, acc, scores  # noqa: E402


def calibrate(nets, X, y, steps=300, lr=0.05, seed=0):
    """板を固定し、各光路の gain・bias だけを融合したスコアで合わせたコピーを返す。"""
    nets = [copy.deepcopy(n) for n in nets]
    dev = next(nets[0].parameters()).device
    params = []
    for n in nets:
        for name, p in n.named_parameters():
            p.requires_grad_(name in ("gain", "bias"))
            if name in ("gain", "bias"):
                params.append(p)
    with torch.no_grad():   # 板の光は変わらないので、窓の明るさを先に計算しておく
        W = [torch.cat([p2.read_windows(n.screen(p2.image_to_field(X[i:i + 500].to(dev), "amp")), n.det)
                        for i in range(0, len(X), 500)]) for n in nets]
    opt = torch.optim.Adam(params, lr=lr)
    yy = y.to(dev)
    for _ in range(steps):
        s = sum(n.gain * w + n.bias for n, w in zip(nets, W)) / len(nets)
        loss = torch.nn.functional.cross_entropy(s, yy)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return nets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="conj", choices=list(SPLITS))
    ap.add_argument("--method", default="xy", choices=["xy", "stripes", "ripple"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--n_cal", type=int, nargs="+", default=[40, 200, 1000])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp08.txt")
    a = ap.parse_args()
    dev = torch.device(a.device)
    C = tasks2d.TASKS[a.task]
    f = lambda v: p2.image_to_field(v, "amp")
    X, y = tasks2d.make(a.task, a.n_train, 64, seed=1000 + a.seed)
    T, t = tasks2d.make(a.task, a.n_test, 64, seed=2000 + a.seed)
    Xc, yc = tasks2d.make(a.task, max(a.n_cal), 64, seed=3000 + a.seed)      # 較正用（訓練・テストとは別）
    ca, cb = SPLITS[a.task]
    ina, inb = torch.isin(y, torch.tensor(ca)), torch.isin(y, torch.tensor(cb))
    t0 = time.perf_counter()
    A = p2.train(p2.Net2D(64, C, PLATES[a.method], seed=a.seed).to(dev), X[ina], y[ina], steps=a.steps, seed=a.seed, to_input=f)
    B = p2.train(p2.Net2D(64, C, PLATES[a.method], seed=a.seed + 100).to(dev), X[inb], y[inb], steps=a.steps, seed=a.seed + 1, to_input=f)
    t_train = time.perf_counter() - t0
    res = {"merged": acc(scores(p2.merge([A, B]), T), t)}
    for n in a.n_cal:
        Ac, Bc = calibrate([A, B], Xc[:n], yc[:n], seed=a.seed)
        res[f"cal{n}"] = acc(scores(p2.merge([Ac, Bc]), T), t)
    rec = dict(exp="08", **vars(a), **res, train_s=t_train, version=p2.__version__, torch=torch.__version__)
    print(json.dumps(rec, ensure_ascii=False))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
