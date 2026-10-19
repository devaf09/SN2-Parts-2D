"""実験06: 学習した板を大きな場（64 → 128）へ移す: 引き伸ばしと敷き詰め

着想: 持ち主の見立て「比率を守れば X・Y が伸びても同じ効果」。回折の働きはフレネル数（模様の細かさ² / 板の間隔）で決まるので、
      模様を s 倍に引き伸ばすなら板の間隔を s² 倍にする必要がある（見立て）。1 次元の実験29 では敷き詰めで完全に同じ働きになった。
方法: 64 × 64 で学習した光路（板 ①②、向き × 位置の課題）を、学習し直さずに 128 × 128 で使う。
  stretch    : 板の模様を 2 倍に引き伸ばし（画素を 2 × 2 に複製）、間隔を 64 → 256（s² 倍）。入力の絵も 2 倍の大きさ。
               スクリーンは中央 64 × 64 を同じ窓で読む（絵が 2 倍なら光の向きは 1/2 になり、画素の番号では同じ所に来る）
  stretch_z64: 同じだが間隔を 64 のまま（s² 倍にしないと崩れるかの確認）
  tile       : 板を 2 × 2 に敷き詰める（間隔 64 のまま）。入力は同じ大きさの絵を 128 × 128 の真ん中に置く（周りは暗い）。
               スクリーンは 1 つおきの画素（偶数番目）を読む
  tile_rep   : 敷き詰めに、入力も 2 × 2 に敷き詰めた絵（理屈の上では元とまったく同じ答えになるはず）
比べるもの: 64 × 64 の元の光路、128 × 128 でゼロから学習した光路（stretch の入力・tile の入力それぞれ）。
結果: results/exp06.txt

実行: python experiments/image/exp06_scale.py --task conj --method xy --seed 0 --device cuda
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
import tasks2d  # noqa: E402
from exp01_basic import PLATES  # noqa: E402


@torch.no_grad()
def run(net64, phases, H, X, y, read, batch=250):
    """固定の板 phases・回折 H で X を通し、read（128 のスクリーン → 64 × 64）で読んで、元の窓・背景光・温度で答える。"""
    dev = net64.det.device
    ok = 0
    for i in range(0, len(X), batch):
        u = p2.image_to_field(X[i:i + batch].to(dev), "amp")
        for ph in phases:
            u = p2.normalize(p2.propagate(p2.apply_plate(u, ph), H))
        I = read(p2.intensity(p2.lens(u)))
        I = I / (I.mean((-2, -1), keepdim=True) + 1e-12)
        out = net64.gain * p2.read_windows(I, net64.det) + net64.bias
        ok += (out.argmax(1).cpu() == y[i:i + batch]).sum().item()
    return 100.0 * ok / len(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="conj", choices=list(tasks2d.TASKS))
    ap.add_argument("--method", default="xy", choices=["xy", "stripes", "ripple"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--scratch", type=int, default=1, help="1 なら 128 × 128 でゼロから学習した比較も測る")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp06.txt")
    a = ap.parse_args()
    dev = torch.device(a.device)
    C = tasks2d.TASKS[a.task]
    f = lambda v: p2.image_to_field(v, "amp")
    # 64 × 64 で学習
    X64, y64 = tasks2d.make(a.task, a.n_train, 64, seed=1000 + a.seed)
    T64, t64 = tasks2d.make(a.task, a.n_test, 64, seed=2000 + a.seed)
    net = p2.Net2D(64, C, PLATES[a.method], seed=a.seed).to(dev)
    p2.train(net, X64, y64, steps=a.steps, seed=a.seed, to_input=f)
    acc = {"orig64": p2.evaluate(net, T64, t64, to_input=f)}
    with torch.no_grad():
        ph64 = [net.plate_phase(d) for d in range(len(net.kinds))]
    # 128 × 128 の入力
    T128s, t128s = tasks2d.make(a.task, a.n_test, 128, seed=2000 + a.seed)      # 絵も 2 倍（引き伸ばし用）
    T128p = F.pad(T64, (32, 32, 32, 32))                                         # 同じ大きさの絵を真ん中に（敷き詰め用）
    T128r = T64.repeat(1, 2, 2)                                                  # 絵も敷き詰め
    crop = lambda I: I[..., 32:96, 32:96]                                        # 中央 64 × 64
    even = lambda I: I[..., ::2, ::2]                                            # 1 つおき（偶数番目）。中心 64 → 32
    up = lambda ph: ph.repeat_interleave(2, 0).repeat_interleave(2, 1)
    H256 = p2.transfer(128, 128, 256.0, device=dev)
    H64 = p2.transfer(128, 128, 64.0, device=dev)
    acc["stretch"] = run(net, [up(p) for p in ph64], H256, T128s, t128s, crop)
    acc["stretch_z64"] = run(net, [up(p) for p in ph64], H64, T128s, t128s, crop)
    acc["tile"] = run(net, [p.repeat(2, 2) for p in ph64], H64, T128p, t64, even)
    acc["tile_rep"] = run(net, [p.repeat(2, 2) for p in ph64], H64, T128r, t64, even)
    # 128 × 128 でゼロから（区画は同じ 8 × 8 = 16 画素四方、学習する数は同じ）
    t_scratch = None
    if a.scratch:
        t0 = time.perf_counter()
        X128s, y128s = tasks2d.make(a.task, a.n_train, 128, seed=1000 + a.seed)
        n1 = p2.Net2D(128, C, PLATES[a.method], seed=a.seed, z=256.0).to(dev)
        p2.train(n1, X128s, y128s, steps=a.steps, seed=a.seed, to_input=f)
        acc["scratch128_big"] = p2.evaluate(n1, T128s, t128s, to_input=f)
        X128p = F.pad(X64, (32, 32, 32, 32))
        n2 = p2.Net2D(128, C, PLATES[a.method], seed=a.seed).to(dev)
        p2.train(n2, X128p, y64, steps=a.steps, seed=a.seed, to_input=f)
        acc["scratch128_pad"] = p2.evaluate(n2, T128p, t64, to_input=f)
        t_scratch = time.perf_counter() - t0
    rec = dict(exp="06", **vars(a), acc=acc, scratch_s=t_scratch, version=p2.__version__, torch=torch.__version__)
    print(json.dumps(rec, ensure_ascii=False))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
