"""実験01: 2 次元の基本部品で、合成した模様を分類する（M1 の主な表）

着想: 持ち主の依頼の M1。板の 3 つの形（① 交点で掛け算・② 斜めの縞・③ 波紋）とその混合を、同じ学習する数でそろえて比べる。
方法: 64 × 64 の合成課題 4 つ（tasks2d.py: 形・縞の向き・位置・向き × 位置）を、板 16 枚・区画 8 × 8 の光路で学習する。
      入力は画像を振幅として載せる（振幅か位相かは実験02）。学習 1500 ステップ × 256 件、Adam lr 0.01。
比べるもの: 偶然、レンズだけ（板 0 枚）、学習しないでたらめな板（窓の背景光・温度だけ学習）、
            ロジスティック回帰、小さな MLP、同じくらいの学習する数の小さな CNN、レンズの後の明るさにロジスティック回帰。
結果: results/exp01.txt（1 回 = 1 行の JSON）。

実行: python experiments/image/exp01_basic.py --task shape --method stripes --seed 0 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import sn2_parts_2d as p2  # noqa: E402
import tasks2d  # noqa: E402
from baselines2d import BASELINES  # noqa: E402

# SinWave 側の条件: 名前 → 板の形の並び
PLATES = {
    "xy": ["xy"] * 16,
    "stripes": ["stripes"] * 16,
    "ripple": ["ripple"] * 16,
    "ripple2": ["ripple2"] * 16,
    "mix_xy_ripple": ["xy"] * 8 + ["ripple"] * 8,
    "sinc": ["sinc"] * 16,         # ④ 減衰する波紋（持ち主の案、実験11）
    "mix_xy_sinc": ["xy"] * 8 + ["sinc"] * 8,
    "sinc_e": ["sinc_e"] * 16,     # ⑤ 楕円で大きさを変えられる減衰する波紋（持ち主の案、実験14）
    "random": ["stripes"] * 16,    # 板は学習しない
    "lens": [],                    # 板 0 枚
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="shape", choices=list(tasks2d.TASKS))
    ap.add_argument("--method", default="stripes", choices=list(PLATES) + list(BASELINES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--patches", type=int, default=8)
    ap.add_argument("--z", type=float, default=64.0)
    ap.add_argument("--input", default="amp", choices=["amp", "phase"])
    ap.add_argument("--n_ripples", type=int, default=None, help="④ の板 1 枚あたりの波紋の数（省略時は区画² / 2）")
    ap.add_argument("--sinc_depth", type=float, default=1.0, help="④ の初期の深さの倍率")
    ap.add_argument("--save", default=None, help="学習した光路を保存するファイル（results/models/ の下。リポジトリには入れない）")
    ap.add_argument("--illum", default=None, help="2 次元の照明。例 tilt:2（縦横 2 × 2 の区画、4 通り）、hadamard:4")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp01.txt")
    a = ap.parse_args()
    dev = torch.device(a.device)
    C = tasks2d.TASKS[a.task]
    # データは seed ごとに作り直す（訓練とテストは別の乱数）
    Xtr, ytr = tasks2d.make(a.task, a.n_train, a.L, seed=1000 + a.seed)
    Xte, yte = tasks2d.make(a.task, a.n_test, a.L, seed=2000 + a.seed)
    torch.manual_seed(a.seed)
    if a.method in PLATES:
        net = p2.Net2D(a.L, C, PLATES[a.method], patches=a.patches, z=a.z, seed=a.seed,
                       freeze=(a.method == "random"), n_ripples=a.n_ripples, sinc_depth=a.sinc_depth,
                       illum=None if a.illum is None else (int(a.illum.split(":")[1]), a.illum.split(":")[0])).to(dev)
        to_input = lambda v: p2.image_to_field(v, a.input)
        n_params = net.n_params()
        lr = a.lr
    else:
        net = BASELINES[a.method](a.L, C).to(dev)
        to_input = None
        n_params = sum(p.numel() for p in net.parameters())
        lr = 0.003 if a.method in ("logreg", "lens_logreg", "mlp") else a.lr
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    p2.train(net, Xtr, ytr, steps=a.steps, lr=lr, batch_size=a.batch, seed=a.seed, to_input=to_input)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t_train = time.perf_counter() - t0
    t0 = time.perf_counter()
    acc = p2.evaluate(net, Xte, yte, to_input=to_input)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t_test = time.perf_counter() - t0
    acc_tr = p2.evaluate(net, Xtr[:2000], ytr[:2000], to_input=to_input)
    if a.save:
        os.makedirs(os.path.dirname(a.save) or ".", exist_ok=True)
        torch.save({"state": net.state_dict(), "args": vars(a), "classes": C}, a.save)
    rec = dict(exp="01", **vars(a), lr_used=lr, classes=C, test_acc=acc, train_acc=acc_tr, params=n_params,
               train_s=t_train, infer_us_per_sample=1e6 * t_test / a.n_test,
               peak_MB=torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else None,
               version=p2.__version__, torch=torch.__version__)
    print(json.dumps(rec, ensure_ascii=False))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
