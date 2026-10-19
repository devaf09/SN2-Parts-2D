"""実験29: 共通する「何か」に反応しないようにする — 差分で読む（持ち主の案、2026-10-20）

着想: 持ち主の見立て。誤反応は、数字の形以外に「白い何か（線がある所）」という共通の特徴も一緒に覚えているためではないか。
      3 と 7 の共通する部分では反応しない仕組みにすれば、3 の特徴・7 の特徴だけで判断できるはず。
      判断が難しい場合は差分でチェックする。（1 次元の Sin-CNN2 実験23 では、似た技能を引くと誤反応が減った）
方法: 実験27・28 と同じ課題（MNIST の数字を 1〜2 個並べた 64 × 64）。光路を 3 本学習する:
      「3」、「7」、「なんでも数字」（どの数字の位置でも光る = 共通する特徴）。読み方を比べる（光路の出力は確率になる前の点数 Z）:
        single   : Z3 だけ（実験27・28 と同じ）
        minus_any: Z3 − λ·Z数字（共通する特徴を引く。λ = 0.5・1）
        minus_7  : Z3 − Z7（相手と比べる。3 と 7 の区別だけを見る）
      4f 系 1 段（fourier）と、2 段 + 光の非線形（fourier2nl、実験28 で最良）の両方で。
指標: AUC（目標の数字の位置と、ほかの数字の位置での点数の最大を比べ、目標の方が高い組の割合）、
      見落とし 5% のときの誤報（目標の 95% が光る閾値で、ほかの数字の位置が光る割合）、3 と 7 だけの間の AUC。
結果: results/exp29.txt

実行: python experiments/image/exp29_diff.py --method fourier2nl --seed 0 --device cuda
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
from exp27_locate import Fourier4f, Fourier4fNL, make_scenes, target_maps  # noqa: E402


def any_maps(objs):
    yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    T = torch.zeros(len(objs), 64, 64)
    for i, o in enumerate(objs):
        for cy, cx, _ in o:
            T[i] = torch.maximum(T[i], ((yy - cy) ** 2 + (xx - cx) ** 2 <= 9).float())
    return T


def train_path(make, X, T, steps, seed, dev, pw=5.0):
    m = make().to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        i = torch.randint(len(X), (64,), generator=g)
        loss = F.binary_cross_entropy_with_logits(m(p2.image_to_field(X[i].to(dev), "amp")), T[i].to(dev),
                                                  pos_weight=torch.tensor(pw, device=dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
    return m


@torch.no_grad()
def logits(m, X, batch=500):
    dev = next(m.parameters()).device
    return torch.cat([m(p2.image_to_field(X[i:i + batch].to(dev), "amp")).cpu() for i in range(0, len(X), batch)])


def peaks(S, objs):
    """各数字の位置（中心から 4 画素以内）での点数の最大と、その数字の種類。"""
    v, d = [], []
    for i, o in enumerate(objs):
        for cy, cx, dig in o:
            v.append(float(S[i, max(cy - 4, 0):cy + 5, max(cx - 4, 0):cx + 5].max()))
            d.append(dig)
    return torch.tensor(v), torch.tensor(d)


def metrics(S, objs, digit, other=None):
    v, d = peaks(S, objs)
    pos = v[d == digit]
    neg = v[d != digit] if other is None else v[d == other]
    auc = float(((pos[:, None] > neg[None]).float() + 0.5 * (pos[:, None] == neg[None]).float()).mean())
    thr = torch.quantile(pos, 0.05)                        # 目標の 95% が光る閾値
    fa = float((neg >= thr).float().mean())
    return 100.0 * auc, 100.0 * fa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="fourier2nl", choices=["fourier", "fourier2nl"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp29.txt")
    a = ap.parse_args()
    dev = torch.device(a.device)
    Xtr, otr = make_scenes(20000, 100 + a.seed, True)
    Xte, ote = make_scenes(2000, 200 + a.seed, False)
    make = (lambda s: (lambda: Fourier4f(1, s))) if a.method == "fourier" else (lambda s: (lambda: Fourier4fNL(s)))
    t0 = time.perf_counter()
    Z = {}
    for name, T, s in [("3", target_maps(otr, 3), 3), ("7", target_maps(otr, 7), 7), ("any", any_maps(otr), 11)]:
        Z[name] = logits(train_path(make(a.seed + s), Xtr, T, a.steps, a.seed, dev), Xte)
    res = {}
    for digit, mine, rival in [(3, "3", "7"), (7, "7", "3")]:
        for rname, S in [("single", Z[mine]), ("minus_any0.5", Z[mine] - 0.5 * Z["any"]), ("minus_any1", Z[mine] - Z["any"]),
                         ("minus_rival", Z[mine] - Z[rival])]:
            auc, fa = metrics(S, ote, digit)
            auc37, fa37 = metrics(S, ote, digit, other=int(rival))
            res[f"d{digit}_{rname}"] = dict(auc=auc, fa95=fa, auc_vs_rival=auc37, fa95_vs_rival=fa37)
    rec = dict(exp="29", **vars(a), **res, total_s=time.perf_counter() - t0, version=p2.__version__, torch=torch.__version__)
    print(json.dumps(rec, ensure_ascii=False))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
