"""実験30: 10 種類の数字の位置を同時に出す（4f 系 2 段 + 光の非線形を、数字ごとに 1 本ずつ、色を分けて融合）

着想: 実験28 で、4f 系 2 段 + 間の光の非線形が、3 と 7 の位置を見分けよく出せた（AUC 97%）。持ち主の目標
      「融合した光路それぞれが反応する場所を、ピクセル単位で出す」を、10 種類に広げる。
方法: 64 × 64 の場に MNIST の数字（20 × 20）を 1〜2 個、0〜9 を同じ割合で置く。数字ごとに光路を 1 本（fourier2nl）、
      それぞれ自分の数字の中心に光を集めるよう別々に学習する（ほかの数字は見せるが、光らせない）。融合 = 10 本の出力を並べ、
      各画素で一番高い光路の数字をその場所の答えにする（色を分けた重ね合わせと同じ）。
指標: 数字ごとの AUC（実験27 と同じ）、数字の位置での答え（その数字の中心から 4 画素以内で一番高い光路の数字が正しいか）、
      絵全体で、ある数字の組を全部当てたか。比較: 普通の分類（板 ① 16 枚、1 つだけの数字、実験15）は位置を出せない。
結果: results/exp30.txt。seed 0 の図は tools/fig_exp30.py。

実行: python experiments/image/exp30_ten.py --seed 0 --device cuda
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
from exp27_locate import Fourier4fNL, make_scenes, target_maps  # noqa: E402
from exp29_diff import logits, metrics, train_path  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp30.txt")
    a = ap.parse_args()
    dev = torch.device(a.device)
    Xtr, otr = make_scenes(20000, 300 + a.seed, True, uniform=True)
    Xte, ote = make_scenes(2000, 400 + a.seed, False, uniform=True)
    t0 = time.perf_counter()
    Z = []
    for d in range(10):
        m = train_path(lambda: Fourier4fNL(a.seed * 10 + d), Xtr, target_maps(otr, d), a.steps, a.seed, dev)
        Z.append(logits(m, Xte))
        if a.seed == 0:
            os.makedirs("results/models", exist_ok=True)
            torch.save({"state": m.state_dict(), "digit": d}, f"results/models/exp30_d{d}_s0.pt")
    Z = torch.stack(Z, 1)                                   # (N, 10, L, L)
    res = {f"auc_{d}": metrics(Z[:, d], ote, d)[0] for d in range(10)}
    ok, tot, scene_ok = 0, 0, 0
    for i, o in enumerate(ote):
        all_ok = True
        for cy, cx, dig in o:
            win = Z[i, :, max(cy - 4, 0):cy + 5, max(cx - 4, 0):cx + 5].flatten(1).max(1).values
            hit = int(win.argmax()) == dig
            ok += hit
            tot += 1
            all_ok &= hit
        scene_ok += all_ok
    res.update(per_digit=100.0 * ok / tot, per_scene=100.0 * scene_ok / len(ote), auc_mean=sum(res[f"auc_{d}"] for d in range(10)) / 10)
    rec = dict(exp="30", **vars(a), **res, params_per_path=8196, total_s=time.perf_counter() - t0, version=p2.__version__, torch=torch.__version__)
    print(json.dumps(rec, ensure_ascii=False))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
