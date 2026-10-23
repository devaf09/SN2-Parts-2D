"""実験73 の図（光の非線形）。
  --curves : 非線形の形（明るさ → 出し直す振幅、カーは位相の遅れ）。学習前の値で描く
  --results: results/exp73.txt の正解率を、課題ごとに非線形の種類で並べる
  --io MODEL --task mnist|char --kind K --stages S --per P : 入力 → 1 段目の後の光 → 非線形の後の光 → 出口の明るさと窓 → 確率
実行: python tools/fig_nl.py --curves --results
      python tools/fig_nl.py --io results/models/exp73_mnist_kerr_2x16_s0.pt --task mnist --kind kerr --stages 2 --per 16
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

from figio import ROOT, plt, show_bars, show_field, show_output, show_plate

sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "image"))
import nl_parts as nl  # noqa: E402
import sn2_parts_2d as p2  # noqa: E402

NAMES = {"none": "なし", "sqrt": "平方根", "square": "2 乗", "log": "対数", "power": "べき（学習）", "sat": "可飽和（学習）",
         "thresh": "しきい値（学習）", "kerr": "カー効果（位相、学習）", "ssat": "可飽和・位相を残す（学習）"}


def curves(path):
    I = torch.linspace(0, 4, 200)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for k in ("sqrt", "square", "log", "power", "sat", "thresh"):
        ax[0].plot(I, nl.NL(k).amplitude(I).detach(), label=NAMES[k])
    ax[0].plot(I, I.sqrt() * 0 + 1, "k:", lw=0.8)
    ax[0].set_xlabel("明るさ I（平均 1）")
    ax[0].set_ylabel("出し直す光の振幅（位相は 0）")
    ax[0].set_title("明るさを測って光を出し直す型（学習前の形）")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].plot(I, 1.0 * I, label="カー効果: 位相の遅れ κ·I（κ = 1、学習）")
    ax[1].plot(I, 1 / (1 + I).sqrt(), label="可飽和・位相を残す: 振幅の倍率 1/√(1 + I/s)")
    ax[1].set_xlabel("明るさ I（平均 1）")
    ax[1].set_title("光の場をそのまま変える型（位相の情報を残す）")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(path)


def results(path):
    R = [json.loads(l) for l in open(os.path.join(ROOT, "results", "exp73.txt"), encoding="utf-8")]
    tasks = [t for t in ("mnist", "char", "charsplit") if any(r["task"] == t for r in R)]
    fig, ax = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 4.8), squeeze=False)
    for a, t in zip(ax[0], tasks):
        rs = [r for r in R if r["task"] == t]
        keys = sorted({(r["kind"], r["stages"], r["per"]) for r in rs}, key=lambda k: (k[1], k[0] != "none", k[0]))
        vals = [np.mean([r["test_acc"] for r in rs if (r["kind"], r["stages"], r["per"]) == k]) for k in keys]
        cols = ["#7f8c8d" if k[0] == "none" else "#2e86c1" for k in keys]
        a.barh(range(len(keys))[::-1], vals, color=cols)
        for i, v in enumerate(vals):
            a.text(v + 0.1, len(keys) - 1 - i, f"{v:.1f}", va="center", fontsize=8)
        a.set_yticks(range(len(keys))[::-1])
        a.set_yticklabels([f"{NAMES[k[0]]}  {k[1]}段×{k[2]}枚" for k in keys], fontsize=8)
        a.set_xlim(min(vals) - 3, max(vals) + 2)
        a.set_xlabel("テストの正解率 %")
        a.set_title({"mnist": "MNIST", "char": "文字（光路 1 本）", "charsplit": "文字（分担の 2DTG 16 個）"}[t])
    fig.suptitle("実験73: 段の間の光の非線形（灰色 = 非線形なし、同じ枚数）")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(path)


def io(model, task, kind, stages, per, path):
    if task != "mnist":
        raise SystemExit("入出力の図は今は mnist だけ")
    from exp15_mnist import load
    _, _, Xt, yt = load("mnist")
    m = nl.StagedNet(64, 10, stages, per, kind)
    m.load_state_dict(torch.load(model))
    m.eval()
    fig, ax = plt.subplots(2, 6, figsize=(21, 7.4))
    with torch.no_grad():
        for r, j in enumerate((0, 1)):
            u = p2.image_to_field(Xt[j:j + 1], "amp")
            I, inter = m.screen(u, keep=True)
            show_field(ax[r, 0], u[0], f"入力（正解 {int(yt[j])}）", kind="amp")
            show_plate(ax[r, 1], m.net.plate_phase(0), "学習後の板 1 枚目")
            show_field(ax[r, 2], inter[0][0][0], "1 段目の後の光（色 = 位相）")
            show_field(ax[r, 3], inter[0][1][0], f"非線形（{NAMES[kind]}）の後の光")
            show_output(ax[r, 4], I[0], m.net.det, "出口の明るさ（対数）と窓 10 個")
            show_bars(ax[r, 5], torch.softmax(m(u)[0], -1), [str(i) for i in range(10)], "確率（赤 = 正解）", int(yt[j]))
    fig.suptitle(f"実験73: 入力 → 1 段目 → 非線形 → 2 段目 → 出口（MNIST、{NAMES[kind]}、{stages} 段 × {per} 枚）")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    print(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", action="store_true")
    ap.add_argument("--results", action="store_true")
    ap.add_argument("--io")
    ap.add_argument("--task", default="mnist")
    ap.add_argument("--kind", default="kerr")
    ap.add_argument("--stages", type=int, default=2)
    ap.add_argument("--per", type=int, default=16)
    a = ap.parse_args()
    img = os.path.join(ROOT, "docs", "images")
    if a.curves:
        curves(os.path.join(img, "exp73_curves.png"))
    if a.results:
        results(os.path.join(img, "exp73_results.png"))
    if a.io:
        io(a.io, a.task, a.kind, a.stages, a.per, os.path.join(img, f"exp73_io_{a.kind}.png"))
