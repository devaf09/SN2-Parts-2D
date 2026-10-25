"""実験87: 2 つの記憶用 Grain を別々の比率で融合できるか（持ち主の依頼 2026-10-26: 一時・短期・中期… と記憶に系列を持たせる前処理。記憶 G1 80% + 記憶 G2 20% など）

着想: 4f 系の周波数の面の板は、ホログラムと同じく「鍵 → 値」の組を学習なしで書き込める（整合フィルタ。鍵の光の共役 × 値の光を重ねる）。
      2 つの記憶 Grain（G1 に組の集まり A、G2 に B）を、比率 α : 1 − α で 1 枚に重ね書き（二重露光、明るさの比で比率を決める）すると、
      A も B も取り出せるか、取り出したときの明るさ（ピークの高さ）が比率を表すか（= どちらの記憶か・どれだけ古いかが読める）を調べる。
方法: 鍵 = 位相だけのでたらめな模様（L × L）、値 = 出口の面の窓 K 個のどれか（その窓の位置に光の点）。
      Grain の板 H_g = Σ_i conj(FFT(鍵_i)) · FFT(点_{値_i})（明るさの合計を 1 にそろえる）。
      融合の仕方:
        field : H = α H_1 + (1 − α) H_2 を 1 枚の板にする（二重露光。光の場で足す）
        phase : field の板の位相だけを残す（位相だけの板 = 空間光変調器でそのまま作れる形。比率が残るかを見る）
        inten : 板は別々のまま、出口の明るさを α : 1 − α で足す（光路 2 本 + 検出器で足す）
      3 つの Grain（比率 0.6 / 0.3 / 0.1 など、系列の下見）も測る。
測るもの: 取り出しの正解率（明るさが最大の窓 = 正しい値）、正しい窓の明るさの平均（A と B の比 = 比率が読めるか）、
          ピークの高さから「どちらの Grain の記憶か」を当てる正解率。
GPU は使わない（CPU で数秒〜数十秒）。結果: results/exp87.txt
実行: python experiments/exp87_memfuse.py
"""
from __future__ import annotations

import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sn2_parts_2d as p2  # noqa: E402

L, K = 64, 256
det = p2.window_grid(L, L, K, 2)                       # 窓 K 個（2 × 2）
cent = torch.stack([torch.stack(torch.where(d > 0)).float().mean(1) for d in det]).round().long()   # 窓の中心


def make_grain(n, g):
    keys = torch.polar(torch.ones(n, L, L), 2 * math.pi * torch.rand(n, L, L, generator=g))
    vals = torch.randperm(K, generator=g)[:n]
    pts = torch.zeros(n, L, L, dtype=torch.complex64)
    pts[torch.arange(n), cent[vals, 0], cent[vals, 1]] = 1
    H = (torch.fft.fft2(keys).conj() * torch.fft.fft2(pts)).sum(0)
    return keys, vals, H / H.abs().pow(2).sum().sqrt()


def recall(H_list, w, keys, mode):
    """鍵 → 窓の明るさ (n, K)。H_list の板を比率 w で融合して取り出す。"""
    Kf = torch.fft.fft2(keys)
    if mode == "inten":
        I = sum(wi * torch.fft.ifft2(Kf * H).abs().pow(2) for wi, H in zip(w, H_list))
    else:
        H = sum(wi * H for wi, H in zip(w, H_list))
        if mode == "phase":
            H = torch.polar(torch.ones_like(H.real), torch.angle(H))
        I = torch.fft.ifft2(Kf * H).abs().pow(2)
    return p2.read_windows(I, det)


def main():
    out = open("results/exp87.txt", "a", encoding="utf-8")
    for seed in range(3):
        for n in (8, 32, 128):
            g = torch.Generator().manual_seed(seed * 1000 + n)
            grains = [make_grain(n, g) for _ in range(3)]
            Hs = [gr[2] for gr in grains]
            cases = [((a, 1 - a), 2) for a in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)] + [((0.6, 0.3, 0.1), 3), ((0.5, 0.3, 0.2), 3), ((1 / 3, 1 / 3, 1 / 3), 3)]
            for mode in ("field", "phase", "inten"):
                for w, ng in cases:
                    # field で比率を明るさの比にしたいときは、振幅に √ をかける（明るさ = 振幅²）
                    wa = [math.sqrt(x) for x in w] if mode != "inten" else list(w)
                    rec = dict(exp="87", seed=seed, n=n, mode=mode, w=[round(x, 3) for x in w], grains=[])
                    peaks_all, src_all = [], []
                    for gi in range(ng):
                        keys, vals, _ = grains[gi]
                        S = recall(Hs[:ng], wa, keys, mode)
                        acc = 100.0 * (S.argmax(1) == vals).float().mean().item()
                        pk = S[torch.arange(n), vals]
                        bg = S.clone()
                        bg[torch.arange(n), vals] = float("nan")
                        rec["grains"].append(dict(acc=acc, peak=float(pk.mean()), peak_sd=float(pk.std()), bg=float(bg.nanmean())))
                        peaks_all.append(pk)
                        src_all.append(torch.full((n,), gi))
                    # ピークの高さだけから、どの Grain の記憶かを当てる（Grain ごとのピークの平均に一番近いもの）
                    P, Sg = torch.cat(peaks_all), torch.cat(src_all)
                    means = torch.tensor([gr["peak"] for gr in rec["grains"]])
                    rec["which_acc"] = 100.0 * ((P[:, None] - means[None]).abs().argmin(1) == Sg).float().mean().item()
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out.close()


if __name__ == "__main__":
    main()
