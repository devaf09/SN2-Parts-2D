"""実験00: 場の大きさごとの計算時間とメモリ（学習 1 ステップ = 順伝播 + 逆伝播）

着想: 持ち主の指示「2 次元は計算とメモリが重いので、大きさ（64×64 → 128×128 → 256×256）ごとの時間とメモリを最初に測る」。
方法: 板 16 枚の光路を、板の形 ①②③ と場の大きさごとに 1 ステップ学習したときの時間と最大メモリを測る。
  ① xy_sep : 交点で掛け算の板、入力も掛け算の形、近軸の回折 → 行と列の 1 次元 FFT だけで計算（軽い計算）
  ① xy_full: 同じ板を 2 次元の場として計算（入力が掛け算の形でないとき）
  ② stripes: 斜めの縞、区画 8 × 8
  ③ ripple : 波紋（等間隔の輪）、区画 8 × 8
比べるもの: 1 次元の SinWave（L=1024、板 16 枚）の同じ計測（Sin-CNN2 の最小版の構成を部品で再現したもの）。
結果: results/exp00.txt（1 回の計測 = 1 行の JSON）。

実行: python experiments/exp00_cost.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sn2_parts_2d as p2  # noqa: E402


def make_params(kind, L, layers, P, device, g):
    """板の形ごとの学習する数（A, φ）と固定の数（ω・中心など）。"""
    r = lambda *s: torch.rand(*s, generator=g).to(device)
    if kind.startswith("xy"):
        prm = dict(Ax=r(layers, P), phix=6.28 * r(layers, P), Ay=r(layers, P), phiy=6.28 * r(layers, P))
        fix = dict(wx=3 * r(layers, P), wy=3 * r(layers, P))
    elif kind == "stripes":
        prm = dict(A=r(layers, P * P), phi=6.28 * r(layers, P * P))
        fix = dict(wx=3 * r(layers, P * P) - 1.5, wy=3 * r(layers, P * P) - 1.5)
    elif kind == "ripple":
        prm = dict(A=r(layers, P * P), phi=6.28 * r(layers, P * P))
        fix = dict(k=2 * r(layers, P * P), cx=L * (3 * r(layers, P * P) - 1), cy=L * (3 * r(layers, P * P) - 1),
                   a=1 + r(layers, P * P), b=1 + r(layers, P * P), theta=3.14 * r(layers, P * P))
    else:  # 1 次元
        prm = dict(A=r(layers, P), phi=6.28 * r(layers, P))
        fix = dict(w=3 * r(layers, P))
    for v in prm.values():
        v.requires_grad_(True)
    return prm, fix


def make_input(kind, L, B, device):
    """入力の光（位相がでたらめな場）。計測の外で 1 回だけ作る。"""
    g = torch.Generator().manual_seed(1)
    ph = lambda *s: torch.polar(torch.ones(*s), 6.28 * torch.rand(*s, generator=g)).to(device)
    if kind == "1d":
        return ph(B, L)
    if kind == "xy_sep":
        return ph(B, L), ph(B, L)
    return ph(B, L, L)


def step(kind, L, u_in, layers, P, prm, fix, device, z=64.0):
    """1 ステップ: 入力 → [板 → 回折 → 正規化] × layers → レンズ → 窓 → 損失 → 逆伝播。"""
    if kind == "1d":
        u = u_in
        t = torch.arange(L, dtype=torch.float32, device=device)
        pidx = p2.band_index(L, P, device)
        kx = 2 * math.pi * torch.fft.fftfreq(L, device=device)
        H = torch.polar(torch.ones(L, device=device), torch.sqrt((2 * math.pi) ** 2 - kx ** 2) * z)
        for d in range(layers):
            ph = prm["A"][d, pidx] * torch.sin(fix["w"][d, pidx] * t + prm["phi"][d, pidx])
            u = u * torch.polar(torch.ones_like(ph), ph)
            u = torch.fft.ifft(torch.fft.fft(u) * H)
            u = u / (u.abs().pow(2).mean(-1, keepdim=True) + 1e-24).sqrt()
        I = torch.fft.fft(u, norm="ortho").abs().pow(2)
        loss = I[:, :16].mean()
    elif kind == "xy_sep":
        uy, ux = u_in
        b = p2.band_index(L, P, device)
        c = torch.arange(L, dtype=torch.float32, device=device)
        Hy, Hx = p2.transfer_sep(L, L, z, device=device)
        for d in range(layers):
            phy, phx = p2.plate_xy_parts(prm["Ax"][d], fix["wx"][d], prm["phix"][d], b,
                                         prm["Ay"][d], fix["wy"][d], prm["phiy"][d], b, c, c)
            uy, ux = uy * torch.polar(torch.ones_like(phy), phy), ux * torch.polar(torch.ones_like(phx), phx)
            uy, ux = p2.normalize_sep(*p2.propagate_sep(uy, ux, Hy, Hx))
        Uy, Ux = p2.lens_sep(uy, ux)
        loss = (Uy.abs().pow(2)[:, :16].mean(-1) * Ux.abs().pow(2)[:, :16].mean(-1)).mean()   # 窓 = 縦と横の帯の積
    else:
        u = u_in
        y, x = p2.grid(L, L, device)
        H = p2.transfer(L, L, z, paraxial=(kind == "xy_full"), device=device)
        if kind == "xy_full":
            b = p2.band_index(L, P, device)
            c = torch.arange(L, dtype=torch.float32, device=device)
        else:
            pid = p2.patch_index(L, L, P, P, device)
        for d in range(layers):
            if kind == "xy_full":
                ph = p2.plate_xy(prm["Ax"][d], fix["wx"][d], prm["phix"][d], b,
                                 prm["Ay"][d], fix["wy"][d], prm["phiy"][d], b, c, c)
            elif kind == "stripes":
                ph = p2.plate_stripes(prm["A"][d], fix["wx"][d], fix["wy"][d], prm["phi"][d], pid, y, x)
            else:
                ph = p2.plate_ripple(prm["A"][d], fix["k"][d], prm["phi"][d], fix["cx"][d], fix["cy"][d],
                                     fix["a"][d], fix["b"][d], fix["theta"][d], pid, y, x)
            u = p2.normalize(p2.propagate(p2.apply_plate(u, ph), H))
        I = p2.intensity(p2.lens(u))
        loss = I[:, :4, :4].mean()
    loss.backward()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--kinds", nargs="+", default=["1d", "xy_sep", "xy_full", "stripes", "ripple"])
    ap.add_argument("--batch", type=int, default=64)  # 1 ステップに入れる例の数
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--patches", type=int, default=8, help="2 次元は縦横 P × P の区画、① は X・Y それぞれ P 本の帯")
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp00.txt")
    a = ap.parse_args()
    dev = torch.device(a.device)
    gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"
    for kind in a.kinds:
        sizes = [1024] if kind == "1d" else a.sizes
        for L in sizes:
            P = 64 if kind == "1d" else a.patches
            g = torch.Generator().manual_seed(0)
            prm, fix = make_params(kind, L, a.layers, P, dev, g)
            n_params = sum(v.numel() for v in prm.values())
            try:
                u_in = make_input(kind, L, a.batch, dev)
                step(kind, L, u_in, a.layers, P, prm, fix, dev)   # 準備運転
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                for _ in range(a.repeat):
                    step(kind, L, u_in, a.layers, P, prm, fix, dev)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) / a.repeat * 1000
                mem = torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else None
                err = None
            except torch.OutOfMemoryError:
                ms, mem, err = None, None, "out of memory"
                torch.cuda.empty_cache()
            rec = dict(exp="00", kind=kind, L=L, batch=a.batch, layers=a.layers, patches=P, params=n_params,
                       ms_per_step=ms, ms_per_sample=None if ms is None else ms / a.batch, peak_MB=mem,
                       error=err, device=gpu, torch=torch.__version__, repeat=a.repeat)
            print(json.dumps(rec, ensure_ascii=False))
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            with open(a.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            del prm, fix
            if dev.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
