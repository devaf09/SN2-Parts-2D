"""2 次元の合成課題（プログラムの中で作る。ダウンロードなし）。

どの課題も 64 × 64（L を変えれば比で拡大縮小）の明るさの画像 (N, L, L)（0〜1）とラベル (N,) を返す。
模様は中央付近に置き、周りは暗くする（周期境界で光が回り込むのを避ける）。

    shape   : 形 8 種（円・正方形・三角・十字・L 字・横棒・リング・X）。位置 ±6、大きさ 0.7〜1.3 倍、回転 ±15°
    orient  : 縞の向き 8 方向（0〜157.5°）。細かさ・位相・包む円の位置はでたらめ
    position: 小さな模様（形はでたらめ）が 3 × 3 のどのマスにあるか（9 クラス）
    conj    : 線の向き 4 種（横・縦・/・\）× 左右どちらの半分か（8 クラス）。「何が」と「どこに」の両方が要る
"""
from __future__ import annotations

import math

import torch

TASKS = {"shape": 8, "orient": 8, "position": 9, "conj": 8}


def _local(L, cx, cy, R, ang):
    """中心 (cx, cy)・大きさ R・回転 ang の局所座標 (u, v)。形 (N, L, L)。"""
    y = torch.arange(L, dtype=torch.float32)[None, :, None]
    x = torch.arange(L, dtype=torch.float32)[None, None, :]
    dx, dy = (x - cx[:, None, None]) / R[:, None, None], (y - cy[:, None, None]) / R[:, None, None]
    c, s = torch.cos(ang)[:, None, None], torch.sin(ang)[:, None, None]
    return c * dx + s * dy, -s * dx + c * dy


def _shape_mask(k, u, v):
    """形 k（0〜7）の中なら 1。u, v は局所座標（大きさ 1 が R 画素）。"""
    r = torch.sqrt(u ** 2 + v ** 2)
    if k == 0:   # 円
        m = r < 1
    elif k == 1:  # 正方形
        m = torch.maximum(u.abs(), v.abs()) < 0.8
    elif k == 2:  # 三角
        m = (v > -0.5) & (v < 1 - 1.732 * u.abs())
    elif k == 3:  # 十字
        m = ((u.abs() < 0.25) & (v.abs() < 1)) | ((v.abs() < 0.25) & (u.abs() < 1))
    elif k == 4:  # L 字
        m = ((u > -0.8) & (u < -0.4) & (v.abs() < 1)) | ((v > 0.6) & (v < 1.0) & (u > -0.8) & (u < 0.8))
    elif k == 5:  # 横棒
        m = (u.abs() < 1) & (v.abs() < 0.25)
    elif k == 6:  # リング
        m = (r > 0.6) & (r < 1)
    else:        # X（十字を 45° 回したもの）
        a, b = (u + v) / 1.4142, (u - v) / 1.4142
        m = ((a.abs() < 0.25) & (b.abs() < 1)) | ((b.abs() < 0.25) & (a.abs() < 1))
    return m.float()


def _draw_shapes(kinds, L, cx, cy, R, ang):
    u, v = _local(L, cx, cy, R, ang)
    img = torch.zeros(len(kinds), L, L)
    for k in range(8):
        sel = kinds == k
        if sel.any():
            img[sel] = _shape_mask(k, u[sel], v[sel])
    return img


def make(task: str, n: int, L: int = 64, seed: int = 0, noise: float = 0.1):
    """課題 task の画像 (n, L, L) とラベル (n,)。seed が同じなら同じデータ。"""
    g = torch.Generator().manual_seed(seed)
    U = lambda lo, hi: lo + (hi - lo) * torch.rand(n, generator=g)
    s = L / 64                                   # 64 × 64 を基準に比で拡大縮小
    y = torch.randint(TASKS[task], (n,), generator=g)
    c = L / 2
    if task == "shape":
        img = _draw_shapes(y, L, c + U(-6, 6) * s, c + U(-6, 6) * s, 7 * s * U(0.7, 1.3), U(-0.26, 0.26))
    elif task == "orient":
        th = y.float() * math.pi / 8
        k = U(0.4, 0.9) / s
        ph = U(0, 2 * math.pi)
        cx, cy = c + U(-6, 6) * s, c + U(-6, 6) * s
        yy = torch.arange(L, dtype=torch.float32)[None, :, None]
        xx = torch.arange(L, dtype=torch.float32)[None, None, :]
        wave = torch.cos(k[:, None, None] * (xx * torch.cos(th)[:, None, None] + yy * torch.sin(th)[:, None, None])
                         + ph[:, None, None])
        env = torch.exp(-((xx - cx[:, None, None]) ** 2 + (yy - cy[:, None, None]) ** 2) / (2 * (9 * s) ** 2))
        img = (0.5 + 0.5 * wave) * env
    elif task == "position":
        kinds = torch.randint(8, (n,), generator=g)
        col, row = y % 3, y // 3
        cx = c + (col.float() - 1) * 16 * s + U(-3, 3) * s
        cy = c + (row.float() - 1) * 16 * s + U(-3, 3) * s
        img = _draw_shapes(kinds, L, cx, cy, 4.5 * s * U(0.8, 1.2), U(-3.14, 3.14))
    elif task == "conj":
        ori, side = y % 4, y // 4
        ang = ori.float() * math.pi / 4 + U(-0.15, 0.15)
        cx = c + (side.float() * 2 - 1) * 12 * s + U(-4, 4) * s
        cy = c + U(-10, 10) * s
        img = _draw_shapes(torch.full((n,), 5), L, cx, cy, 7 * s * U(0.8, 1.2), ang)   # 横棒を回して線にする
    else:
        raise ValueError(f"task は {list(TASKS)} のどれか")
    img = img + noise * torch.randn(n, L, L, generator=g)
    return img.clamp(0, 1), y
