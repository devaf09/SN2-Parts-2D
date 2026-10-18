"""sn2_parts_2d の部品の確認（CPU で数秒）。実行: python -m pytest tests -q"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sn2_parts_2d as p2  # noqa: E402


def _rand_field(n, Ly, Lx, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(n, Ly, Lx, generator=g), torch.randn(n, Ly, Lx, generator=g))


def test_patch_index():
    pid = p2.patch_index(8, 12, 2, 3)
    assert pid.shape == (8, 12)
    assert pid.min() == 0 and pid.max() == 5
    assert pid[0, 0] == 0 and pid[0, 11] == 2 and pid[7, 0] == 3


def test_propagate_keeps_energy():
    """回折は光の量を変えない（届かない成分のない近軸の伝達関数で確かめる）。"""
    u = _rand_field(2, 32, 48)
    H = p2.transfer(32, 48, z=64.0, paraxial=True)
    v = p2.propagate(u, H)
    assert torch.allclose(p2.intensity(u).sum(), p2.intensity(v).sum(), rtol=1e-4)


def test_lens_keeps_energy():
    u = _rand_field(1, 16, 16)
    assert torch.allclose(p2.intensity(u).sum(), p2.intensity(p2.lens(u)).sum(), rtol=1e-4)


def test_transfer_sep_is_product():
    Hy, Hx = p2.transfer_sep(16, 24, z=30.0)
    H = p2.transfer(16, 24, z=30.0, paraxial=True)
    assert torch.allclose(Hy[:, None] * Hx[None, :], H, atol=1e-5)


def test_separable_path_equals_full_2d():
    """① の板 + 近軸の回折 + 掛け算の形の入力なら、行と列に分けた計算は 2 次元の計算と一致する。"""
    Ly, Lx, layers = 24, 32, 4
    g = torch.Generator().manual_seed(1)
    uy = torch.polar(torch.ones(Ly), 2 * math.pi * torch.rand(Ly, generator=g))[None]
    ux = torch.polar(torch.ones(Lx), 2 * math.pi * torch.rand(Lx, generator=g))[None]
    by, bx = p2.band_index(Ly, 3), p2.band_index(Lx, 4)
    y1, x1 = torch.arange(Ly, dtype=torch.float32), torch.arange(Lx, dtype=torch.float32)
    Hy, Hx = p2.transfer_sep(Ly, Lx)
    H = p2.transfer(Ly, Lx, paraxial=True)
    u = p2.outer(uy, ux)
    for _ in range(layers):
        prm = [torch.rand(4, generator=g), torch.rand(4, generator=g), 6 * torch.rand(4, generator=g), bx,
               torch.rand(3, generator=g), torch.rand(3, generator=g), 6 * torch.rand(3, generator=g), by, x1, y1]
        phy, phx = p2.plate_xy_parts(*prm)
        uy, ux = p2.normalize_sep(*p2.propagate_sep(uy * torch.polar(torch.ones_like(phy), phy),
                                                    ux * torch.polar(torch.ones_like(phx), phx), Hy, Hx))
        u = p2.normalize(p2.propagate(p2.apply_plate(u, p2.plate_xy(*prm)), H))
    assert torch.allclose(p2.outer(*p2.lens_sep(uy, ux)), p2.lens(u), atol=1e-4)


def test_ripple_far_center_becomes_stripes():
    """③ の中心を遠くへ置くと、② の斜めの縞になる（中心の向きに進む縞）。"""
    Ly = Lx = 32
    y, x = p2.grid(Ly, Lx)
    pid = p2.patch_index(Ly, Lx, 1, 1)
    R, ang, k = 1e6, 0.3, 0.7
    cx, cy = torch.tensor([-R * math.cos(ang)]), torch.tensor([-R * math.sin(ang)])
    one, zero = torch.ones(1), torch.zeros(1)
    phi = torch.tensor([-k * R])   # 中心からの距離 R の分の位相を打ち消す
    rip = p2.plate_ripple(one, torch.tensor([k]), phi, cx, cy, one, one, zero, pid, y.double(), x.double())
    stp = p2.plate_stripes(one, torch.tensor([k * math.cos(ang)]), torch.tensor([k * math.sin(ang)]), zero, pid, y, x)
    assert (rip.float() - stp).abs().max() < 1e-2


def test_window_grid():
    det = p2.window_grid(32, 32, 10, w=4)
    assert det.shape == (10, 32, 32)
    assert torch.allclose(det.sum((1, 2)), torch.ones(10))
    I = torch.ones(3, 32, 32)
    assert torch.allclose(p2.read_windows(I, det), torch.ones(3, 10))


def test_net2d_forward_all_kinds():
    """どの板の形でも、スコアの形がそろい、学習する数が同じ。"""
    X = torch.rand(3, 32, 32)
    n = None
    for k in p2.PLATE_KINDS:
        net = p2.Net2D(32, 5, [k] * 2, patches=4)
        out = net(p2.image_to_field(X, "phase"))
        assert out.shape == (3, 5) and torch.isfinite(out).all()
        n = n or net.n_params()
        if k == "sinc_e":   # 波紋 1 つ 7 個の数なので、ちょうどにはそろわない
            assert abs(net.n_params() - n) <= 0.15 * n
        else:
            assert net.n_params() == n


def test_illumination_cancels_cross_block_interference():
    """照明で平均すると、明るさは区画ごとに別々に通した明るさの和になる（違う区画どうしの干渉が消える）。"""
    torch.manual_seed(0)
    L, b = 16, 2
    for kind in ("tilt", "hadamard"):
        net = p2.Net2D(L, 3, ["stripes"] * 2, patches=2, illum=(b, kind))
        u = _rand_field(1, L, L)
        m = p2.patch_index(L, L, b, b)
        H = net.H

        def raw(v):   # 正規化なしの光路（線形）
            for d in range(2):
                v = p2.propagate(p2.apply_plate(v, net.plate_phase(d)), H)
            return p2.intensity(p2.lens(v))
        with torch.no_grad():
            avg = raw(u[:, None] * net.illum[None]).mean(1)
            sep = sum(raw(u * (m == k)) for k in range(b * b))
        assert torch.allclose(avg, sep, rtol=1e-3, atol=1e-4)


def test_sinc_ripple_center_and_decay():
    """④ 減衰する波紋: 中心で深さ A、外へ行くほど浅くなる。"""
    y, x = p2.grid(33, 33)
    ph = p2.plate_sinc(torch.tensor([2.0]), torch.tensor([0.8]), torch.tensor([16.0]), torch.tensor([16.0]), y, x)
    assert abs(ph[16, 16].item() - 2.0) < 1e-4
    assert ph[16, 32].abs() < 0.2 * ph[16, 16]
