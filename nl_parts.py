"""光の非線形の部品（実験73〜、持ち主の依頼 2026-10-23: 実験28 の非線形をもっと掘る。今はソフトウェアの処理だけで考え、ハードの都合で幅を狭めない）。

板 ① を段に分け、段の間で光の場 u に非線形をかける。どの段も板の枚数は同じ（実験36 では段に分けると 1 段の板が減ったので、それを外して測る）。
非線形の候補（a = 出し直す光の振幅、I = |u|² を平均 1 にしたもの、φ = u の位相）。学習する数があるものは段ごとに持つ。
  none   : 非線形なし（段の境目で何もしない = 板を続けて並べたのと同じ）
  sqrt   : a = √I、位相は 0（実験28・36 と同じ「明るさを測って光を出し直す」）
  square : a = I、位相は 0（2 乗。明るい所をさらに強める。二次の高調波のような形）
  log    : a = log(1 + I)、位相は 0（明るさの差を縮める）
  power  : a = I^γ（γ を学習、0.1〜2）、位相は 0（sqrt と square の間を学習で選ぶ）
  sat    : a = √(I / (1 + I/s))（s を学習）、位相は 0（可飽和吸収: 明るすぎると頭打ち）
  thresh : a = √(softplus(k·(I − t)) / k)（t を学習、k = 8）、位相は 0（しきい値より暗い光を切る）
  kerr   : u' = u · exp(j·κ·I)（κ を学習）。明るさは変えず、明るい所ほど位相を遅らせる（光カー効果・自己位相変調）。位相の情報を捨てない唯一の候補
  ssat   : u' = u / √(1 + I/s)（s を学習）、位相はそのまま（可飽和吸収を、位相を残して）
  fold   : a = |I − t|（t を学習）、位相は 0。しきい値より暗い分を折り返して明るさに変える（持ち主の案「しきい値以下を反転」、V 字）
  signed : a = I − t（t を学習）。しきい値より暗い所は振幅が負 = 位相が π（反対向きの光）になる（持ち主の案「反転」のもう 1 つの読み方）
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

import sn2_parts_2d as p2

KINDS = ("none", "sqrt", "square", "log", "power", "sat", "thresh", "kerr", "ssat", "fold", "signed")


class NL(torch.nn.Module):
    """段の間の非線形 1 つ。"""

    def __init__(self, kind):
        super().__init__()
        assert kind in KINDS, kind
        self.kind = kind
        init = {"power": 0.5, "sat": 1.0, "thresh": 0.5, "kerr": 1.0, "ssat": 1.0, "fold": 0.5, "signed": 0.5}.get(kind, 0.0)
        self.p = torch.nn.Parameter(torch.tensor(float(init)), requires_grad=kind in ("power", "sat", "thresh", "kerr", "ssat", "fold", "signed"))

    def amplitude(self, I):
        """明るさ I（平均 1）→ 出し直す振幅（位相 0 の候補だけ）。図にも使う。"""
        k = self.kind
        if k == "sqrt":
            return I.clamp(min=0).sqrt()
        if k == "square":
            return I
        if k == "log":
            return torch.log1p(I)
        if k == "power":
            return I.clamp(min=1e-12) ** self.p.clamp(0.1, 2.0)
        if k == "sat":
            return (I / (1 + I / self.p.abs().clamp(min=1e-3))).clamp(min=0).sqrt()
        if k == "thresh":
            return (F.softplus(8.0 * (I - self.p)) / 8.0).sqrt()
        if k == "fold":
            return (I - self.p).abs()
        if k == "signed":
            return I - self.p
        raise ValueError(k)

    def forward(self, u):
        if self.kind == "none":
            return u
        I = p2.intensity(u)
        I = I / (I.mean((-2, -1), keepdim=True) + 1e-12)
        if self.kind == "kerr":
            return u * torch.polar(torch.ones_like(I), self.p * I)
        if self.kind == "ssat":
            return u / (1 + I / self.p.abs().clamp(min=1e-3)).sqrt()
        return p2.normalize(self.amplitude(I).to(torch.complex64))


class StagedNet(torch.nn.Module):
    """光の場 → [板 per 枚（回折しながら）→ 非線形] × stages（最後の段の後は非線形なし）→ レンズ → 窓 n_out 個。"""

    def __init__(self, L, n_out, stages=2, per=16, kind="sqrt", patches=8, seed=0):
        super().__init__()
        self.net = p2.Net2D(L, n_out, ["xy"] * (stages * per), patches=patches, seed=seed)
        self.stages, self.per = stages, per
        self.nls = torch.nn.ModuleList([NL(kind) for _ in range(stages - 1)])

    def screen(self, u, keep=False):
        """keep=True なら、段ごとの光（非線形の前と後）も返す（図のため）。"""
        net, inter = self.net, []
        d = 0
        for s in range(self.stages):
            for _ in range(self.per):
                u = p2.normalize(p2.propagate(p2.apply_plate(u, net.plate_phase(d)), net.H))
                d += 1
            if s < self.stages - 1:
                before = u
                u = self.nls[s](u)
                if keep:
                    inter.append((before, u))
        I = p2.intensity(p2.lens(u))
        I = I / (I.mean((-2, -1), keepdim=True) + 1e-12)
        return (I, inter) if keep else I

    def forward(self, u):
        return self.net.gain * p2.read_windows(self.screen(u), self.net.det) + self.net.bias
