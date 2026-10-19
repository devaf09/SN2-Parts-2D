"""実験27: どこに何があるかを出す — 光学相関器（4f 系）で、数字の位置に光を集める（MNIST の 2 個並べ）

着想: 持ち主の案（2026-10-19）。犬の光路と猫の光路を融合したとき、「このピクセルからこのピクセルが猫、ここが犬」と出せるのが一番良い。
      確率で「犬 48%、猫 72%」のように、クラスごとに独立した答えでもよい。
      光学では、レンズ → 周波数の面に模様の板 → もう一度レンズ、という 4f 系（VanderLugt の相関器）で、
      出力の面の「探す物がある位置」に明るい点が出る。周波数の面の板は、絵のどこに物があっても同じ働きをする（位置をずらしても同じ）。
      いまの SinWave は最後のレンズで位置を捨てているが、位置を残したまま読めば「どこに」が分かるはず。
方法: 64 × 64 の場に、20 × 20 に縮めた MNIST の数字を 1〜2 個、重ならないようランダムな位置に置く（3・7・ほかの数字が 1/3 ずつ）。
      技能「3」の光路と技能「7」の光路を別々に学習する。出力 = 出力の面の明るさの地図。目標 = その数字の中心に置いた丸い山（半径 3 画素）。
      学習: 地図の各画素を「その数字の中心か」の確率として（温度・背景光を学習）、二値のクロスエントロピー（山の画素に重み）。学習率 0.01 → 0、3000 ステップ。
  fourier  : 4f 系。周波数の面に位相だけの板 1 枚（64 × 64 画素すべてを学習、4096 個）
  fourier4 : 4f 系を 4 段（周波数の面の板 → 逆のレンズ → 回折 → …）。学習する数 16384 個
  plates   : いまの板 ① 16 枚（区画 8 × 8）を回折しながら通し、最後のレンズを使わずに出力の面で読む（学習する数 2057）
指標: ある・なし（地図の最大の確率 > 0.5）の正解率、位置（その数字の中心から 4 画素以内に最大の点があるか）、2 個並んだときにそれぞれ光るか。
      融合: 2 本の光路を別の色として 1 枚の出力に重ねる（3 は赤・7 は緑）。図は tools/fig_exp27.py。
結果: results/exp27.txt。seed 0 の光路は results/models/ に保存。

実行: python experiments/exp27_locate.py --method fourier --seed 0 --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sn2_parts_2d as p2  # noqa: E402
from exp15_mnist import ROOT  # noqa: E402

L, S = 64, 20          # 場の大きさ、数字の大きさ


def make_scenes(n, seed, train=True):
    """絵 (n, L, L)、数字の中心と種類のリスト。"""
    from torchvision import datasets
    d = datasets.MNIST(ROOT, train=train, download=False)
    dig = F.interpolate(d.data[:, None].float() / 255.0, size=(S, S), mode="bilinear", align_corners=False)[:, 0]
    lab = d.targets
    g = torch.Generator().manual_seed(seed)
    idx3, idx7 = (lab == 3).nonzero()[:, 0], (lab == 7).nonzero()[:, 0]
    idxo = ((lab != 3) & (lab != 7)).nonzero()[:, 0]
    X = torch.zeros(n, L, L)
    objs = []
    for i in range(n):
        k = 1 + int(torch.rand(1, generator=g) < 0.5)
        cs, o, tries = [], [], 0
        while len(cs) < k:
            cy, cx = [int(v) for v in torch.randint(S // 2 + 2, L - S // 2 - 2, (2,), generator=g)]
            if all(max(abs(cy - a), abs(cx - b)) >= S + 1 for a, b in cs):
                cs.append((cy, cx))
            tries += 1
            if tries > 50:              # 1 個目が真ん中だと 2 個目を置けないので、置き直す
                cs, tries = [], 0
        for cy, cx in cs:
            r = int(torch.randint(3, (1,), generator=g))
            pool = [idx3, idx7, idxo][r]
            j = pool[int(torch.randint(len(pool), (1,), generator=g))]
            X[i, cy - S // 2:cy + S // 2, cx - S // 2:cx + S // 2] = torch.maximum(X[i, cy - S // 2:cy + S // 2, cx - S // 2:cx + S // 2], dig[j])
            o.append((cy, cx, int(lab[j])))
        objs.append(o)
    return X, objs


def target_maps(objs, digit):
    yy, xx = torch.meshgrid(torch.arange(L), torch.arange(L), indexing="ij")
    T = torch.zeros(len(objs), L, L)
    for i, o in enumerate(objs):
        for cy, cx, d in o:
            if d == digit:
                T[i] = torch.maximum(T[i], ((yy - cy) ** 2 + (xx - cx) ** 2 <= 9).float())
    return T


class Fourier4f(torch.nn.Module):
    """4f 系: レンズ（フーリエ変換）→ 周波数の面の位相の板 → 逆のレンズ。stages 段重ね、段の間は回折（距離 64）。"""

    def __init__(self, stages=1, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.phase = torch.nn.Parameter(0.1 * torch.randn(stages, L, L, generator=g))
        self.register_buffer("H", p2.transfer(L, L, 64.0))
        self.gain = torch.nn.Parameter(torch.tensor(1.0))
        self.bias = torch.nn.Parameter(torch.tensor(-4.0))

    def screen(self, u):
        for s in range(self.phase.shape[0]):
            if s > 0:
                u = p2.propagate(u, self.H)
            U = torch.fft.fft2(u, norm="ortho") * torch.polar(torch.ones_like(self.phase[s]), self.phase[s])
            u = p2.normalize(torch.fft.ifft2(U, norm="ortho"))
        return p2.intensity(u)

    def forward(self, u):
        I = self.screen(u)
        return self.gain * I / (I.mean((-2, -1), keepdim=True) + 1e-12) + self.bias


class Fourier4fNL(torch.nn.Module):
    """4f 系を 2 段、間に「明るさを測って、その明るさで光を出し直す」段（カメラと表示器のつなぎ = 光の非線形）を入れる（実験28）。
    1 段目は数字の形の部品に光を集め、2 段目がその組み合わせを見る。線形の 4f 系を何段重ねても 1 枚の板と同じなので、間に非線形が要る。"""

    def __init__(self, seed=0):
        super().__init__()
        self.s1 = Fourier4f(1, seed)
        self.s2 = Fourier4f(1, seed + 1)
        self.gain, self.bias = self.s2.gain, self.s2.bias

    def forward(self, u):
        I1 = self.s1.screen(u)
        I1 = I1 / (I1.mean((-2, -1), keepdim=True) + 1e-12)
        return self.s2(torch.sqrt(I1 + 1e-12).to(torch.complex64))


class PlatesNoLens(torch.nn.Module):
    """板 ① 16 枚を回折しながら通し、最後のレンズを使わずに出力の面（位置）で読む。"""

    def __init__(self, seed=0):
        super().__init__()
        self.net = p2.Net2D(L, 2, ["xy"] * 16, seed=seed)
        self.gain = torch.nn.Parameter(torch.tensor(1.0))
        self.bias = torch.nn.Parameter(torch.tensor(-4.0))

    def screen(self, u):
        for d in range(16):
            u = p2.normalize(p2.propagate(p2.apply_plate(u, self.net.plate_phase(d)), self.net.H))
        return p2.intensity(u)

    def forward(self, u):
        I = self.screen(u)
        return self.gain * I / (I.mean((-2, -1), keepdim=True) + 1e-12) + self.bias


def n_params(m):
    return sum(p.numel() for n, p in m.named_parameters() if p.requires_grad and not n.startswith("net.gain") and not n.startswith("net.bias"))


@torch.no_grad()
def evaluate(model, X, objs, digit, batch=500):
    dev = next(model.parameters()).device
    Z = torch.cat([model(p2.image_to_field(X[i:i + batch].to(dev), "amp")).cpu() for i in range(0, len(X), batch)])
    P = torch.sigmoid(Z)
    pres_true = torch.tensor([any(d == digit for _, _, d in o) for o in objs])
    pmax = P.flatten(1).max(1).values
    pres_acc = 100.0 * ((pmax > 0.5) == pres_true).float().mean().item()
    hit, tot = 0, 0                       # 目標の数字ごとに、中心から 4 画素以内の確率の最大が 0.5 を超えるか
    for i, o in enumerate(objs):
        for cy, cx, d in o:
            if d == digit:
                tot += 1
                hit += int(P[i, max(cy - 4, 0):cy + 5, max(cx - 4, 0):cx + 5].max() > 0.5)
    fp, nfp = 0, 0                        # ほかの数字の位置で誤って光る割合
    for i, o in enumerate(objs):
        for cy, cx, d in o:
            if d != digit:
                nfp += 1
                fp += int(P[i, max(cy - 4, 0):cy + 5, max(cx - 4, 0):cx + 5].max() > 0.5)
    # 目標の数字の位置と、ほかの数字の位置での確率の最大を比べる（目標の方が高い組の割合 = AUC。閾値によらない見分けの強さ）
    pos, neg = [], []
    for i, o in enumerate(objs):
        for cy, cx, d in o:
            v = float(P[i, max(cy - 4, 0):cy + 5, max(cx - 4, 0):cx + 5].max())
            (pos if d == digit else neg).append(v)
    pt, nt = torch.tensor(pos), torch.tensor(neg)
    auc = float(((pt[:, None] > nt[None]).float() + 0.5 * (pt[:, None] == nt[None]).float()).mean()) if len(pos) and len(neg) else float("nan")
    return dict(presence=pres_acc, located=100.0 * hit / max(tot, 1), false_alarm=100.0 * fp / max(nfp, 1), auc=100.0 * auc), P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="fourier", choices=["fourier", "fourier4", "fourier2nl", "plates"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--pos_weight", type=float, default=20.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp27.txt")
    a = ap.parse_args()
    dev = torch.device(a.device)
    Xtr, otr = make_scenes(a.n_train, 100 + a.seed, True)
    Xte, ote = make_scenes(a.n_test, 200 + a.seed, False)
    res, t0 = {}, time.perf_counter()
    for digit in (3, 7):
        T = target_maps(otr, digit)
        m = {"fourier": lambda: Fourier4f(1, a.seed + digit), "fourier4": lambda: Fourier4f(4, a.seed + digit),
             "fourier2nl": lambda: Fourier4fNL(a.seed + digit),
             "plates": lambda: PlatesNoLens(a.seed + digit)}[a.method]().to(dev)
        opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=0.01)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
        g = torch.Generator().manual_seed(a.seed)
        pw = torch.tensor(a.pos_weight, device=dev)
        for _ in range(a.steps):
            i = torch.randint(len(Xtr), (64,), generator=g)
            z = m(p2.image_to_field(Xtr[i].to(dev), "amp"))
            loss = F.binary_cross_entropy_with_logits(z, T[i].to(dev), pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
        r, _ = evaluate(m, Xte, ote, digit)
        for k, v in r.items():
            res[f"d{digit}_{k}"] = v
        res[f"d{digit}_params"] = n_params(m)
        if a.seed == 0:
            os.makedirs("results/models", exist_ok=True)
            torch.save({"state": m.state_dict(), "args": vars(a), "digit": digit}, f"results/models/exp27_{a.method}_d{digit}_s0.pt")
    rec = dict(exp="27", **vars(a), **res, total_s=time.perf_counter() - t0, version=p2.__version__, torch=torch.__version__)
    print(json.dumps(rec, ensure_ascii=False))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
