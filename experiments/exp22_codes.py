"""実験22: 窓の数に縛られない読み方 — クラスを窓の明るさの模様（符号）で表す（Fashion-MNIST）

着想: 持ち主の案（2026-10-19）。「1 クラス = 1 窓」では、融合を続けると窓が足りなくなる。窓の数 K は固定のまま、
      クラスを K 個の窓の明るさの模様（符号）で表せば、クラスは実質無制限に増やせる。
        1. 模様で判断: 各クラスに符号を割り当て、一番似た符号のクラスを答えにする
        2. 共通の模様から想定: 例の平均の模様（典型）を符号にして、学習なしでクラスを足す
        3. 部品の組み合わせ: 符号 = 部品の符号の和（上半身 + 長袖 + 前開き → コート）。例を見せないクラスを部品だけで定義する
方法: 板 ① 16 枚、学習率 0.01 → 0（cosine）、3000 ステップ。窓の明るさ w（K 個）から平均を引いて長さ 1 にし、
      符号 c（長さ 1）との内積（似ている度合い）× 温度 をクラスの点数にする。
  A onehot   : 1 クラス 1 窓（K = 10、いままでの方式）
    random K   : でたらめな ±1 の符号（K = 16・32）
    learned 16 : 符号も学習する（K = 16）
  B add      : 8 クラスででたらめな符号（K = 32）で学習したあと、残り 2 クラス（コート・ブーツ）を、板を動かさず
               例 20 枚の平均の模様を符号にして足す（学習なし）。10 クラスで答える
  C parts    : 部品の符号の和を符号にして 9 クラスで学習（コートは見せない）。コート = 上半身 + 長袖 + 前開き と定義だけして 10 クラスで答える
  D merge    : 技能 A（0〜4）と B（5〜9）を、同じ 32 個の窓・でたらめな符号で別々に学習し（担当なしの絵も見せる）、明るさを足して 10 クラス
比べるもの: A の onehot、同時に学習。
結果: results/exp22.txt

実行: python experiments/exp22_codes.py --seed 0 --device cuda
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sn2_parts_2d as p2  # noqa: E402
from exp15_mnist import load  # noqa: E402

# Fashion-MNIST の部品（人が決めた特徴）。0 T シャツ 1 ズボン 2 プルオーバー 3 ドレス 4 コート 5 サンダル 6 シャツ 7 スニーカー 8 バッグ 9 ブーツ
PARTS = ["上半身", "長袖", "前開き", "下半身", "全身", "足", "運動靴", "丈が高い", "持ち物"]
CLASS_PARTS = {0: [0], 1: [3], 2: [0, 1], 3: [4], 4: [0, 1, 2], 5: [5], 6: [0, 2], 7: [5, 6], 8: [8], 9: [5, 7]}


class CodeNet(torch.nn.Module):
    """Net2D の窓の明るさ（K 個）を、符号との似ている度合いでクラスの点数にする。"""

    def __init__(self, codes, K, seed, learn_codes=False):
        super().__init__()
        self.net = p2.Net2D(64, K, ["xy"] * 16, seed=seed)
        self.codes = torch.nn.Parameter(codes.clone(), requires_grad=learn_codes)
        self.temp = torch.nn.Parameter(torch.tensor(10.0))
        self.net.gain.requires_grad_(False)
        self.net.bias.requires_grad_(False)

    def windows(self, u):
        w = p2.read_windows(self.net.screen(u), self.net.det)
        w = w - w.mean(1, keepdim=True)
        return w / (w.norm(dim=1, keepdim=True) + 1e-9)

    def forward(self, u):
        c = self.codes - self.codes.mean(1, keepdim=True)
        c = c / (c.norm(dim=1, keepdim=True) + 1e-9)
        return self.temp * self.windows(u) @ c.T


def rand_codes(n, K, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 2, (n, K), generator=g) * 2 - 1).float()


def train(model, X, y, steps, seed, reject=None, w=0.3):
    dev = next(model.parameters()).device
    f = lambda v: p2.image_to_field(v.to(dev), "amp")
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        i = torch.randint(len(X), (256,), generator=g)
        loss = F.cross_entropy(model(f(X[i])), y[i].to(dev))
        if reject is not None:
            j = torch.randint(len(reject), (128,), generator=g)
            loss = loss - w * F.log_softmax(model(f(reject[j])), 1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
    return model


@torch.no_grad()
def scores(model, X, batch=1000):
    dev = next(model.parameters()).device
    return torch.cat([model(p2.image_to_field(X[i:i + batch].to(dev), "amp")).cpu() for i in range(0, len(X), batch)])


@torch.no_grad()
def windows(model, X, batch=1000):
    dev = next(model.parameters()).device
    return torch.cat([model.windows(p2.image_to_field(X[i:i + batch].to(dev), "amp")).cpu() for i in range(0, len(X), batch)])


acc = lambda s, y: 100.0 * (s.argmax(1) == y).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--parts", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--holdout", type=int, default=4, help="C で例を見せないクラス（部品だけで定義する。実験24）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp22.txt")
    a = ap.parse_args()
    dev = torch.device(a.device)
    Xtr, ytr, Xte, yte = load("fashion", 64)
    res, t0 = {}, time.perf_counter()
    if "A" in a.parts:   # 読み方の比べ合い（10 クラス）
        for name, K, codes, learn in [("onehot", 10, torch.eye(10) * 2 - 1, False), ("random16", 16, rand_codes(10, 16, a.seed), False),
                                      ("random32", 32, rand_codes(10, 32, a.seed), False), ("learned16", 16, rand_codes(10, 16, a.seed), True)]:
            m = train(CodeNet(codes, K, a.seed, learn).to(dev), Xtr, ytr, a.steps, a.seed)
            res[f"A_{name}"] = acc(scores(m, Xte), yte)
    if "B" in a.parts:   # 学習なしでクラスを足す（典型の模様）
        new = [4, 9]
        old = [c for c in range(10) if c not in new]
        codes = rand_codes(10, 32, a.seed)
        mtr = torch.isin(ytr, torch.tensor(old))
        m = train(CodeNet(codes, 32, a.seed).to(dev), Xtr[mtr], ytr[mtr], a.steps, a.seed)
        res["B_old8"] = acc(scores(m, Xte[torch.isin(yte, torch.tensor(old))]), yte[torch.isin(yte, torch.tensor(old))])
        for n in (5, 20, 100):
            g = torch.Generator().manual_seed(a.seed)
            with torch.no_grad():
                for c in new:
                    idx = (ytr == c).nonzero()[:, 0]
                    idx = idx[torch.randperm(len(idx), generator=g)[:n]]
                    m.codes[c] = windows(m, Xtr[idx]).mean(0).to(dev)
            s = scores(m, Xte)
            res[f"B_add{n}_all10"] = acc(s, yte)
            tn = torch.isin(yte, torch.tensor(new))
            res[f"B_add{n}_new"] = acc(s[tn], yte[tn])
        # 古いクラスの符号も、同じく例の平均の模様に置き換える（全部のクラスを同じ作り方にそろえる。実験23）
        for n in (20, 100):
            g = torch.Generator().manual_seed(a.seed + 7)
            with torch.no_grad():
                for c in range(10):
                    idx = (ytr == c).nonzero()[:, 0]
                    idx = idx[torch.randperm(len(idx), generator=g)[:n]]
                    m.codes[c] = windows(m, Xtr[idx]).mean(0).to(dev)
            s = scores(m, Xte)
            res[f"B_allproto{n}_all10"] = acc(s, yte)
            tn = torch.isin(yte, torch.tensor(new))
            res[f"B_allproto{n}_new"] = acc(s[tn], yte[tn])
    if "C" in a.parts:   # 部品の符号の和（コートは見せない）
        pc = rand_codes(len(PARTS), 32, a.seed + 1)
        codes = torch.stack([pc[CLASS_PARTS[c]].sum(0) for c in range(10)])
        ho = a.holdout
        seen = [c for c in range(10) if c != ho]
        mtr = torch.isin(ytr, torch.tensor(seen))
        m = train(CodeNet(codes, 32, a.seed).to(dev), Xtr[mtr], ytr[mtr], a.steps, a.seed)
        s = scores(m, Xte)
        res["C_all10"] = acc(s, yte)
        res["C_coat_zero_shot"] = acc(s[yte == ho], yte[yte == ho])     # 名前は実験22 のまま（例を見せないクラスの正解率）
        res["C_seen9"] = acc(s[yte != ho][:, seen], torch.tensor([seen.index(v) for v in yte[yte != ho].tolist()]))
        res["C_unseen_pred"] = torch.bincount(s[yte == ho].argmax(1), minlength=10).tolist()   # 例を見せないクラスを何と答えたか
    if "D" in a.parts:   # 同じ 32 個の窓を共有して融合
        codes = rand_codes(10, 32, a.seed)
        sk = []
        for i, cls in enumerate([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]):
            mtr = torch.isin(ytr, torch.tensor(cls))
            G = p2.generic_images(4000, 64, seed=5000 + 100 * i + a.seed)
            sk.append(train(CodeNet(codes, 32, a.seed + 100 * i).to(dev), Xtr[mtr], ytr[mtr], a.steps, a.seed + i, reject=G))
        with torch.no_grad():   # 融合: 窓の明るさの模様を足してから符号と照らし合わせる
            w = sum(windows(m, Xte) for m in sk)
            w = w / (w.norm(dim=1, keepdim=True) + 1e-9)
            c = codes - codes.mean(1, keepdim=True)
            c = c / c.norm(dim=1, keepdim=True)
        res["D_merged_windows"] = acc(w @ c.T, yte)
        res["D_merged_scores"] = acc(sum(scores(m, Xte) for m in sk), yte)
    rec = dict(exp="22", **vars(a), **res, total_s=time.perf_counter() - t0, version=p2.__version__, torch=torch.__version__)
    print(json.dumps(rec, ensure_ascii=False))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
