"""実験73: 光の非線形をもっと掘る（持ち主の依頼、2026-10-23）

着想: 実験28 で、4f 系 2 段の間に光の非線形（明るさの平方根を振幅として出し直す）を 1 つ入れると、見分けの強さが AUC 89 / 82 → 97 / 97 になった。
      一方、言語（実験36）では −1〜2 点だったが、段に分けると 1 段の板が減っていた（切り分けていない）。言語の弱点（組み合わせを表せない、
      語どうしが混ざる）は、ふつうのニューラルネットでは非線形が担う部分。持ち主の方針: 今はソフトウェアの処理だけ。ハードの都合で幅を狭めない。
方法: nl_parts.StagedNet。板 ① を stages 段 × 1 段 per 枚に分け、段の間に非線形（nl_parts.KINDS の 9 通り）。1 段の板の数は減らさない。
  課題:
    mnist     : 手書き数字（28 × 28 を 64 × 64 の真ん中に振幅で）、10 クラス
    char      : 文字の課題（直前 8 文字 → 次の文字 64 クラス、帯に並べ位置ごとの明るさを学習、訓練は全部）
    charsplit : 文字の課題を分担の 2DTG（直前の文字の余りで 16 個、担当の決め打ち、各 2DTG が StagedNet、直前 4 文字、窓 = 65）
  比べる: 非線形なし（none）の同じ枚数。
条件: 3000 ステップ、学習率 0.01 → 0。seed 0（良いものは seed を足して確かめる）。
結果: results/exp73.txt。--save で光路を results/models/ に保存（図のため）。
実行: python experiments/exp73_nl.py --task mnist --kind sqrt --stages 2 --per 16 --seed 0 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "image"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "text"))
import nl_parts as nl  # noqa: E402
import sn2_parts_2d as p2  # noqa: E402


class TextNL(torch.nn.Module):
    """文字を帯に並べて入れる（実験33・40 の EncNet と同じ入れ方）→ StagedNet。"""

    def __init__(self, n, V, C, stages, per, kind, patches, seed):
        super().__init__()
        from exp40_inputs import EncNet
        self.enc = EncNet("bands", n, V, C, 1, 8, seed)        # 入れ方と位置ごとの明るさだけを使う（中の板は使わない）
        self.body = nl.StagedNet(64, C, stages, per, kind, patches, seed)

    def field(self, X):
        return self.enc.field(X)

    def forward(self, X):
        return self.body(self.field(X))

    def opt_params(self):
        return [self.enc.w] + list(self.body.parameters())


def train(m, params, X, y, steps, dev, seed, to_input=lambda x: x):
    opt = torch.optim.Adam([p for p in params if p.requires_grad], lr=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        i = torch.randint(len(y), (256,), generator=g)
        loss = F.cross_entropy(m(to_input(X[i].to(dev))), y[i].to(dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
    return m.eval()


def evaluate(m, Xt, yt, dev, to_input=lambda x: x, bs=250):
    with torch.no_grad():
        S = torch.cat([m(to_input(Xt[j:j + bs].to(dev))).cpu() for j in range(0, len(yt), bs)])
    return 100.0 * (S.argmax(1) == yt).float().mean().item(), F.cross_entropy(S, yt).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="mnist", choices=["mnist", "char", "charsplit"])
    ap.add_argument("--kind", default="sqrt", choices=nl.KINDS)
    ap.add_argument("--stages", type=int, default=2)
    ap.add_argument("--per", type=int, default=16)
    ap.add_argument("--patches", type=int, default=8)
    ap.add_argument("--G", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp73.txt")
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    t0 = time.perf_counter()
    extra = {}
    if a.task == "mnist":
        from exp15_mnist import load
        X, y, Xt, yt = load("mnist")
        to_in = lambda x: p2.image_to_field(x, "amp")
        m = nl.StagedNet(64, 10, a.stages, a.per, a.kind, a.patches, a.seed).to(dev)
        train(m, m.parameters(), X, y, a.steps, dev, a.seed, to_in)
        acc, ce = evaluate(m, Xt, yt, dev, to_in)
        tr_acc, _ = evaluate(m, X[:10000], y[:10000], dev, to_in)
        params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    else:
        import textdata
        n = 8 if a.task == "char" else 4
        (X, y), (Xt, yt), info = textdata.load(8, 64, 600000, 10000, a.seed)
        V, C = info["V_in"], info["C"]
        if a.task == "char":
            m = TextNL(n, V, C, a.stages, a.per, a.kind, a.patches, a.seed).to(dev)
            train(m, m.opt_params(), X, y, a.steps, dev, a.seed)
            acc, ce = evaluate(m, Xt, yt, dev)
            tr_acc, _ = evaluate(m, X[:10000], y[:10000], dev)
            params = sum(p.numel() for p in m.opt_params() if p.requires_grad)
        else:   # 分担の 2DTG（直前の文字の余りで担当を決め打ち）
            Xn, Xtn = X[:, -n:], Xt[:, -n:]
            grp, grpt = Xn[:, -1] % a.G, Xtn[:, -1] % a.G
            S = torch.zeros(len(yt), C)
            ms, params, tr_hit, tr_n = [], 0, 0, 0
            for gi in range(a.G):
                own, ownt = (grp == gi).nonzero().squeeze(1), (grpt == gi).nonzero().squeeze(1)
                torch.manual_seed(a.seed * 100 + gi)
                m = TextNL(n, V, C, a.stages, a.per, a.kind, a.patches, a.seed * 100 + gi).to(dev)
                train(m, m.opt_params(), Xn[own], y[own], a.steps, dev, a.seed * 100 + gi)
                with torch.no_grad():
                    if len(ownt):
                        S[ownt] = torch.cat([m(Xtn[ownt][j:j + 250].to(dev)).cpu() for j in range(0, len(ownt), 250)])
                    o = own[:1000]
                    tr_hit += int((m(Xn[o].to(dev)).cpu().argmax(1) == y[o]).sum())
                    tr_n += len(o)
                params += sum(p.numel() for p in m.opt_params() if p.requires_grad)
                ms.append(m)
            acc, ce = 100.0 * (S.argmax(1) == yt).float().mean().item(), F.cross_entropy(S, yt).item()
            tr_acc = 100.0 * tr_hit / max(1, tr_n)
            m = ms[0]
    if hasattr(m, "body"):
        extra["nl_params"] = [float(q.p.detach()) for q in m.body.nls]
    elif hasattr(m, "nls"):
        extra["nl_params"] = [float(q.p.detach()) for q in m.nls]
    train_s = time.perf_counter() - t0
    rec = dict(exp="73", **vars(a), test_acc=acc, test_ce=ce, train_acc=tr_acc, params=params, train_s=train_s, **extra,
               version=p2.__version__, torch=torch.__version__)
    print(json.dumps(rec, ensure_ascii=False))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if a.save:
        os.makedirs("results/models", exist_ok=True)
        torch.save({k: v.cpu() for k, v in m.state_dict().items()}, f"results/models/exp73_{a.task}_{a.kind}_{a.stages}x{a.per}_s{a.seed}.pt")


if __name__ == "__main__":
    main()
