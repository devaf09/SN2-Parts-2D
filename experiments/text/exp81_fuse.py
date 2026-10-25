"""実験81: 分担の 2DTG（4f 系）と光路 1 本（4f 系）の確率を足し合わせる。保存したテストの対数確率を読むだけ（GPU は使わない）

着想: 分担の 2DTG（4f 系）は覚えすぎる（訓練 68〜85%、テスト 49〜53%、実験80）。光路 1 本（4f 系）は覚えすぎない（訓練 55.5%、テスト 53.0%）。
      性格の違う 2 つの確率を足すと、互いの誤りを打ち消すかもしれない（実験65b・67 の「融合」と同じ考え）。
方法: log p = Σ w_i log p_i（重みの和を 1 にして正規化）。重みは 0〜1 を 0.05 刻みで、テストの前半で選び、後半で測る（選び方の上振れを避ける）。
実行: python experiments/text/exp81_fuse.py results/logits/a.pt results/logits/b.pt
"""
import itertools
import json
import os
import sys

import torch
import torch.nn.functional as F

paths = sys.argv[1:]
Ls = [torch.load(p) for p in paths]
y = Ls[0]["y"]
assert all(torch.equal(d["y"], y) for d in Ls), "テストの並びが違う"
P = torch.stack([d["logp"] for d in Ls])
half = len(y) // 2
grid = torch.arange(0, 1.0001, 0.05)


def score(w, sl):
    lp = F.log_softmax((w[:, None, None] * P[:, sl]).sum(0), 1)
    return 100.0 * (lp.argmax(1) == y[sl]).float().mean().item(), F.nll_loss(lp, y[sl]).item()


def choose(sl):
    best = None
    for ws in itertools.product(grid, repeat=len(paths) - 1):
        if sum(ws) > 1 + 1e-6:
            continue
        w = torch.tensor(list(ws) + [1 - sum(ws)])
        ce = score(w, sl)[1]
        if best is None or ce < best[0]:
            best = (ce, w)
    return best[1]


A, B = slice(0, half), slice(half, None)
w = choose(A)
# 2 つに分けて入れ替える（実験82）: 前半で選んで後半で測る・後半で選んで前半で測る、の平均 = テスト全部の値
wB = choose(B)
cross = [(score(w, B)[i] + score(wB, A)[i]) / 2 for i in range(2)]
full = lambda i: score(torch.eye(len(paths))[i], slice(None))
rec = dict(exp="81", files=paths, weights=[round(float(x), 2) for x in w], weights_B=[round(float(x), 2) for x in wB],
           alone=[score(torch.eye(len(paths))[i], B) for i in range(len(paths))],
           fused=score(w, B), alone_full=[full(i) for i in range(len(paths))], fused_cross=cross)
print(json.dumps(rec, ensure_ascii=False))
with open(os.environ.get("FUSE_OUT", "results/exp81.txt"), "a", encoding="utf-8") as fo:
    fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
