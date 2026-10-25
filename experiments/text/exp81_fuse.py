"""実験81: 分担の 2DTG（4f 系）と光路 1 本（4f 系）の確率を足し合わせる。保存したテストの対数確率を読むだけ（GPU は使わない）

着想: 分担の 2DTG（4f 系）は覚えすぎる（訓練 68〜85%、テスト 49〜53%、実験80）。光路 1 本（4f 系）は覚えすぎない（訓練 55.5%、テスト 53.0%）。
      性格の違う 2 つの確率を足すと、互いの誤りを打ち消すかもしれない（実験65b・67 の「融合」と同じ考え）。
方法: log p = Σ w_i log p_i（重みの和を 1 にして正規化）。重みは 0〜1 を 0.05 刻みで、テストの前半で選び、後半で測る（選び方の上振れを避ける）。
実行: python experiments/text/exp81_fuse.py results/logits/a.pt results/logits/b.pt
"""
import itertools
import json
import sys

import torch
import torch.nn.functional as F

paths = sys.argv[1:]
Ls = [torch.load(p) for p in paths]
y = Ls[0]["y"]
P = torch.stack([d["logp"] for d in Ls])
half = len(y) // 2
grid = torch.arange(0, 1.0001, 0.05)


def score(w, sl):
    lp = F.log_softmax((w[:, None, None] * P[:, sl]).sum(0), 1)
    return 100.0 * (lp.argmax(1) == y[sl]).float().mean().item(), F.nll_loss(lp, y[sl]).item()


best = None
for ws in itertools.product(grid, repeat=len(paths) - 1):
    if sum(ws) > 1:
        continue
    w = torch.tensor(list(ws) + [1 - sum(ws)])
    ce = score(w, slice(0, half))[1]
    if best is None or ce < best[0]:
        best = (ce, w)
w = best[1]
rec = dict(exp="81", files=paths, weights=[round(float(x), 2) for x in w],
           alone=[score(torch.eye(len(paths))[i], slice(half, None)) for i in range(len(paths))],
           fused=score(w, slice(half, None)))
print(json.dumps(rec, ensure_ascii=False))
with open("results/exp81.txt", "a", encoding="utf-8") as fo:
    fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
