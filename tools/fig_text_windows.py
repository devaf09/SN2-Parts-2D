"""言語の出力の窓のイメージ（docs/images/text_windows.png）。実験40 の形（帯 + 位置ごとの明るさ + 板 ①）を短く学習し、
スクリーン（レンズの後の明るさ）に 64 個の窓（1 クラス 1 窓）を重ねて描く。

    python tools/fig_text_windows.py            （GPU で 1〜2 分）
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments", "text")]
import sn2_parts_2d as p2  # noqa: E402
import textdata  # noqa: E402
from exp40_inputs import EncNet  # noqa: E402

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
(X, y), (Xt, yt), info = textdata.load(8, 64, 60000, 10000, 0)
voc_out = sorted(info["out_idx"], key=info["out_idx"].get) + ["他"]
voc_in = info["in_vocab"]
torch.manual_seed(0)
m = EncNet("bands", 8, info["V_in"], 64, layers=32, patches=16, seed=0).to(dev)
opt = torch.optim.Adam(m.parameters(), lr=0.01)
steps = 1500
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
g = torch.Generator().manual_seed(0)
for _ in range(steps):
    i = torch.randint(len(y), (256,), generator=g)
    loss = F.cross_entropy(m(X[i].to(dev)), y[i].to(dev))
    opt.zero_grad()
    loss.backward()
    opt.step()
    sch.step()
with torch.no_grad():
    S = torch.cat([m(Xt[j:j + 500].to(dev)).cpu() for j in range(0, 2000, 500)])
acc = 100.0 * (S.argmax(1) == yt[:2000]).float().mean().item()
det = m.net.det.cpu()
with torch.no_grad():   # 窓に入る光の割合（スクリーン全体の明るさのうち、64 個の窓の中にある分）
    Iall = torch.cat([m.net.screen(m.field(Xt[j:j + 250].to(dev))).cpu() for j in range(0, 1000, 250)])
    frac = float((Iall * (det.sum(0) > 0)).sum((-2, -1)).mean() / Iall.sum((-2, -1)).mean())
print("窓に入る光の割合", frac)
picks = [k for k in range(200) if S[k].argmax() == yt[k]][:2] + [k for k in range(200) if S[k].argmax() != yt[k]][:2]
fig, ax = plt.subplots(1, 4, figsize=(20, 5.6))
with torch.no_grad():
    for a, k in zip(ax, picks):
        I = m.net.screen(m.field(Xt[k:k + 1].to(dev)))[0].cpu()
        a.imshow(torch.log1p(I), cmap="magma")
        P = S[k].softmax(0)
        top = P.argsort(descending=True)[:3].tolist()
        for c in range(64):
            ys, xs = det[c].nonzero(as_tuple=True)
            y0, x0 = ys.min().item(), xs.min().item()
            col = "lime" if c == int(yt[k]) else ("cyan" if c == top[0] else "white")
            lw = 2.0 if col != "white" else 0.5
            a.add_patch(patches.Rectangle((x0 - 0.5, y0 - 0.5), 4, 4, fill=False, edgecolor=col, lw=lw))
            a.text(x0 + 1.5, y0 - 1.2, voc_out[c], color=col, fontsize=6 if col == "white" else 9, ha="center")
        ctx = "".join(voc_in[c] if c < len(voc_in) else "？" for c in Xt[k].tolist())
        ans = voc_out[int(yt[k])]
        guess = "・".join(f"{voc_out[c]} {float(P[c]) * 100:.0f}%" for c in top)
        a.set_title(f"「{ctx}」→ 正解「{ans}」\n上位: {guess}", fontsize=10)
        a.axis("off")
fig.suptitle(f"出力の窓（64 個、1 窓 = 1 文字）。背景 = スクリーンの明るさ（対数）。緑の枠 = 正解、水色 = 一番明るかった窓（短い学習 1500 ステップ、テスト {acc:.1f}%）", y=1.08)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "docs", "images", "text_windows.png"), dpi=100, bbox_inches="tight")
print("docs/images/text_windows.png", acc, frac)
