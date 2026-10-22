"""実験48 の図: 短冊形（持ち主の案）の入力 → 学習後の板 → 出力。

実験と同じ条件（短冊 208 × 32、板 16 枚、1500 ステップ、seed 0、マスクあり・プリズム 4・照らす範囲 3）で学習し直して描く。
実行: python tools/fig_exp48.py
"""
import os

import torch

from figio import ROOT, plt, show_bars, show_field, show_output, show_plate, train
import sn2_parts_2d as p2
import textdata
from exp48_strip import StripNet


def main():
    torch.manual_seed(0)
    (X, y), (Xt, yt), info = textdata.load(8, 64, 60000, 10000, 0)
    V, C = info["V_in"], info["C"]
    inv = info["in_vocab"] + ["〓"]
    out = [k for k, _ in sorted(info["out_idx"].items(), key=lambda t: t[1])] + ["その他"]
    m = train(StripNet(8, V, C), X, y)
    j = 7
    x, t = Xt[j:j + 1].cuda(), int(yt[j])
    ctx = "".join(inv[i] for i in Xt[j].tolist())
    fig, ax = plt.subplots(1, 7, figsize=(17, 8.5), gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1, 1, 2.2]))
    with torch.no_grad():
        # 文字ごとの照らす範囲
        a = torch.exp(m.w)
        a = a / a.pow(2).mean().sqrt()
        Mw = (m.M * a[:, None, None]).sum(0)
        ax[0].imshow(Mw.cpu().numpy(), cmap="viridis")
        for k in range(8):
            ax[0].text(m.Lx + 1, (0.5 * k + 0.2) * m.U, f"{k + 1}:{inv[int(x[0, k])]}", fontsize=8, va="top")
        ax[0].set_title("文字ごとの照らす範囲\n（明るさ = 学んだ重み）", fontsize=9)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        u = p2.normalize(m.field(x))[0]
        show_field(ax[1], u, "入力: 短冊の光\n（色 = 位相、濃さ = 明るさ）")
        nb = m.nb
        phases = [p2.plate_xy(m.A[d, :nb], m.wv[d, :nb], m.phi[d, :nb], m.bx, m.A[d, nb:], m.wv[d, nb:], m.phi[d, nb:], m.by, m.x1, m.y1)
                  for d in range(m.A.shape[0])]
        show_plate(ax[2], phases[0], "学習後の板 1 枚目")
        show_plate(ax[3], phases[-1], "学習後の板 16 枚目")
        for P in phases:
            u = p2.normalize(p2.propagate(p2.apply_plate(u, P), m.H))
        show_field(ax[4], u, "最後の板の後の光")
        I = p2.intensity(p2.lens(u))
        show_output(ax[5], I, m.det, "出力: レンズの後の明るさ\n（対数）と窓 64 個")
        show_bars(ax[6], torch.softmax(m(x)[0], -1), out, f"次の文字の確率（正解「{out[t]}」は赤）", t, k=8)
    fig.suptitle(f"実験48: 短冊形（持ち主の案）の入力 → 学習後の板 → 出力。直前 8 文字「{ctx}」→ 次の文字（テスト 41.7%）", fontsize=12)
    fig.tight_layout()
    path = os.path.join(ROOT, "docs", "images", "exp48_io.png")
    fig.savefig(path, dpi=110)
    print(path)


if __name__ == "__main__":
    main()
