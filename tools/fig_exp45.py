"""実験45 の図: 入力 → 学習後の板 → 出力。足し合わせる（帯）と、文字を板として掛け算で通す形を並べる。

実験と同じ条件（板 16 枚 × 区画 8、1500 ステップ、seed 0）で学習し直して描く（実験の時は光路を保存していなかったため）。
実行: python tools/fig_exp45.py
"""
import math
import os

import torch

from figio import ROOT, plt, show_bars, show_field, show_output, show_plate, train
import sn2_parts_2d as p2
import textdata
from exp40_inputs import EncNet
from exp45_modulate import ModNet


def main():
    torch.manual_seed(0)
    (X, y), (Xt, yt), info = textdata.load(8, 64, 60000, 10000, 0)
    V, C = info["V_in"], info["C"]
    inv = info["in_vocab"] + ["〓"]
    out = [k for k, _ in sorted(info["out_idx"].items(), key=lambda t: t[1])] + ["その他"]
    add = train(EncNet("bands", 8, V, C, 16, 8, 0), X, y)
    mul = train(ModNet(8, V, C, 2, 8, "rand", 0), X, y)
    j = 7
    x, t = Xt[j:j + 1].cuda(), int(yt[j])
    ctx = "".join(inv[i] for i in Xt[j].tolist())
    fig, ax = plt.subplots(2, 5, figsize=(17, 7.2), gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1.1]))
    with torch.no_grad():
        # 足し合わせる
        u = add.field(x)[0]
        show_field(ax[0, 0], u, "入力: 8 文字を帯に足し合わせた光\n（色 = 位相、濃さ = 明るさ）")
        show_plate(ax[0, 1], add.net.plate_phase(0), "学習後の板 1 枚目（位相の遅れ）")
        show_plate(ax[0, 2], add.net.plate_phase(15), "学習後の板 16 枚目")
        I = add.net.screen(add.field(x))[0]
        show_output(ax[0, 3], I, add.net.det, "出力: レンズの後の明るさ（対数）\n白い枠 = 答えごとの窓 64 個")
        show_bars(ax[0, 4], torch.softmax(add(x)[0], -1), out, f"次の文字の確率（正解「{out[t]}」は赤）", t)
        # 掛け算
        u = torch.ones(1, 64, 64, dtype=torch.complex64, device="cuda")
        ph = mul.masks[x[0, 0]]
        u1 = u * torch.polar(torch.ones_like(ph), ph)
        show_field(ax[1, 0], u1[0], f"入力: 1 文字目「{inv[int(x[0, 0])]}」の板を通した光\n（文字ごとのでたらめな位相の模様）")
        show_plate(ax[1, 1], mul.net.plate_phase(0), "学習後の板 1 枚目（文字の板の間）")
        show_plate(ax[1, 2], mul.net.plate_phase(15), "学習後の板 16 枚目")
        uu = torch.ones(1, 64, 64, dtype=torch.complex64, device="cuda")
        d = 0
        for mpos in range(8):
            ph = mul.masks[x[:, mpos]]
            uu = uu * torch.polar(torch.ones_like(ph), ph)
            for _ in range(2):
                uu = p2.normalize(p2.propagate(p2.apply_plate(uu, mul.net.plate_phase(d)), mul.net.H))
                d += 1
        I = p2.intensity(p2.lens(uu))[0]
        show_output(ax[1, 3], I, mul.net.det, "出力: 明るさ（対数）。砂嵐のように散る")
        show_bars(ax[1, 4], torch.softmax(mul(x)[0], -1), out, f"次の文字の確率（正解「{out[t]}」は赤）", t)
    ax[0, 0].set_ylabel("足し合わせる（帯）\n42.8%", fontsize=11)
    ax[1, 0].set_ylabel("掛け算で通す\n36.3%（学べない）", fontsize=11)
    fig.suptitle(f"実験45: 入力 → 学習後の板 → 出力。直前 8 文字「{ctx}」→ 次の文字", fontsize=12)
    fig.tight_layout()
    path = os.path.join(ROOT, "docs", "images", "exp45_io.png")
    fig.savefig(path, dpi=110)
    print(path)


if __name__ == "__main__":
    main()
