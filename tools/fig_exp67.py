"""実験67 の図: 使った 4 本の光路（位相差も読む 2DTG、seed 0〜3）それぞれの、入力 → 学習後の板 → 出力、と逆向きの反応。

穴埋めの例（テストの文の連続 3 語 w x D）で、
  順向き: w を入れて、次の語 x の確率（出口の明るさと、参照光との位相差を窓で読む）
  逆向き: D を出口から入れて、入口に戻ってきた光の、語ごとの向きの強さ（D の前に来やすい語）
を 1 本ずつ並べ、最後の段に 4 本を合わせた結果（逆向きの掛け合わせ、左 × 右）を出す。
実行: python tools/fig_exp67.py
"""
import os
import random

import torch

from figio import ROOT, plt, show_bars, show_field, show_output, show_plate
import sn2_parts_2d as p2
import worddata
from exp63_amp_phase import AmpPhaseNet
from exp65b_combine import reverse_map

V, C, L = 4096, 1024, 64


def main():
    (X, y), _, info = worddata.load(1, C, 400000, 100, 0, V)
    iv, ov, invoc, outvoc = info["in_idx"], info["out_idx"], info["in_vocab"], info["out_vocab"]
    models = []
    for s in range(4):
        m = AmpPhaseNet("phase", "int_phase", 1, V, C)
        m.load_state_dict(torch.load(os.path.join(ROOT, "results", "models", f"exp63_phase_int_phase_s{s}.pt")))
        models.append(m.eval())
    # 入力の語の番号 → 答えの番号（同じ語が答えの語彙にあるとき）
    in2out = torch.full((V,), -1, dtype=torch.long)
    for w, i in iv.items():
        if w in ov:
            in2out[i] = ov[w]
    k = int(len(worddata.docs_words()) * 0.85)
    te = worddata.docs_words()[k:]
    trip = [(d[i - 1], d[i], d[i + 1]) for d in te for i in range(1, len(d) - 1)
            if d[i - 1] in iv and d[i] in ov and d[i] in iv and d[i + 1] in ov and len(d[i]) >= 2]
    random.Random(1).shuffle(trip)
    with torch.no_grad():
        def scores(w, D):
            left = torch.stack([torch.softmax(m(torch.tensor([[iv[w]]])), -1)[0] for m in models])          # (4, C)
            right = torch.stack([reverse_map(m, "phase", ov[D]) for m in models])                               # (4, V)
            return left, right
        pick = None
        for w, x, D in trip[:400]:   # 左だけでは外れ、左 × 右で当たる例を探す
            left, right = scores(w, D)
            lv = torch.full((V,), -30.0)
            ok = in2out >= 0
            lv[ok] = torch.log(left.mean(0)[in2out[ok]] + 1e-12)
            both = lv + torch.log(right + 1e-12).mean(0)
            if int(both.argmax()) == iv[x] and int(lv.argmax()) != iv[x]:
                pick = (w, x, D, left, right, lv, both)
                break
        w, x, D, left, right, lv, both = pick
        fig, ax = plt.subplots(5, 7, figsize=(22, 16))
        for s, m in enumerate(models):
            net = m.net
            u = m.field(torch.tensor([[iv[w]]]))[0]
            show_field(ax[s, 0], u, f"入力:「{w}」の光（向き = 語）" if s == 0 else "入力（同じ）")
            show_plate(ax[s, 1], net.plate_phase(0), "学習後の板 1 枚目")
            show_plate(ax[s, 2], net.plate_phase(len(net.kinds) - 1), "学習後の板 32 枚目")
            uu = u[None]
            for d in range(len(net.kinds)):
                uu = p2.normalize(p2.propagate(p2.apply_plate(uu, net.plate_phase(d)), net.H))
            U = p2.lens(uu)[0]
            show_output(ax[s, 3], p2.intensity(U), net.det, "出力: 明るさ（対数）と窓")
            ax[s, 4].imshow(torch.angle(U).numpy(), cmap="twilight")
            ax[s, 4].set_title("出力: 参照光との位相差", fontsize=9)
            ax[s, 4].set_xticks([])
            ax[s, 4].set_yticks([])
            show_bars(ax[s, 5], left[s], outvoc, f"順向き: 「{w}」の次の語", ov[x])
            show_bars(ax[s, 6], right[s], invoc, f"逆向き: 「{D}」の前に来やすい語", iv[x])
            ax[s, 0].set_ylabel(f"2DTG（seed {s}）", fontsize=12)
        # 合わせた結果
        grid = torch.zeros(L, L)
        prod = torch.softmax(torch.log(right + 1e-12).mean(0), 0)
        grid[models[0].ky.long(), models[0].kx.long()] = prod
        ax[4, 0].imshow(torch.log1p(grid / grid.mean()).numpy(), cmap="inferno")
        ax[4, 0].set_title(f"逆向き 4 本を掛け合わせた反応\n（入口の向きの格子、明るい = 「{D}」の前らしい語）", fontsize=9)
        ax[4, 0].set_xticks([])
        ax[4, 0].set_yticks([])
        for c in range(1, 5):
            ax[4, c].axis("off")
        note = (f"穴埋めの例: 「{w}  ＿  {D}」の ＿ を当てる（正解「{x}」）\n\n"
                      "左の手がかり = 順向き（「{0}」を入れて次の語）\n右の手がかり = 逆向き（「{1}」を出口から入れて前の語）\n\n"
                      "2000 問の正解率: 左だけ 21.3% → 左 × 右（4 本）33.8%".format(w, D))
        show_bars(ax[4, 5], torch.softmax(lv, 0), invoc, "左だけ（4 本の平均）", iv[x])
        show_bars(ax[4, 6], torch.softmax(both, 0), invoc, "左 × 右（4 本）", iv[x])
    fig.suptitle("実験67: 位相差も読む 2DTG 4 本それぞれの、入力 → 学習後の板 → 出力（順向き）と、逆向きの反応。赤 = 正解", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p0, p4 = ax[4, 1].get_position(), ax[4, 4].get_position()
    fig.text(p0.x0, (p0.y0 + p0.y1) / 2, note, fontsize=13, va="center")
    path = os.path.join(ROOT, "docs", "images", "exp67_io.png")
    fig.savefig(path, dpi=100)
    print(path, w, x, D)


if __name__ == "__main__":
    main()
