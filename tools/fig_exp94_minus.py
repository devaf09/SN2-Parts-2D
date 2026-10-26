"""実験94 のマイナスポイントの説明図: 段の選び間違い・各段の板の例・似た単語の並びの鍵。実行: python tools/fig_exp94_minus.py"""
import os
import sys

import numpy as np
import torch

from figio import ROOT, plt

sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "text"))
import exp88_memkeys as m88  # noqa: E402
import exp94_flush as e  # noqa: E402

res = {k: e.run(k, 0, "r1.5", 10, "sum", False) for k in ("rand", "word")}
st = dict(e.LAST)                                   # 最後に回した word の中身
words = m88._word["last_X"]
voc = m88._word["in_vocab"]
txt = lambda x: "".join(voc[int(i)] if int(i) < len(voc) - 2 else ("〈他〉" if int(i) == len(voc) - 2 else "〈始〉") for i in x)
keys = st["keys"] - st["kmean"]
kn = keys.flatten(1) / keys.flatten(1).norm(dim=1, keepdim=True)
sim = (kn @ kn.conj().T).abs()
sim.fill_diagonal_(0)
seen = st["seen"]
Ss, chosen, cur = st["Ss"], st["chosen"], st["cur"]
correct_plate = Ss.argmax(-1) == cur[None]          # (段, 組)
wrong = seen & ~(Ss.argmax(-1)[chosen, torch.arange(len(cur))] == cur) & correct_plate.any(0)

fig = plt.figure(figsize=(19, 11))
gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 1.15])
# (1) 選び間違い
a = fig.add_subplot(gs[0, 0:2])
for j, k in enumerate(("rand", "word")):
    a.bar(j - 0.18, res[k]["acc"], 0.36, color="#2e86c1", label="確信度で段を選ぶ（実際）" if j == 0 else None)
    a.bar(j + 0.18, res[k]["acc_oracle"], 0.36, color="#27ae60", label="どれかの段では正しい（選び方が完全なら）" if j == 0 else None)
    a.text(j - 0.18, res[k]["acc"], f"{res[k]['acc']:.0f}", ha="center", va="bottom")
    a.text(j + 0.18, res[k]["acc_oracle"], f"{res[k]['acc_oracle']:.0f}", ha="center", va="bottom")
a.set_xticks([0, 1])
a.set_xticklabels(["でたらめな鍵", "単語の並びの鍵"])
a.set_ylim(0, 105)
a.set_ylabel("取り出し %（seed 0）")
a.set_title("マイナス 1: 単語の鍵では、記憶は残っているのに読む段を選び間違える", fontsize=10)
a.legend(fontsize=8, loc="lower left")
# (2) 各段の板（周波数の面の明るさ）
nonempty = [(j, H) for j, H in enumerate(st["series"][0]) if H.abs().sum() > 0]
for c, (j, H) in enumerate(nonempty[:4]):
    a = fig.add_subplot(gs[0, 2 + c])
    a.imshow(np.fft.fftshift(H.abs().numpy()) ** 0.5, cmap="magma")
    a.set_title(f"段 {j}（幅 {st['W'][j]:.1f}、たまった量 {st['amount'][0][j]:.0f} 区切り）", fontsize=8)
    a.axis("off")
# (3) 似た単語の並びの鍵
q = int(torch.nonzero(wrong)[0])
order = sim[q].argsort(descending=True)[:4]
a = fig.add_subplot(gs[1, 0:2])
a.axis("off")
lines = [f"問い合わせ: 「{txt(words[q])}」 → 正しい値の窓 {int(cur[q])}"] + [f"似た鍵 {i + 1}: 「{txt(words[o])}」 似ている度合い {float(sim[q, o]):.2f}" for i, o in enumerate(order)]
lines += ["", f"鍵どうしの似ている度合いの平均: {float(sim[seen][:, seen].mean()):.3f}"]
a.text(0, 1, "\n".join(lines), va="top", fontsize=10, family="IPAexGothic" if False else None)
a.set_title("似た単語の並びの鍵（直前 4 語、拡散の入れ方）", fontsize=10)
for c, o in enumerate([q] + order[:3].tolist()):
    a = fig.add_subplot(gs[1, 2 + c])
    a.imshow(np.angle(m88.make_keys.__globals__["torch"].tensor(0) if False else st["keys"][o].numpy()), cmap="twilight")
    a.set_title(("問い合わせ" if c == 0 else f"似た鍵 {c}") + f"\n「{txt(words[o])}」", fontsize=8)
    a.axis("off")
# (4) 正しい段と選ばれた段の読み出し
cp = int(torch.nonzero(correct_plate[:, q])[0])
cb = int(chosen[q])
for c, (j, nm) in enumerate(((cp, "正しい段"), (cb, "選ばれた段（間違い）"))):
    a = fig.add_subplot(gs[2, c * 3:(c + 1) * 3])
    v = Ss[j, q].numpy()
    a.bar(range(len(v)), v, color="#bbbbbb", width=1.0)
    a.bar([int(cur[q])], [v[int(cur[q])]], color="#27ae60", width=3, label="正しい値の窓")
    top2 = np.sort(v)[-2:]
    a.bar([int(v.argmax())], [v.max()], color="#c0392b" if v.argmax() != int(cur[q]) else "#27ae60", width=3, label="一番明るい窓")
    a.set_title(f"{nm}: 段 {j}、確信度（一番 / 二番）= {top2[1] / top2[0]:.2f}", fontsize=10)
    a.set_xlabel("窓（値）")
    a.legend(fontsize=8)
fig.suptitle("実験94 のマイナスポイント: 満ちたら送る段（1.5 倍ずつ 10 段）、単語の鍵。正しい段では正しい窓が光るのに、似た記憶を持つ別の段の方が自信ありげに光る")
fig.tight_layout()
path = os.path.join(ROOT, "docs", "images", "exp94_minus.png")
fig.savefig(path, dpi=105)
print(path, "q", q, "correct plate", cp, "chosen", cb)
