"""実験84 の図: 単語の課題で、4f 系（光路 1 本・分担の 2DTG・足し合わせ）と数える方式を比べる（seed 0）。実行: python tools/fig_exp84.py"""
import json
import os

from figio import ROOT, plt

R = [json.loads(l) for l in open(os.path.join(ROOT, "results", "exp84.txt"), encoding="utf-8")]
Fz = [json.loads(l) for l in open(os.path.join(ROOT, "results", "exp84_fuse.txt"), encoding="utf-8")]
ng = next(r for r in R if r.get("ngram2"))
rows = [("単語の 2-gram", *ng["ngram2"][:2], "#e67e22"), ("単語の 3-gram", *ng["ngram3"][:2], "#e67e22"), ("単語の 4-gram", *ng["ngram4"][:2], "#e67e22"),
        ("GRU（実験57、実験70 の表の値）", 36.9, 3.04, "#7f8c8d")]
name = lambda r: "光路 1 本 " + f"{r['stages']} 段 × {r['width']}" if r["G"] == 1 else f"分担の 2DTG 担当 {r['G']}・直前 {r['n']} 語"
rows += [(name(r), r["test_acc"], r["test_ce"], "#2e86c1") for r in R]
lab = {"w_g4_n4": "担当 4", "w_g8_n2": "担当 8", "w_g16_n2": "担当 16"}
rows += [("足し合わせ: 光路 1 本 + " + " + ".join(lab[os.path.basename(p)[:-3]] for p in f["files"][1:]), *f["fused_cross"], "#c0392b") for f in Fz]
fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
y = range(len(rows))[::-1]
for a, k, t in ((ax[0], 1, "テストの正解率 %（高いほど良い）"), (ax[1], 2, "クロスエントロピー（低いほど良い）")):
    a.barh(list(y), [r[k] for r in rows], color=[r[3] for r in rows])
    for yy, r in zip(y, rows):
        a.text(r[k], yy, f" {r[k]:.2f}" if k == 2 else f" {r[k]:.1f}", va="center", fontsize=8)
    a.set_yticks(list(y))
    a.set_yticklabels([r[0] for r in rows] if k == 1 else [], fontsize=8)
    a.set_xlabel(t)
ax[0].set_xlim(30, 39)
ax[1].set_xlim(2.8, 4.4)
fig.suptitle("実験84: 単語（直前 4 語 → 次の語、答え 1024 種類）。橙 = 数える方式、青 = 4f 系 + 平方根、赤 = 足し合わせ（seed 0、テスト全部）")
fig.tight_layout()
path = os.path.join(ROOT, "docs", "images", "exp84_word4f.png")
fig.savefig(path, dpi=110)
print(path)
