"""現時点のまとめの図（2026-10-26）: 目的ごとに、前の最良 → いまの最良 → 比べる相手。実行: python tools/fig_overview.py"""
import os

from figio import ROOT, plt

panels = [
    ("画像: 位置がでたらめな形（8 クラス）", "%", [("板を重ねる（実験01）", 70.6, "#95a5a6"), ("4f 系 4 段 + 平方根（実験76）", 98.4, "#2e86c1"), ("小さな CNN", 98.6, "#e67e22")]),
    ("画像: MNIST", "%", [("④ 波紋（実験18）", 96.3, "#95a5a6"), ("4f 系 3 段 + 平方根（実験76）", 98.0, "#2e86c1"), ("小さな CNN", 98.1, "#e67e22")]),
    ("文字: 直前 8 文字 → 次の文字", "%", [("光路 1 本・板（実験73）", 47.0, "#95a5a6"), ("光路 1 本・4f 系（実験79）", 53.0, "#5dade2"),
                                      ("4f 系 3 つの足し合わせ（実験82）", 55.7, "#2e86c1"), ("4-gram", 55.4, "#e67e22")]),
    ("単語: 直前 4 語 → 次の語", "%", [("分担の 2DTG・板（実験62）", 36.0, "#95a5a6"), ("4f 系 3 つの足し合わせ（実験85、3 seed）", 37.4, "#2e86c1"),
                                    ("3-gram", 36.4, "#e67e22"), ("GRU", 36.9, "#e67e22")]),
    ("記憶: 似た鍵 128 組の取り出し（似ている度合い 0.75）", "%", [("そのまま重ね書き（実験88）", 3, "#95a5a6"), ("差分記憶法（実験89）", 100, "#2e86c1")]),
    ("記憶: 短期 → 長期、300 組の取り出し（単語の鍵）", "%", [("そのまま書く（実験89）", 12.4, "#95a5a6"), ("差分 + 足りない分だけ書く", 43.4, "#2e86c1")]),
]
fig, ax = plt.subplots(2, 3, figsize=(17, 8))
for a, (title, unit, rows) in zip(ax.flat, panels):
    y = range(len(rows))[::-1]
    a.barh(list(y), [v for _, v, _ in rows], color=[c for _, _, c in rows])
    lo = min(v for _, v, _ in rows)
    for yy, (_, v, _) in zip(y, rows):
        a.text(v, yy, f" {v:g}", va="center", fontsize=9)
    a.set_yticks(list(y))
    a.set_yticklabels([n for n, _, _ in rows], fontsize=8)
    a.set_title(title, fontsize=10)
    a.set_xlim(max(0, lo - (lo * 0.3 if lo > 30 else lo)), max(v for _, v, _ in rows) * 1.12)
fig.suptitle("SN2-Parts-2D の現時点のまとめ（2026-10-26）: 灰 = 前の最良、青 = いまの最良、橙 = 比べる相手")
fig.tight_layout()
path = os.path.join(ROOT, "docs", "images", "overview_20261026.png")
fig.savefig(path, dpi=110)
print(path)
