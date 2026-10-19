"""実験21 の図: 50 ステップごとの学習曲線（テスト・訓練・移動平均）とクラスごとの正解率（docs/images/exp21_dips.png）。

    python tools/fig_exp21.py
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
R = [json.loads(l) for l in open(os.path.join(ROOT, "results", "exp21.txt"), encoding="utf-8")]
CLS = ["Tシャツ", "ズボン", "プルオーバー", "ドレス", "コート", "サンダル", "シャツ", "スニーカー", "バッグ", "ブーツ"]
fig, ax = plt.subplots(2, 4, figsize=(17, 7.5), sharex=True)
for j, (m, s, t) in enumerate([("xy", "const", "板 ①・学習率一定"), ("xy", "cosine", "板 ①・0 まで下げる"),
                               ("sinc", "const", "④ 波紋 128 個・学習率一定"), ("sinc", "cosine", "④ 波紋 128 個・0 まで下げる")]):
    r = [r for r in R if r["method"] == m and r["sched"] == s and r["seed"] == 0][0]
    c = r["curve"]
    st = [x[0] for x in c]
    ax[0, j].plot(st, [x[3] for x in c], color="#ff7f0e", lw=1, label="訓練")
    ax[0, j].plot(st, [x[2] for x in c], color="#1f77b4", lw=1, label="テスト")
    ax[0, j].plot(st, [x[5] for x in c], color="k", lw=1.5, label="テスト（板の移動平均）")
    ax[0, j].axvline(r["best_step"], color="gray", ls=":", label="確かめ用で最良の時点")
    ax[0, j].set_ylim(55, 92)
    ax[0, j].set_title(t, fontsize=10)
    ax[0, j].grid(alpha=0.3)
    for k in range(10):
        hl = k in (0, 2, 4, 6)
        ax[1, j].plot(st, [x[7 + k] for x in c], lw=1.2 if hl else 0.6, alpha=1 if hl else 0.4, label=CLS[k])
    ax[1, j].set_ylim(0, 100)
    ax[1, j].set_xlabel("学習のステップ")
    ax[1, j].grid(alpha=0.3)
ax[0, 0].set_ylabel("正解率（%）")
ax[1, 0].set_ylabel("クラスごとのテスト正解率（%）")
ax[0, 0].legend(fontsize=8, loc="lower right")
ax[1, 3].legend(fontsize=7, loc="lower right", ncol=2)
fig.suptitle("Fashion-MNIST の 50 ステップごとの学習曲線（seed 0）。谷では訓練も一緒に落ち、似た上着どうし（太線）が入れ替わる")
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "docs", "images", "exp21_dips.png"), dpi=100, bbox_inches="tight")
print("docs/images/exp21_dips.png")
