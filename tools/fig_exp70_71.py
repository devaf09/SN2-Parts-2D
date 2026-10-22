"""実験70（デジタルの重みを減らす）と実験71（2DWIG）の図。実行: python tools/fig_exp70_71.py"""
import json
import os

from figio import ROOT, plt

fig, ax = plt.subplots(1, 3, figsize=(17, 5))
# 実験70: 正解率とデジタルの重み
pts = {"単語": [("1 本、窓 256（実験57）", 0.26, 32.7), ("分担 16、窓 256（実験58）", 4.20, 36.0), ("分担 16、窓 32", 0.52, 34.6),
                ("分担 16、窓 32、担当の決め打ち", 0.52, 35.7), ("頻度の分担 64、窓 32、決め打ち", 2.10, 34.9), ("GRU（全部）", 0.76, 36.9)],
       "文字": [("1 本、窓 256（実験66）", 0.017, 51.3), ("分担 16、窓 256（実験62）", 0.27, 54.0), ("分担 16、窓 32", 0.033, 53.0),
                ("分担 16、窓 32、担当の決め打ち", 0.033, 53.3), ("頻度の分担 64、窓 32、決め打ち", 0.13, 53.5), ("GRU（全部）", 0.17, 53.6)]}
for a, (task, ps) in zip(ax[:2], pts.items()):
    for name, d, acc in ps:
        col = "#e67e22" if "GRU" in name else ("#c0392b" if "決め打ち" in name and "頻度" not in name else "#2e86c1")
        a.scatter(d, acc, s=60, color=col)
        a.annotate(name, (d, acc), textcoords="offset points", xytext=(5, 4), fontsize=8)
    a.set_xscale("log")
    from matplotlib.ticker import NullFormatter
    a.xaxis.set_minor_formatter(NullFormatter())
    a.set_xlabel("学習する数のうちデジタルの部分（百万、対数）")
    a.set_ylabel("正解率 %")
    a.set_title(f"実験70（{task}）: 窓を減らし担当を決め打ちにしても精度を保つ")
    a.grid(alpha=0.3)
# 実験71
R = [json.loads(l) for l in open(os.path.join(ROOT, "results", "exp71.txt"), encoding="utf-8")]
avg = lambda m, k: sum(r[k] for r in R if r["method"] == m) / sum(1 for r in R if r["method"] == m)
names = ["いつも一番多い類", "光の 2DWIG", "ロジスティック回帰", "数えた表\n（ナイーブベイズ）"]
te = [R[0]["majority"], avg("plates", "test_acc"), avg("logreg", "test_acc"), 51.9]
tr = [None, avg("plates", "train_acc"), avg("logreg", "train_acc"), None]
ax[2].bar(range(4), te, color=["#7f8c8d", "#2e86c1", "#e67e22", "#27ae60"])
for i, (a_, b_) in enumerate(zip(te, tr)):
    ax[2].text(i, a_ + 1, f"{a_:.1f}%" + (f"\n（訓練 {b_:.0f}%）" if b_ else ""), ha="center", fontsize=9)
ax[2].set_xticks(range(4))
ax[2].set_xticklabels(names, fontsize=9)
ax[2].set_ylim(0, 70)
ax[2].set_ylabel("テストの正解率 %（12 類）")
ax[2].set_title("実験71: 2DWIG（会話文の語尾から話者の一人称）")
fig.tight_layout()
path = os.path.join(ROOT, "docs", "images", "exp70_71.png")
fig.savefig(path, dpi=110)
print(path)
