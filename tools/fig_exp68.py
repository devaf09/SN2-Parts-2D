"""実験68 の図: 「その他」に光が集まる問題（案 1 = 表示で除く目安、案 2 = その他の中を受け持つ 2DTG）。
実行: python tools/fig_exp68.py
"""
import json
import os

from figio import ROOT, plt

R = [json.loads(l) for l in open(os.path.join(ROOT, "results", "exp68.txt"), encoding="utf-8")]
avg = lambda f: sum(f(r) for r in R) / len(R)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
e = lambda k: avg(lambda r: r["results"]["exclude_other"][k])
v = [e("other_share_of_answers"), e("other_is_top1"), e("other_in_top5")]
ax[0].bar(range(3), v, color=["#7f8c8d", "#c0392b", "#e67e22"])
ax[0].set_xticks(range(3))
ax[0].set_xticklabels(["答えが本当に\n「その他」", "一番上に\n「その他」を出す", "上位 5 に\n「その他」が入る"], fontsize=9)
ax[0].set_ylabel("%")
ax[0].set_title("(a) 光路は「その他」を出しすぎる")
v = [e("acc_with_other_on_not_other"), e("acc_excluding_other_on_not_other")]
ax[1].bar(range(2), v, color=["#7f8c8d", "#2e86c1"])
ax[1].set_xticks(range(2))
ax[1].set_xticklabels(["そのまま", "「その他」を除いて\n並べ直す（案 1）"], fontsize=9)
ax[1].set_ylabel("正解率 %（答えが頻出語の位置）")
ax[1].set_title("(b) 案 1: 除くだけで +4 点")
names = ["1 段だけ", "その他の中を受け持つ\n2DTG を足す（案 2）", "2048 語を直接"]
keys = ["stage1_only", "tree", "direct"]
ce = [sum(r["results"][k]["all"]["ce"] for r in R if k in r["results"]) / sum(1 for r in R if k in r["results"]) for k in keys]
rare = [sum(r["results"][k]["rare"]["top5"] for r in R if k in r["results"]) / sum(1 for r in R if k in r["results"]) for k in keys]
ax[2].bar(range(3), ce, color=["#7f8c8d", "#2e86c1", "#e67e22"])
for i, (c, t) in enumerate(zip(ce, rare)):
    ax[2].text(i, c + 0.05, f"CE {c:.2f}\nまれな語の上位5 {t:.1f}%", ha="center", fontsize=8)
ax[2].set_xticks(range(3))
ax[2].set_xticklabels(names, fontsize=9)
ax[2].set_ylabel("確信度 CE（2048 語、小さいほど良い）")
ax[2].set_ylim(0, 5.4)
ax[2].set_title("(c) 案 2: 確信度が大きく良くなり、直接と同等")
fig.suptitle("実験68: 「その他」に光が集まる問題（単語、分担の 2DTG 16 個 × 板 8 枚、seed 2 本の平均）", fontsize=12)
fig.tight_layout()
path = os.path.join(ROOT, "docs", "images", "exp68_other.png")
fig.savefig(path, dpi=110)
print(path)
