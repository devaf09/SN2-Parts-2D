"""目標への到達度（ポイント表）。合計 100 点。

各項目: 点 = 配点 × clip((今の値 − ゼロ点の基準) / (目標 − ゼロ点の基準), 0, 1)
小さいほど良い指標も同じ式（目標 < ゼロ点の基準になる）。桁で変わるもの（時間など）は log10 で比べる。
未測定の項目は 0 点。新しい結果が出たら ITEMS の「今の値」と「出典」を書き換えて
    python tools/scoreboard.py --md
で docs/scoreboard.md を作り直す。
"""
from __future__ import annotations

import argparse
import math
import os
import sys

# (軸, 項目, 配点, ゼロ点の基準, 目標, 今の値, 対数か, 目標の根拠, 出典)
ITEMS = [
    ("M1 基本", "形の分類（8 クラス、%）", 8, 12.5, 98.6, 79.9, False, "同じくらいの学習する数の小さな CNN（実験01）", "実験04 板 ① + tilt 照明 16 通り"),
    ("M1 基本", "縞の向き（8 クラス、%）", 4, 12.5, 100, 99.9, False, "小さな CNN（実験01）", "実験01 板 ①②③"),
    ("M1 基本", "位置（3×3 のマス、%）", 8, 11.1, 99.0, 99.8, False, "小さな CNN（実験01）", "実験01 板 ①"),
    ("M1 基本", "向き × 位置（8 クラス、%）", 6, 12.5, 100, 99.8, False, "小さな CNN（実験01）", "実験04 板 ① + tilt 照明 4 通り"),
    ("M1 基本", "板の形 ①②③ と混合の比較がそろう（0/1）", 4, 0, 1, 1, False, "4 課題 × 5 種 × 3 seed", "実験01"),
    ("M2 得意分野", "2 技能の融合（同時学習との差、点。小さいほど良い）", 7, 10, 0, 0, False, "差なし", "実験10 担当なしの絵 100.0 / 同時学習 99.4"),
    ("M2 得意分野", "取消の厳密さ（外して元に戻るか、0/1）", 3, 0, 1, 1, False, "完全に戻る", "実験07"),
    ("M2 得意分野", "足し算で 1 光路に融合（並列光路との差、点）", 7, 10, 0, 28.2, False, "差なし（1 次元の実験23）", "実験09 向き × 位置 59.9 / 88.1"),
    ("M2 得意分野", "未知の物（例 0 個）の認識（既知と未知の調和平均、%）", 8, 0, 80, None, False, "既知・未知とも 80%", "未測定"),
    ("M3 本物の画像", "手書き数字 MNIST（%）", 10, 10, 92.5, 94.4, False, "ロジスティック回帰", "実験15 板 ①"),
    ("M3 本物の画像", "Fashion-MNIST（%）", 10, 10, 84.4, 84.4, False, "ロジスティック回帰（実験15）", "実験15 板 ①"),
    ("M4 受け渡し", "SN2-Acoustics の 2 次元の入力の項目が測れる（0/1）", 10, 0, 1, None, False, "部品として呼べる", "未測定"),
    ("効率・アナログ", "推論の速さ（µs/件、64×64、GPU、小さいほど良い）", 5, 1000, 6, 45, True, "小さな CNN（実験01）", "実験01 板 ②"),
    ("効率・アナログ", "位相 4 ビットでの正解率の保ち（%、元に対する割合）", 5, 0, 98, 98.2, False, "1 次元の実験48 並み", "実験05 板 ① 向き × 位置 97.6 / 99.4"),
    ("効率・アナログ", "板が 1 画素ずれたときの保ち（%、元に対する割合）", 5, 0, 90, 18.5, False, "1 次元は 28%（実験48）", "実験05 板 ① 向き × 位置 18.4 / 99.4"),
]


def score(w, zero, goal, now, log):
    if now is None:
        return 0.0
    f = (lambda v: math.log10(v)) if log else (lambda v: v)
    r = (f(now) - f(zero)) / (f(goal) - f(zero))
    return w * min(max(r, 0.0), 1.0)


def to_md() -> str:
    rows, axes = [], {}
    total = measured = 0.0
    for ax, name, w, zero, goal, now, log, why, src in ITEMS:
        s = score(w, zero, goal, now, log)
        total += s
        measured += w if now is not None else 0
        a = axes.setdefault(ax, [0.0, 0.0])
        a[0] += s
        a[1] += w
        rows.append((ax, name, w, zero, goal, "未測定" if now is None else now, s, w - s, why, src))
    out = ["# SN2-Parts-2D の到達度（ポイント）", "",
           "各項目に「ゼロ点の基準」（何もしない場合）と「目標」（最低限の指標＝定番の手法の値）を置き、今の値がその間のどこにあるかで採点しています。",
           "計算式と数値は `tools/scoreboard.py` にあり、`python tools/scoreboard.py --md` で更新します。", "",
           f"## 合計: **{total:.1f} / 100 点**（測定済みの項目の配点 {measured:.0f} 点）", "",
           "| 軸 | 点 | 配点 | 達成率 |", "|---|---|---|---|"]
    for ax, (s, w) in axes.items():
        out.append(f"| {ax} | {s:.1f} | {w:g} | {100 * s / w:.0f}% |")
    out += ["", "## 項目ごと", "",
            "| 軸 | 項目 | 配点 | ゼロ点の基準 | 目標 | 今の値 | 点 | あと | 目標の根拠 | 出典 |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append("| {} | {} | {:g} | {:g} | {:g} | {} | {:.1f} | {:.1f} | {} | {} |".format(*r))
    out += ["", "## 伸びしろが大きい項目（残りの点が多い順）", ""]
    for r in sorted(rows, key=lambda r: -r[7])[:5]:
        out.append(f"- **{r[1]}**（{r[0]}）: あと {r[7]:.1f} 点")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true", help="docs/scoreboard.md を書き出す")
    a = ap.parse_args()
    text = to_md()
    if a.md:
        path = os.path.join(os.path.dirname(__file__), "..", "docs", "scoreboard.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.buffer.write(text.encode("utf-8"))   # Windows の端末の文字コードでも落ちないように
