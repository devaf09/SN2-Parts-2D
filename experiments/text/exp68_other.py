"""実験68: 「その他」に光が集まる問題（持ち主の依頼、2026-10-22）

着想: 単語の課題では、答えの 20%（文字では 36%）が「その他」（頻出語の外）で、自信がない場面ほど光路は「その他」を一番に出す。
      1 次元（Sin-CNN2）でも根本的には解決していない（窓ごとの背景光で事前分布を表せるようにしただけ、実験06b）。
方法: 単語の課題（直前の語 → 次の語、訓練は全部）。答えの語彙を 2048 に広げて測る（頻出 2047 語 + その他）。
  1 段目 = 実験58 の分担の 2DTG（16 個 × 板 8、頻出 1023 語 + その他）。
  案 1（表示で除く、目安）: 1 段目の確率から「その他」を除いて並べ直す。測るもの:
        ・その他でない答えの位置での正解率（その他を除いた一番上が当たるか）
        ・出力の上位 5 に「その他」が入る割合
  案 2（その他の中を受け持つ 2DTG を足す）: 2 段目 = 答えが「その他」の文脈だけで学習する分担の 2DTG（16 個 × 板 8）。
        答えは次の頻出 1024 語（順位 1024〜2047）+「まだその他」。
        融合: P(頻出語) = 1 段目、P(まれな語) = 1 段目の P(その他) × 2 段目の P(まれな語 | その他)、P(まだその他) = 同じく。
  比べる相手: 1 段だけ（まれな語は全部「その他」、2048 語彙の物差しでは外れ）、1 本で 2048 語を直接当てる分担の 2DTG（16 個 × 板 8、C = 2048）。
指標: 2048 語彙での正解率・CE・上位 5、その他でない位置での正解率、まれな語（順位 1024〜2047）の位置での正解率。
結果: results/exp68.txt
実行: python experiments/text/exp68_other.py --seed 0 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import sn2_parts_2d as p2  # noqa: E402
import worddata  # noqa: E402
from exp52_words import PROMPTS  # noqa: E402
from exp61_tree import level_probs, train_grain  # noqa: E402

C1, C2 = 1024, 1024           # 1 段目の答え（頻出 1023 + その他）、2 段目の答え（順位 1024〜2046 の 1023 語 + まだその他）
CF = C1 + C2 - 1              # 2048 語彙（0〜2046 が語、2047 がまだその他）


def split_train(X, y, C, G, layers, steps, seed, dev, mask=None):
    """直前の語の余りで G 個に分担した 2DTG を、担当なしの学習で別々に学ぶ。mask があれば、その位置だけを使う。"""
    if mask is not None:
        X, y = X[mask], y[mask]
    grp = X[:, -1] % G
    return [train_grain(X, y, (grp == g).nonzero().squeeze(1), (grp != g).nonzero().squeeze(1), C, layers, 16, 256,
                        seed * 1000 + G * 10 + g, steps, 0.3, dev) for g in range(G)]


def ev(p, yy, mask=None):
    if mask is not None:
        p, yy = p[mask], yy[mask]
    nll = -torch.log(p[torch.arange(len(yy)), yy] + 1e-12)
    return dict(acc=100.0 * (p.argmax(1) == yy).float().mean().item(), ce=nll.mean().item(),
                top5=100.0 * (p.topk(5, 1).indices == yy[:, None]).any(1).float().mean().item(), n=len(yy))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--G", type=int, default=16)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--max_train", type=int, default=400000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--direct", action="store_true", help="比べる相手（2048 語を直接当てる分担の 2DTG）も学習する")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp68.txt")
    a = ap.parse_args()
    dev = torch.device(a.device)
    (X, yF), (Xt, ytF), info = worddata.load(2, CF, a.max_train, 10000, a.seed, 4096)     # yF: 0〜2047
    to1 = lambda yy: torch.where(yy < C1 - 1, yy, torch.full_like(yy, C1 - 1))             # 1 段目の答え
    to2 = lambda yy: (yy - (C1 - 1)).clamp(min=0)                                          # 2 段目の答え（0〜1024）
    t0 = time.perf_counter()
    stage1 = split_train(X, to1(yF), C1, a.G, a.layers, a.steps, a.seed, dev)
    other_mask = yF >= C1 - 1
    stage2 = split_train(X, to2(yF), C2, a.G, a.layers, a.steps, a.seed + 50, dev, mask=other_mask)
    direct = split_train(X, yF, CF, a.G, a.layers, a.steps, a.seed + 90, dev) if a.direct else None
    train_s = time.perf_counter() - t0

    P1 = level_probs(stage1, Xt, C1, dev)                      # (N, 1024)
    P2 = level_probs(stage2, Xt, C2, dev)                      # (N, 1024)
    only1 = torch.zeros(len(ytF), CF)
    only1[:, :C1 - 1] = P1[:, :C1 - 1]
    only1[:, CF - 1] = P1[:, C1 - 1]                            # 1 段だけ: まれな語は全部「まだその他」に入れる
    tree = torch.zeros(len(ytF), CF)
    tree[:, :C1 - 1] = P1[:, :C1 - 1]
    tree[:, C1 - 1:] = P1[:, C1 - 1:C1] * P2                    # P(その他) × P(まれな語 | その他)
    not_other, rare = ytF < CF - 1, (ytF >= C1 - 1) & (ytF < CF - 1)
    res = {}
    for name, p in (("stage1_only", only1), ("tree", tree)) + ((("direct", level_probs(direct, Xt, CF, dev)),) if direct else ()):
        res[name] = dict(all=ev(p, ytF), not_other=ev(p, ytF, not_other), rare=ev(p, ytF, rare))
    # 案 1: 1 段目から「その他」を除いて並べ直す（目安）
    y1 = to1(ytF)
    m1 = y1 < C1 - 1
    q = P1[:, :C1 - 1] / P1[:, :C1 - 1].sum(1, keepdim=True)
    res["exclude_other"] = dict(
        other_share_of_answers=100.0 * (~m1).float().mean().item(),
        other_is_top1=100.0 * (P1.argmax(1) == C1 - 1).float().mean().item(),
        other_in_top5=100.0 * (P1.topk(5, 1).indices == C1 - 1).any(1).float().mean().item(),
        acc_with_other_on_not_other=100.0 * (P1.argmax(1)[m1] == y1[m1]).float().mean().item(),
        acc_excluding_other_on_not_other=100.0 * (q.argmax(1)[m1] == y1[m1]).float().mean().item(),
        top5_excluding_other_on_not_other=100.0 * (q.topk(5, 1).indices[m1] == y1[m1][:, None]).any(1).float().mean().item())
    demo = {}
    for ptxt in PROMPTS:
        xp, ws = worddata.encode_prompt(ptxt, info, 2)
        pt = torch.zeros(CF)
        p1 = level_probs(stage1, xp, C1, dev)[0]
        pr2 = level_probs(stage2, xp, C2, dev)[0]
        pt[:C1 - 1] = p1[:C1 - 1]
        pt[C1 - 1:] = p1[C1 - 1] * pr2
        v, i = pt.topk(5)
        v1, i1 = (p1[:C1 - 1] / p1[:C1 - 1].sum()).topk(5)
        demo[ptxt] = dict(words=ws, tree=[(info["out_vocab"][j], round(100 * float(x), 1)) for x, j in zip(v, i)],
                          exclude_other=[(info["out_vocab"][j], round(100 * float(x), 1)) for x, j in zip(v1, i1)])
    rec = dict(exp="68", **vars(a), results=res, demo=demo, train_s=train_s, version=p2.__version__, torch=torch.__version__)
    print(json.dumps(rec, ensure_ascii=False))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
