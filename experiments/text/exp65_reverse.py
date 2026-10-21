"""実験65: 光路を逆向きに通すと、「D の前に来やすい語」が光るか（持ち主の問い、2026-10-21）

着想: 持ち主の問い。A B C の次が D と学習した光路に、出口から D を入れると A B C の反応が出るか。
      位相の板・回折・レンズは光を失わない変換なので、出口の光を位相まで逆に送れば入口の光に戻る（位相共役）。
      ただし答え D は窓の明るさの模様（符号）でしか表していないので、位相の情報がない。
方法: 実験63 で保存した光路（直前 1 語 → 次の語、板 32 枚）を使う。
      出口に「D の符号のうち正の値を、その窓に置いた明るさ」の光（位相 0）を置き、レンズ・回折・板を逆向き（随伴: 共役の位相、逆 FFT）に通す。
      入口の光を語ごとの光の向きに分け（入口でのフーリエ変換）、その向きの強さを「逆向きの反応」とする。
      比べるもの: 本当の「D の前に来やすい語」の割合 P(前の語 | 次 = D)（訓練の数えた表）。
        reverse  = 逆向きに通した反応
        forward  = 順向きの光路で、全部の語を 1 つずつ入れて P(D | 前の語) を出し、前の語の頻度を掛けたもの（ベイズで裏返す。計算は語の数だけ）
        random   = でたらめ（偶然の基準）
      指標: 上位 20 語の重なり（本当の上位 20 語のうち何語が入るか）、順位相関（スピアマン）。D は頻度の高い 50 語。
結果: results/exp65.txt

実行: python experiments/text/exp65_reverse.py --model results/models/exp63_phase_int_s0.pt --plate phase --read int
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import sn2_parts_2d as p2  # noqa: E402
import worddata  # noqa: E402
from exp63_amp_phase import AmpPhaseNet  # noqa: E402


def spearman(a, b):
    ra, rb = a.argsort().argsort().float(), b.argsort().argsort().float()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--plate", default="phase")
    ap.add_argument("--read", default="int")
    ap.add_argument("--targets", type=int, default=50)
    ap.add_argument("--out", default="results/exp65.txt")
    a = ap.parse_args()
    (X, y), _, info = worddata.load(1, 1024, 400000, 100, 0, 4096)
    V, C, L = 4096, 1024, 64
    m = AmpPhaseNet(a.plate, a.read, 1, V, C)
    m.load_state_dict(torch.load(a.model))
    m.eval()
    net = m.net
    D = len(net.kinds)
    K = net.det.shape[0]
    # 本当の P(前の語 | 次 = t)
    cnt = torch.zeros(C, V)
    cnt.index_put_((y, X[:, 0]), torch.ones(len(y)), accumulate=True)
    prev_freq = torch.bincount(X[:, 0], minlength=V).float()
    xs = torch.arange(L, dtype=torch.float32)
    res = dict(reverse=[], forward=[], random=[])
    examples = {}
    with torch.no_grad():
        # 順向き: 全部の語を入れたときの P(t | 前の語)
        S = torch.cat([torch.softmax(m(torch.arange(j, min(j + 256, V))[:, None]), -1) for j in range(0, V, 256)])   # (V, C)
        for t in range(min(a.targets, C - 1)):
            truth = cnt[t]
            if truth.sum() < 50:
                continue
            code = m.codes[t, :K].clamp(min=0)
            A = (code[:, None, None] * net.det).sum(0)                   # 出口に置く明るさ
            u = torch.sqrt(A).to(torch.complex64)
            u = torch.fft.ifft2(torch.fft.ifftshift(u, dim=(-2, -1)), norm="ortho")    # レンズの逆
            for d in reversed(range(D)):
                u = torch.fft.ifft2(torch.fft.fft2(u) * net.H.conj())    # 回折の逆
                if a.plate in ("amp", "both"):
                    u = u * torch.exp(0.5 * m.ampnet.plate_phase(d))
                if a.plate in ("phase", "both"):
                    u = u * torch.polar(torch.ones(L, L), -net.plate_phase(d))
            F = torch.fft.fft2(u)                                         # 入口の光を向きごとに分ける
            # 語 w の入口の光は exp(j2π(kx·x + ky·y)/L) なので、向き (kx, ky) の成分の強さが反応
            rev = F.abs()[m.ky.long(), m.kx.long()] ** 2
            fwd = S[:, t] * prev_freq
            rnd = torch.rand(V, generator=torch.Generator().manual_seed(t))
            top = set(truth.topk(20).indices.tolist())
            for name, sc in (("reverse", rev), ("forward", fwd), ("random", rnd)):
                res[name].append(dict(top20=len(top & set(sc.topk(20).indices.tolist())), spearman=spearman(sc, truth)))
            if t < 8:
                iv = info["in_vocab"]
                examples[info["out_vocab"][t]] = dict(truth=[iv[i] for i in truth.topk(8).indices.tolist()],
                                                      reverse=[iv[i] for i in rev.topk(8).indices.tolist()],
                                                      forward=[iv[i] for i in fwd.topk(8).indices.tolist()])
    summ = {k: dict(top20=sum(r["top20"] for r in v) / len(v), spearman=sum(r["spearman"] for r in v) / len(v), n=len(v)) for k, v in res.items()}
    rec = dict(exp="65", **vars(a), summary=summ, examples=examples)
    print(json.dumps(rec, ensure_ascii=False))
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
