"""数えた表での見積もり: 持ち主の案（2026-10-21）。2DTG-k =「k 語前の語 → 次の語」を別々に学び、
欲しい語の前の n 語それぞれの出力を掛け算して確率を出す（専門家の積・ナイーブベイズ）。
P(D | A, B, C) ∝ P(D) · Π_k [P_k(D | k 語前) / P(D)]^w_k 。w_k は重み（1 ならナイーブベイズそのもの）。
単語・文字の両方で。実行: python experiments/text/est_product_counts.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import textdata  # noqa: E402
import worddata  # noqa: E402


def run(name, X, y, Xt, yt, V, C, n):
    prior = torch.bincount(y, minlength=C).float() + 1
    prior = prior / prior.sum()
    logs = []
    for k in range(1, n + 1):
        T = torch.zeros(V, C)
        T.index_put_((X[:, n - k], y), torch.ones(len(y)), accumulate=True)
        P = (T + 0.1) / (T.sum(1, keepdim=True) + 0.1 * C)
        logs.append(torch.log(P[Xt[:, n - k]]))
    lp = torch.log(prior)

    def ev(s):
        p = torch.softmax(s, 1)
        return round(100 * (p.argmax(1) == yt).float().mean().item(), 1), round(-torch.log(p[torch.arange(len(yt)), yt]).mean().item(), 3)
    print(f"== {name}")
    print("  2DTG-1 だけ（直前 1 単位）", ev(logs[0]))
    for K in range(2, n + 1):
        nb = lp + sum(logs[k] - lp for k in range(K))
        print(f"  掛け算 {K} 個（ナイーブベイズ、重み 1）", ev(nb))
        best = None
        for w in (0.1, 0.2, 0.3, 0.5, 0.7):
            s = logs[0] + sum(w * (logs[k] - lp) for k in range(1, K))
            r = ev(s)
            if best is None or r[1] < best[1][1]:
                best = (w, r)
        print(f"  掛け算 {K} 個（2 つ目以降の重み {best[0]}）", best[1])


(X, y), (Xt, yt), info = worddata.load(4, 1024, 400000, 10000, 0, 4096)
run("単語（直前 4 語、1024 クラス）比べる: 2-gram 33.7・3-gram 36.4・GRU 36.9", X, y, Xt, yt, 4096, 1024, 4)
(X, y), (Xt, yt), info = textdata.load(8, 64, 600000, 10000, 0)
run("文字（直前 8 文字、64 クラス）比べる: 2-gram 42.9・3-gram 53.7・4-gram 55.4・GRU 54.1", X, y, Xt, yt, 512, 64, 8)


def run_pair(name, X, y, Xt, yt, V, C, n):
    """2 単位を見る表（直前 + 2 つ前 = 実験58 の 2DTG と同じ情報）に、遠い単位の 2DTG を掛け算で足す。"""
    prior = torch.bincount(y, minlength=C).float() + 1
    lp = torch.log(prior / prior.sum())
    key = X[:, n - 1] * V + X[:, n - 2]
    keyt = Xt[:, n - 1] * V + Xt[:, n - 2]
    uniq, inv = torch.unique(torch.cat([key, keyt]), return_inverse=True)
    T = torch.zeros(len(uniq), C)
    T.index_put_((inv[:len(key)], y), torch.ones(len(y)), accumulate=True)
    T1 = torch.zeros(V, C)
    T1.index_put_((X[:, n - 1], y), torch.ones(len(y)), accumulate=True)
    P1 = (T1 + 0.1) / (T1.sum(1, keepdim=True) + 0.1 * C)
    tt = T[inv[len(key):]]
    tot = tt.sum(1, keepdim=True)
    base = torch.log((tt + P1[Xt[:, n - 1]]) / (tot + 1))       # 2 単位の表（見ていない組は 1 単位に後退）
    far = []
    for k in range(3, n + 1):
        Tk = torch.zeros(V, C)
        Tk.index_put_((X[:, n - k], y), torch.ones(len(y)), accumulate=True)
        far.append(torch.log((Tk + 0.1) / (Tk.sum(1, keepdim=True) + 0.1 * C))[Xt[:, n - k]] - lp)

    def ev(s):
        p = torch.softmax(s, 1)
        return round(100 * (p.argmax(1) == yt).float().mean().item(), 1), round(-torch.log(p[torch.arange(len(yt)), yt]).mean().item(), 3)
    print(f"== {name}: 2 単位の表 × 遠い 2DTG の掛け算")
    print("  2 単位の表だけ", ev(base))
    for w in (0.1, 0.2, 0.3, 0.5):
        print(f"  × 遠い {len(far)} 個（重み {w}）", ev(base + w * sum(far)))


(X, y), (Xt, yt), info = worddata.load(4, 1024, 400000, 10000, 0, 4096)
run_pair("単語", X, y, Xt, yt, 4096, 1024, 4)
(X, y), (Xt, yt), info = textdata.load(8, 64, 600000, 10000, 0)
run_pair("文字", X, y, Xt, yt, 512, 64, 8)
