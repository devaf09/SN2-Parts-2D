"""言語の実験の課題（Sin-CNN2 の exp06_nextchar.load と同じ作り方。1 次元の結果とそのまま比べるため）。

    文書（1 行 = 1 文書）の先頭 85% を訓練、残りをテスト。直前 n 文字から次の 1 文字を当てる。
    答えは訓練データで頻出の 63 字 + その他 の 64 クラス。訓練・テストの組をそれぞれシャッフルして先頭を使う。
    データ: data/cc100_ja.txt（CC-100 日本語の先頭 約 60 万字。Sin-CNN2 が取得したものを写して使う。リポジトリには入れない）
"""
from __future__ import annotations

import os
import random
from collections import Counter, defaultdict

import torch

PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cc100_ja.txt")


def load(n=8, C=64, max_train=60000, max_test=10000, seed=0, V_in=512, path=PATH):
    """戻り値: 訓練 (X, y)・テスト (X, y)、X は文字の番号 (N, n)（入力の文字は頻出 V_in−1 字 + その他）、y は 0〜C−1。
    ほかに n-gram 用の訓練文書と、テストの (文脈の文字列, 答え) の組。"""
    docs = [l.strip() for l in open(path, encoding="utf-8") if l.strip()]
    k = int(len(docs) * 0.85)
    tr_docs, te_docs = docs[:k], docs[k:]
    freq = Counter("".join(tr_docs))
    out_vocab = [c for c, _ in freq.most_common(C - 1)]
    out_idx = {c: i for i, c in enumerate(out_vocab)}
    in_vocab = [c for c, _ in freq.most_common(V_in - 1)]
    in_idx = {c: i for i, c in enumerate(in_vocab)}

    def pairs(ds):
        out = []
        for d in ds:
            for i in range(n, len(d)):
                out.append((d[i - n:i], out_idx.get(d[i], C - 1)))
        return out

    r = random.Random(seed)
    tr, te = pairs(tr_docs), pairs(te_docs)
    r.shuffle(tr)
    r.shuffle(te)
    tr, te = tr[:max_train], te[:max_test]
    enc = lambda ps: (torch.tensor([[in_idx.get(c, V_in - 1) for c in x] for x, _ in ps]), torch.tensor([y for _, y in ps]))
    return enc(tr), enc(te), dict(tr_docs=tr_docs, te_pairs=te, out_idx=out_idx, C=C, V_in=V_in, in_vocab=in_vocab)


def ngram(info, order):
    """後退つき n-gram（加算スムージング、Sin-CNN2 と同じ）。テストの正解率（%）とクロスエントロピー（nats）。"""
    import math
    C, idx = info["C"], info["out_idx"]
    tab = [defaultdict(Counter) for _ in range(order)]
    for d in info["tr_docs"]:
        ys = [idx.get(c, C - 1) for c in d]
        for i in range(len(d)):
            for k in range(order):
                if i - k < 0:
                    break
                tab[k][d[i - k:i]][ys[i]] += 1
    acc = ce = 0.0
    for x, y in info["te_pairs"]:
        for k in range(order - 1, -1, -1):
            ctx = x[len(x) - k:] if k else ""
            c = tab[k].get(ctx)
            if c and sum(c.values()) >= 2:
                break
        tot = sum(c.values())
        acc += c.most_common(1)[0][0] == y
        ce -= math.log((c.get(y, 0) + 0.1) / (tot + 0.1 * C))
    n = len(info["te_pairs"])
    return 100.0 * acc / n, ce / n
