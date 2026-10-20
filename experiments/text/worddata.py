"""単語（形態素）単位の言語の課題（実験52〜）。

    データは文字の課題と同じ data/cc100_ja.txt（文書の先頭 85% を訓練、残りをテスト）。
    文を MeCab（fugashi + unidic-lite）で単語に切り、直前 n 語から次の 1 語を当てる。
    入力の語彙: 訓練で頻出の V_in−2 語 + その他 + 文書の始まり（BOS）。文書の始まりは BOS で埋めるので、短い文（「やぁ、」）からも予測できる。
    答え: 頻出の C−1 語 + その他。
    単語に切った結果は data/cc100_ja_words.txt に保存して使い回す（1 行 = 1 文書、単語はタブ区切り。リポジトリには入れない）。
"""
from __future__ import annotations

import os
import random
from collections import Counter, defaultdict

import torch

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")
PATH = os.path.join(DATA, "cc100_ja.txt")
WPATH = os.path.join(DATA, "cc100_ja_words.txt")

_tagger = None


def tokenize(text):
    global _tagger
    if _tagger is None:
        import fugashi
        _tagger = fugashi.Tagger()
    return [w.surface for w in _tagger(text) if w.surface.strip()]


def docs_words():
    if not os.path.exists(WPATH):
        docs = [l.strip() for l in open(PATH, encoding="utf-8") if l.strip()]
        with open(WPATH, "w", encoding="utf-8") as fo:
            for d in docs:
                fo.write("\t".join(w.replace("\t", " ") for w in tokenize(d)) + "\n")
    return [l.rstrip("\n").split("\t") for l in open(WPATH, encoding="utf-8") if l.strip()]


def load(n=4, C=1024, max_train=60000, max_test=10000, seed=0, V_in=4096):
    """戻り値: 訓練 (X, y)・テスト (X, y)・info。X は単語の番号 (N, n)、y は 0〜C−1。"""
    docs = docs_words()
    k = int(len(docs) * 0.85)
    tr_docs, te_docs = docs[:k], docs[k:]
    freq = Counter(w for d in tr_docs for w in d)
    out_vocab = [w for w, _ in freq.most_common(C - 1)] + ["<その他>"]
    out_idx = {w: i for i, w in enumerate(out_vocab[:-1])}
    in_vocab = [w for w, _ in freq.most_common(V_in - 2)] + ["<その他>", "<始まり>"]
    in_idx = {w: i for i, w in enumerate(in_vocab[:-2])}
    OTHER, BOS = V_in - 2, V_in - 1

    def pairs(ds):
        out = []
        for d in ds:
            ids = [BOS] * n + [in_idx.get(w, OTHER) for w in d]
            for i in range(len(d)):
                out.append((ids[i:i + n], out_idx.get(d[i], C - 1), d[max(0, i - n):i], d[i]))
        return out

    r = random.Random(seed)
    tr, te = pairs(tr_docs), pairs(te_docs)
    n_tr, n_te = len(tr), len(te)
    r.shuffle(tr)
    r.shuffle(te)
    tr, te = tr[:max_train], te[:max_test]
    enc = lambda ps: (torch.tensor([p[0] for p in ps]), torch.tensor([p[1] for p in ps]))
    info = dict(tr_docs=tr_docs, te_pairs=te, C=C, V_in=V_in, in_vocab=in_vocab, out_vocab=out_vocab, in_idx=in_idx, out_idx=out_idx,
                OTHER=OTHER, BOS=BOS, n_pairs_train=n_tr, n_pairs_test=n_te, n_word_types=len(freq),
                n_words_train=sum(len(d) for d in tr_docs), n_words_test=sum(len(d) for d in te_docs))
    return enc(tr), enc(te), info


def encode_prompt(text, info, n):
    """文 → 直前 n 語の入力の番号 (1, n)（足りない分は文書の始まり）。"""
    ws = tokenize(text)
    ids = [info["BOS"]] * n + [info["in_idx"].get(w, info["OTHER"]) for w in ws]
    return torch.tensor([ids[-n:]]), ws


def ngram(info, order, n):
    """後退つき n-gram（単語）。テストの正解率（%）・クロスエントロピー（nats）・上位 5 の正解率。"""
    import math
    C, idx = info["C"], info["out_idx"]
    tab = [defaultdict(Counter) for _ in range(order)]
    for d in info["tr_docs"]:
        pad = ["<始まり>"] * n + d
        for i in range(len(d)):
            y = idx.get(d[i], C - 1)
            for k in range(order):
                tab[k][tuple(pad[n + i - k:n + i])][y] += 1
    acc = ce = top5 = 0.0
    for _, y, ctxw, _ in info["te_pairs"]:
        ctx = ["<始まり>"] * n + list(ctxw)
        for k in range(order - 1, -1, -1):
            c = tab[k].get(tuple(ctx[len(ctx) - k:]) if k else ())
            if c and sum(c.values()) >= 2:
                break
        tot = sum(c.values())
        mc = [w for w, _ in c.most_common(5)]
        acc += mc[0] == y
        top5 += y in mc
        ce -= math.log((c.get(y, 0) + 0.1) / (tot + 0.1 * C))
    m = len(info["te_pairs"])
    return 100.0 * acc / m, ce / m, 100.0 * top5 / m
