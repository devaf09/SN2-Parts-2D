"""話者を変えた水増しと、不自然な水増しの除外（持ち主の案、2026-10-23）。

1. 置き換え: 会話文の一人称（MeCab で代名詞のもの）を、目標の話者の類の一人称に置き換える。文末が次の 3 つの組のどれかで終わるときは、
   文末も目標の話者の言い方に置き換える（言い方は、本物の会話文で目標の類が実際に使った頻度で選ぶ）。
     断定: だ / です / じゃ / でござる / でさ / だわ / ですわ
     説明: のだ / んだ / のです / んです / のじゃ / んじゃ / のでござる / のよ / んでさ
     推量: だろう / でしょう / じゃろう / でござろう / だろ
   文末が組のどれでもない文、文末の言い方が変わらない文は使わない（一人称は 2DWIG の入力では隠すので、一人称だけ変えた文は答えだけ違う雑音になる）。
   「ません」→「ない」のような活用の変化は扱わない。
2. 除外: 本物の会話文（訓練の作品だけ）で数えた文字の 5-gram で、置き換えた文の末尾 10 文字の自然さ（1 文字あたりの対数確率）を測り、
   本物の会話文の同じ値の下位 10% より低いものを捨てる。さらに、置き換えた後の「一人称の類と文末の言い方」の組が、
   本物の会話文に 3 回以上出てこないものも捨てる。
"""
from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict

import aozoradata as ad

ENDGROUPS = {
    "断定": ["でござる", "ですわ", "だわ", "です", "じゃ", "でさ", "だ"],
    "説明": ["のでござる", "んでさ", "のです", "んです", "のじゃ", "んじゃ", "のだ", "んだ", "のよ"],
    "推量": ["でござろう", "じゃろう", "でしょう", "だろう", "だろ"],
}
_TAIL = r"([よねなぜさわいかぞ]{0,2})([。！？!?…―ー、\s]*)$"
TARGETS = ["私", "僕", "俺", "わし", "あたし", "武士", "わたくし", "あっし", "おいら"]


def ending(u):
    """(組, 言い方, 前の部分, 後ろの終助詞と記号)。組に当てはまらなければ None。"""
    for grp, forms in ENDGROUPS.items():
        for f in sorted(forms, key=len, reverse=True):
            m = re.search(re.escape(f) + _TAIL, u)
            if m:
                return grp, f, u[:m.start()], m.group(1) + m.group(2)
    return None


class CharLM:
    def __init__(self, texts, n=5):
        self.n = n
        self.c = [defaultdict(Counter) for _ in range(n)]
        for t in texts:
            s = "^" * (n - 1) + t
            for i in range(n - 1, len(s)):
                for k in range(n):
                    self.c[k][s[i - k:i]][s[i]] += 1
        self.V = len(self.c[0][""])

    def logp(self, t, last=10):
        s = "^" * (self.n - 1) + t
        lp, cnt = 0.0, 0
        for i in range(max(self.n - 1, len(s) - last), len(s)):
            p, w = 0.0, 1.0
            for k in range(self.n - 1, -1, -1):   # 素朴な後退の混ぜ合わせ
                c = self.c[k].get(s[i - k:i])
                if c:
                    tot = sum(c.values())
                    p += w * 0.6 * c.get(s[i], 0) / tot
                    w *= 0.4
            p += w / self.V
            lp += math.log(p)
            cnt += 1
        return lp / max(1, cnt)


def build(train_items, seed=0, per_item=2):
    """train_items: [(会話文（一人称はそのまま）, 類)]。戻り値: 水増しした [(文, 類)] と、除外の内訳。"""
    from exp71_wig import GROUP
    rnd = random.Random(seed)
    pron = defaultdict(Counter)            # 類 → 一人称の書き方の頻度
    endf = defaultdict(lambda: defaultdict(Counter))   # 類 → 組 → 言い方の頻度
    for u, g in train_items:
        for p in ad.first_persons(u):
            pron[g][p] += 1
        e = ending(u)
        if e:
            endf[g][e[0]][e[1]] += 1
    surf = {g: Counter() for g in pron}
    for u, g in train_items:               # 一人称の表記（ぼく・おれ など）も本物の頻度で
        for w in ad._tagger(u):
            if w.surface in ad.FIRST and w.feature.pos1 == "代名詞":
                surf[g][w.surface] += 1
    lm = CharLM([u for u, _ in train_items])
    real_scores = sorted(lm.logp(u) for u, _ in train_items[:3000])
    thr = real_scores[len(real_scores) // 10]
    combo = Counter((g, e[1]) for u, g in train_items if (e := ending(u)))
    out, stat = [], Counter()

    def pick(c):
        items, w = zip(*c.items())
        return rnd.choices(items, weights=w)[0]
    for u, g in train_items:
        for t in rnd.sample([x for x in TARGETS if x != g and surf.get(x)], k=min(per_item, len(TARGETS) - 1)):
            toks = [(w.surface, w.feature.pos1) for w in ad._tagger(u)]
            v = "".join(pick(surf[t]) if (s in ad.FIRST and p == "代名詞") else s for s, p in toks)
            e = ending(v)
            if not (e and endf[t].get(e[0])):
                stat["文末を置き換えられない"] += 1   # 一人称だけ変えた文は、一人称を隠すと元の文と同じになり、答えだけ違う雑音になるので使わない
                continue
            f2 = pick(endf[t][e[0]])
            if f2 == e[1]:
                stat["文末が同じ"] += 1
                continue
            v = e[2] + f2 + e[3]
            if combo[(t, f2)] < 3:
                stat["組み合わせが本物にほぼない"] += 1
                continue
            if lm.logp(v) < thr:
                stat["末尾が不自然"] += 1
                continue
            stat["採用"] += 1
            out.append((v, t))
    return out, stat
