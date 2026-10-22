"""青空文庫の会話文と話者の一人称（実験71〜、持ち主の 2DWIG 案、2026-10-22）。

データ: 青空文庫（www.aozora.gr.jp、公式の作品一覧 list_person_all_extended_utf8.zip）から、作品・人物とも著作権フラグ「なし」、
       分類 913（日本の小説）、新字新仮名の作品を、会話の多い作家 17 人から 150 作品取得（data/aozora/raw/、リポジトリには入れない）。
       青空文庫の「収録ファイルの取り扱い規準」により、著作権の切れた作品は自由に利用できる。
処理: Shift_JIS のテキストを読み、ルビ《》・注記［＃…］・前後の解説を除き、「」の中を会話文として取り出す。
      会話文の中の一人称（下の FIRST）を「話者の正解」にする。一人称が 2 種類以上出る会話文は使わない。
"""
from __future__ import annotations

import json
import os
import re
import zipfile
from collections import Counter

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data", "aozora")

# 一人称（表記ゆれをまとめる）。長いものから順に照合する
FIRST = {
    "私": "私", "わたし": "わたし", "わたくし": "わたくし", "あたし": "あたし", "あたい": "あたい", "あっし": "あっし",
    "わっし": "わっし", "わし": "わし", "儂": "わし", "僕": "僕", "ぼく": "僕", "俺": "俺", "おれ": "俺", "おいら": "おいら",
    "おら": "おら", "己": "俺", "拙者": "拙者", "それがし": "それがし", "某": "それがし", "身共": "身共", "手前": "手前",
    "吾輩": "吾輩", "我輩": "吾輩", "わらわ": "わらわ", "妾": "わらわ", "小生": "小生", "自分": "自分", "うち": "うち",
    "我々": "我々", "われわれ": "我々", "私たち": "私たち", "僕ら": "僕ら", "俺たち": "俺たち", "わちき": "わちき", "朕": "朕",
}
_PAT = re.compile("|".join(sorted(map(re.escape, FIRST), key=len, reverse=True)))


def clean(text):
    text = re.sub(r"《[^》]*》", "", text)
    text = re.sub(r"［＃[^］]*］", "", text)
    text = text.replace("｜", "")
    parts = re.split(r"-{20,}", text)
    body = parts[2] if len(parts) >= 3 else text
    body = body.split("底本：")[0]
    return body


def works():
    meta = json.load(open(os.path.join(DATA, "picked.json"), encoding="utf-8"))
    out = []
    for m in meta:
        fn = os.path.join(DATA, "raw", f"{m['id']}.zip")
        if not os.path.exists(fn):
            continue
        z = zipfile.ZipFile(fn)
        txt = [n for n in z.namelist() if n.endswith(".txt")]
        if not txt:
            continue
        out.append(dict(m, text=clean(z.read(txt[0]).decode("cp932", errors="ignore"))))
    return out


def utterances(text):
    return [u.strip() for u in re.findall(r"「([^「」]{2,200})」", text)]


_tagger = None


def first_persons(u):
    """会話文の中の一人称。MeCab（unidic）で代名詞と判定されたものだけ（「うち割って」「おられた」などを除く）。"""
    global _tagger
    if _tagger is None:
        import fugashi
        _tagger = fugashi.Tagger()
    out = []
    for w in _tagger(u):
        if w.surface in FIRST and w.feature.pos1 == "代名詞":
            out.append(FIRST[w.surface])
    return out


def label(u):
    """会話文の一人称（1 種類だけのとき）。なければ None、2 種類以上なら False。"""
    found = set(first_persons(u))
    if not found:
        return None
    return found.pop() if len(found) == 1 else False


def stats():
    ws = works()
    n_utt = Counter()
    lab = Counter()
    for w in ws:
        us = utterances(w["text"])
        n_utt[w["author"]] += len(us)
        for u in us:
            l = label(u)
            if l:
                lab[l] += 1
    return ws, n_utt, lab
