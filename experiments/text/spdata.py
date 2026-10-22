"""サブワード（SentencePiece）単位の言語の課題（持ち主の判断 2026-10-22: 「その他」をなくすため、LLM と同じ割り当て方を流用する）。

    データは文字・単語の課題と同じ data/cc100_ja.txt（文書の先頭 85% を訓練、残りをテスト）。
    訓練の文書だけで SentencePiece（unigram、Apache 2.0、sentencepiece 0.2.2）を学び、語彙 V のサブワードに切る。
    byte_fallback = True なので、語彙にない文字はバイト（UTF-8 の 1 バイト = 1 単位）に分かれ、「その他」は出ない。
    直前 n 単位 → 次の 1 単位。入力・答えとも同じ語彙 V（入力にだけ「文書の始まり」を足す）。
    SentencePiece は文字の網羅率を 0.98 より下げられない（このデータでは 1376 種の文字・バイトが要る）ので、語彙は 2048 か 4096 にする。
    学んだ分け方は data/spm_{V}.model に保存して使い回す（リポジトリには入れない）。
"""
from __future__ import annotations

import os
import random

import torch

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")
PATH = os.path.join(DATA, "cc100_ja.txt")


def docs():
    ds = [l.strip() for l in open(PATH, encoding="utf-8") if l.strip()]
    k = int(len(ds) * 0.85)
    return ds[:k], ds[k:]


def model(V=2048, byte_fallback=True):
    import sentencepiece as spm
    path = os.path.join(DATA, f"spm_{V}{'' if byte_fallback else '_nobyte'}")
    if not os.path.exists(path + ".model"):
        tr, _ = docs()
        txt = os.path.join(DATA, "spm_train.txt")
        with open(txt, "w", encoding="utf-8") as fo:
            fo.write("\n".join(tr))
        # 語彙が小さいと全部の文字を入れられないので、文字の網羅率を下げていく（入らない文字はバイトに分かれる）
        for cov in ((0.9995, 0.999, 0.995, 0.99, 0.98) if byte_fallback else (1.0,)):
            try:
                spm.SentencePieceTrainer.train(input=txt, model_prefix=path, vocab_size=V, model_type="unigram",
                                               character_coverage=cov, byte_fallback=byte_fallback, add_dummy_prefix=False,
                                               max_sentence_length=100000, bos_id=-1, eos_id=-1, unk_id=0, pad_id=-1,
                                               minloglevel=2)
                with open(path + ".coverage", "w") as fo:
                    fo.write(str(cov))
                break
            except RuntimeError:
                continue
    return spm.SentencePieceProcessor(model_file=path + ".model")


def load(n=4, V=2048, max_train=60000, max_test=10000, seed=0, byte_fallback=True):
    """戻り値: 訓練 (X, y)・テスト (X, y)・info。X は単位の番号 (N, n)（0〜V−1、文書の始まり = V）、y は 0〜V−1。"""
    sp = model(V, byte_fallback)
    tr_docs, te_docs = docs()
    BOS = V

    def pairs(ds):
        out = []
        for d in ds:
            ids = sp.encode(d)
            pad = [BOS] * n + ids
            for i in range(len(ids)):
                out.append((pad[i:i + n], ids[i]))
        return out
    r = random.Random(seed)
    tr, te = pairs(tr_docs), pairs(te_docs)
    n_tr, n_te = len(tr), len(te)
    r.shuffle(tr)
    r.shuffle(te)
    tr, te = tr[:max_train], te[:max_test]
    enc = lambda ps: (torch.tensor([p[0] for p in ps]), torch.tensor([p[1] for p in ps]))
    pieces = [sp.id_to_piece(i) for i in range(V)]
    n_chars = sum(len(d) for d in tr_docs)
    covf = os.path.join(DATA, f"spm_{V}{'' if byte_fallback else '_nobyte'}.coverage")
    info = dict(V_in=V + 1, C=V, char_coverage=float(open(covf).read()) if os.path.exists(covf) else None, BOS=BOS, pieces=pieces, sp=sp, n_pairs_train=n_tr, n_pairs_test=n_te,
                chars_per_unit=n_chars / n_tr, unk_share=100.0 * sum(1 for _, yy in te if yy == 0) / max(1, len(te)),
                byte_share=100.0 * sum(1 for _, yy in te if pieces[yy].startswith("<0x")) / max(1, len(te)), tr_docs=tr_docs)
    return enc(tr), enc(te), info


def encode_prompt(text, info, n):
    ids = info["sp"].encode(text)
    pad = [info["BOS"]] * n + ids
    return torch.tensor([pad[-n:]]), [info["pieces"][i] for i in ids]
