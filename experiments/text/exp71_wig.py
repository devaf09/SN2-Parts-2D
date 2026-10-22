"""実験71: 2DWIG（2D Who am I Grain、持ち主の案、2026-10-22）— 会話文から、話者の一人称を当てる

データ: 青空文庫の著作権切れの小説 150 作品（aozoradata.py）。「」の会話文のうち、一人称（MeCab で代名詞と判定）が 1 種類だけのもの 3032 件。
        一人称を 12 の類にまとめる（私・僕・俺・わし・あたし・あっし・我々・わたくし・武士（拙者・身共・それがし）・おいら・わらわ・わちき）。
        作品で訓練・テストを分ける（同じ人物の台詞が両方に入らないように、30 作品をテスト）。
入力: 会話文の一人称を「＊」に置き換えて隠し、末尾の 12 文字（語尾・敬語・呼び方が多く出る所）を、帯に 1 文字ずつ並べる（実験33 と同じ入れ方）。
光路: 板 ① 32 枚 × 区画 16、窓 = 12 類（1 類 1 窓）、位置ごとの明るさを学習。3000 ステップ、学習率 0.01 → 0。
比べる相手: いつも一番多い類（34.2%）、数えた表のナイーブベイズ（語尾 1〜3 文字 + 文字 2-gram、51.9%）、ロジスティック回帰（同じ入力の one-hot）。
結果: results/exp71.txt
実行: python experiments/text/exp71_wig.py --method plates --seed 0 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import aozoradata as ad  # noqa: E402
import sn2_parts_2d as p2  # noqa: E402
from exp31_text2d import LogReg  # noqa: E402
from exp40_inputs import EncNet  # noqa: E402

GROUP = {"私": "私", "わたし": "私", "わたくし": "わたくし", "あたし": "あたし", "あたい": "あたし", "僕": "僕", "僕ら": "僕", "俺": "俺",
         "おいら": "おいら", "おら": "おいら", "俺たち": "俺", "わし": "わし", "拙者": "武士", "身共": "武士", "それがし": "武士", "吾輩": "わし",
         "わらわ": "わらわ", "あっし": "あっし", "わっし": "あっし", "わちき": "わちき", "我々": "我々", "私たち": "我々"}


def mask(u):
    for p in sorted(ad.FIRST, key=len, reverse=True):
        u = u.replace(p, "＊")
    return u


def load(n=12, V=512, seed=0, aug=0):
    """aug > 0: 訓練の会話文から、話者を変えた水増しを 1 文あたり aug 通り作って足す（aozora_aug.py、実験72）。テストは本物だけ。"""
    ws = ad.works()
    data = []
    for wi, w in enumerate(ws):
        for u in ad.utterances(w["text"]):
            l = ad.label(u)
            if l:
                data.append((wi, u, GROUP.get(l, l)))
    ids = list(range(len(ws)))
    random.Random(0).shuffle(ids)
    te_w = set(ids[:30] if len(ws) <= 150 else ids[:len(ws) * 15 // 100])
    tr = [d for d in data if d[0] not in te_w]
    te = [(wi, mask(u), g) for wi, u, g in data if wi in te_w]
    n_aug, aug_stat = 0, {}
    if aug:
        import aozora_aug
        extra, aug_stat = aozora_aug.build([(u, g) for _, u, g in tr], seed=seed, per_item=aug)
        n_aug = len(extra)
        tr = tr + [(-1, v, g) for v, g in extra]
    tr = [(wi, mask(u), g) for wi, u, g in tr]
    classes = [g for g, _ in Counter(g for _, _, g in tr).most_common()]
    ci = {g: i for i, g in enumerate(classes)}
    chars = [c for c, _ in Counter(c for _, u, _ in tr for c in u).most_common(V - 2)]
    vi = {c: i for i, c in enumerate(chars)}
    PAD, OTHER = V - 1, V - 2

    def enc(ds):
        X = torch.tensor([[PAD] * max(0, n - len(u)) + [vi.get(c, OTHER) for c in u[-n:]] for _, u, _ in ds])
        y = torch.tensor([ci[g] for _, _, g in ds])
        return X, y
    return enc(tr), enc(te), dict(classes=classes, n_train=len(tr), n_test=len(te), V=V, n_aug=n_aug, aug_stat=dict(aug_stat), tr=tr, te=te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="plates", choices=["plates", "logreg", "nb"])
    ap.add_argument("--aug", type=int, default=0, help="話者を変えた水増し（1 文あたりの数、実験72）")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--patches", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp71.txt")
    a = ap.parse_args()
    dev = torch.device(a.device)
    (X, y), (Xt, yt), info = load(a.n, seed=a.seed, aug=a.aug)
    C = len(info["classes"])
    if a.method == "nb":   # 数えた表のナイーブベイズ（語尾 1〜3 文字 + 文字の 2 つ組の袋、会話文の全体）
        import math
        feats = lambda u: ["E" + u[-k:] for k in (1, 2, 3)] + [u[i:i + 2] for i in range(len(u) - 1)]
        cls = Counter(g for _, _, g in info["tr"])
        Fq = {g: Counter() for g in cls}
        for _, u, g in info["tr"]:
            Fq[g].update(feats(u))
        tot = {g: sum(Fq[g].values()) for g in cls}
        Vf = len({f for g in Fq for f in Fq[g]})
        pred = [max(cls, key=lambda g: math.log(cls[g]) + sum(math.log((Fq[g][f] + 0.5) / (tot[g] + 0.5 * Vf)) for f in feats(u))) for _, u, _ in info["te"]]
        acc = 100.0 * sum(p == g for p, (_, _, g) in zip(pred, info["te"])) / len(pred)
        rec = dict(exp="71", **vars(a), classes=info["classes"], n_train=info["n_train"], n_test=info["n_test"], n_aug=info["n_aug"],
                   aug_stat=info["aug_stat"], test_acc=acc, majority=100.0 * (yt == 0).float().mean().item())
        print(json.dumps(rec, ensure_ascii=False))
        with open(a.out, "a", encoding="utf-8") as fo:
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return
    torch.manual_seed(a.seed)
    m = (EncNet("bands", a.n, info["V"], C, a.layers, a.patches, a.seed) if a.method == "plates" else LogReg(info["V"], C, a.n)).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=0.01 if a.method == "plates" else 0.003)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    g = torch.Generator().manual_seed(a.seed)
    t0 = time.perf_counter()
    for _ in range(a.steps):
        i = torch.randint(len(y), (128,), generator=g)
        loss = F.cross_entropy(m(X[i].to(dev)), y[i].to(dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
    train_s = time.perf_counter() - t0
    with torch.no_grad():
        S = m(Xt.to(dev)).cpu()
        Str = m(X.to(dev)).cpu()
    pred = S.argmax(1)
    conf = torch.zeros(C, C, dtype=torch.long)
    conf.index_put_((yt, pred), torch.ones(len(yt), dtype=torch.long), accumulate=True)
    rec = dict(exp="71", **vars(a), classes=info["classes"], n_train=info["n_train"], n_test=info["n_test"], n_aug=info["n_aug"], aug_stat=info["aug_stat"],
               test_acc=100.0 * (pred == yt).float().mean().item(), test_ce=F.cross_entropy(S, yt).item(),
               train_acc=100.0 * (Str.argmax(1) == y).float().mean().item(),
               majority=100.0 * (yt == 0).float().mean().item(), confusion=conf.tolist(),
               params=sum(p.numel() for p in m.parameters()), train_s=train_s, version=p2.__version__, torch=torch.__version__)
    print(json.dumps(rec, ensure_ascii=False))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
