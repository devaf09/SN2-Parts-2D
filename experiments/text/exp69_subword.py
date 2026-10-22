"""実験69: サブワード（SentencePiece）で「その他」をなくした課題の比べる相手（持ち主の判断、2026-10-22）

方法: spdata.py（SentencePiece unigram、語彙 V = 2048 / 4096、byte_fallback でその他なし）。直前 4 単位 → 次の 1 単位（V クラス）。
      ここでは比べる相手（n-gram、GRU、光路 1 本）を測る。分担の 2DTG は exp58_split.py --task sp。
      物差し: 単位の正解率・CE に加えて、単位の長さが V で違うので「1 文字あたりの CE」（CE ÷ 1 単位あたりの文字数）。
結果: results/exp69.txt
実行: python experiments/text/exp69_subword.py --method gru --V 4096 --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import sn2_parts_2d as p2  # noqa: E402
import spdata  # noqa: E402
from exp31_text2d import GRU  # noqa: E402
from exp40_inputs import EncNet  # noqa: E402
from exp52_words import PROMPTS  # noqa: E402


def ngram(X, y, Xt, yt, C, order):
    """後退つき n-gram（加算スムージング）。order = 2 なら直前 1 単位。"""
    n = X.shape[1]
    tab = [defaultdict(Counter) for _ in range(order)]
    for x, t in zip(X.tolist(), y.tolist()):
        for k in range(order):
            tab[k][tuple(x[n - k:]) if k else ()][t] += 1
    acc = ce = 0.0
    for x, t in zip(Xt.tolist(), yt.tolist()):
        for k in range(order - 1, -1, -1):
            c = tab[k].get(tuple(x[n - k:]) if k else ())
            if c and sum(c.values()) >= 2:
                break
        tot = sum(c.values())
        acc += c.most_common(1)[0][0] == t
        ce -= math.log((c.get(t, 0) + 0.1) / (tot + 0.1 * C))
    return 100.0 * acc / len(yt), ce / len(yt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="gru", choices=["ngram", "gru", "plates"])
    ap.add_argument("--V", type=int, default=4096)
    ap.add_argument("--layers", type=int, default=128)
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--max_train", type=int, default=600000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp69.txt")
    a = ap.parse_args()
    dev = torch.device(a.device)
    (X, y), (Xt, yt), info = spdata.load(4, a.V, a.max_train, 10000, a.seed)
    C, cpu_ = info["C"], info["chars_per_unit"]
    base = dict(exp="69", **vars(a), chars_per_unit=cpu_, char_coverage=info["char_coverage"], byte_share=info["byte_share"],
                n_pairs_train=info["n_pairs_train"])
    if a.method == "ngram":
        rec = dict(base)
        for o in (2, 3, 4):
            acc, ce = ngram(X, y, Xt, yt, C, o)
            rec.update({f"{o}gram_acc": acc, f"{o}gram_ce": ce, f"{o}gram_ce_per_char": ce / cpu_})
    else:
        torch.manual_seed(a.seed)
        m = (GRU(info["V_in"], C) if a.method == "gru" else EncNet("bands", 4, info["V_in"], C, a.layers, 16, a.seed, a.K)).to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=0.003 if a.method == "gru" else 0.01)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
        g = torch.Generator().manual_seed(a.seed)
        t0 = time.perf_counter()
        for _ in range(a.steps):
            i = torch.randint(len(y), (256,), generator=g)
            loss = F.cross_entropy(m(X[i].to(dev)), y[i].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        train_s = time.perf_counter() - t0
        with torch.no_grad():
            S = torch.cat([m(Xt[j:j + 250].to(dev)).cpu() for j in range(0, len(yt), 250)])
            demo = {}
            for ptxt in PROMPTS:
                xp, ws = spdata.encode_prompt(ptxt, info, 4)
                v, i = torch.softmax(m(xp.to(dev))[0].cpu(), -1).topk(5)
                demo[ptxt] = dict(units=ws, top5=[(info["pieces"][j], round(100 * float(x), 1)) for x, j in zip(v, i)])
        ce = F.cross_entropy(S, yt).item()
        rec = dict(base, test_acc=100.0 * (S.argmax(1) == yt).float().mean().item(), test_ce=ce, test_ce_per_char=ce / cpu_,
                   test_top5=100.0 * (S.topk(5, 1).indices == yt[:, None]).any(1).float().mean().item(),
                   params=sum(p.numel() for p in m.parameters()), train_s=train_s, demo=demo, version=p2.__version__, torch=torch.__version__)
    print(json.dumps(rec, ensure_ascii=False))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
