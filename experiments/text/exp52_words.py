"""実験52: 単語（形態素）を単位にして、次の単語の確率を出す（持ち主の案、2026-10-20）

着想: 持ち主の案。レイヤへの入力を単語ごとのトークンにして、通った光から次の単語の確率を出す（「やぁ、→ こんにちは 20%、おはよう 15%」）。
      文字の実験は直前 2 文字ぶんしか使えずに頭打ち（実験40・41）。単位を単語にすれば、同じ「直前 2 単位」でも見える文脈が長くなる。
方法: 文を MeCab（fugashi + unidic-lite）で単語に切る（worddata.py）。直前 n = 4 語 → 次の 1 語。
      入力: 頻出 4094 語 + その他 + 文書の始まり（64 × 64 の 4096 通りの向きに 1 語ずつ）。帯に並べ、位置ごとの明るさを学習（実験33・40 と同じ）。
      光路: 板 ① 32 枚 × 区画 16 + レンズ + 窓 256 個の符号読み（実験42）。答えは頻出 1023 語 + その他 = 1024 クラス（窓の数に縛られない）。
      出口の明るさを softmax で確率にする（文字の実験と同じ）。
比べる相手: 単語の n-gram（2・3）、GRU、ロジスティック回帰（同じ入力・答え）。
指標: 正解率（一番上が当たる）、上位 5 に入る割合、クロスエントロピー。最後に例の文の、次の単語の上位 5 と確率を出す。
結果: results/exp52.txt

実行: python experiments/text/exp52_words.py --method plates --seed 0 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import sn2_parts_2d as p2  # noqa: E402
import worddata  # noqa: E402
from exp31_text2d import GRU, LogReg  # noqa: E402
import exp40_inputs  # noqa: E402
from exp40_inputs import EncNet  # noqa: E402

PROMPTS = ["やぁ、", "こんにちは。今日は", "よろしくお願い", "ありがとう", "東京の", "私は", "この商品は", "明日の天気は"]


def topk(S, info, k=5):
    p = torch.softmax(S, -1)
    v, i = p.topk(k)
    return [(info["out_vocab"][j], round(100 * float(x), 1)) for x, j in zip(v, i)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="plates", choices=["plates", "gru", "logreg", "ngram"])
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--C", type=int, default=1024)
    ap.add_argument("--V", type=int, default=4096)
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--layout", default="stride", choices=["stride", "embed", "embed_rand"], help="語を光の向きに並べる方法（実験60）。embed = 使われ方の近い語を近い向きに、embed_rand = 同じ並べ方を語の間でかき混ぜたもの（比較）")
    ap.add_argument("--z", type=float, default=64.0, help="板の間隔（実験59: 厚いホログラムに近づける）")
    ap.add_argument("--L", type=int, default=64, help="場の大きさ（実験56: 大きくして頭打ちを確かめる）")
    ap.add_argument("--patches", type=int, default=16)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--max_train", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp52.txt")
    a = ap.parse_args()
    exp40_inputs.L = a.L
    dev = torch.device(a.device)
    (X, y), (Xt, yt), info = worddata.load(a.n, a.C, a.max_train, 10000, a.seed, a.V)
    base = dict(exp="52", **vars(a), other_share=100.0 * (yt == a.C - 1).float().mean().item(),
                n_pairs_train=info["n_pairs_train"], n_word_types=info["n_word_types"])
    if a.method == "ngram":
        rec = dict(base)
        for o in (2, 3):
            acc, ce, t5 = worddata.ngram(info, o, a.n)
            rec.update({f"{o}gram_acc": acc, f"{o}gram_ce": ce, f"{o}gram_top5": t5})
    else:
        torch.manual_seed(a.seed)
        if a.method == "plates":
            m = EncNet("bands", a.n, a.V, a.C, a.layers, a.patches, a.seed, a.K)
            if a.layout != "stride":
                kx, ky, _ = worddata.embed_layout(info, a.L)
                if a.layout == "embed_rand":
                    perm = torch.randperm(len(kx), generator=torch.Generator().manual_seed(7))
                    kx, ky = kx[perm], ky[perm]
                m.kx.copy_(kx)
                m.ky.copy_(ky)
            if a.z != 64.0:
                m.net.H = p2.transfer(a.L, a.L, a.z)
        elif a.method == "gru":
            m = GRU(a.V, a.C)
        else:
            m = LogReg(a.V, a.C, a.n)
        m = m.to(dev)
        lr = 0.01 if a.method == "plates" else 0.003
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
        g = torch.Generator().manual_seed(a.seed)
        history, t_train, t_mark = [], 0.0, time.perf_counter()

        def scores(XX):
            with torch.no_grad():
                return torch.cat([m(XX[j:j + 250].to(dev)).cpu() for j in range(0, len(XX), 250)])

        def peek():
            r = {}
            for name, XX, yy in (("test", Xt[:2000], yt[:2000]), ("train", X[:2000], y[:2000])):
                S = scores(XX)
                r[f"{name}_acc"] = 100.0 * (S.argmax(1) == yy).float().mean().item()
                r[f"{name}_ce"] = F.cross_entropy(S, yy).item()
            return r
        for step in range(1, a.steps + 1):
            i = torch.randint(len(y), (256,), generator=g)
            loss = F.cross_entropy(m(X[i].to(dev)), y[i].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
            if step % 250 == 0 or step == a.steps:
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t_train += time.perf_counter() - t_mark
                history.append(dict(step=step, train_s=round(t_train, 2), loss=float(loss.detach()), lr=opt.param_groups[0]["lr"], **peek()))
                t_mark = time.perf_counter()
        S = scores(Xt)
        demo = {}
        for ptxt in PROMPTS:
            xp, ws = worddata.encode_prompt(ptxt, info, a.n)
            demo[ptxt] = dict(words=ws, top5=topk(scores(xp)[0], info))
        rec = dict(base, test_acc=100.0 * (S.argmax(1) == yt).float().mean().item(), test_ce=F.cross_entropy(S, yt).item(),
                   test_top5=100.0 * (S.topk(5, 1).indices == yt[:, None]).any(1).float().mean().item(),
                   params=sum(p.numel() for p in m.parameters() if p.requires_grad), train_s=t_train, history=history, demo=demo,
                   peak_MB=(torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else 0),
                   version=p2.__version__, torch=torch.__version__)
        if a.method == "plates":
            rec["pos_brightness"] = torch.exp(m.w.detach()).cpu().tolist()
    print(json.dumps(rec, ensure_ascii=False))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
