"""実験54: 関連度グレイン — 話題の近さで、内容語の確率を掛け直す（持ち主の案、2026-10-20）

着想: 持ち主の案。確率だけでなく「関連度」（海・湖・池は近く、山・街・砂漠は遠い）を並行して出し、確率 × 関連度で決める。
      数えた表での見積もり（est_relatedness_counts.py）: 同じ文の近くに一緒に出る回数から作った意味のベクトル（64 個の数）で、
      直前 50 語の話題との近さを関連度にすると、内容語の位置の正解率が 11 → 13〜22%。ただし次の語の 65% は機能語（、の て）で、
      全部に掛けると機能語の正しい予測を押しのける。「内容語の確率の合計は変えず、内容語の中だけ並べ替える」が一番良かった（全体 37.2 → 37.3%、CE 3.48 → 3.43）。
方法: 関連度グレイン = 直前 H = 50 語のうち内容語（訓練で出現 2000 回未満の語）を、語ごとの向きの平面波として場全体に重ねて入れる（話題の袋。光の足し算そのもの）
      → 板 ① 16 枚 × 区画 16 → レンズ → 窓 64 個の明るさ h。答えの語ごとに符号（64 個の数、学習）を持ち、関連度 = h と符号のコサイン。
      学習: 次の語が内容語の位置で、内容語の中での softmax（関連度 × 温度）のクロスエントロピー。
      合わせ方: 確率（2DTG の改良形を数えた表で作ったもの、実験53 の見積もりと同じ）の、内容語の部分だけを exp(λ·関連度) で掛け直し、
      内容語の確率の合計は元のまま。λ は訓練から除いた 5000 件で選ぶ。
比べる相手: 確率だけ、数えた表の関連度（共起 → PPMI → 64 次元）。
結果: results/exp54.txt

実行: python experiments/text/exp54_related.py --seed 0 --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import sn2_parts_2d as p2  # noqa: E402
import worddata  # noqa: E402
from exp31_text2d import stride  # noqa: E402

L = 64


class RelGrain(torch.nn.Module):
    def __init__(self, V, C, K=64, layers=16, patches=16, seed=0):
        super().__init__()
        self.net = p2.Net2D(L, K, ["xy"] * layers, patches=patches, seed=seed)
        g = stride(L * L)
        pos = (torch.arange(V) * g) % (L * L)
        self.register_buffer("kx", (pos % L).float())
        self.register_buffer("ky", (pos // L).float())
        y = torch.arange(L, dtype=torch.float32)
        self.register_buffer("yy", y[:, None].expand(L, L).clone())
        self.register_buffer("xx", y[None, :].expand(L, L).clone())
        self.codes = torch.nn.Parameter(torch.randn(C, K, generator=torch.Generator().manual_seed(seed)))
        self.temp = torch.nn.Parameter(torch.tensor(10.0))

    def topic(self, H, M):
        """H (B, h) 語の番号、M (B, h) 有効か → 窓の明るさ (B, K)。話題の袋を場全体に重ねて入れる。"""
        ph = 2 * math.pi * (self.kx[H][..., None, None] * self.xx + self.ky[H][..., None, None] * self.yy) / L
        u = (torch.polar(torch.ones_like(ph), ph) * M[..., None, None]).sum(1)
        u = u / (u.abs().pow(2).mean((-2, -1), keepdim=True) + 1e-12).sqrt()
        return p2.read_windows(self.net.screen(u), self.net.det)

    def related(self, H, M):
        """関連度（コサイン）(B, C)。"""
        return p2.unit_rows(self.topic(H, M)) @ p2.unit_rows(self.codes).T


def build(n, C, V, Hn, seed, max_train, n_val, n_test):
    docs = worddata.docs_words()
    k = int(len(docs) * 0.85)
    tr, te = docs[:k], docs[k:]
    freq = Counter(x for d in tr for x in d)
    out_vocab = [x for x, _ in freq.most_common(C - 1)] + ["<その他>"]
    oi = {x: i for i, x in enumerate(out_vocab[:-1])}
    vocab = [x for x, _ in freq.most_common(V - 1)]
    vi = {x: i for i, x in enumerate(vocab)}
    content = torch.tensor([freq[x] < 2000 for x in out_vocab[:-1]] + [False])

    def positions(ds):
        out = [(di, i) for di, d in enumerate(ds) for i in range(len(d))]
        random.Random(seed).shuffle(out)
        return out

    def hist(d, i):
        h = [vi[x] for x in d[max(0, i - Hn):i] if x in vi and freq[x] < 2000]
        return (h + [0] * Hn)[:Hn], ([1.0] * len(h) + [0.0] * Hn)[:Hn]

    # 確率（2DTG の改良形 = (直前, k 語前) → 次の語、数えた表、割り当て 0.4・0.3・0.2・0.1）
    Ts = [defaultdict(Counter) for _ in range(n)]
    U1 = defaultdict(Counter)
    for d in tr:
        pad = ["<s>"] * n + d
        for i in range(len(d)):
            yy = oi.get(d[i], C - 1)
            U1[pad[n + i - 1]][yy] += 1
            for kk in range(1, n + 1):
                Ts[kk - 1][(pad[n + i - 1], pad[n + i - kk])][yy] += 1

    def base(d, i):
        pad = ["<s>"] * n + d
        u = torch.full((C,), 0.1 / C)
        for yy, v in U1.get(pad[n + i - 1], {}).items():
            u[yy] += v
        u = u / u.sum()
        p = torch.zeros(C)
        for kk, wt in zip(range(1, n + 1), (0.4, 0.3, 0.2, 0.1)):
            c = Ts[kk - 1].get((pad[n + i - 1], pad[n + i - kk]))
            if c:
                t = sum(c.values())
                q = torch.zeros(C)
                for yy, v in c.items():
                    q[yy] = v
                q = (q + u) / (t + 1)
            else:
                q = u
            p += wt * q
        return p / p.sum()

    def pack(ds, pos, with_base):
        Hs, Ms, Y, B = [], [], [], []
        for di, i in pos:
            h, m = hist(ds[di], i)
            Hs.append(h)
            Ms.append(m)
            Y.append(oi.get(ds[di][i], C - 1))
            if with_base:
                B.append(base(ds[di], i))
        return torch.tensor(Hs), torch.tensor(Ms), torch.tensor(Y), (torch.stack(B) if with_base else None)

    ptr = positions(tr)
    train = pack(tr, ptr[:max_train], False)
    val = pack(tr, ptr[max_train:max_train + n_val], True)   # 訓練の文書だが、関連度グレインの学習には使わない位置
    test = pack(te, positions(te)[:n_test], True)
    return train, val, test, content, dict(freq=freq, vocab=vocab, vi=vi, out_vocab=out_vocab, tr=tr)


def count_related(info, V, C, Hn):
    """比べる相手: 数えた表の関連度（前後 10 語の共起 → PPMI → 64 次元）。"""
    vi, vocab, freq = info["vi"], info["vocab"], info["freq"]
    co = torch.zeros(V - 1, V - 1)
    for d in info["tr"]:
        ids = [vi.get(x, -1) for x in d]
        for i, a in enumerate(ids):
            if a < 0:
                continue
            for b in ids[max(0, i - 10):i]:
                if b >= 0:
                    co[a, b] += 1
                    co[b, a] += 1
    tot = co.sum()
    pmi = torch.log(co * tot / (co.sum(1, keepdim=True) * co.sum(0, keepdim=True)).clamp(min=1e-9) + 1e-12).clamp(min=0)
    U, S, _ = torch.linalg.svd(pmi)
    E = p2.unit_rows(U[:, :64] * S[:64].sqrt())
    Eo = torch.cat([torch.stack([E[vi[x]] for x in info["out_vocab"][:-1]]), torch.zeros(1, 64)])

    def rel(H, M):
        h = p2.unit_rows((E[H] * M[..., None]).sum(1))
        return h @ Eo.T
    return rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--C", type=int, default=1024)
    ap.add_argument("--V", type=int, default=4096)
    ap.add_argument("--H", type=int, default=50, help="話題に使う直前の語の数")
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--patches", type=int, default=16)
    ap.add_argument("--max_train", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp54.txt")
    a = ap.parse_args()
    dev = torch.device(a.device)
    (Htr, Mtr, Ytr, _), (Hv, Mv, Yv, Bv), (Ht, Mt, Yt, Bt), content, info = build(a.n, a.C, a.V, a.H, a.seed, a.max_train, 5000, 10000)
    keep = content[Ytr] & (Mtr.sum(1) > 0)
    Htr, Mtr, Ytr = Htr[keep], Mtr[keep], Ytr[keep]
    torch.manual_seed(a.seed)
    m = RelGrain(a.V, a.C, a.K, a.layers, a.patches, a.seed).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    g = torch.Generator().manual_seed(a.seed)
    cmask = torch.where(content, 0.0, -1e9).to(dev)
    t0 = time.perf_counter()
    history = []
    for step in range(1, a.steps + 1):
        i = torch.randint(len(Ytr), (256,), generator=g)
        s = m.related(Htr[i].to(dev), Mtr[i].to(dev)) * m.temp + cmask
        loss = F.cross_entropy(s, Ytr[i].to(dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
        if step % 500 == 0:
            history.append(dict(step=step, loss=float(loss.detach())))
    if dev.type == "cuda":
        torch.cuda.synchronize()
    train_s = time.perf_counter() - t0

    def rel_optical(H, M):
        with torch.no_grad():
            return torch.cat([m.related(H[j:j + 250].to(dev), M[j:j + 250].to(dev)).cpu() for j in range(0, len(H), 250)])

    def rerank(B, R, lam):
        Bn = B / B.sum(1, keepdim=True)
        mass = (Bn * content).sum(1, keepdim=True)
        q = Bn * content * torch.exp(lam * R)
        q = q / q.sum(1, keepdim=True).clamp(min=1e-12) * mass
        return Bn * ~content + q

    def ev(p, Y):
        cm = content[Y]
        hit = p.argmax(1) == Y
        nll = -torch.log(p[torch.arange(len(Y)), Y] + 1e-12)
        return dict(acc=100.0 * hit.float().mean().item(), ce=nll.mean().item(),
                    content_acc=100.0 * hit[cm].float().mean().item(), content_ce=nll[cm].mean().item())

    res = dict(base=ev(Bt / Bt.sum(1, keepdim=True), Yt))
    for name, relf in (("optical", rel_optical), ("counts", count_related(info, a.V, a.C, a.H))):
        Rv, Rt = relf(Hv, Mv), relf(Ht, Mt)
        lam = min((0.5, 1, 2, 4, 8, 16), key=lambda l: ev(rerank(Bv, Rv, l), Yv)["ce"])
        res[name] = dict(lam=lam, **ev(rerank(Bt, Rt, lam), Yt))
        # 関連度だけで内容語を当てる（内容語の位置で）
        cm = content[Yt]
        res[name]["related_only_content_acc"] = 100.0 * ((Rt + torch.where(content, 0.0, -1e9)).argmax(1) == Yt)[cm].float().mean().item()
    # 近い語の例（光の関連度グレインの符号どうしの近さ）
    cod = p2.unit_rows(m.codes.detach().cpu())
    near = {}
    for q in ["海", "山", "仕事", "美味しい", "東京", "猫"]:
        j = info["out_vocab"].index(q) if q in info["out_vocab"] else -1
        if j >= 0:
            s = cod @ cod[j] + torch.where(content, 0.0, -1e9)
            near[q] = [info["out_vocab"][t] for t in s.topk(9).indices.tolist() if t != j][:8]
    rec = dict(exp="54", **vars(a), results=res, test_acc=res["optical"]["acc"], test_ce=res["optical"]["ce"],
               params=sum(p.numel() for p in m.parameters() if p.requires_grad), train_s=train_s, history=history, near=near,
               content_share=100.0 * content[Yt].float().mean().item(), version=p2.__version__, torch=torch.__version__)
    print(json.dumps(rec, ensure_ascii=False))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
