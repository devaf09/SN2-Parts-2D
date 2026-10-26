"""実験97: 環状の板・精緻化リハーサル（持ち主のメモ 2026-10-26、docs/memory_design.md）

場面は実験95・96 と同じ（300 組から 16 組ずつ、600 区切り、Zipf、後半で 15% の値が変わる、10% に重要の印）。
読むときはどれも「確信度 × なじみ（関連度）で重みを付けて全部の板を足す」（実験96 の一番）。

A. 環状の板（ring）: N 枚の板を輪に並べ、書く位置（針）を区切りごとに 1 つ進める。針が来た板の「次の板」の中身を、
   さらに次の板へ比率 β で融合して移し（少しずつ融合、β = 1 なら全部）、針の板に新しい記憶を書く。
   板は輪になっているので、全体として記憶を持ち、古い記憶は融合を重ねて輪を回る。N = 8・16、β = 1・0.5。
B. 精緻化リハーサル（elab）: 実験94〜96 の「満ちたら送る」段（短期 10 段 + 中期 10 段、1.5 倍ずつ）に、
   「長期（中期の系列）に対するなじみが高い記憶は、1 回でも中期の段 0 へ直接書く（深く覚える）」を足す。
   なじみの高さは、中期の板のなじみの和が、書いた量に比べて上位 q（q = 0.2・0.5）に入るか。
比べる: 実験96 の形（elab なし）、時間で分けて倍々（実験92）。鍵: rand・word_hash。CPU、3 seed。結果: results/exp97.txt
"""
from __future__ import annotations

import json
import os
import sys

import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from exp88_memkeys import K, L, make_keys, read  # noqa: E402
from exp89_consolidate import M, P  # noqa: E402
from exp91_series import write  # noqa: E402
from exp93_shift import widths  # noqa: E402

T = 600


def setup(kind, seed):
    g = torch.Generator().manual_seed(seed)
    keys = make_keys(kind, P, g, seed * 7 + 1)
    val = torch.randint(K, (P,), generator=g)
    freq = 1.0 / torch.arange(1, P + 1).float()
    freq = freq[torch.randperm(P, generator=g)]
    important = torch.rand(P, generator=g) < 0.1
    changed = torch.rand(P, generator=g) < 0.15
    new_val = (val + 1 + torch.randint(K - 1, (P,), generator=g)) % K
    return g, keys, val, freq, important, changed, new_val


def readout(plates, kq):
    """全部の板を、確信度 × なじみで重みを付けて足す。返り値: (答え, 確信度の最大)"""
    Ss = torch.stack([read(H[0], kq) for H in plates])
    fam = torch.stack([read(H[1], kq)[:, 0] for H in plates])
    fam = fam / fam.sum(0, keepdim=True).clamp(min=1e-12)
    top2 = Ss.topk(2, -1).values
    mg = top2[..., 0] / top2[..., 1].clamp(min=1e-12)
    Pn = Ss / Ss.sum(-1, keepdim=True)
    return (Pn * ((mg[..., None] - 1).clamp(min=0) * fam[..., None])).sum(0).argmax(-1), mg.max(0).values


def pair(kk, vv, w):
    return torch.stack([write(kk, vv, w), write(kk, torch.zeros_like(vv), w)])


def score(am, cur, seen, count, last, important, changed, new_val):
    hit = (am == cur) & seen
    age = (T - 1 - last).float()
    grp = lambda m_: float(100.0 * hit[m_ & seen].float().mean()) if (m_ & seen).any() else None
    ch = changed & seen & (last >= T // 2)
    return dict(acc=grp(seen), acc_once=grp(count == 1), acc_2to4=grp((count >= 2) & (count <= 4)), acc_5up=grp(count >= 5),
                acc_important=grp(important), acc_recent=grp(age < 10), acc_old=grp(age >= 30),
                changed_new=float(100.0 * (am == new_val)[ch].float().mean()) if ch.any() else None)


def run_ring(kind, seed, N, beta):
    g, keys, val, freq, important, changed, new_val = setup(kind, seed)
    plates = [torch.zeros(2, L, L, dtype=torch.complex64) for _ in range(N)]
    kmean, nk = torch.zeros(L, L, dtype=torch.complex64), 0
    count, last = torch.zeros(P), torch.full((P,), -1)
    for t in range(T):
        cur = torch.where(changed & (t >= T // 2), new_val, val)
        idx = torch.multinomial(freq, M, replacement=False, generator=g)
        count[idx] += 1
        last[idx] = t
        kmean = (kmean * nk + keys[idx].sum(0)) / (nk + M)
        nk += M
        kk, vv = keys[idx] - kmean, cur[idx]
        w = torch.where(important[idx], 3.0, 1.0)
        if t > 0:
            am, mg = readout([H for H in plates if H[0].abs().sum() > 0], kk)
            w = w * torch.where((am == vv) & (mg > 2.0), 0.1, 1.0)
        p = t % N
        a, b = (p + 1) % N, (p + 2) % N
        move = beta * plates[a]                       # 針の次の板の中身を、さらに次の板へ（少しずつ）融合して移す
        plates[b] = plates[b] + move
        plates[a] = plates[a] - move
        plates[p] = plates[p] + pair(kk, vv, w)
    seen = count > 0
    am, _ = readout([H for H in plates if H[0].abs().sum() > 0], keys - kmean)
    return score(am, torch.where(changed, new_val, val), seen, count, last, important, changed, new_val)


def run_elab(kind, seed, q, n_levels=10, wkind="r1.5"):
    g, keys, val, freq, important, changed, new_val = setup(kind, seed)
    W = widths(wkind, n_levels)
    zero = lambda: torch.zeros(2, L, L, dtype=torch.complex64)
    series = [[zero() for _ in range(n_levels)] for _ in range(2)]
    amount = [[0.0] * n_levels for _ in series]
    kmean, nk = torch.zeros(L, L, dtype=torch.complex64), 0
    count, last = torch.zeros(P), torch.full((P,), -1)
    n_elab = 0

    def push(si, k, H, s):
        lv, am = series[si], amount[si]
        top = k == n_levels - 1
        if am[k] > 0 and am[k] + s > W[k] + 1e-9 and (not top or si + 1 < len(series)):
            Hk, sk = lv[k], am[k]
            lv[k], am[k] = zero(), 0.0
            push(si + 1, 0, Hk, sk) if top else push(si, k + 1, Hk, sk)
        lv[k] = lv[k] + H
        am[k] += s

    for t in range(T):
        cur = torch.where(changed & (t >= T // 2), new_val, val)
        idx = torch.multinomial(freq, M, replacement=False, generator=g)
        count[idx] += 1
        last[idx] = t
        kmean = (kmean * nk + keys[idx].sum(0)) / (nk + M)
        nk += M
        kk, vv = keys[idx] - kmean, cur[idx]
        w = torch.where(important[idx], 3.0, 1.0)
        plates = [H for lv in series for H in lv if H[0].abs().sum() > 0]
        if plates:
            am, mg = readout(plates, kk)
            w = w * torch.where((am == vv) & (mg > 2.0), 0.1, 1.0)
        series[-1][-1] = 0.995 * series[-1][-1]
        push(0, 0, pair(kk, vv, w), 1.0)
        mids = [H for H in series[1] if H[0].abs().sum() > 0]
        if q > 0 and mids:                            # 精緻化: 中期（長期）になじむ記憶は、1 回でも中期の段 0 へ直接書く
            fam = sum(read(H[1], kk)[:, 0] for H in mids)
            thr = torch.quantile(fam, 1 - q)
            deep = (fam >= thr) & (w >= 0.5)
            if deep.any():
                n_elab += int(deep.sum())
                series[1][0] = series[1][0] + pair(kk[deep], vv[deep], w[deep])
                amount[1][0] += float(deep.sum()) / M
    seen = count > 0
    plates = [H for lv in series for H in lv if H[0].abs().sum() > 0]
    am, _ = readout(plates, keys - kmean)
    rec = score(am, torch.where(changed, new_val, val), seen, count, last, important, changed, new_val)
    rec["n_elab"] = n_elab
    return rec


def main():
    with open(os.path.join(ROOT, "results", "exp97.txt"), "a", encoding="utf-8") as out:
        for seed in range(3):
            for kind in ("rand", "word_hash"):
                for q in (0.0, 0.2, 0.5):
                    out.write(json.dumps(dict(exp="97", seed=seed, keys=kind, method=f"elab_q{q}", **run_elab(kind, seed, q)), ensure_ascii=False) + "\n")
                for N in (8, 16):
                    for beta in (1.0, 0.5):
                        out.write(json.dumps(dict(exp="97", seed=seed, keys=kind, method=f"ring_N{N}_b{beta}", **run_ring(kind, seed, N, beta)), ensure_ascii=False) + "\n")
                print(seed, kind, flush=True)


if __name__ == "__main__":
    main()
