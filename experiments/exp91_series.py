"""実験91: 記憶の板を系列（短期 → 中期 → 長期）にして容量を増やす（持ち主と決めた順 1 → 4 → 5 の 5、2026-10-26）

実験89 の場面（300 組から 16 組 × 60 回、Zipf、後半で 15% の値が変わる、10% に重要の印）で、板 1 枚（差分 + 足りない分だけ書く、55%）を、
3 枚の系列に分ける:
  短期 : 毎回の 16 組を全部書く。薄れが速い（λ = 0.7）
  中期 : 2 回以上来た組を書く（差分 + 足りない分だけ）。λ = 0.95
  長期 : 4 回以上来た組・重要の印の組を書く（差分 + 足りない分だけ）。λ = 0.995（ほとんど薄れない）
読むとき: 3 枚それぞれで取り出し、確信度（一番明るい窓 / 二番目）が一番高い板の答えを使う（確信度で優先、持ち主の案）。
比べる: 板 1 枚（実験89 の gated）、3 枚を同じ作り（どれも毎回全部を書く、λ 0.97）で確信度で選ぶもの（容量が 3 倍になっただけの効果を分ける）。
鍵: rand・word・sim0.5。3 seed、CPU。結果: results/exp91.txt
"""
from __future__ import annotations

import json
import os
import sys

import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from exp88_memkeys import K, L, conf, make_keys, read  # noqa: E402
from exp89_consolidate import M, P, T, run, wmap  # noqa: E402


def write(kk, vals, w):
    d = torch.zeros(len(vals), K)
    d[torch.arange(len(vals)), vals] = w
    return (torch.fft.fft2(kk).conj() * torch.fft.fft2(wmap(d))).sum(0) / (L * L)


def gate(H, kk, vv):
    S = read(H, kk)
    _, mg = conf(S)
    return torch.where((S.argmax(1) == vv) & (mg > 2.0), 0.1, 1.0)


def run_series(kind, seed, mode):
    g = torch.Generator().manual_seed(seed)
    keys = make_keys(kind, P, g, seed * 7 + 1)
    val = torch.randint(K, (P,), generator=g)
    freq = 1.0 / torch.arange(1, P + 1).float()
    freq = freq[torch.randperm(P, generator=g)]
    important = torch.rand(P, generator=g) < 0.1
    changed = torch.rand(P, generator=g) < 0.15
    new_val = (val + 1 + torch.randint(K - 1, (P,), generator=g)) % K
    lam = (0.7, 0.95, 0.995) if mode == "series" else (0.97, 0.97, 0.97)
    Hs = [torch.zeros(L, L, dtype=torch.complex64) for _ in range(3)]
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
        wimp = torch.where(important[idx], 3.0, 1.0)
        for j in range(3):
            if mode == "series":
                sel = torch.ones(M, dtype=torch.bool) if j == 0 else (count[idx] >= 2) if j == 1 else (count[idx] >= 4) | important[idx]
            else:
                sel = torch.arange(M) % 3 == j            # 同じ作り: 組を 3 枚に振り分けるだけ
            w = wimp * sel.float()
            if j > 0 or mode != "series":
                w = w * (gate(Hs[j], kk, vv) if t > 0 else 1.0)
            Hs[j] = lam[j] * Hs[j] + write(kk, vv, w)
    seen = count > 0
    cur = torch.where(changed, new_val, val)
    kq = keys - kmean
    Ss = torch.stack([read(H, kq) for H in Hs])                   # (3, P, K)
    top = Ss.topk(2, -1).values
    mg = top[..., 0] / top[..., 1].clamp(min=1e-12)
    best = mg.argmax(0)                                           # 確信度が一番高い板
    am = Ss.argmax(-1)[best, torch.arange(P)]
    hit = (am == cur) & seen
    age = (T - 1 - last).float()
    grp = lambda m_: float(100.0 * hit[m_ & seen].float().mean()) if (m_ & seen).any() else None
    ch = changed & seen & (last >= T // 2)
    return dict(acc=grp(seen), acc_once=grp(count == 1), acc_2to4=grp((count >= 2) & (count <= 4)), acc_5up=grp(count >= 5),
                acc_important=grp(important), acc_recent=grp(age < 10), acc_old=grp(age >= 30),
                changed_new=float(100.0 * (am == new_val)[ch].float().mean()) if ch.any() else None,
                plate_share=[float((best[seen] == j).float().mean()) for j in range(3)])


def main():
    with open(os.path.join(ROOT, "results", "exp91.txt"), "a", encoding="utf-8") as out:
        for seed in range(3):
            for kind in ("rand", "word", "sim0.5"):
                rec = dict(exp="91", seed=seed, keys=kind, method="single_gated", **run(kind, "gated", seed))
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                for mode in ("series", "split3"):
                    rec = dict(exp="91", seed=seed, keys=kind, method=mode, **run_series(kind, seed, mode))
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
