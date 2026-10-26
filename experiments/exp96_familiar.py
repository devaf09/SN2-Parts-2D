"""実験96: 段の選び方に「確信度以外」を足す — 関連度（なじみ）、〈他〉を別々にした単語の鍵（持ち主の依頼 2026-10-26）

実験94・95 の「満ちたら送る」段（1.5 倍ずつ 10 段、短期 + 中期）で、単語の鍵は「選び方が完全なら 84〜90%」なのに実際は 50〜58%。
足すもの:
  なじみの板（関連度）: 段ごとに、記憶の板と一緒に「なじみの板」を持つ。書く鍵をすべて同じ 1 点（値に関係なく）に結ぶ板で、
    問い合わせの鍵を通すと、その段に似た鍵がどれだけ書かれているか（関連度・なじみ）がその点の明るさで分かる。段を送るときは一緒に送る。
  読み方: conf_sum = Σ_段 (確信度 − 1) × 明るさ（実験95 の一番）、fam = Σ_段 なじみ × 明るさ、conf_fam = Σ_段 (確信度 − 1) × なじみ × 明るさ
鍵: word（〈他〉を 1 つの印にまとめる、今まで）、word_hash（〈他〉の語を表記ごとに別の向きに）、rand。
流れ: 60 区切り（短期だけ）と 600 区切り（短期 + 中期）。CPU、3 seed。結果: results/exp96.txt
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
from exp93_shift import pick, widths  # noqa: E402


def run(keys_kind, seed, T, nested, n_levels=10, wkind="r1.5"):
    g = torch.Generator().manual_seed(seed)
    keys = make_keys(keys_kind, P, g, seed * 7 + 1)
    val = torch.randint(K, (P,), generator=g)
    freq = 1.0 / torch.arange(1, P + 1).float()
    freq = freq[torch.randperm(P, generator=g)]
    important = torch.rand(P, generator=g) < 0.1
    changed = torch.rand(P, generator=g) < 0.15
    new_val = (val + 1 + torch.randint(K - 1, (P,), generator=g)) % K
    W = widths(wkind, n_levels)
    zero = lambda: torch.zeros(2, L, L, dtype=torch.complex64)      # [記憶の板, なじみの板]
    series = [[zero() for _ in range(n_levels)] for _ in range(2 if nested else 1)]
    amount = [[0.0] * n_levels for _ in series]
    kmean, nk = torch.zeros(L, L, dtype=torch.complex64), 0
    count, last = torch.zeros(P), torch.full((P,), -1)

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
        allp = [H[0] for lv in series for H in lv if H[0].abs().sum() > 0]
        if allp:
            am_, mg, _ = pick(allp, kk)
            w = w * torch.where((am_ == vv) & (mg > 2.0), 0.1, 1.0)
        series[-1][-1] = 0.995 * series[-1][-1]
        push(0, 0, torch.stack([write(kk, vv, w), write(kk, torch.zeros_like(vv), w)]), 1.0)
    seen = count > 0
    cur = torch.where(changed, new_val, val)
    plates = [H for lv in series for H in lv if H[0].abs().sum() > 0]
    kq = keys - kmean
    Ss = torch.stack([read(H[0], kq) for H in plates])                     # (段, 組, 窓)
    fam = torch.stack([read(H[1], kq)[:, 0] for H in plates])              # (段, 組) なじみ = 点 0 の明るさ
    fam = fam / fam.sum(0, keepdim=True).clamp(min=1e-12)
    top2 = Ss.topk(2, -1).values
    mg = top2[..., 0] / top2[..., 1].clamp(min=1e-12)
    P_ = Ss / Ss.sum(-1, keepdim=True)
    sel = {"conf1": Ss.argmax(-1)[mg.argmax(0), torch.arange(P)],
           "conf_sum": (P_ * (mg[..., None] - 1).clamp(min=0)).sum(0).argmax(-1),
           "fam": (P_ * fam[..., None]).sum(0).argmax(-1),
           "fam1": Ss.argmax(-1)[fam.argmax(0), torch.arange(P)],
           "conf_fam": (P_ * ((mg[..., None] - 1).clamp(min=0) * fam[..., None])).sum(0).argmax(-1)}
    oracle = (Ss.argmax(-1) == cur[None]).any(0)
    rec = dict(oracle=float(100.0 * oracle[seen].float().mean()), n_plates=len(plates))
    age = (T - 1 - last).float()
    for nm, am in sel.items():
        hit = (am == cur) & seen
        rec[f"acc_{nm}"] = float(100.0 * hit[seen].float().mean())
        if nm == "conf_fam":
            grp = lambda m_: float(100.0 * hit[m_ & seen].float().mean()) if (m_ & seen).any() else None
            ch = changed & seen & (last >= T // 2)
            rec.update(cf_recent=grp(age < 10), cf_old=grp(age >= 30), cf_5up=grp(count >= 5), cf_important=grp(important),
                       cf_changed_new=float(100.0 * (am == new_val)[ch].float().mean()) if ch.any() else None)
    # なじみが正しい段（その組の記憶がある段）を一番に指す割合
    right = (Ss.argmax(-1) == cur[None])
    rec["fam_points_right"] = float(100.0 * right[fam.argmax(0), torch.arange(P)][seen & oracle].float().mean())
    rec["conf_points_right"] = float(100.0 * right[mg.argmax(0), torch.arange(P)][seen & oracle].float().mean())
    return rec


def main():
    with open(os.path.join(ROOT, "results", "exp96.txt"), "a", encoding="utf-8") as out:
        for T, nested in ((60, False), (600, True)):
            for seed in range(3):
                for kind in ("rand", "word", "word_hash"):
                    out.write(json.dumps(dict(exp="96", T=T, seed=seed, keys=kind, **run(kind, seed, T, nested)), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
