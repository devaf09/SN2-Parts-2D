"""実験94: 記憶の段を「満ちたらまとめて次へ送る」（実験93 の改善案、持ち主の依頼 2026-10-26）

実験93（各段の 1 / 幅 を毎回少しずつ送る）は、1 つの記憶が何段にも薄く分かれ、単語の鍵で負けた。
案: 段 k は「幅 W_k 区切り分」までためる板。
  1. 新しい記憶（区切り 1 つ分）は段 0 に書く（差分記憶法: 鍵の共通部分を引く。足りない分だけ書く）。段 0 の量 s_0 += 1
  2. 段 k に新しく入れると幅 W_k を超えるなら（満ちたら）、先に段 k の板を**まるごと**段 k+1 へ送り、段 k+1 の板と融合する。段 k は空にしてから入れる
     融合の比率（--fuse）:
       sum   : そのまま足す（記憶 1 つ分の明るさを保つ。比率は自然に「ためた量」の比になる）
       ratio : 持ち主の比率。今ある中身（古い方）: 入ってくる板（新しい方）= ためた量の比（例 2 : 3）で、明るさをそろえてから足す
  3. 一番上の段は送らずに積み重ねる（長期、λ = 0.995）。中期の系列（_mid）を重ねるときは、短期の一番上が満ちたら中期の段 0 へまるごと送る
  幅 W: fib = 1, 1, 2, 3, 5, 8, 13, 21, 34, 55 / r1.5・r2 = 1.5 倍・2 倍ずつ
読むとき: 確信度が一番高い段の答え。比べる: 時間で分けて倍々に融合（実験92 の blocks）、少しずつ送る形（実験93）。
場面は実験89 と同じ（300 組から 16 組 × 60 回）。CPU、3 seed。結果: results/exp94.txt
"""
from __future__ import annotations

import json
import os
import sys

import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from exp88_memkeys import K, L, make_keys, read  # noqa: E402
from exp89_consolidate import M, P, T  # noqa: E402
from exp91_series import write  # noqa: E402
from exp93_shift import pick, widths  # noqa: E402


LAST = {}          # 最後の run の中身（図のため）


def unit(H):
    n = H.abs().pow(2).sum().sqrt()
    return H / n if n > 0 else H


def run(keys_kind, seed, wkind, n_levels, fuse, nested, T=T):
    g = torch.Generator().manual_seed(seed)
    keys = make_keys(keys_kind, P, g, seed * 7 + 1)
    val = torch.randint(K, (P,), generator=g)
    freq = 1.0 / torch.arange(1, P + 1).float()
    freq = freq[torch.randperm(P, generator=g)]
    important = torch.rand(P, generator=g) < 0.1
    changed = torch.rand(P, generator=g) < 0.15
    new_val = (val + 1 + torch.randint(K - 1, (P,), generator=g)) % K
    W = widths(wkind, n_levels)
    zero = lambda: torch.zeros(L, L, dtype=torch.complex64)
    series = [[zero() for _ in range(n_levels)] for _ in range(2 if nested else 1)]
    amount = [[0.0] * n_levels for _ in series]
    kmean, nk = zero(), 0
    count, last = torch.zeros(P), torch.full((P,), -1)

    def push(si, k, H, s):
        """系列 si の段 k に、ためた量 s の板 H を入れる。入れると幅 W_k を超えるなら、先に今の中身をまるごと次へ送る（満ちたら送る）。"""
        lv, am = series[si], amount[si]
        top = k == n_levels - 1
        if am[k] > 0 and am[k] + s > W[k] + 1e-9 and (not top or si + 1 < len(series)):
            Hk, sk = lv[k], am[k]
            lv[k], am[k] = zero(), 0.0
            if top:
                push(si + 1, 0, Hk, sk)
            else:
                push(si, k + 1, Hk, sk)
        if fuse == "ratio" and am[k] > 0:
            # 持ち主の比率: 今ある中身（古い方）: 入ってくる板（新しい方）= ためた量の比（例 2 : 3）。明るさの合計は足した量に合わせる
            a, b = am[k], s
            tot = a + b
            lv[k] = (a * unit(lv[k]) + b * unit(H)) / tot * (lv[k].abs().pow(2).sum().sqrt() + H.abs().pow(2).sum().sqrt())
        else:
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
        allp = [H for lv in series for H in lv if H.abs().sum() > 0]
        if allp:
            am_, mg, _ = pick(allp, kk)
            w = w * torch.where((am_ == vv) & (mg > 2.0), 0.1, 1.0)
        series[-1][-1] = 0.995 * series[-1][-1]
        push(0, 0, write(kk, vv, w), 1.0)
    seen = count > 0
    cur = torch.where(changed, new_val, val)
    allp = [H for lv in series for H in lv if H.abs().sum() > 0]
    am, _, b = pick(allp, keys - kmean)
    hit = (am == cur) & seen
    Ss = torch.stack([read(H, keys - kmean) for H in allp])
    oracle = ((Ss.argmax(-1) == cur[None]).any(0) & seen)
    # 段の選び方の比べ（実験95）: 確信度・新しい段を優先・全段の明るさを足す・全段の割合を足す（多数決）
    top2 = Ss.topk(2, -1).values
    mgs = top2[..., 0] / top2[..., 1].clamp(min=1e-12)                  # (段, 組)
    lvl = torch.tensor([float(j) for j, (si, H) in enumerate([(si, H) for si, lv in enumerate(series) for H in lv if H.abs().sum() > 0])])
    sel = {}
    for nm, score in (("recent0.9", mgs * 0.9 ** lvl[:, None]), ("recent0.7", mgs * 0.7 ** lvl[:, None])):
        bb = score.argmax(0)
        sel[nm] = Ss.argmax(-1)[bb, torch.arange(len(cur))]
    sel["sum"] = Ss.sum(0).argmax(-1)
    sel["vote"] = (Ss / Ss.sum(-1, keepdim=True)).sum(0).argmax(-1)
    sel["conf_sum"] = (Ss * (mgs[..., None] - 1).clamp(min=0)).sum(0).argmax(-1)   # 確信度で重みを付けて足す
    LAST.update(series=series, amount=amount, keys=keys, kmean=kmean, cur=cur, seen=seen, chosen=b, Ss=Ss, idx_kind=keys_kind, W=W)               # どれかの段では正しく取り出せる（段の選び方が完全なら）
    age = (T - 1 - last).float()
    grp = lambda m_: float(100.0 * hit[m_ & seen].float().mean()) if (m_ & seen).any() else None
    ch = changed & seen & (last >= T // 2)
    return dict(acc=grp(seen), acc_once=grp(count == 1), acc_5up=grp(count >= 5), acc_important=grp(important),
                acc_recent=grp(age < 10), acc_mid=grp((age >= 10) & (age < 30)), acc_old=grp(age >= 30),
                changed_new=float(100.0 * (am == new_val)[ch].float().mean()) if ch.any() else None, n_plates=len(allp),
                acc_oracle=float(100.0 * oracle[seen].float().mean()),
                **{f"acc_sel_{k}": float(100.0 * ((v == cur) & seen)[seen].float().mean()) for k, v in sel.items()})


def main():
    with open(os.path.join(ROOT, "results", "exp94.txt"), "a", encoding="utf-8") as out:
        for seed in range(3):
            for kind in ("rand", "word"):
                for wk, n in (("fib", 10), ("r1.5", 10), ("r2", 7)):
                    for fuse in ("sum", "ratio"):
                        for nested in (False, True):
                            rec = dict(exp="94", seed=seed, keys=kind, method=f"flush_{wk}_{fuse}" + ("_mid" if nested else ""),
                                       widths=widths(wk, n), **run(kind, seed, wk, n, fuse, nested))
                            out.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
