"""実験92: 記憶の系列を細かく刻む（持ち主の見立て 2026-10-26: 秒 → 分 → 10 分 → … → 1 日、刻みはフィボナッチ数列っぽく）

実験89・91 の場面（300 組から 16 組 × 60 回、Zipf、後半で 15% の値が変わる、10% に重要の印）。板を 7 枚の系列にし、
板 j の記憶の半減期を フィボナッチ 1, 2, 3, 5, 8, 13, 21 区切り（λ_j = 0.5^(1/τ_j)）にする。読むときは確信度が一番高い板の答え。
送り方:
  all     : 来た組を全部の板に書く（足りない分だけ書く）。板ごとに薄れる速さだけが違う
  cascade : 板 j には、その組が来た回数が F_j 回以上のときだけ書く（F = 1, 1, 2, 3, 5, 8, 13）。重要の印は 1 段上から入る
  blocks  : 時間で板を分ける。区切りごとに新しい板に書き、同じ長さの時間をまとめた板が 2 枚そろったら 1 枚に融合する
            （古いほど長い時間をまとめた板になる: 1, 1, 2, 4, 8, … 区切り。2 の累乗の刻み。板の数は約 log2(区切りの数)）
  blocks_fib : 同じだが、まとめる長さをフィボナッチ数にする（板の長さがいつも違うフィボナッチ数の和になるよう、隣り合うフィボナッチ数の板を融合）
比べる: 板 7 枚を同じ作り（λ 0.97、組を順に振り分け）、板 1 枚・3 枚（実験91）。
結果: results/exp92.txt
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
from exp91_series import gate, write  # noqa: E402

FIB_T = (1, 2, 3, 5, 8, 13, 21)
FIB_N = (1, 1, 2, 3, 5, 8, 13)


def run_blocks(kind, seed, fib):
    g = torch.Generator().manual_seed(seed)
    keys = make_keys(kind, P, g, seed * 7 + 1)
    val = torch.randint(K, (P,), generator=g)
    freq = 1.0 / torch.arange(1, P + 1).float()
    freq = freq[torch.randperm(P, generator=g)]
    important = torch.rand(P, generator=g) < 0.1
    changed = torch.rand(P, generator=g) < 0.15
    new_val = (val + 1 + torch.randint(K - 1, (P,), generator=g)) % K
    plates, spans = [], []
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
        if plates:                                    # 足りない分だけ書く: 今ある板（確信度で選んで）で取り出せる組は軽く
            Ss = torch.stack([read(H, kk) for H in plates])
            top = Ss.topk(2, -1).values
            mg = top[..., 0] / top[..., 1].clamp(min=1e-12)
            b = mg.argmax(0)
            ok = (Ss.argmax(-1)[b, torch.arange(M)] == vv) & (mg.max(0).values > 2.0)
            w = w * torch.where(ok, 0.1, 1.0)
        plates = [0.97 * H for H in plates] + [write(kk, vv, w)]
        spans.append(1)
        while len(spans) >= 2:                        # 融合: 2 の累乗なら同じ長さ 2 枚、フィボナッチなら隣り合う F_k・F_{k-1}
            a, c = spans[-2], spans[-1]
            fibs = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
            if fib:   # ゼッケンドルフの形（違うフィボナッチ数の和）に保つ: 1 と 1、または隣り合うフィボナッチ数（F_{k+1}, F_k）が並んだら融合
                merge = (a == 1 and c == 1) or (a in fibs and c in fibs and fibs.index(a) == fibs.index(c) + 1)
            else:     # 2 の累乗: 同じ長さが 2 枚並んだら融合
                merge = a == c
            if not merge:
                break
            plates[-2:] = [plates[-2] + plates[-1]]
            spans[-2:] = [a + c]
    seen = count > 0
    cur = torch.where(changed, new_val, val)
    Ss = torch.stack([read(H, keys - kmean) for H in plates])
    top = Ss.topk(2, -1).values
    best = (top[..., 0] / top[..., 1].clamp(min=1e-12)).argmax(0)
    am = Ss.argmax(-1)[best, torch.arange(P)]
    hit = (am == cur) & seen
    age = (T - 1 - last).float()
    grp = lambda m_: float(100.0 * hit[m_ & seen].float().mean()) if (m_ & seen).any() else None
    ch = changed & seen & (last >= T // 2)
    return dict(acc=grp(seen), acc_once=grp(count == 1), acc_2to4=grp((count >= 2) & (count <= 4)), acc_5up=grp(count >= 5),
                acc_important=grp(important), acc_recent=grp(age < 10), acc_mid=grp((age >= 10) & (age < 30)), acc_old=grp(age >= 30),
                changed_new=float(100.0 * (am == new_val)[ch].float().mean()) if ch.any() else None, spans=spans)


def run(kind, seed, mode, n_plates=7):
    g = torch.Generator().manual_seed(seed)
    keys = make_keys(kind, P, g, seed * 7 + 1)
    val = torch.randint(K, (P,), generator=g)
    freq = 1.0 / torch.arange(1, P + 1).float()
    freq = freq[torch.randperm(P, generator=g)]
    important = torch.rand(P, generator=g) < 0.1
    changed = torch.rand(P, generator=g) < 0.15
    new_val = (val + 1 + torch.randint(K - 1, (P,), generator=g)) % K
    lam = [0.5 ** (1 / t) for t in FIB_T] if mode in ("all", "cascade") else [0.97] * n_plates
    Hs = [torch.zeros(L, L, dtype=torch.complex64) for _ in range(n_plates)]
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
        for j in range(n_plates):
            if mode == "all":
                sel = torch.ones(M, dtype=torch.bool)
            elif mode == "cascade":
                sel = (count[idx] >= FIB_N[j]) | (important[idx] & (count[idx] >= FIB_N[max(0, j - 1)]))
            else:
                sel = torch.arange(M) % n_plates == j
            w = wimp * sel.float() * (gate(Hs[j], kk, vv) if t > 0 else 1.0)
            Hs[j] = lam[j] * Hs[j] + write(kk, vv, w)
    seen = count > 0
    cur = torch.where(changed, new_val, val)
    Ss = torch.stack([read(H, keys - kmean) for H in Hs])
    top = Ss.topk(2, -1).values
    best = (top[..., 0] / top[..., 1].clamp(min=1e-12)).argmax(0)
    am = Ss.argmax(-1)[best, torch.arange(P)]
    hit = (am == cur) & seen
    age = (T - 1 - last).float()
    grp = lambda m_: float(100.0 * hit[m_ & seen].float().mean()) if (m_ & seen).any() else None
    ch = changed & seen & (last >= T // 2)
    return dict(acc=grp(seen), acc_once=grp(count == 1), acc_2to4=grp((count >= 2) & (count <= 4)), acc_5up=grp(count >= 5),
                acc_important=grp(important), acc_recent=grp(age < 10), acc_mid=grp((age >= 10) & (age < 30)), acc_old=grp(age >= 30),
                changed_new=float(100.0 * (am == new_val)[ch].float().mean()) if ch.any() else None,
                plate_share=[float((best[seen] == j).float().mean()) for j in range(n_plates)])


def main():
    with open(os.path.join(ROOT, "results", "exp92.txt"), "a", encoding="utf-8") as out:
        for seed in range(3):
            for kind in ("rand", "word"):
                for mode in ("all", "cascade", "split7"):
                    out.write(json.dumps(dict(exp="92", seed=seed, keys=kind, method=mode, **run(kind, seed, mode)), ensure_ascii=False) + "\n")
                for mode, fib in (("blocks", False), ("blocks_fib", True)):
                    out.write(json.dumps(dict(exp="92", seed=seed, keys=kind, method=mode, **run_blocks(kind, seed, fib)), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
