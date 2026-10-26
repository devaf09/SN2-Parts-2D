"""実験99: 板の数をそろえて比べる・融合による忘却（持ち主の依頼 2026-10-26）

1. 流れ型（満ちたら送る + 中期の系列）も板 64 枚に: 短期 32 段 + 中期 32 段、幅の比率 r = 1.1・1.2（64 枚に中身が入るように細かく刻む）。
2. 環状 64 枚に忘却を足す（融合しない形は、流れが長いと 1 枚が満杯になり、忘却もない）:
     keep      : 融合も忘却もしない（実験98 の一番）
     decay     : 全部の板を区切りごとに 0.995 倍（ただ薄れる）
     lap       : 針が 1 周して戻ってきた板は、古い中身を 0.5 倍にしてから新しい記憶を重ねる（1 周ごとに半分に。融合による忘却）
     consol    : 針が戻ってきた板の古い中身を、長期の環（16 枚）の板へ融合して移し、空にしてから書く。長期の環も 1 周ごとに 0.5 倍
                 （短期の環 → 長期の環。長期へ移る分は「足りない分だけ書く」でもう選ばれている）
流れの長さ: 600 区切りと 2400 区切り（1 枚あたり約 600 組、容量を超える）。読み方は確信度 × なじみ（実験96）。
鍵: rand・word_hash。CPU、3 seed。結果: results/exp99.txt
実行: python experiments/exp99_plates64.py <part> [modes]   part = flow・ring、modes = keep,decay,lap,consol（分けて同時に回せる）
"""
from __future__ import annotations

import json
import os
import sys

import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import exp97_ring_elab as e97  # noqa: E402
from exp88_memkeys import L  # noqa: E402
from exp89_consolidate import M, P  # noqa: E402


def run_ring2(kind, seed, T, mode, N=64, NL=16):
    e97.T = T
    g, keys, val, freq, important, changed, new_val = e97.setup(kind, seed)
    zero = lambda: torch.zeros(2, L, L, dtype=torch.complex64)
    plates = [zero() for _ in range(N)]
    longs = [zero() for _ in range(NL)] if mode == "consol" else []
    kmean, nk = torch.zeros(L, L, dtype=torch.complex64), 0
    count, last = torch.zeros(P), torch.full((P,), -1)
    li = 0
    for t in range(T):
        cur = torch.where(changed & (t >= T // 2), new_val, val)
        idx = torch.multinomial(freq, M, replacement=False, generator=g)
        count[idx] += 1
        last[idx] = t
        kmean = (kmean * nk + keys[idx].sum(0)) / (nk + M)
        nk += M
        kk, vv = keys[idx] - kmean, cur[idx]
        w = torch.where(important[idx], 3.0, 1.0)
        live = [H for H in plates + longs if H[0].abs().sum() > 0]
        if live:
            am, mg = e97.readout(live, kk)
            w = w * torch.where((am == vv) & (mg > 2.0), 0.1, 1.0)
        p = t % N
        if mode == "decay":
            plates = [0.995 * H for H in plates]
        elif mode == "lap" and t >= N:
            plates[p] = 0.5 * plates[p]
        elif mode == "consol" and t >= N:
            if t % (N // NL) == 0:                    # 長期の環は N / NL 回に 1 回進む
                li = (li + 1) % NL
                if t >= N + N:                         # 長期の環も 1 周ごとに半分（融合による忘却）
                    longs[li] = 0.5 * longs[li]
            longs[li] = longs[li] + plates[p]
            plates[p] = zero()
        plates[p] = plates[p] + e97.pair(kk, vv, w)
    seen = count > 0
    am, _ = e97.readout([H for H in plates + longs if H[0].abs().sum() > 0], keys - kmean)
    return e97.score(am, torch.where(changed, new_val, val), seen, count, last, important, changed, new_val) | age_bins(am, torch.where(changed, new_val, val), seen, last, T)


def age_bins(am, cur, seen, last, T):
    hit = (am == cur) & seen
    age = (T - 1 - last).float()
    out = {}
    for lo, hi in ((0, 64), (64, 300), (300, 10 ** 9)):
        m = seen & (age >= lo) & (age < hi)
        out[f"acc_age{lo}"] = float(100.0 * hit[m].float().mean()) if m.any() else None
    return out


def run_flow(kind, seed, T, r):
    e97.T = T
    rec = e97.run_elab(kind, seed, 0.0, n_levels=32, wkind=f"r{r}")
    return rec


def main():
    part = sys.argv[1]
    modes = sys.argv[2].split(",") if len(sys.argv) > 2 else ["keep", "decay", "lap", "consol"]
    with open(os.path.join(ROOT, "results", f"exp99_{part}.txt"), "a", encoding="utf-8") as out:
        for T in (600, 2400):
            for seed in range(3):
                for kind in ("rand", "word_hash"):
                    if part == "flow":
                        for r in (1.1, 1.2):
                            out.write(json.dumps(dict(exp="99", T=T, seed=seed, keys=kind, method=f"flow64_r{r}", **run_flow(kind, seed, T, r)), ensure_ascii=False) + "\n")
                    else:
                        for mode in modes:
                            out.write(json.dumps(dict(exp="99", T=T, seed=seed, keys=kind, method=f"ring64_{mode}", **run_ring2(kind, seed, T, mode)), ensure_ascii=False) + "\n")
                    print(part, T, seed, kind, flush=True)


if __name__ == "__main__":
    main()
