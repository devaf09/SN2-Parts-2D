"""実験93: 記憶の段を「ずらしながら融合」する（持ち主の構想 2026-10-26）

持ち主の構想: 短期記憶野は 1（今）、1（1 単位前）、2（1 と 1 の差分記憶）、3（前の 2 と現在の差分記憶、比率 2 : 3）、5（比率 3 : 5）… と 55 くらいまで。
  短期記憶全体を 1 つとして、中期記憶も同じ形（1 今の短期、1 前回の短期、2 …）、長期も同様。新しい記憶にははっきりした曲線を残しつつ、長期に積み重ねを残す。
ここでの作り（私の読み方）: 段 k は「古さが幅 W_k の範囲の記憶」を持つ板。W = 1, 1, 2, 3, 5, 8, 13, 21, 34, 55（フィボナッチ）。
  区切りごとに: 新しい記憶を段 0 に書く（足りない分だけ書く・鍵の共通部分を引く）。各段の中身のうち 1 / W_k を次の段へ送る
  （段 k の中身は平均して W_k 区切りのあいだそこに留まる。送るときは、段 k の板と送り先の板を比率で融合 = 持ち主の「比率 2 : 3」などに当たる）。
  一番上の段は送らずに積み重ねる（長期、λ = 0.995 で少しだけ薄れる）。読むときは確信度が一番高い段の答え。
比率の調整（持ち主の「間隔の比率調整」）: 幅を r 倍ずつにする W_k = r^k（r = 1.5, 2, 3）も比べる。
二重の系列（短期 → 中期）: 短期の段の一番上から送り出す分を、中期の系列の段 0 に入れる（mid）。
比べる: 板 1 枚、時間で板を分けて倍々に融合（実験92 の blocks、板 4 枚）。
場面は実験89 と同じ（300 組から 16 組 × 60 回）。CPU、3 seed。結果: results/exp93.txt
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
from exp92_fibtiers import run_blocks  # noqa: E402


def widths(kind, n):
    if kind == "fib":
        w = [1, 1]
        while len(w) < n:
            w.append(w[-1] + w[-2])
        return w[:n]
    r = float(kind[1:])
    return [max(1.0, r ** k) for k in range(n)]


def pick(Hs, kk):
    Ss = torch.stack([read(H, kk) for H in Hs])
    top = Ss.topk(2, -1).values
    mg = top[..., 0] / top[..., 1].clamp(min=1e-12)
    b = mg.argmax(0)
    return Ss.argmax(-1)[b, torch.arange(kk.shape[0])], mg.max(0).values, b


def run(keys_kind, seed, wkind, n_levels, nested):
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
    S = [zero() for _ in range(n_levels)]                       # 短期の段
    Mid = [zero() for _ in range(n_levels)] if nested else []   # 中期の段（短期の上から送られてくる）
    kmean, nk = zero(), 0
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
            am, mg, _ = pick(S + Mid, kk)
            w = w * torch.where((am == vv) & (mg > 2.0), 0.1, 1.0)
        # 各段の 1 / W_k を次の段へ送る（上から順に）
        def shift(levels):
            out_top = levels[-1] * 0
            for k in range(len(levels) - 1, -1, -1):
                move = levels[k] / W[k] if k < len(levels) - 1 else levels[k] * 0
                if k + 1 < len(levels):
                    levels[k + 1] = levels[k + 1] + move
                levels[k] = levels[k] - move
            return levels
        if nested:
            sent = S[-1] / W[-1]                                 # 短期の一番上から中期へ
            S[-1] = S[-1] - sent
            Mid = shift(Mid)
            Mid[0] = Mid[0] + sent
            Mid[-1] = 0.995 * Mid[-1]
        else:
            S[-1] = 0.995 * S[-1]
        S = shift(S)
        S[0] = S[0] + write(kk, vv, w)
    seen = count > 0
    cur = torch.where(changed, new_val, val)
    am, _, b = pick(S + Mid, keys - kmean)
    hit = (am == cur) & seen
    age = (T - 1 - last).float()
    grp = lambda m_: float(100.0 * hit[m_ & seen].float().mean()) if (m_ & seen).any() else None
    ch = changed & seen & (last >= T // 2)
    return dict(acc=grp(seen), acc_once=grp(count == 1), acc_5up=grp(count >= 5), acc_important=grp(important),
                acc_recent=grp(age < 10), acc_mid=grp((age >= 10) & (age < 30)), acc_old=grp(age >= 30),
                changed_new=float(100.0 * (am == new_val)[ch].float().mean()) if ch.any() else None,
                level_share=[float((b[seen] == j).float().mean()) for j in range(len(S) + len(Mid))])


def main():
    with open(os.path.join(ROOT, "results", "exp93.txt"), "a", encoding="utf-8") as out:
        for seed in range(3):
            for kind in ("rand", "word"):
                out.write(json.dumps(dict(exp="93", seed=seed, keys=kind, method="blocks", **run_blocks(kind, seed, False)), ensure_ascii=False) + "\n")
                for wk, n in (("fib", 10), ("r1.5", 10), ("r2", 7), ("r3", 5)):
                    for nested in (False, True):
                        rec = dict(exp="93", seed=seed, keys=kind, method=f"shift_{wk}" + ("_mid" if nested else ""), widths=widths(wk, n),
                                   **run(kind, seed, wk, n, nested))
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
