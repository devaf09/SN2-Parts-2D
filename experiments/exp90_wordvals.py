"""実験90: 記憶の値を実際の言葉にする（持ち主と決めた順 1 → 4 → 5 の 4、2026-10-26）

実験87〜89 は値がでたらめな窓だった。ここでは「直前 4 語 → 次の語」の組を、実際の文章（cc100 の単語、worddata）から作って記憶する。
  鍵 = 直前 4 語を実験78 の拡散の入れ方で光にしたもの（学習なし）、値 = 次の語（頻出 255 語 + その他 = 窓 256 個）。
  同じ値（「の」「、」など）が多くの鍵に付き、値の出方が大きく偏る。同じ鍵に違う次の語が来ることもある（文脈が同じでも次は 1 つに決まらない）。
課題 2 つ:
  once   : 文章から N 組（違う鍵）を取り、1 回で書く（実験87 と同じ）。そのまま / 差分（鍵の共通部分を引く）。
           比べる: 値をでたらめな窓にした同じ鍵（値の偏りの影響を分ける）、いつも一番多い値を答える（多数派）
  stream : 文章を頭から順に、16 組ずつ短期記憶に入れて長期記憶に移す（実験89 の移し方: そのまま・差分・差分 + 足りない分だけ書く）。
           最後に、出てきた組を取り出す（同じ鍵で次の語が違う組は、最後に来た語を正解とする）
GPU は使わない。結果: results/exp90.txt（実験89 の作り直し分は results/exp90_logdiff.txt）
"""
from __future__ import annotations

import json
import os
import sys

import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "text"))
from exp88_memkeys import K, L, conf, read  # noqa: E402
from exp89_consolidate import wmap  # noqa: E402


def load_pairs():
    import worddata
    from exp40_inputs import EncNet
    (X, y), _, info = worddata.load(4, K, 10 ** 9, 10, 0, 4096)      # 訓練の組を全部（文章の順は shuffle されるので、順序は文書から作り直す）
    enc = EncNet("spread", 4, info["V_in"], K, 1, 8, 0)
    # 文章の順に並べ直す（stream 用）: 訓練の文書を頭から
    in_idx, out_idx, OTHER, BOS = info["in_idx"], info["out_idx"], info["OTHER"], info["BOS"]
    seq_x, seq_y = [], []
    for d in info["tr_docs"][:3000]:
        ids = [BOS] * 4 + [in_idx.get(w, OTHER) for w in d]
        for i in range(len(d)):
            seq_x.append(ids[i:i + 4])
            seq_y.append(out_idx.get(d[i], K - 1))
    return torch.tensor(seq_x), torch.tensor(seq_y), enc


def field(enc, X):
    with torch.no_grad():
        return torch.cat([enc.field(X[i:i + 512]) for i in range(0, len(X), 512)]).to(torch.complex64)


def write(keys, vals, w=None):
    d = torch.zeros(len(vals), K)
    d[torch.arange(len(vals)), vals] = 1.0 if w is None else w
    return (torch.fft.fft2(keys).conj() * torch.fft.fft2(wmap(d))).sum(0) / (L * L)


def main():
    Xs, Ys, enc = load_pairs()
    out = open(os.path.join(ROOT, "results", "exp90.txt"), "a", encoding="utf-8")
    # once: 違う鍵の組を N 個
    uniq, inv = torch.unique(Xs, dim=0, return_inverse=True)
    first = torch.full((len(uniq),), -1, dtype=torch.long)
    first[inv.flip(0)] = torch.arange(len(Xs)).flip(0)
    for seed in range(3):
        g = torch.Generator().manual_seed(seed)
        perm = first[torch.randperm(len(uniq), generator=g)]
        for N in (32, 128, 512):
            idx = perm[:N]
            keys, vals = field(enc, Xs[idx]), Ys[idx]
            rvals = torch.randint(K, (N,), generator=g)
            major = 100.0 * (vals == vals.mode().values).float().mean().item()
            for mode in ("plain", "diff"):
                kk = keys - keys.mean(0) if mode == "diff" else keys
                for vname, vv in (("word", vals), ("rand", rvals)):
                    S = read(write(kk, vv), kk)
                    am = S.argmax(1)
                    _, mg = conf(S)
                    hit = am == vv
                    rec = dict(exp="90", part="once", seed=seed, N=N, mode=mode, values=vname, acc=100.0 * hit.float().mean().item(),
                               majority=major, n_distinct_vals=int(torch.unique(vv).numel()),
                               conf_hit=float(mg[hit].mean()) if hit.any() else None, conf_miss=float(mg[~hit].mean()) if (~hit).any() else None,
                               acc_rare=100.0 * hit[torch.bincount(vv, minlength=K)[vv] <= 2].float().mean().item())
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # stream: 文章を順に 16 組ずつ
    T, M, LAM = 60, 16, 0.97
    for seed in range(3):
        start = 5000 * seed
        Xw, Yw = Xs[start:start + T * M], Ys[start:start + T * M]
        keys_all = field(enc, Xw)
        for method in ("plain", "diff", "gated"):
            H = torch.zeros(L, L, dtype=torch.complex64)
            kmean, nk = torch.zeros(L, L, dtype=torch.complex64), 0
            for t in range(T):
                kk, vv = keys_all[t * M:(t + 1) * M], Yw[t * M:(t + 1) * M]
                if method != "plain":
                    kmean = (kmean * nk + kk.sum(0)) / (nk + M)
                    nk += M
                    kk = kk - kmean
                w = torch.ones(M)
                if method == "gated" and t > 0:
                    S = read(H, kk)
                    _, mg = conf(S)
                    w = torch.where((S.argmax(1) == vv) & (mg > 2.0), 0.1, 1.0)
                H = LAM * H + write(kk, vv, w)
            # 取り出し: 同じ鍵は最後に来た値が正解。古さ別も
            uq, inv2 = torch.unique(Xw, dim=0, return_inverse=True)
            last = torch.zeros(len(uq), dtype=torch.long)
            last[inv2] = torch.arange(len(Xw))
            kq = keys_all[last] - (kmean if method != "plain" else 0)
            S = read(H, kq)
            am = S.argmax(1)
            hit = am == Yw[last]
            age = (T * M - 1 - last) // M
            cnt = torch.bincount(inv2)
            rec = dict(exp="90", part="stream", seed=seed, method=method, acc=100.0 * hit.float().mean().item(),
                       acc_recent=100.0 * hit[age < 10].float().mean().item(), acc_old=100.0 * hit[age >= 30].float().mean().item(),
                       acc_repeated=100.0 * hit[cnt >= 2].float().mean().item() if (cnt >= 2).any() else None,
                       majority=100.0 * (Yw[last] == Yw[last].mode().values).float().mean().item(), n_keys=len(uq))
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out.close()


if __name__ == "__main__":
    main()
