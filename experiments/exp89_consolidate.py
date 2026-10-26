"""実験89: 短期記憶から長期記憶へ移す方法（持ち主の構想 2026-10-26、docs/memory_design.md の 1 繰り返し・2 関連・3 重要度）

場面: 記憶の候補（鍵 → 値）が P 組あり、話の区切り（エピソード）ごとに M 組が短期記憶に入る。よく出る組は何度も来る（出る頻度は Zipf）。
      区切りのたびに、短期記憶（4f 系の板、鍵の共役 × 値の点）を長期記憶の板に足す。長期記憶は区切りごとに薄れる（λ = 0.97、忘却）。
      後半で一部の組の値が変わる（新旧の食い違い）。一部の組には「重要」の印。
移し方（どれも長期 ← λ · 長期 + 短期の書き込み）:
  plain    : 短期の組をそのまま書く（比べる相手）
  diff     : 差分記憶法（持ち主の案 1 の前処理）。長期記憶が覚えている「鍵の共通部分」（今までの鍵の平均、長期記憶と一緒に持つ）を、
             書くときの鍵からも読むときの鍵からも引く。似た鍵どうしの共通成分が背景として積み上がらない
  repeat   : diff + 繰り返し記憶（案 1）。長期から同じ値が取り出せる組（= 前にも来た、共通の組）の書く重みを 3 倍にする（強度を上げる）
  relate   : diff + 関連記憶（案 2）の簡単な形。長期から読んだときの確信度が高い組（今の記憶と関連が強い）ほど重く書く
  gated    : diff + 足りない分だけ書く繰り返し記憶。すでに確信度 2 以上で正しく取り出せる組は書く重みを 0.1 に（強度が上がりすぎて、ほかを埋もれさせない）
  gated_upd: gated + update
  logdiff  : diff + 案 1 の手順（実験90）。差分マップ = log(前段の値の地図 + ε) − log(長期から読んだ地図 + ε)。値の窓の差分の大きさだけ書く
  logdiff_soft: 同じ差分マップで、softmax(差分マップ + 現在の記憶) − 現在の記憶 を書く（引く窓もある）
  update   : diff + 上書き。長期から読んだ値が新しい値と違えば、古い値の窓を引く（読み出した光の差分を書く。不可逆で OK、持ち主）
重要度（案 3）: 印の付いた組は、どの方法でも書く重みを 3 倍にする。
測るもの: 最後に、出てきた全部の組を長期から取り出す正解率（最新の値が正解）。出てきた回数別・重要の印別・古さ別、
          値が変わった組で新しい値 / 古い値が出る割合、確信度（一番明るい窓 / 二番目）。
鍵: rand、word（単語の並び、拡散）、sim0.25・sim0.5・sim0.75（共通の模様を混ぜて似せた）。3 seed、CPU。結果: results/exp89.txt
"""
from __future__ import annotations

import json
import os
import sys

import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from exp88_memkeys import K, L, cent, conf, make_keys, read  # noqa: E402

P, T, M, LAM = 300, 60, 16, 0.97
METHODS = ("plain", "diff", "repeat", "relate", "update", "gated", "gated_upd", "logdiff", "logdiff_soft")


def wmap(d):
    """窓ごとの数 (n, K) → 窓の中心に光の点を置いた場 (n, L, L)。"""
    D = torch.zeros(d.shape[0], L, L, dtype=torch.complex64)
    D[:, cent[:, 0], cent[:, 1]] = d.to(torch.complex64)
    return D


def run(kind, method, seed):
    g = torch.Generator().manual_seed(seed)
    keys = make_keys(kind, P, g, seed * 7 + 1)
    val = torch.randint(K, (P,), generator=g)
    freq = 1.0 / torch.arange(1, P + 1).float()
    freq = freq[torch.randperm(P, generator=g)]
    important = torch.rand(P, generator=g) < 0.1
    changed = torch.rand(P, generator=g) < 0.15
    new_val = (val + 1 + torch.randint(K - 1, (P,), generator=g)) % K
    H = torch.zeros(L, L, dtype=torch.complex64)
    kmean, nk = torch.zeros(L, L, dtype=torch.complex64), 0
    count, last = torch.zeros(P), torch.full((P,), -1)
    use_diff = method != "plain"
    for t in range(T):
        cur = torch.where(changed & (t >= T // 2), new_val, val)
        idx = torch.multinomial(freq, M, replacement=False, generator=g)
        count[idx] += 1
        last[idx] = t
        kk = keys[idx]
        if use_diff:                                   # 鍵の共通部分（長期記憶が持つ平均）を更新して引く
            kmean = (kmean * nk + kk.sum(0)) / (nk + M)
            nk += M
            kk = kk - kmean
        w = torch.ones(M)
        w[important[idx]] *= 3.0
        d = torch.zeros(M, K)
        d[torch.arange(M), cur[idx]] = 1.0
        if t > 0 and method in ("repeat", "relate", "update", "gated", "gated_upd", "logdiff", "logdiff_soft"):
            S = read(H, kk)
            am = S.argmax(1)
            if method == "repeat":
                w = w * (1 + 2.0 * (am == cur[idx]).float())
            elif method == "relate":
                _, mg = conf(S)
                w = w * (1 + torch.log(mg).clamp(0, 2))
            elif method in ("logdiff", "logdiff_soft"):
                # 案 1 の手順（実験90 で追加）: 前段（短期）の値の地図 d と、現在（長期）から読んだ地図 p（窓の明るさを合計 1 に）の対数の差 = 差分マップ。
                # 差分マップ log(d + ε) − log(p + ε) を、softmax（差分マップ + 現在の記憶）で窓ごとの書く量にして融合する。
                # 長期がすでに値の窓を強く出していれば差分は小さく、書く量も小さい（「足りない分だけ書く」を対数で連続にした形）
                pL = S / S.sum(1, keepdim=True)
                eps = 1.0 / K
                Dm = torch.log(d + eps) - torch.log(pL + eps)                    # 差分マップ（対数）
                if method == "logdiff":
                    a = (Dm[torch.arange(M), cur[idx]] / torch.log(torch.tensor(1.0 + K))).clamp(0, 1)
                    w = w * a                                                   # 値の窓の差分の大きさだけ書く
                else:
                    q = torch.softmax(Dm + torch.log(pL + eps), 1)              # softmax(差分マップ + 現在の記憶)
                    d = q - pL                                                  # 現在の記憶から q へ近づける分（引く窓もある）
            elif method in ("gated", "gated_upd"):
                # 繰り返し記憶を「足りない分だけ書く」形に: すでに確信度 2 以上で正しく取り出せる組は書く重みを 0.1 にする（強度が上がりすぎない）
                _, mg = conf(S)
                w = w * torch.where((am == cur[idx]) & (mg > 2.0), 0.1, 1.0)
                if method == "gated_upd":
                    wrong = am != cur[idx]
                    d[torch.arange(M)[wrong], am[wrong]] -= 1.0
            else:
                wrong = am != cur[idx]
                d[torch.arange(M)[wrong], am[wrong]] -= 1.0     # 古い値の窓を引く
        H = LAM * H + (torch.fft.fft2(kk).conj() * torch.fft.fft2(wmap(d * w[:, None]))).sum(0) / (L * L)
    seen = count > 0
    cur = torch.where(changed, new_val, val)
    kq = keys - kmean if use_diff else keys
    S = read(H, kq)
    am = S.argmax(1)
    hit = (am == cur) & seen
    _, mg = conf(S)
    age = (T - 1 - last).float()
    grp = lambda m_: float(100.0 * hit[m_ & seen].float().mean()) if (m_ & seen).any() else None
    ch = changed & seen & (last >= T // 2)
    return dict(acc=grp(seen), acc_once=grp(count == 1), acc_2to4=grp((count >= 2) & (count <= 4)), acc_5up=grp(count >= 5),
                acc_important=grp(important), acc_normal=grp(~important), acc_recent=grp(age < 10), acc_old=grp(age >= 30),
                changed_new=float(100.0 * (am == new_val)[ch].float().mean()) if ch.any() else None,
                changed_old=float(100.0 * (am == val)[ch].float().mean()) if ch.any() else None,
                conf_hit=float(mg[hit].mean()) if hit.any() else None, conf_miss=float(mg[seen & ~hit].mean()) if (seen & ~hit).any() else None,
                n_seen=int(seen.sum()))


def main():
    with open(os.path.join(ROOT, "results", "exp89.txt"), "a", encoding="utf-8") as out:
        for seed in range(3):
            for kind in ("rand", "word", "sim0.25", "sim0.5", "sim0.75"):
                for method in METHODS:
                    rec = dict(exp="89", seed=seed, keys=kind, method=method, **run(kind, method, seed))
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
