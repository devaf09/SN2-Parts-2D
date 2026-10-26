"""実験88: 記憶 Grain の融合を、似た鍵で試す・確信度を測る（持ち主の依頼 2026-10-26: 似た鍵から。確信度で短期記憶を優先させたい）

実験87 の続き。記憶 Grain = 4f 系の板に「鍵 → 値」を学習なしで書き込んだもの。G1（短期、比率 α）と G2（長め、1 − α）を 1 枚に重ね書き。
鍵の種類:
  rand      : 位相だけのでたらめな模様（実験87 と同じ。互いにほぼ直交）
  sim<ρ>    : 全部の鍵に共通の模様を混ぜる。鍵 = √ρ · 共通 + √(1 − ρ) · でたらめ（ρ = 鍵どうしの似ている度合い）
  word      : 実際の単語の並び（直前 4 語、worddata）を、実験78 の拡散の入れ方（EncNet.field、学習なし）で光にしたもの。同じ語を含む鍵は似る
確信度（1 つの問い合わせごと）:
  share  : 一番明るい窓の明るさ / 窓全部の明るさの合計
  margin : 一番明るい窓 / 二番目に明るい窓
課題 2 つ:
  別々の記憶 : G1 と G2 に別の鍵を書く。両方を取り出せるか、確信度は G1・G2 でどう違うか、確信度と正解の関係
  食い違い   : 同じ鍵に、G1 は新しい値・G2 は古い値を書く（短期記憶で上書き）。どちらの値が出るか、そのときの確信度
GPU は使わない。結果: results/exp88.txt
"""
from __future__ import annotations

import json
import math
import os
import sys

import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "text"))
import sn2_parts_2d as p2  # noqa: E402

L, K = 64, 256
det = p2.window_grid(L, L, K, 2)
cent = torch.stack([torch.stack(torch.where(d > 0)).float().mean(1) for d in det]).round().long()


def points(vals):
    pts = torch.zeros(len(vals), L, L, dtype=torch.complex64)
    pts[torch.arange(len(vals)), cent[vals, 0], cent[vals, 1]] = 1
    return pts


def write(keys, vals):
    H = (torch.fft.fft2(keys).conj() * torch.fft.fft2(points(vals))).sum(0)
    return H / H.abs().pow(2).sum().sqrt()


def read(H, keys):
    return p2.read_windows(torch.fft.ifft2(torch.fft.fft2(keys) * H).abs().pow(2), det)


_word = {}


def word_keys(n_total, seed):
    if "X" not in _word:
        import worddata
        from exp40_inputs import EncNet
        (_, _), (Xt, _), info = worddata.load(4, 1024, 1000, 10000, 0, 4096)
        _word["X"] = torch.unique(Xt, dim=0)
        _word["enc"] = EncNet("spread", 4, info["V_in"], 1024, 1, 8, 0)
        _word["in_vocab"] = info["in_vocab"]
    X = _word["X"]
    i = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed))[:n_total]
    _word["last_X"] = X[i]                                   # 図のため: いま使った鍵の単語の番号
    with torch.no_grad():
        return _word["enc"].field(X[i]).to(torch.complex64)


def make_keys(kind, n_total, g, seed):
    if kind == "word":
        return word_keys(n_total, seed)
    rnd = torch.polar(torch.ones(n_total, L, L), 2 * math.pi * torch.rand(n_total, L, L, generator=g))
    if kind == "rand":
        return rnd
    rho = float(kind[3:])
    common = torch.polar(torch.ones(L, L), 2 * math.pi * torch.rand(L, L, generator=g))
    return math.sqrt(rho) * common + math.sqrt(1 - rho) * rnd


def conf(S):
    top = S.topk(2, 1).values
    return top[:, 0] / S.sum(1), top[:, 0] / top[:, 1].clamp(min=1e-12)


def main():
    out = open(os.path.join(ROOT, "results", "exp88.txt"), "a", encoding="utf-8")
    for seed in range(3):
        for kind in ("rand", "sim0.25", "sim0.5", "sim0.75", "word"):
            for n in (8, 32, 128):
                g = torch.Generator().manual_seed(seed * 1000 + n)
                keys = make_keys(kind, 2 * n, g, seed * 1000 + n)
                k1, k2 = keys[:n], keys[n:]
                v1, v2 = torch.randint(K, (n,), generator=g), torch.randint(K, (n,), generator=g)
                vold = (v1 + 1 + torch.randint(K - 1, (n,), generator=g)) % K        # 食い違い用の古い値（新しい値と必ず違う）
                kn = keys.flatten(1) / keys.flatten(1).norm(dim=1, keepdim=True)
                sim = (kn @ kn.conj().T).abs()
                sim_mean = float((sim.sum() - sim.diagonal().sum()) / (2 * n * (2 * n - 1)))
                H1, H2, H2c = write(k1, v1), write(k2, v2), write(k1, vold)
                for a in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
                    rec = dict(exp="88", seed=seed, keys=kind, n=n, alpha=a, key_sim=sim_mean)
                    H = math.sqrt(a) * H1 + math.sqrt(1 - a) * H2
                    for name, kk, vv in (("G1", k1, v1), ("G2", k2, v2)):
                        S = read(H, kk)
                        hit = S.argmax(1) == vv
                        sh, mg = conf(S)
                        rec[name] = dict(acc=100.0 * hit.float().mean().item(), share=float(sh.mean()), margin=float(mg.mean()),
                                         share_hit=float(sh[hit].mean()) if hit.any() else None, share_miss=float(sh[~hit].mean()) if (~hit).any() else None)
                    # 食い違い: 同じ鍵 k1 に、G1 は新しい値 v1、G2 は古い値 vold
                    S = read(math.sqrt(a) * H1 + math.sqrt(1 - a) * H2c, k1)
                    am = S.argmax(1)
                    sh, mg = conf(S)
                    rec["conflict"] = dict(new=100.0 * (am == v1).float().mean().item(), old=100.0 * (am == vold).float().mean().item(),
                                           share=float(sh.mean()), margin=float(mg.mean()),
                                           new_over_old=float((S[torch.arange(n), v1] / S[torch.arange(n), vold].clamp(min=1e-12)).median()))
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out.close()


if __name__ == "__main__":
    main()
