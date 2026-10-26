"""実験100: 段階的記憶（歯車）・新しい記憶が次々来る流れ・崩れた鍵（持ち主の依頼 2026-10-27）

持ち主の案（docs/memory_design.md）: 16 枚ほどの記憶域に連続して記憶し、ラスト 4 枚か抽出 4 枚を融合して、次の記憶域 16 枚に記憶させる
（16 丁の歯車と連動する 4 丁の歯車が、別の 16 丁の歯車に噛み合う）。
作り（gears）: 記憶域 4 段（各 16 枚の環、合わせて 64 枚）。段 0 は区切りごとに針を進めて書く（足りない分だけ書く）。
  段 s の針が 4 つ進むたびに、4 枚を融合して 1 枚にし、段 s+1 の針の板へ書く（段 s+1 の針を 1 つ進める。歯車比 4）。
    last4    : 直前に書いた 4 枚（ラスト 4 枚）
    extract4 : 段 s の 16 枚のうち、繰り返しの多い記憶をたくさん持つ 4 枚（抽出 4 枚。選ばれた板の「繰り返しの多さ」は半分にする）
  段 s の針が戻ってきた板は上書きする（中身はもう上へ融合済み、または選ばれずに忘れる）。一番上の段は 0.5 倍して重ねる。
  1 周の長さ: 段 0 は 16 区切り、段 1 は 64、段 2 は 256、段 3 は 1024 区切り。
比べる: 環状 64 枚・忘却なし / ただ薄れる（実験99 の一番）。デジタルの辞書（覚えた鍵に一番近い鍵の値、容量無限 = 上限の目安）。
場面（実験99 の弱点を直す）: 記憶の候補 3000 組（Zipf）から 16 組 × 2400 区切り。新しい記憶が次々来る。
崩れた鍵: 取り出すとき鍵を崩して問い合わせる。rand は 30% をでたらめな模様に置き換え、word_hash は直前 4 語のうち 1 語を別の語に。
計算は GPU（板をまとめて計算）。鍵: rand・word_hash。3 seed。結果: results/exp100.txt
実行: python experiments/exp100_gears.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "text"))
import exp88_memkeys as m88  # noqa: E402

K, L = m88.K, m88.L


class Mem:
    """板（記憶 + なじみ）をまとめて持ち、GPU でまとめて読む。"""

    def __init__(self, n, dev):
        self.H = torch.zeros(n, 2, L, L, dtype=torch.complex64, device=dev)
        self.det = m88.det.to(dev)
        self.cent = m88.cent.to(dev)
        self.dev = dev

    def write_pair(self, kk, vv, w):
        f = torch.zeros(len(vv), L, L, dtype=torch.complex64, device=self.dev)
        f[torch.arange(len(vv)), self.cent[vv, 0], self.cent[vv, 1]] = w.to(torch.complex64)
        f0 = torch.zeros_like(f)
        f0[:, self.cent[0, 0], self.cent[0, 1]] = w.to(torch.complex64)
        Kc = torch.fft.fft2(kk).conj()
        return torch.stack([(Kc * torch.fft.fft2(f)).sum(0), (Kc * torch.fft.fft2(f0)).sum(0)]) / (L * L)

    def readout(self, kq, chunk=128):
        """全部の板を、確信度 × なじみで重みを付けて足す（実験96）。返り値: 答え (n,)、確信度の最大 (n,)"""
        live = self.H[:, 0].abs().flatten(1).sum(1) > 0
        H = self.H[live]
        ans, best = [], []
        for i in range(0, len(kq), chunk):
            Kf = torch.fft.fft2(kq[i:i + chunk])[None]
            S = torch.einsum("pqyx,kyx->pqk", torch.fft.ifft2(Kf * H[:, 0, None]).abs().pow(2), self.det)
            fam = torch.fft.ifft2(Kf * H[:, 1, None]).abs().pow(2)[..., self.cent[0, 0], self.cent[0, 1]]
            fam = fam / fam.sum(0, keepdim=True).clamp(min=1e-12)
            top2 = S.topk(2, -1).values
            mg = top2[..., 0] / top2[..., 1].clamp(min=1e-12)
            Pn = S / S.sum(-1, keepdim=True).clamp(min=1e-12)
            ans.append((Pn * ((mg[..., None] - 1).clamp(min=0) * fam[..., None])).sum(0).argmax(-1))
            best.append(mg.max(0).values)
        return torch.cat(ans), torch.cat(best)


def corrupt_keys(kind, keys, sidx, g):
    """崩れた鍵: rand は 30% をでたらめな模様に、word_hash は 4 語のうち 1 語を別の語に。"""
    if kind == "rand":
        n = len(sidx)
        rnd = torch.polar(torch.ones(n, L, L), 2 * math.pi * torch.rand(n, L, L, generator=g))
        return math.sqrt(0.7) * keys[sidx] + math.sqrt(0.3) * rnd
    X = m88._word["last_X"][sidx].clone()
    pos = torch.randint(4, (len(X),), generator=g)
    X[torch.arange(len(X)), pos] = torch.randint(4094, (len(X),), generator=g)
    with torch.no_grad():
        return m88._word["enc"].field(X).to(torch.complex64)


def run(kind, seed, method, P=3000, T=2400, M=16, dev="cuda"):
    g = torch.Generator().manual_seed(seed)
    keys = m88.make_keys(kind, P, g, seed * 7 + 1)
    val = torch.randint(K, (P,), generator=g)
    freq = 1.0 / torch.arange(1, P + 1).float()
    freq = freq[torch.randperm(P, generator=g)]
    important = torch.rand(P, generator=g) < 0.1
    changed = torch.rand(P, generator=g) < 0.15
    new_val = (val + 1 + torch.randint(K - 1, (P,), generator=g)) % K
    keys_d = keys.to(dev)
    stages = 4
    mem = Mem(64, dev)
    ptr, since = [0] * stages, [0] * stages
    rep = torch.zeros(64, device=dev)                       # 板ごとの「繰り返しの多さ」（抽出 4 枚用）
    last_written = [[] for _ in range(stages)]
    kmean, nk = torch.zeros(L, L, dtype=torch.complex64, device=dev), 0
    count, last = torch.zeros(P), torch.full((P,), -1)

    def put(s, H, r):
        j = s * 16 + ptr[s]
        mem.H[j] = 0.5 * mem.H[j] + H if s == stages - 1 else H
        rep[j] = r
        last_written[s] = (last_written[s] + [j])[-4:]
        ptr[s] = (ptr[s] + 1) % 16
        since[s] += 1
        if s + 1 < stages and since[s] % 4 == 0:            # 4 つ進むたびに 4 枚を融合して上の段へ
            if method == "gears_last4":
                pick = last_written[s]
            else:
                cand = list(range(s * 16, s * 16 + 16))
                pick = [cand[i] for i in rep[cand].topk(4).indices.tolist()]
            rr = float(rep[pick].sum())
            rep[pick] = rep[pick] * 0.5
            put(s + 1, mem.H[pick].sum(0), rr)

    for t in range(T):
        cur = torch.where(changed & (t >= T // 2), new_val, val)
        idx = torch.multinomial(freq, M, replacement=False, generator=g)
        count[idx] += 1
        last[idx] = t
        kmean = (kmean * nk + keys_d[idx].sum(0)) / (nk + M)
        nk += M
        kk, vv = keys_d[idx] - kmean, cur[idx].to(dev)
        w = torch.where(important[idx], 3.0, 1.0).to(dev)
        if t > 0:
            am, mg = mem.readout(kk)
            w = w * torch.where((am == vv) & (mg > 2.0), 0.1, 1.0)
        new = mem.write_pair(kk, vv, w)
        if method == "ring64_keep":
            mem.H[t % 64] += new
        elif method == "ring64_decay":
            mem.H *= 0.995
            mem.H[t % 64] += new
        else:
            put(0, new, float((count[idx] - 1).sum()))
    seen = count > 0
    cur = torch.where(changed, new_val, val)
    sidx = torch.nonzero(seen).squeeze(1)
    am, _ = mem.readout(keys_d[sidx] - kmean)
    hit = am.cpu() == cur[sidx]
    kc = corrupt_keys(kind, keys, sidx, torch.Generator().manual_seed(seed + 99))
    amc, _ = mem.readout(kc.to(dev) - kmean)
    hitc = amc.cpu() == cur[sidx]
    # デジタルの辞書（容量無限）: 崩れた鍵に一番近い、覚えた鍵の値
    kn = (keys[sidx] - kmean.cpu()).flatten(1)
    kn = kn / kn.norm(dim=1, keepdim=True)
    qc = (kc - kmean.cpu()).flatten(1)
    qc = qc / qc.norm(dim=1, keepdim=True)
    nn = torch.cat([(qc[i:i + 512] @ kn.conj().T).abs().argmax(1) for i in range(0, len(qc), 512)])
    dict_fuzzy = 100.0 * (cur[sidx][nn] == cur[sidx]).float().mean().item()
    age = (T - 1 - last[sidx]).float()
    cnt = count[sidx]
    grp = lambda m_, h=hit: float(100.0 * h[m_].float().mean()) if m_.any() else None
    ch = changed[sidx] & (last[sidx] >= T // 2)
    return dict(n_seen=int(seen.sum()), acc=grp(torch.ones_like(hit)), acc_fuzzy=grp(torch.ones_like(hitc), hitc), dict_fuzzy=dict_fuzzy,
                acc_once=grp(cnt == 1), acc_2to4=grp((cnt >= 2) & (cnt <= 4)), acc_5up=grp(cnt >= 5), acc_important=grp(important[sidx]),
                acc_age0=grp(age < 16), acc_age16=grp((age >= 16) & (age < 256)), acc_age256=grp(age >= 256),
                fuzzy_5up=grp(cnt >= 5, hitc), changed_new=grp(ch) if ch.any() else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--T", type=int, default=2400)
    ap.add_argument("--P", type=int, default=3000)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--methods", default="ring64_keep,ring64_decay,gears_last4,gears_extract4")
    ap.add_argument("--out", default="results/exp100.txt")
    a = ap.parse_args()
    with open(a.out, "a", encoding="utf-8") as out:
        for seed in map(int, a.seeds.split(",")):
            for kind in ("rand", "word_hash"):
                for method in a.methods.split(","):
                    rec = dict(exp="100", seed=seed, keys=kind, method=method, T=a.T, P=a.P, **run(kind, seed, method, a.P, a.T, dev=a.device))
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    print(seed, kind, method, round(rec["acc"], 1), round(rec["acc_fuzzy"], 1), flush=True)


if __name__ == "__main__":
    main()
