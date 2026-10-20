"""数えた表での見積もり: 2DT グレイン（持ち主の案）を足す方法（実験53 の前、2026-10-20）。実行: python experiments/text/est_grains_counts.py"""
import sys, math, torch
sys.path.insert(0, __import__("os").path.dirname(__file__))
import worddata as w
n = 4
(X, y), (Xt, yt), info = w.load(n, 1024, 10**7, 10000, 0, 4096)
V, C = info["V_in"], info["C"]
# 2DTG-k の代わりに、数えた表 P_k(y | k 語前) を使う（光路の天井の見積もり）
P = []
for k in range(1, n + 1):
    T = torch.zeros(V, C)
    T.index_put_((X[:, n - k], y), torch.ones(len(y)), accumulate=True)
    P.append((T + 0.1) / (T.sum(1, keepdim=True) + 0.1 * C))
Pt = torch.stack([P[k][Xt[:, n - 1 - k]] for k in range(n)])  # (n, N, C)
def ev(p):
    return round(100 * (p.argmax(1) == yt).float().mean().item(), 1), round(-torch.log(p[torch.arange(len(yt)), yt]).mean().item(), 3)
for k in range(n):
    print("only k=%d" % (k + 1), ev(Pt[k]))
for K in range(2, n + 1):
    for wts in ([1] * K, [0.5, 0.2, 0.2, 0.1][:K], [0.6, 0.25, 0.1, 0.05][:K]):
        wt = torch.tensor(wts, dtype=torch.float)[:, None, None]; wt = wt / wt.sum()
        lin = (Pt[:K] * wt).sum(0)
        lg = (torch.log(Pt[:K]) * wt * K).sum(0); prod = torch.softmax(lg, 1)
        print(K, wts, "sum", ev(lin), "product", ev(prod))
    conf = Pt[:K].max(2).values[..., None]  # 確信度 = 一番上の確率
    cw = conf / conf.sum(0, keepdim=True)
    print(K, "confidence-weighted sum", ev((Pt[:K] * cw).sum(0)))
    ent = -(Pt[:K] * torch.log(Pt[:K])).sum(2, keepdim=True)
    ew = torch.softmax(-ent, 0)
    print(K, "entropy-weighted sum", ev((Pt[:K] * ew).sum(0)))
print("--- pair grains: (1 語前, k 語前) -> y")
from collections import defaultdict, Counter
def pair_table(k):
    T = defaultdict(Counter)
    for a, b, yy in zip(X[:, n - 1].tolist(), X[:, n - k].tolist(), y.tolist()):
        T[(a, b)][yy] += 1
    return T
def pair_prob(T, k, lam=1.0):
    out = torch.zeros(len(yt), C)
    for i, (a, b) in enumerate(zip(Xt[:, n - 1].tolist(), Xt[:, n - k].tolist())):
        c = T.get((a, b))
        if c:
            tot = sum(c.values())
            for yy, v in c.items():
                out[i, yy] = v / (tot + lam)
            out[i] += lam / (tot + lam) * P[0][a]
        else:
            out[i] = P[0][a]
    return out
Q = [Pt[0]] + [pair_prob(pair_table(k), k) for k in range(2, n + 1)]
for k, q in enumerate(Q):
    print("pair grain k=%d" % (k + 1), ev(q))
for wts in ([0.4, 0.4, 0.1, 0.1], [0.2, 0.5, 0.2, 0.1], [0.25, 0.25, 0.25, 0.25]):
    print(wts, ev(sum(wv * q for wv, q in zip(wts, Q))))
