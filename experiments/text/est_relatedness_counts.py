"""関連度（単語の近さ）で確率を掛け直す見積もり。数えた表だけで（光路なし）。"""
import sys, math, random, torch
from collections import Counter, defaultdict
sys.path.insert(0, __import__("os").path.dirname(__file__))
import worddata as w
docs = w.docs_words()
k = int(len(docs) * 0.85)
tr, te = docs[:k], docs[k:]
C, V = 1024, 4096
freq = Counter(x for d in tr for x in d)
out_vocab = [x for x, _ in freq.most_common(C - 1)]
oi = {x: i for i, x in enumerate(out_vocab)}
vocab = [x for x, _ in freq.most_common(V)]
vi = {x: i for i, x in enumerate(vocab)}
# 関連度: 同じ文書の ±10 語に一緒に出る回数 → PPMI → SVD 64 次元 → コサイン
W = 10
co = torch.zeros(V, V)
for d in tr:
    ids = [vi.get(x, -1) for x in d]
    for i, a in enumerate(ids):
        if a < 0: continue
        for b in ids[max(0, i - W):i]:
            if b >= 0:
                co[a, b] += 1; co[b, a] += 1
tot = co.sum(); r = co.sum(1, keepdim=True); c = co.sum(0, keepdim=True)
pmi = torch.log((co * tot) / (r * c).clamp(min=1e-9) + 1e-12).clamp(min=0)
U, S, _ = torch.linalg.svd(pmi)
E = U[:, :64] * S[:64].sqrt()
E = E / (E.norm(dim=1, keepdim=True) + 1e-9)
def near(word, m=8):
    s = E @ E[vi[word]]
    return [vocab[j] for j in s.topk(m + 1).indices.tolist()[1:]]
for q in ["海", "山", "東京", "猫", "仕事", "美味しい"]:
    if q in vi: print(q, "->", near(q))
# 基準: 2DTG の改良形（直前 + k 語前の組、数えた表）。関連度: 過去 H 語の平均ベクトルと候補語のコサイン
n = 4
def table(k):
    T = defaultdict(Counter)
    for d in tr:
        pad = ["<s>"] * n + d
        for i in range(len(d)):
            T[(pad[n + i - 1], pad[n + i - k])][oi.get(d[i], C - 1)] += 1
    return T
Ts = [table(k) for k in range(1, n + 1)]
U1 = defaultdict(Counter)
for d in tr:
    pad = ["<s>"] * n + d
    for i in range(len(d)):
        U1[pad[n + i - 1]][oi.get(d[i], C - 1)] += 1
def dist(c, back):
    p = torch.full((C,), 0.1 / C)
    if c:
        t = sum(c.values()); lam = 1.0
        for yy, v in c.items(): p[yy] += v / (t + lam)
        p = p * (t + lam) / (t + lam) ; p = p / p.sum()
        p = (t * p + lam * back) / (t + lam)
    else:
        p = back
    return p / p.sum()
pos = [(di, i) for di, d in enumerate(te) for i in range(len(d))]
random.Random(0).shuffle(pos); pos = pos[:5000]
Eo = torch.cat([torch.stack([E[vi[x]] for x in out_vocab]), torch.zeros(1, 64)])  # 答えの語のベクトル
base_p, rel_s, ys = [], [], []
for di, i in pos:
    d = te[di]; pad = ["<s>"] * n + d
    u = torch.zeros(C) + 0.1 / C
    c1 = U1.get(pad[n + i - 1])
    if c1:
        for yy, v in c1.items(): u[yy] += v
    u = u / u.sum()
    ps = [dist(Ts[k - 1].get((pad[n + i - 1], pad[n + i - k])), u) for k in range(1, n + 1)]
    base_p.append(0.4 * ps[0] + 0.3 * ps[1] + 0.2 * ps[2] + 0.1 * ps[3])
    hist = [vi[x] for x in d[max(0, i - 50):i] if x in vi and freq[x] < 2000]   # よく出る語（助詞など）は除く
    if hist:
        h = E[hist].mean(0); h = h / (h.norm() + 1e-9)
        rel_s.append(Eo @ h)
    else:
        rel_s.append(torch.zeros(C))
    ys.append(oi.get(d[i], C - 1))
B, R, Y = torch.stack(base_p), torch.stack(rel_s), torch.tensor(ys)
def ev(p):
    p = p / p.sum(1, keepdim=True)
    return round(100 * (p.argmax(1) == Y).float().mean().item(), 2), round(-torch.log(p[torch.arange(len(Y)), Y]).mean().item(), 3)
print("確率だけ", ev(B))
for lam in [0.5, 1, 2, 4, 8]:
    print("確率 × exp(%.1f × 関連度)" % lam, ev(B * torch.exp(lam * R)))
for lam in [0.5, 1, 2]:
    print("確率 × (1 + %.1f × 関連度+)" % lam, ev(B * (1 + lam * R.clamp(min=0))))
content = torch.tensor([freq[x] < 2000 for x in out_vocab] + [False])
Rc = R * content
mask = content[Y]
def ev2(p):
    p = p / p.sum(1, keepdim=True)
    a = (p.argmax(1) == Y)
    return round(100 * a.float().mean().item(), 2), round(-torch.log(p[torch.arange(len(Y)), Y]).mean().item(), 3), "内容語の位置", round(100 * a[mask].float().mean().item(), 2), round(-torch.log(p[torch.arange(len(Y)), Y])[mask].mean().item(), 3)
print("内容語の割合", round(100 * mask.float().mean().item(), 1))
print("確率だけ", ev2(B))
for lam in [1, 2, 4, 8]:
    print("内容語だけ 確率 × exp(%.0f × 関連度)" % lam, ev2(B * torch.exp(lam * Rc)))
print("--- 内容語の確率の合計は変えず、内容語の中だけ関連度で並べ替え")
Bn = B / B.sum(1, keepdim=True)
mass = (Bn * content).sum(1, keepdim=True)
for lam in [1, 2, 4, 8, 16]:
    q = Bn * content * torch.exp(lam * R)
    q = q / q.sum(1, keepdim=True) * mass
    print(lam, ev2(Bn * ~content + q))
