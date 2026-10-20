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
for q in ["王","女王","亭主","女房","男","女","父","母","息子","娘","彼","彼女","夫","妻","兄","姉","弟","妹","男性","女性","日本","東京","大阪","行く","行っ","来る","来","大きい","小さい","食べる","食べ","飲む","飲ま","見る","見"]:
    print(q, q in vi, freq[q])
def ana(a, b, c, m=5):
    if not all(x in vi for x in (a, b, c)): return None
    v = E[vi[b]] - E[vi[a]] + E[vi[c]]
    s = E @ (v / v.norm())
    for x in (a, b, c): s[vi[x]] = -9
    return [vocab[j] for j in s.topk(m).indices.tolist()]
for a,b,c in [("男","女","父"),("男","女","彼"),("男性","女性","父"),("夫","妻","父"),("父","母","息子"),("兄","姉","弟"),("食べる","飲む","食べ"),("食べる","食べ","見る"),("行く","行っ","来る"),("大きい","小さい","多い"),("日本","東京","大阪")]:
    print(a,"→",b," = ",c,"→",ana(a,b,c))
# 差分ベクトルどうしの向きの近さ
def diff(a,b): v=E[vi[b]]-E[vi[a]]; return v/v.norm()
import itertools
pairs=[("男","女"),("男性","女性"),("父","母"),("夫","妻"),("兄","姉"),("彼","彼女"),("食べる","食べ"),("見る","見"),("大きい","小さい")]
pairs=[p for p in pairs if all(x in vi for x in p)]
for p,q in itertools.combinations(pairs,2):
    print(p,q,round(float(diff(*p)@diff(*q)),2))
