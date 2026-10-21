"""実験67: 逆向きの「ぼんやり」を固める（持ち主の案、2026-10-21）

実験65 の続き。位相差も読む光路（実験63、位相だけの板、直前 1 語 → 次の語）を逆向きに通すと、「D の前に来やすい語」がぼんやり光る（上位 20 語のうち 7 語）。
これを固める 3 つの方法:
  1. ensemble: 同じ種類の光路（seed 0〜3）の逆向きを掛け合わせる（対数の和）
  2. refine  : 逆向きの上位 50 語だけを順向きに通して確かめる（P(D | 語) × 語の頻度。全部の語を通すより 80 倍軽い）
  3. cloze   : 穴埋め。テストの文の連続 3 語 (w, x, D) で、x を当てる。左の手がかり = 順向き（w を入れて x の確率）、
               右の手がかり = 逆向き（D を出口から入れて x の反応）、両方を掛け合わせる
結果: results/exp67.txt
実行: python experiments/text/exp67_firm.py
"""
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import worddata  # noqa: E402
from exp63_amp_phase import AmpPhaseNet  # noqa: E402
from exp65b_combine import reverse_map  # noqa: E402

V, C = 4096, 1024


def main():
    (X, y), _, info = worddata.load(1, C, 400000, 100, 0, V)
    cnt = torch.zeros(C, V)
    cnt.index_put_((y, X[:, 0]), torch.ones(len(y)), accumulate=True)
    prev_freq = torch.bincount(X[:, 0], minlength=V).float()
    models = []
    for s in range(4):
        path = f"results/models/exp63_phase_int_phase_s{s}.pt"
        if os.path.exists(path):
            m = AmpPhaseNet("phase", "int_phase", 1, V, C)
            m.load_state_dict(torch.load(path))
            models.append(m.eval())
    rec = dict(exp="67", n_models=len(models))
    with torch.no_grad():
        fwd = [torch.cat([torch.softmax(m(torch.arange(j, min(j + 256, V))[:, None]), -1) for j in range(0, V, 256)]) for m in models]  # (V, C)
        # 1・2: 上位 20 語の重なり
        ens = {k: [] for k in ("rev_1", f"rev_prod_{len(models)}", "refine_top50", "forward_bayes_1", f"forward_bayes_{len(models)}")}
        rev_cache = {}
        for t in range(50):
            truth = cnt[t]
            if truth.sum() < 50:
                continue
            top = set(truth.topk(20).indices.tolist())
            maps = [reverse_map(m, "phase", t) for m in models]
            rev_cache[t] = maps
            prod = sum(torch.log(r + 1e-12) for r in maps)
            fb1 = fwd[0][:, t] * prev_freq
            fbn = torch.stack([f[:, t] for f in fwd]).mean(0) * prev_freq
            cand = prod.topk(50).indices
            ref = torch.full((V,), -1.0)
            ref[cand] = fbn[cand]
            for name, sc in (("rev_1", maps[0]), (f"rev_prod_{len(models)}", prod), ("refine_top50", ref),
                             ("forward_bayes_1", fb1), (f"forward_bayes_{len(models)}", fbn)):
                ens[name].append(len(top & set(sc.topk(20).indices.tolist())))
        rec["top20_overlap"] = {k: sum(v) / len(v) for k, v in ens.items()}
        # 3: 穴埋め
        k = int(len(worddata.docs_words()) * 0.85)
        te = worddata.docs_words()[k:]
        iv, ov = info["in_idx"], info["out_idx"]
        trip = [(d[i - 1], d[i], d[i + 1]) for d in te for i in range(1, len(d) - 1)
                if d[i - 1] in iv and d[i] in iv and d[i + 1] in ov]
        random.Random(0).shuffle(trip)
        trip = trip[:2000]
        prior = torch.log(prev_freq + 1)
        hits = {"left_forward": 0, "right_reverse": 0, "right_reverse_prod": 0, "both": 0, "both_prod": 0}
        for w, x, dd in trip:
            xi, wi, di = iv[x], iv[w], ov[dd]
            # 左: 順向き。x の候補 c について P(c | w)。x は入力の語なので、出力の語にある場合だけ番号を合わせる
            left = torch.full((V,), -30.0)
            for cword, ci in ov.items():
                if cword in iv:
                    left[iv[cword]] = torch.log(torch.stack([f[wi, ci] for f in fwd]).mean() + 1e-12)
            if di not in rev_cache:
                rev_cache[di] = [reverse_map(m, "phase", di) for m in models]
            r1 = torch.log(rev_cache[di][0] + 1e-12)
            rp = sum(torch.log(r + 1e-12) for r in rev_cache[di]) / len(models)
            for name, sc in (("left_forward", left), ("right_reverse", r1 + 0 * prior), ("right_reverse_prod", rp),
                             ("both", left + r1), ("both_prod", left + rp)):
                hits[name] += int(sc.argmax()) == xi
        rec["cloze_acc"] = {k2: 100.0 * v / len(trip) for k2, v in hits.items()}
        rec["cloze_n"] = len(trip)
    print(json.dumps(rec, ensure_ascii=False))
    with open("results/exp67.txt", "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
