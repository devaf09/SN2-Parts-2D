"""実験58: 小さな 2DTG に表を分けて受け持たせ、バラバラに学習して足すだけで融合する（持ち主の方針、2026-10-21）

着想: 持ち主の方針。光路 1 本は覚えられる対応表が小さい（実験55: 直前 1 語 → 次の語でも板 64 枚で 29.6%、表は 33.6%）。
      1 本を大きくしても伸びは遅い（実験56: 板を倍にして +1 点）。光路の特性（並べる・足すのはただ）を生かし、
      小さな 2DTG を多数並べて表を分担させ、別々に学習して足すだけで融合する。
方法: 直前の語の番号（頻度順）を G で割った余りでグループに分ける（頻度が偏らないように）。2DTG-g は直前の語がグループ g の文脈を受け持つ。
      入力は (2 語前, 直前) の 2 語（実験53 の 2DTG-2 と同じ入れ方）。答えは 1024 クラス + 「担当なし」。
      学習: 自分の担当の文脈は本当の答え、担当外の文脈は「担当なし」（画像の実験10 の担当なしの学習）。担当外は 1 バッチの reject 割合。
      各 2DTG は自分のデータと担当外のデータしか見ない（ほかの 2DTG の重みも見ない）。
      融合 add: 確率を「担当なしでない度合い」(1 − p_担当なし) で重みをつけて足す（足すだけ。担当の情報は使わない）。
      融合 route: 直前の語のグループで 1 つを選ぶ（担当が分かっている場合の上限）。
条件: G = 8、各 2DTG は板 ① 16 枚 × 区画 16 + 符号読み 256（合計 128 枚 = 実験56 の 1 本 128 枚、32.3% と同じ計算）。
      訓練は全部（33 万件、各 2DTG の担当は約 4 万件）。3000 ステップずつ。
結果: results/exp58.txt

実行: python experiments/text/exp58_split.py --seed 0 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import sn2_parts_2d as p2  # noqa: E402
import textdata  # noqa: E402
import worddata  # noqa: E402
from exp40_inputs import EncNet  # noqa: E402
from exp52_words import PROMPTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="word", choices=["word", "char"], help="char = 文字の課題（実験62。直前 8 文字、64 クラス、入力 512 字）")
    ap.add_argument("--m", type=int, default=2, help="各 2DTG に見せる直前の単位の数（実験62）")
    ap.add_argument("--fixed_codes", action="store_true", help="符号を学習せず、全部の 2DTG で同じでたらめな符号にする（デジタルの重みを増やさない対照、実験64）")
    ap.add_argument("--G", type=int, default=8, help="2DTG の数（グループの数）")
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--patches", type=int, default=16)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--C", type=int, default=1024)
    ap.add_argument("--V", type=int, default=4096)
    ap.add_argument("--reject", type=float, default=0.3, help="1 バッチのうち担当外の文脈の割合")
    ap.add_argument("--max_train", type=int, default=400000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp58.txt")
    a = ap.parse_args()
    dev = torch.device(a.device)
    G = a.G
    if a.task == "word":
        n, C = 4, a.C
        (X, y), (Xt, yt), info = worddata.load(n, C, a.max_train, 10000, a.seed, a.V)
    else:
        n = 8
        (X, y), (Xt, yt), info = textdata.load(n, 64, a.max_train, 10000, a.seed)
        C, a.V = info["C"], info["V_in"]
        a.C = C
    pair = lambda Z: Z[:, -a.m:]                     # 直前 m 単位（既定: 2 語前, 直前）
    grp, grpt = X[:, -1] % G, Xt[:, -1] % G
    grains, t_all, per = [], 0.0, []
    for gi in range(G):
        own, other = (grp == gi).nonzero().squeeze(1), (grp != gi).nonzero().squeeze(1)
        torch.manual_seed(a.seed * 100 + gi)
        m = EncNet("bands", a.m, a.V, C + 1, a.layers, a.patches, a.seed * 100 + gi, a.K).to(dev)
        if a.fixed_codes:   # 全部の 2DTG で同じ符号（学習しない）。学習するのは光路（板）と温度だけ
            with torch.no_grad():
                m.codes.copy_(torch.randn(C + 1, a.K, generator=torch.Generator().manual_seed(4321)))
            m.codes.requires_grad_(False)
        opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=0.01)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
        g = torch.Generator().manual_seed(a.seed * 100 + gi)
        nr = int(256 * a.reject)
        t0 = time.perf_counter()
        for step in range(a.steps):
            io = own[torch.randint(len(own), (256 - nr,), generator=g)]
            ir = other[torch.randint(len(other), (nr,), generator=g)]
            xb = torch.cat([X[io], X[ir]])
            yb = torch.cat([y[io], torch.full((nr,), C)])
            loss = F.cross_entropy(m(pair(xb).to(dev)), yb.to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_all += time.perf_counter() - t0
        grains.append(m.eval())

    def probs(XX):
        with torch.no_grad():
            return torch.stack([torch.cat([torch.softmax(m(pair(XX[j:j + 250]).to(dev)), -1).cpu()
                                           for j in range(0, len(XX), 250)]) for m in grains])   # (G, N, C+1)

    def ev(p, yy):
        nll = -torch.log(p[torch.arange(len(yy)), yy] + 1e-12)
        return dict(acc=100.0 * (p.argmax(1) == yy).float().mean().item(), ce=nll.mean().item(),
                    top5=100.0 * (p.topk(5, 1).indices == yy[:, None]).any(1).float().mean().item())

    P = probs(Xt)
    resp = 1 - P[:, :, C]                                                     # 担当なしでない度合い (G, N)
    add = (P[:, :, :C] * resp[..., None]).sum(0)
    add = add / add.sum(1, keepdim=True)
    route = P[grpt, torch.arange(len(yt)), :C]
    route = route / route.sum(1, keepdim=True)
    chosen = resp.argmax(0)
    for gi in range(G):
        mk = grpt == gi
        q = P[gi][mk][:, :C]
        per.append(dict(g=gi, n_test=int(mk.sum()), own=ev(q / q.sum(1, keepdim=True), yt[mk]),
                        reject_on_other=100.0 * (P[gi][~mk].argmax(1) == C).float().mean().item()))
    res = dict(add=ev(add, yt), route=ev(route, yt), responsibility_correct=100.0 * (chosen == grpt).float().mean().item())
    demo = {}
    for ptxt in (PROMPTS if a.task == "word" else []):
        xp, ws = worddata.encode_prompt(ptxt, info, n)
        pp = probs(xp)
        r = 1 - pp[:, :, C]
        q = (pp[:, :, :C] * r[..., None]).sum(0)[0]
        q = q / q.sum()
        v, i = q.topk(5)
        demo[ptxt] = dict(words=ws, top5=[(info["out_vocab"][j], round(100 * float(x), 1)) for x, j in zip(v, i)])
    rec = dict(exp="58", **vars(a), results=res, per_grain=per, test_acc=res["add"]["acc"], test_ce=res["add"]["ce"], test_top5=res["add"]["top5"],
               params=sum(p.numel() for m in grains for p in m.parameters() if p.requires_grad), digital_code_params=sum(m.codes.numel() for m in grains if m.codes.requires_grad), train_s=t_all, demo=demo,
               version=p2.__version__, torch=torch.__version__)
    print(json.dumps(rec, ensure_ascii=False))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
