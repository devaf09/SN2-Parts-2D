"""実験84: 単語の課題で、拡散 + 4f 系 + 平方根（光路 1 本・分担の 2DTG）と足し合わせ（持ち主と話した「数える方式が苦手な土俵」）

着想: 文字の課題（64 種類、データが多い）では、4f 系の足し合わせが 4-gram と並んだ（実験82）。文字の種類が少なくデータが多い課題は、
      数える方式（n-gram）の得意な土俵。単語（入力 4096 種類、答え 1024 種類、見たことのない組み合わせが多い）なら差が出るかもしれない。
方法: 入れ方は実験40 の EncNet.field（拡散）、中身は exp75_4f.Multi4f（平方根、レンズの後で読む）、窓 K 個 → クラスの符号（実験52・58 と同じ符号読み）。
      --G 1 で光路 1 本、--G > 1 で直前の語の番号の余りで担当を決め打ちにした分担の 2DTG。--save_logits でテストの対数確率を保存（exp81_fuse.py で足し合わせ）。
課題: 直前 4 語 → 次の語（worddata、答え 1023 語 + その他、訓練 40 万件、テスト 1 万件）。
比べる: 板の分担の 2DTG 36.0%（実験58・62）、単語の 3-gram 36.4%（実験52）、GRU 36.9%。
結果: results/exp84.txt
実行: python experiments/text/exp84_word4f.py --G 1 --stages 6 --width 8 --steps 9000 --seed 0 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import sn2_parts_2d as p2  # noqa: E402
import worddata  # noqa: E402
from exp40_inputs import EncNet  # noqa: E402
from exp75_4f import Multi4f  # noqa: E402


class Word4f(torch.nn.Module):
    def __init__(self, enc, n, V, C, K, stages, width, kind, read, seed):
        super().__init__()
        self.enc = EncNet(enc, n, V, C, 1, 8, seed)      # 入れ方と位置ごとの明るさだけを使う
        self.body = Multi4f(64, K, stages, width, kind, read, seed)
        self.codes = torch.nn.Parameter(torch.randn(C, K, generator=torch.Generator().manual_seed(seed)))
        self.temp = torch.nn.Parameter(torch.tensor(10.0))

    def forward(self, X):
        w = p2.read_windows(self.body.screen(self.enc.field(X)), self.body.det)
        return p2.code_scores(w, self.codes, 1.0) * self.temp

    def opt_params(self):
        return [self.enc.w, self.codes, self.temp] + [q for n, q in self.body.named_parameters() if q.requires_grad and n not in ("gain", "bias")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--G", type=int, default=1)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--C", type=int, default=1024)
    ap.add_argument("--enc", default="spread")
    ap.add_argument("--stages", type=int, default=4)
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--kind", default="sqrt")
    ap.add_argument("--read", default="lens", choices=["lens", "plane"])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--max_train", type=int, default=400000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp84.txt")
    ap.add_argument("--save_logits", default="")
    ap.add_argument("--ngram", action="store_true", help="同じテストで単語の 2〜4-gram も測る（CPU）")
    a = ap.parse_args()
    dev = torch.device(a.device)
    (X, y), (Xt, yt), info = worddata.load(max(4, a.n), a.C, a.max_train, 10000, a.seed, 4096)
    V, C = info["V_in"], a.C
    X, Xt = X[:, -a.n:], Xt[:, -a.n:]
    grp, grpt = X[:, -1] % a.G, Xt[:, -1] % a.G
    S = torch.zeros(len(yt), C)
    params, tr_hit, tr_n = 0, 0, 0
    t0 = time.perf_counter()
    for gi in range(a.G):
        own, ownt = (grp == gi).nonzero().squeeze(1), (grpt == gi).nonzero().squeeze(1)
        sd = a.seed * 100 + gi
        torch.manual_seed(sd)
        m = Word4f(a.enc, a.n, V, C, a.K, a.stages, a.width, a.kind, a.read, sd).to(dev)
        opt = torch.optim.Adam(m.opt_params(), lr=0.01)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
        g = torch.Generator().manual_seed(sd)
        for _ in range(a.steps):
            i = own[torch.randint(len(own), (256,), generator=g)]
            loss = F.cross_entropy(m(X[i].to(dev)), y[i].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
        with torch.no_grad():
            if len(ownt):
                S[ownt] = torch.cat([m(Xt[ownt][j:j + 250].to(dev)).cpu() for j in range(0, len(ownt), 250)])
            o = own[:2000]
            tr_hit += int((m(X[o].to(dev)).cpu().argmax(1) == y[o]).sum())
            tr_n += len(o)
        params += sum(q.numel() for q in m.opt_params())
    train_s = time.perf_counter() - t0
    acc, ce = 100.0 * (S.argmax(1) == yt).float().mean().item(), F.cross_entropy(S, yt).item()
    top5 = 100.0 * (S.topk(5, 1).indices == yt[:, None]).any(1).float().mean().item()
    extra = {}
    if a.ngram:
        for k in (2, 3, 4):
            extra[f"ngram{k}"] = worddata.ngram(info, k, 4)
    rec = dict(exp="84", **vars(a), test_acc=acc, test_ce=ce, top5=top5, train_acc=100.0 * tr_hit / max(1, tr_n), params=params,
               digital=C * a.K * a.G, train_s=train_s, **extra, version=p2.__version__, torch=torch.__version__)
    print(json.dumps(rec, ensure_ascii=False))
    if a.save_logits:
        os.makedirs(os.path.dirname(a.save_logits), exist_ok=True)
        torch.save({"logp": F.log_softmax(S, 1), "y": yt}, a.save_logits)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
