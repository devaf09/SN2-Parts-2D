"""実験65b: 逆向きの「ぼんやりした反応」を複数合わせて固める（持ち主の案、2026-10-21）

位相差も読む光路（実験63 の int_phase）では、出口の符号に位相の部分があるので、逆向きに通すと「D の前に来やすい語」がぼんやり光る（実験65）。
別々に学習した光路（位相だけの板・振幅 + 位相の板・振幅だけの板）の逆向きの反応を、
  sum = 割合にして足す、prod = 掛け算（対数の和）
で合わせ、上位 20 語の重なりが増えるか（固まるか）を見る。
実行: python experiments/text/exp65b_combine.py
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import worddata  # noqa: E402
from exp63_amp_phase import AmpPhaseNet  # noqa: E402

L, V, C = 64, 4096, 1024


def reverse_map(m, plate, t):
    net = m.net
    K = net.det.shape[0]
    cre, cim = m.codes[t, K:2 * K], m.codes[t, 2 * K:3 * K]
    u = ((cre[:, None, None] + 1j * cim[:, None, None]) * net.det).sum(0).to(torch.complex64)
    u = torch.fft.ifft2(torch.fft.ifftshift(u, dim=(-2, -1)), norm="ortho")
    for d in reversed(range(len(net.kinds))):
        u = torch.fft.ifft2(torch.fft.fft2(u) * net.H.conj())
        if plate in ("amp", "both"):
            u = u * torch.exp(0.5 * m.ampnet.plate_phase(d))
        if plate in ("phase", "both"):
            u = u * torch.polar(torch.ones(L, L), -net.plate_phase(d))
    F = torch.fft.fft2(u)
    r = F.abs()[m.ky.long(), m.kx.long()] ** 2
    return r / r.sum()


def main():
    (X, y), _, info = worddata.load(1, C, 400000, 100, 0, V)
    cnt = torch.zeros(C, V)
    cnt.index_put_((y, X[:, 0]), torch.ones(len(y)), accumulate=True)
    models = {}
    for pl in ("phase", "both", "amp"):
        m = AmpPhaseNet(pl, "int_phase", 1, V, C)
        m.load_state_dict(torch.load(f"results/models/exp63_{pl}_int_phase_s0.pt"))
        models[pl] = m.eval()
    combos = {"phase": ["phase"], "both": ["both"], "amp": ["amp"], "sum(phase,both,amp)": ["phase", "both", "amp"], "prod(phase,both,amp)": ["phase", "both", "amp"],
              "sum(phase,both)": ["phase", "both"], "prod(phase,both)": ["phase", "both"]}
    res = {k: [] for k in combos}
    with torch.no_grad():
        for t in range(50):
            truth = cnt[t]
            if truth.sum() < 50:
                continue
            top = set(truth.topk(20).indices.tolist())
            maps = {pl: reverse_map(m, pl, t) for pl, m in models.items()}
            for name, pls in combos.items():
                if name.startswith("prod"):
                    sc = sum(torch.log(maps[p] + 1e-12) for p in pls)
                else:
                    sc = sum(maps[p] for p in pls)
                res[name].append(len(top & set(sc.topk(20).indices.tolist())))
    summ = {k: sum(v) / len(v) for k, v in res.items()}
    rec = dict(exp="65b", top20_overlap=summ, n_targets=len(res["phase"]))
    print(json.dumps(rec, ensure_ascii=False))
    with open("results/exp65.txt", "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
