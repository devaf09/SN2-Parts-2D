"""実験63: 板の種類（位相だけ・振幅だけ・振幅 + 位相）と、読み方（明るさだけ・位相差も読む）の比較（持ち主の依頼、2026-10-21）

着想: 持ち主の問い。今の板は位相だけ（明るさを変えずに位相を A·sin(ω·x + φ) 遅らせる）。ホログラムには、位相で光を曲げる型（位相型）と、
      場所ごとに光を減らす型（振幅型・吸収型）がある。両方を比べ、両方を持つ板（振幅 + 位相）も試す。
      また、今は明るさ |U|² しか読まないので、出口の光の位相は捨てている。参照光と干渉させれば位相差も読める（干渉計）。
      レンズの後の光 U を、明るさ |U|² に加えて、参照光 R（平らな光）との干渉 Re(U·R*)、Im(U·R*)（= 位相差の cos・sin × 強さ）でも窓で読む。
方法: 直前 1 語 → 次の語（実験55 と同じ、数えた表なら 33.6%）。入力は語ごとの向きの平面波（帯 1 本）。
      板 ① 32 枚 × 区画 16。板 d の透過 = T_d·exp(j·φ_d)、φ_d = A·sin(...)（位相）、T_d = exp(B·sin(...))（振幅、帯ごとに学習。光を減らす・強めるのどちらも
      学べるが、板ごとに明るさの平均を 1 に戻すので、実質は場所ごとの減らし方の比）。
        phase = 位相だけ（今まで）、amp = 振幅だけ、both = 振幅 + 位相
      読み方: int = 明るさの窓 256 個、int_phase = 明るさ 256 + Re 256 + Im 256 の窓の値をまとめて符号と比べる。
条件: 6 万件、3000 ステップ、学習率 0.01 → 0、seed 0。
結果: results/exp63.txt

実行: python experiments/text/exp63_amp_phase.py --plate both --read int_phase --seed 0 --device cuda
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
import worddata  # noqa: E402
from exp40_inputs import EncNet  # noqa: E402


class AmpPhaseNet(EncNet):
    def __init__(self, plate, read, n, V, C, layers=32, patches=16, seed=0, K=256):
        super().__init__("bands", n, V, C, layers, patches, seed, K)
        self.plate, self.read = plate, read
        self.ampnet = p2.Net2D(64, 1, ["xy"] * layers, patches=patches, seed=seed + 999)   # 振幅の板の模様（位相の板と同じ形）
        if read == "int_phase":
            self.codes = torch.nn.Parameter(torch.randn(C, 3 * K, generator=torch.Generator().manual_seed(seed)))

    def forward(self, X):
        u = self.field(X)
        net = self.net
        for d in range(len(net.kinds)):
            if self.plate in ("phase", "both"):
                u = p2.apply_plate(u, net.plate_phase(d))
            if self.plate in ("amp", "both"):
                u = u * torch.exp(0.5 * self.ampnet.plate_phase(d))
            u = p2.normalize(p2.propagate(u, net.H))
        U = p2.lens(u)
        I = p2.intensity(U)
        s = I.mean((-2, -1), keepdim=True) + 1e-12
        w = p2.read_windows(I / s, net.det)
        if self.read == "int_phase":   # 参照光（平らな光）との干渉で、位相差の cos・sin 成分も読む
            Un = U / s.sqrt()
            w = torch.cat([w, p2.read_windows(Un.real, net.det), p2.read_windows(Un.imag, net.det)], 1)
        return p2.code_scores(w, self.codes, 1.0) * self.temp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plate", default="both", choices=["phase", "amp", "both"])
    ap.add_argument("--read", default="int", choices=["int", "int_phase"])
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--patches", type=int, default=16)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--max_train", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp63.txt")
    a = ap.parse_args()
    dev = torch.device(a.device)
    (X, y), (Xt, yt), info = worddata.load(a.n, 1024, a.max_train, 10000, a.seed, 4096)
    torch.manual_seed(a.seed)
    m = AmpPhaseNet(a.plate, a.read, a.n, 4096, 1024, a.layers, a.patches, a.seed, a.K).to(dev)
    params = [p for name, p in m.named_parameters() if not (a.plate == "phase" and name.startswith("ampnet"))
              and not (a.plate == "amp" and name.startswith("net.") and name.split(".")[1] in ("A", "phi"))]
    opt = torch.optim.Adam(params, lr=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    g = torch.Generator().manual_seed(a.seed)
    history, t_train, t_mark = [], 0.0, time.perf_counter()

    def scores(XX):
        with torch.no_grad():
            return torch.cat([m(XX[j:j + 250].to(dev)).cpu() for j in range(0, len(XX), 250)])
    for step in range(1, a.steps + 1):
        i = torch.randint(len(y), (256,), generator=g)
        loss = F.cross_entropy(m(X[i].to(dev)), y[i].to(dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
        if step % 500 == 0 or step == a.steps:
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t_train += time.perf_counter() - t_mark
            r = {}
            for name, XX, yy in (("test", Xt[:2000], yt[:2000]), ("train", X[:2000], y[:2000])):
                S = scores(XX)
                r[f"{name}_acc"] = 100.0 * (S.argmax(1) == yy).float().mean().item()
                r[f"{name}_ce"] = F.cross_entropy(S, yy).item()
            history.append(dict(step=step, train_s=round(t_train, 2), loss=float(loss.detach()), **r))
            t_mark = time.perf_counter()
    S = scores(Xt)
    rec = dict(exp="63", **vars(a), test_acc=100.0 * (S.argmax(1) == yt).float().mean().item(), test_ce=F.cross_entropy(S, yt).item(),
               test_top5=100.0 * (S.topk(5, 1).indices == yt[:, None]).any(1).float().mean().item(),
               params=sum(p.numel() for p in params), train_s=t_train, history=history, version=p2.__version__, torch=torch.__version__)
    print(json.dumps(rec, ensure_ascii=False))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.makedirs("results/models", exist_ok=True)
    torch.save({k: v.cpu() for k, v in m.state_dict().items()}, f"results/models/exp63_{a.plate}_{a.read}_s{a.seed}.pt")


if __name__ == "__main__":
    main()
