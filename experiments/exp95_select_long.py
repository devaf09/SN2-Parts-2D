"""実験95: 実験94 のマイナスポイント 2 つへの手当て（持ち主の依頼 2026-10-26）

1. 段の選び方: 確信度で 1 段を選ぶ（実験94）、新しい段を優先（確信度 × γ^段の番号、γ = 0.9・0.7）、全段の明るさを足す、
   全段の割合を足す（多数決）、確信度で重みを付けて足す。比べる: 選び方が完全なら（どれかの段で正しい）
2. 長い流れ: 60 区切り → 600 区切り。短期の一番上が満ちて中期の系列（_mid）が使われるか。比べる: 時間で分けて倍々（実験92）、板 1 枚
鍵: rand・word。満ちたら送る・1.5 倍ずつ 10 段・そのまま足す。CPU、3 seed。結果: results/exp95.txt
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import exp92_fibtiers as e92  # noqa: E402
import exp94_flush as e94  # noqa: E402


def main():
    with open(os.path.join(ROOT, "results", "exp95.txt"), "a", encoding="utf-8") as out:
        for T in (60, 600):
            for seed in range(3):
                for kind in ("rand", "word"):
                    e92.T = T
                    out.write(json.dumps(dict(exp="95", T=T, seed=seed, keys=kind, method="blocks", **e92.run_blocks(kind, seed, False)), ensure_ascii=False) + "\n")
                    for nested in (False, True):
                        r = e94.run(kind, seed, "r1.5", 10, "sum", nested, T=T)
                        out.write(json.dumps(dict(exp="95", T=T, seed=seed, keys=kind, method="flush_r1.5" + ("_mid" if nested else ""), **r), ensure_ascii=False) + "\n")
                        print(T, seed, kind, nested, round(r["acc"], 1), flush=True)


if __name__ == "__main__":
    main()
