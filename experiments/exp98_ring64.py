"""実験98: 環状の板を 64 枚に（持ち主の想定「板の数は 64 枚くらい」2026-10-26）

実験97 の環状の板（8・16 枚）は流れ型に負けた。持ち主の想定の 64 枚（と 32 枚）で、融合の比率 β = 1・0.5・0（融合しない = 板に順に振り分けるだけ）を比べる。
場面・読み方は実験97 と同じ（600 区切り、確信度 × なじみ）。CPU、3 seed。結果: results/exp98.txt
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from exp97_ring_elab import run_ring  # noqa: E402


def main():
    with open(os.path.join(ROOT, "results", "exp98.txt"), "a", encoding="utf-8") as out:
        for seed in range(3):
            for kind in ("rand", "word_hash"):
                for N in (32, 64):
                    for beta in (1.0, 0.5, 0.0):
                        out.write(json.dumps(dict(exp="98", seed=seed, keys=kind, method=f"ring_N{N}_b{beta}", **run_ring(kind, seed, N, beta)), ensure_ascii=False) + "\n")
                print(seed, kind, flush=True)


if __name__ == "__main__":
    main()
