"""MNIST・Fashion-MNIST を data/ に取得する（torchvision のダウンロード機能を使う）。データはリポジトリに入れない。

    python tools/fetch_mnist.py              # 両方
    python tools/fetch_mnist.py mnist        # MNIST だけ

出典とライセンス（README の「データの出典」にも書く）:
    MNIST         : Yann LeCun, Corinna Cortes（CC BY-SA 3.0）。取得元は torchvision の置き場（ossci-datasets.s3.amazonaws.com/mnist）
    Fashion-MNIST : Zalando Research（MIT）。取得元は github.com/zalandoresearch/fashion-mnist の置き場
"""
from __future__ import annotations

import os
import sys

from torchvision import datasets

ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
names = sys.argv[1:] or ["mnist", "fashion"]
for n in names:
    cls = {"mnist": datasets.MNIST, "fashion": datasets.FashionMNIST}[n]
    for train in (True, False):
        d = cls(ROOT, train=train, download=True)
        print(n, "train" if train else "test", len(d))
