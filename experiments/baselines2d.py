"""比較相手（重みの表を持つ定番の学習器）。どれも画像 (N, L, L) を受け取り、クラスごとのスコアを返す。

    logreg     : ロジスティック回帰（画素 → クラス）。学習する数 = 画素数 × クラス数
    mlp        : 隠れ層 1 つ（32 個）の小さなニューラルネット
    cnn        : SinWave と同じくらいの学習する数（約 3000 個）の小さな畳み込みネット
    lens_logreg: 入力をレンズで見た明るさ |FFT|²（平行移動で変わらない）にロジスティック回帰。
                 「レンズの後の明るさ」だけで何が分かるかの基準
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LogReg(nn.Module):
    def __init__(self, L, C):
        super().__init__()
        self.fc = nn.Linear(L * L, C)

    def forward(self, x):
        return self.fc(x.flatten(1))


class MLP(nn.Module):
    def __init__(self, L, C, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(L * L, hidden), nn.ReLU(), nn.Linear(hidden, C))

    def forward(self, x):
        return self.net(x)


class SmallCNN(nn.Module):
    """3 段の畳み込み → 2 × 2 に平均（位置の情報を少し残す）→ 全結合。64 × 64 で約 3000 個。"""

    def __init__(self, L, C):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, 8, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(8, 12, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(12, 12, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Linear(48, C))

    def forward(self, x):
        return self.f(x[:, None])


class LensLogReg(nn.Module):
    def __init__(self, L, C):
        super().__init__()
        self.fc = nn.Linear(L * L, C)

    def forward(self, x):
        I = torch.fft.fft2(x, norm="ortho").abs().pow(2)
        I = I / (I.mean((-2, -1), keepdim=True) + 1e-12)
        return self.fc(torch.log1p(I).flatten(1))


BASELINES = {"logreg": LogReg, "mlp": MLP, "cnn": SmallCNN, "lens_logreg": LensLogReg}
