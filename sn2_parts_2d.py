"""sn2_parts_2d.py — 2 次元の場を正弦の板と回折で処理する部品（単独で動く 1 ファイル版）

Sin-CNN2 の SinWave（1 次元の光の場、https://github.com/devaf09/Sin-CNN2）を 2 次元（Ly × Lx の場）へ広げるための部品集です。
依存は PyTorch のみ。部品は関数として独立させ、組み合わせて使います。

    import sn2_parts_2d as p2
    y, x = p2.grid(64, 64)
    pid = p2.patch_index(64, 64, 4, 4)                       # 4 × 4 の区画
    ph = p2.plate_stripes(A, wx, wy, phi, pid, y, x)         # ② 斜めの縞の板の位相の遅れ (64, 64)
    H = p2.transfer(64, 64, z=64.0)                          # 板の間隔 64 画素の回折
    u = p2.normalize(p2.propagate(p2.apply_plate(u, ph), H)) # 板 → 回折 → 明るさを平均 1 に
    I = p2.intensity(p2.lens(u))                             # レンズの後のスクリーンの明るさ

しくみ（光の実験装置に例えると）
    入力 → 2 次元の光の場 → [Sin の板 → 回折 → 明るさを平均 1 に] × 枚数 → レンズ → スクリーンの窓 → 答え

板の 3 つの形（どれも「位相の遅れ」の模様。区画ごとに振幅 A・初期位相 φ などを持つ）
    ① 交点で掛け算（plate_xy）  : A1·sin(ωx·x + φ1) + A2·sin(ωy·y + φ2)。位相の和 = 波の掛け算なので、
                                  入力も X × Y の掛け算の形なら、2 次元の計算が 1 次元の計算 2 本に分かれる（*_sep の関数）
    ② 斜めの縞（plate_stripes）  : A·sin(ωx·x + ωy·y + φ)
    ③ 波紋（plate_ripple）       : A·sin(k·r + φ)（等間隔の輪、アキシコン）または A·sin(k·r² + φ)（ゾーンプレート）。
                                  r は中心 (cx, cy)・楕円の軸 a, b・角度 θ で決まる距離。中心を遠くへ置くと ② になる

変更履歴（改良したらここを更新し、__version__ を上げる）
    0.3.0 (2026-10-18)  2 次元の照明 illumination（tilt・アダマール。区画どうしの干渉を消す）と Net2D(illum=...) を追加。
    0.2.0 (2026-10-17)  入力 image_to_field（振幅として・位相として）、1 本の光路 Net2D（板の形を 1 枚ごとに選べる）、
                        train・evaluate を追加。window_grid はレンズの後の中心（入力の平均の明るさ）を避けて窓を置く。
    0.1.0 (2026-10-17)  最初の版。格子、区画、板 ①②③ の位相、2 次元の回折（角スペクトル法・近軸）、レンズ、
                        明るさの正規化、① の行と列に分けた計算、窓の読み出し。計算量は実験00 で測定。

ライセンス: MIT
"""
from __future__ import annotations

import math

import torch

__version__ = "0.3.0"


# =============================================================================================
# 1. 場の座標と区画
# =============================================================================================

def grid(Ly: int, Lx: int, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """場の画素の座標。y は (Ly, 1)、x は (1, Lx) の形（足し算で (Ly, Lx) に広がる）。単位は画素。"""
    y = torch.arange(Ly, dtype=torch.float32, device=device)[:, None]
    x = torch.arange(Lx, dtype=torch.float32, device=device)[None, :]
    return y, x


def patch_index(Ly: int, Lx: int, py: int, px: int, device=None) -> torch.Tensor:
    """場を縦 py × 横 px の格子の区画に分けたときの、各画素の区画番号 (Ly, Lx)。番号は 行 × px + 列。"""
    iy = torch.arange(Ly, device=device) * py // Ly
    ix = torch.arange(Lx, device=device) * px // Lx
    return iy[:, None] * px + ix[None, :]


def band_index(L: int, p: int, device=None) -> torch.Tensor:
    """長さ L の 1 列を p 本の帯に分けたときの、各画素の帯の番号 (L,)。① の板の X 用・Y 用に使う。"""
    return torch.arange(L, device=device) * p // L


# =============================================================================================
# 2. 板（位相の遅れの模様）
# =============================================================================================
# どの関数も「位相の遅れ」(Ly, Lx) または 1 次元の部品を返す。光に掛けるのは apply_plate。
# 引数の A, φ などは区画ごとの値（学習する数）。ω・k・中心は固定にも学習にもできる（Sin-CNN2 では ω 固定でよかった）。

def plate_xy_parts(Ax, wx, phix, bx, Ay, wy, phiy, by, x1, y1) -> tuple[torch.Tensor, torch.Tensor]:
    """① 交点で掛け算の板を、X 用と Y 用の 1 次元の位相に分けて返す: (phase_y (Ly,), phase_x (Lx,))。
    bx, by は band_index の帯の番号、x1, y1 は 1 次元の座標 (Lx,), (Ly,)。
    2 次元の位相は phase_y[:, None] + phase_x[None, :]（plate_xy）。位相の和なので、光には exp の積として掛かる。"""
    phase_x = Ax[bx] * torch.sin(wx[bx] * x1 + phix[bx])
    phase_y = Ay[by] * torch.sin(wy[by] * y1 + phiy[by])
    return phase_y, phase_x


def plate_xy(Ax, wx, phix, bx, Ay, wy, phiy, by, x1, y1) -> torch.Tensor:
    """① 交点で掛け算の板の位相 (Ly, Lx) = A1·sin(ωx·x + φ1) + A2·sin(ωy·y + φ2)（帯ごとに値を持つ）。"""
    py_, px_ = plate_xy_parts(Ax, wx, phix, bx, Ay, wy, phiy, by, x1, y1)
    return py_[:, None] + px_[None, :]


def plate_stripes(A, wx, wy, phi, pid, y, x) -> torch.Tensor:
    """② 斜めの縞の板の位相 (Ly, Lx) = A·sin(ωx·x + ωy·y + φ)。pid は patch_index の区画番号。"""
    return A[pid] * torch.sin(wx[pid] * x + wy[pid] * y + phi[pid])


def ripple_radius(cx, cy, a, b, theta, pid, y, x) -> torch.Tensor:
    """③ 波紋の「距離」r (Ly, Lx)。区画ごとの中心 (cx, cy)（場の外でもよい）、楕円の軸 a, b、角度 θ。
    r = √((x'/a)² + (y'/b)²)、(x', y') は中心から見た座標を θ だけ回したもの。"""
    dx, dy = x - cx[pid], y - cy[pid]
    c, s = torch.cos(theta[pid]), torch.sin(theta[pid])
    xr, yr = c * dx + s * dy, -s * dx + c * dy
    return torch.sqrt((xr / a[pid]) ** 2 + (yr / b[pid]) ** 2 + 1e-12)


def plate_ripple(A, k, phi, cx, cy, a, b, theta, pid, y, x, power: int = 1) -> torch.Tensor:
    """③ 波紋の板の位相 (Ly, Lx) = A·sin(k·r^power + φ)。
    power=1: 等間隔の輪（円錐レンズ = アキシコンと同じ形）、power=2: 外ほど詰まる輪（ゾーンプレート = レンズと同じ形）。"""
    r = ripple_radius(cx, cy, a, b, theta, pid, y, x)
    return A[pid] * torch.sin(k[pid] * r ** power + phi[pid])


def apply_plate(u: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """光の場 u (..., Ly, Lx) に位相の遅れ phase を掛ける: u·exp(j·phase)。明るさは変えない（位相だけの板）。"""
    return u * torch.polar(torch.ones_like(phase), phase)


# =============================================================================================
# 3. 回折・明るさの正規化・レンズ
# =============================================================================================
# 波長 1 画素、画素の間隔 1 として計算する（Sin-CNN2 の 1 次元と同じ。板の間隔は既定 64 画素）。
# 周期境界（場の端から出た光は反対側から入る）。

def _k(L: int, device=None) -> torch.Tensor:
    return 2 * math.pi * torch.fft.fftfreq(L, device=device)


def transfer(Ly: int, Lx: int, z: float = 64.0, wavelength: float = 1.0, paraxial: bool = False,
             device=None) -> torch.Tensor:
    """距離 z だけ進む回折の伝達関数 H (Ly, Lx)（周波数領域で掛ける）。
    paraxial=False: 角スペクトル法 exp(j·kz·z)、kz = √(k² − kx² − ky²)（届かない成分は減衰させる）。
    paraxial=True : 近軸（フレネル）近似 exp(−j·z·(kx² + ky²)/(2k))。X と Y の積に分かれる（transfer_sep）。"""
    k = 2 * math.pi / wavelength
    ky, kx = _k(Ly, device)[:, None], _k(Lx, device)[None, :]
    if paraxial:
        return torch.polar(torch.ones(Ly, Lx, device=device), -z * (kx ** 2 + ky ** 2) / (2 * k))
    kz2 = k ** 2 - kx ** 2 - ky ** 2
    kz = torch.sqrt(kz2.clamp(min=0))
    decay = torch.exp(-z * torch.sqrt((-kz2).clamp(min=0)))
    return torch.polar(decay, kz * z)


def transfer_sep(Ly: int, Lx: int, z: float = 64.0, wavelength: float = 1.0, device=None):
    """近軸の伝達関数を Y 用 (Ly,) と X 用 (Lx,) に分けたもの。transfer(paraxial=True) = Hy[:, None]·Hx[None, :]。"""
    k = 2 * math.pi / wavelength
    ky, kx = _k(Ly, device), _k(Lx, device)
    return (torch.polar(torch.ones(Ly, device=device), -z * ky ** 2 / (2 * k)),
            torch.polar(torch.ones(Lx, device=device), -z * kx ** 2 / (2 * k)))


def propagate(u: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """回折: 2 次元 FFT → H を掛ける → 逆 FFT。u は (..., Ly, Lx)。"""
    return torch.fft.ifft2(torch.fft.fft2(u) * H)


def normalize(u: torch.Tensor) -> torch.Tensor:
    """明るさの平均を 1 に戻す（増幅）。全部 0 でも微分が壊れないように小さな数を足す。"""
    return u / (u.abs().pow(2).mean((-2, -1), keepdim=True) + 1e-24).sqrt()


def lens(u: torch.Tensor) -> torch.Tensor:
    """レンズ = 2 次元フーリエ変換。同じ向きに進む光を 1 点に集める。中心が向き 0（fftshift）。"""
    return torch.fft.fftshift(torch.fft.fft2(u, norm="ortho"), dim=(-2, -1))


def intensity(u: torch.Tensor) -> torch.Tensor:
    """明るさ = 振幅の 2 乗。"""
    return u.abs().pow(2)


# =============================================================================================
# 4. ① の板の軽い計算（行と列に分ける）
# =============================================================================================
# 入力が uy(y)·ux(x) の掛け算の形で、板が ①、回折が近軸なら、場はずっと掛け算の形のまま。
# 2 次元の FFT（Ly·Lx·log）の代わりに、長さ Ly と Lx の 1 次元 FFT を 1 本ずつで済む。

def propagate_sep(uy: torch.Tensor, ux: torch.Tensor, Hy: torch.Tensor, Hx: torch.Tensor):
    """掛け算の形の場 (uy (..., Ly), ux (..., Lx)) の回折。"""
    return torch.fft.ifft(torch.fft.fft(uy) * Hy), torch.fft.ifft(torch.fft.fft(ux) * Hx)


def normalize_sep(uy: torch.Tensor, ux: torch.Tensor):
    """掛け算の形の場で、明るさの平均を 1 に戻す（Y と X を別々に 1 にすれば、積の平均も 1）。"""
    ny = (uy.abs().pow(2).mean(-1, keepdim=True) + 1e-24).sqrt()
    nx = (ux.abs().pow(2).mean(-1, keepdim=True) + 1e-24).sqrt()
    return uy / ny, ux / nx


def lens_sep(uy: torch.Tensor, ux: torch.Tensor):
    """掛け算の形の場のレンズ。2 次元のレンズの結果は Uy[:, None]·Ux[None, :]。"""
    f = lambda v: torch.fft.fftshift(torch.fft.fft(v, norm="ortho"), dim=-1)
    return f(uy), f(ux)


def outer(uy: torch.Tensor, ux: torch.Tensor) -> torch.Tensor:
    """掛け算の形の場を 2 次元に戻す: (..., Ly, Lx)。確認と表示用（計算の途中では作らない）。"""
    return uy[..., :, None] * ux[..., None, :]


# =============================================================================================
# 5. 読み出し（窓）
# =============================================================================================

def window_grid(Ly: int, Lx: int, n: int, w: int = 4, device=None) -> torch.Tensor:
    """スクリーンに n 個の w × w の窓を、ほぼ正方の格子に等間隔に並べた重み (n, Ly, Lx)。窓の中の平均を読む。
    レンズの後の中心（向き 0 = 入力の平均の明るさが集まる所）にかかるマスは使わない。"""
    g = math.ceil(math.sqrt(n + 1))
    centers = []
    for r in range(g):
        for c in range(g):
            cy, cx = int((r + 0.5) * Ly / g), int((c + 0.5) * Lx / g)
            if abs(cy - Ly // 2) < w and abs(cx - Lx // 2) < w:
                continue                                   # 中心のマスは飛ばす
            centers.append((cy, cx))
    det = torch.zeros(n, Ly, Lx, device=device)
    for i, (cy, cx) in enumerate(centers[:n]):
        iy = (torch.arange(w, device=device) + cy - w // 2) % Ly
        ix = (torch.arange(w, device=device) + cx - w // 2) % Lx
        det[i, iy[:, None], ix[None, :]] = 1.0 / (w * w)
    return det


def read_windows(I: torch.Tensor, det: torch.Tensor) -> torch.Tensor:
    """スクリーンの明るさ I (N, Ly, Lx) → 窓ごとの明るさ (N, n)。"""
    return torch.einsum("nyx,cyx->nc", I, det)


# =============================================================================================
# 6. 入力（画像 → 光の場）
# =============================================================================================

def image_to_field(img: torch.Tensor, mode: str = "amp") -> torch.Tensor:
    """明るさの画像 (N, Ly, Lx)（0〜1）→ 光の場（複素数）。
    amp  : 振幅として。u = 画像の値（暗い所には光がない）。スライド写真を光で照らすのと同じ
    phase: 位相として。u = exp(j·π·画像の値)（どこも明るさ 1、模様は位相の遅れだけ）。透明な凹凸の板と同じ"""
    if mode == "amp":
        return img.to(torch.complex64)
    if mode == "phase":
        return torch.polar(torch.ones_like(img), math.pi * img)
    raise ValueError('mode は "amp" か "phase"')


def illumination(L: int, blocks: int, kind: str = "tilt", device=None) -> torch.Tensor:
    """2 次元の照明の符号 (P, L, L)。場を縦横 blocks × blocks の区画に分け、区画 m の光に、照明 j ごとに違う符号を掛ける。
    P 通りの照明で明るさを平均すると、違う区画どうしの干渉（位置の雑音）がちょうど 0 になる（1 次元の実験10・11 の 2 次元版）。
    tilt    : 符号 exp(j·2π·j·m / n)（n = blocks²）。照明の向きを少しずつ傾けるのと同じ。P = n
    hadamard: アダマール行列の ±1（n は 2 の累乗）。位相 0 か π だけで作れる。P = n
    計算は P 倍になる。"""
    n = blocks * blocks
    m = patch_index(L, L, blocks, blocks, device)                       # 各画素の区画番号 (L, L)
    j = torch.arange(n, device=device, dtype=torch.float32)[:, None, None]
    if kind == "tilt":
        return torch.polar(torch.ones(n, L, L, device=device), 2 * math.pi * j * m[None] / n)
    if kind == "hadamard":
        if n & (n - 1):
            raise ValueError("hadamard は blocks² が 2 の累乗のとき")
        Hm = torch.ones(1, 1)
        while Hm.shape[0] < n:
            Hm = torch.cat([torch.cat([Hm, Hm], 1), torch.cat([Hm, -Hm], 1)], 0)
        return Hm.to(device)[:, m].to(torch.complex64)
    raise ValueError('kind は "tilt" か "hadamard"')


# =============================================================================================
# 7. 学習器（1 本の光路）
# =============================================================================================

PLATE_KINDS = ("xy", "stripes", "ripple", "ripple2")


class Net2D(torch.nn.Module):
    """2 次元の 1 本の光路: 入力の場 → [板 → 回折 → 明るさを平均 1 に] × 枚数 → レンズ → 窓 → スコア。

    plates  : 板の形の並び。例 ["stripes"] * 16、["xy"] * 8 + ["ripple"] * 8。
              "xy" = ① 交点で掛け算、"stripes" = ② 斜めの縞、"ripple" = ③ 等間隔の輪、"ripple2" = ③ ゾーンプレート
    patches : ②③ は縦横 patches × patches の区画、① は X 用・Y 用それぞれ patches² / 2 本の帯（学習する数をそろえる）
    学習するのは区画ごとの振幅 A と初期位相 φ、窓ごとの背景光、全体の温度だけ。
    ω・k・中心・楕円は固定の乱数（Sin-CNN2 では ω 固定でよかった）。freeze=True なら板は学習しない（比較用）。
    illum   : (blocks, kind) を与えると、2 次元の照明（illumination）で位置の雑音を消す。計算は blocks² 倍。"""

    def __init__(self, L: int, n_classes: int, plates=("stripes",) * 16, patches: int = 8, z: float = 64.0,
                 seed: int = 0, window: int = 4, freeze: bool = False, illum=None):
        super().__init__()
        self.L, self.kinds = L, list(plates)
        if any(k not in PLATE_KINDS for k in self.kinds):
            raise ValueError(f"板の形は {PLATE_KINDS} のどれか")
        g = torch.Generator().manual_seed(seed)
        r = lambda *s: torch.rand(*s, generator=g)
        logu = lambda lo, hi, *s: torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * r(*s))
        D, P2 = len(self.kinds), patches * patches
        self.nb = P2 // 2
        # 学習する数: 板ごとに (A, φ) を P2 組（① は X 用 nb 本 + Y 用 nb 本の帯）
        self.A = torch.nn.Parameter(0.5 + 0.5 * r(D, P2), requires_grad=not freeze)
        self.phi = torch.nn.Parameter(2 * math.pi * r(D, P2), requires_grad=not freeze)
        # 固定の数（どの形の分も作り、板の形に合うものだけ使う。乱数の並びを形によらずそろえるため）
        ang = 2 * math.pi * r(D, P2)
        wmag = logu(2 * math.pi / L, math.pi / 2, D, P2)
        self.register_buffer("wx", wmag * torch.cos(ang))                  # ② の縞の細かさと向き
        self.register_buffer("wy", wmag * torch.sin(ang))
        self.register_buffer("w1", logu(2 * math.pi / L, math.pi, D, P2))   # ① の X 用・Y 用の細かさ
        self.register_buffer("k1", logu(2 * math.pi / L, math.pi, D, P2))   # ③ 等間隔の輪の細かさ
        self.register_buffer("k2", logu(math.pi / L ** 2, math.pi / (4 * L), D, P2))   # ③ ゾーンプレート
        self.register_buffer("cx", L * (3 * r(D, P2) - 1))                 # ③ の中心（場の外でもよい）
        self.register_buffer("cy", L * (3 * r(D, P2) - 1))
        self.register_buffer("ea", torch.ones(D, P2))                      # ③ の楕円の軸（a = 1、b = 0.5〜2）
        self.register_buffer("eb", logu(0.5, 2.0, D, P2))
        self.register_buffer("th", math.pi * r(D, P2))                     # ③ の楕円の角度
        y, x = grid(L, L)
        self.register_buffer("y", y)
        self.register_buffer("x", x)
        self.register_buffer("pid", patch_index(L, L, patches, patches))
        self.register_buffer("band", band_index(L, self.nb))
        self.register_buffer("H", transfer(L, L, z))
        self.register_buffer("det", window_grid(L, L, n_classes, window))
        self.gain = torch.nn.Parameter(torch.tensor(5.0))                  # 明るさ → スコアの倍率（温度）
        self.bias = torch.nn.Parameter(torch.zeros(n_classes))            # 窓ごとの背景光
        self.register_buffer("illum", None if illum is None else illumination(L, *illum))

    def plate_phase(self, d: int) -> torch.Tensor:
        """板 d の位相の遅れ (L, L)。"""
        k, A, phi, nb = self.kinds[d], self.A[d], self.phi[d], self.nb
        if k == "xy":
            c = self.x[0]
            return plate_xy(A[:nb], self.w1[d, :nb], phi[:nb], self.band,
                            A[nb:], self.w1[d, nb:], phi[nb:], self.band, c, c)
        if k == "stripes":
            return plate_stripes(A, self.wx[d], self.wy[d], phi, self.pid, self.y, self.x)
        kk, pw = (self.k1[d], 1) if k == "ripple" else (self.k2[d], 2)
        return plate_ripple(A, kk, phi, self.cx[d], self.cy[d], self.ea[d], self.eb[d], self.th[d],
                            self.pid, self.y, self.x, power=pw)

    def screen(self, u: torch.Tensor) -> torch.Tensor:
        """光の場 (N, L, L) → レンズの後の明るさ (N, L, L)（平均 1）。照明があれば照明ごとの明るさを平均する。"""
        if self.illum is not None:
            u = u[:, None] * self.illum[None]                              # (N, P, L, L)
        for d in range(len(self.kinds)):
            u = normalize(propagate(apply_plate(u, self.plate_phase(d)), self.H))
        I = intensity(lens(u))
        if self.illum is not None:
            I = I.mean(1)
        return I / (I.mean((-2, -1), keepdim=True) + 1e-12)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """光の場 (N, L, L) → クラスごとのスコア (N, n_classes)。"""
        return self.gain * read_windows(self.screen(u), self.det) + self.bias

    def n_params(self) -> int:
        """学習する数（板の A・φ、背景光、温度）。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _device(net: torch.nn.Module):
    for t in list(net.buffers()) + list(net.parameters()):
        return t.device
    return torch.device("cpu")


def train(net: torch.nn.Module, X: torch.Tensor, y: torch.Tensor, steps: int = 1500, lr: float = 0.01,
          batch_size: int = 256, seed: int = 0, to_input=None) -> torch.nn.Module:
    """クロスエントロピーで学習する（Adam）。X の一部を to_input で変換してから net に渡す（例: image_to_field）。"""
    dev = _device(net)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr)
    g = torch.Generator().manual_seed(seed)
    to_input = to_input or (lambda v: v)
    for _ in range(steps):
        idx = torch.randint(len(X), (batch_size,), generator=g)
        loss = torch.nn.functional.cross_entropy(net(to_input(X[idx].to(dev))), y[idx].to(dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


@torch.no_grad()
def evaluate(net: torch.nn.Module, X: torch.Tensor, y: torch.Tensor, batch_size: int = 500, to_input=None) -> float:
    """正解率（%）。"""
    dev = _device(net)
    to_input = to_input or (lambda v: v)
    ok = 0
    for i in range(0, len(X), batch_size):
        ok += (net(to_input(X[i:i + batch_size].to(dev))).argmax(1).cpu() == y[i:i + batch_size]).sum().item()
    return 100.0 * ok / len(X)
