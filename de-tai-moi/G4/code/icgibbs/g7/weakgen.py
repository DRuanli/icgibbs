"""Sinh nhãn yếu cho G7 (giao thức Mục 3.1) — cùng thuật toán và cùng cách dùng seed như các script thí điểm, để
cách sinh nhãn đã dùng trên tập phát triển trùng với cách dùng trên tập đánh giá.

Seed: rng = default_rng(1000·seed + did + offset), offset = 0 cho PU và LLP (như kaggle_g4_pilot.py), 7 cho SSL, nhãn
nhiễu, PLL (như kaggle_g4_rq4_pilot.py). Khác thí điểm: G7 không cắt train về 2.000 hàng (G7_RUNNER_DESIGN R3), nên
với dataset có train > 2.000 hàng dãy số ngẫu nhiên không trùng thí điểm — điều này không ảnh hưởng tập phát triển.

Kết quả `WeakData` không chứa nhãn sạch (R6): phương pháp chỉ nhận đối tượng này.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SAR_SLOPE = 2.0
OFFSET = {"pu_scar": 0, "pu_sar": 0, "llp": 0, "ssl": 7, "noisy": 7, "pll": 7}


@dataclass
class WeakData:
    setting: str
    level: float
    n_classes: int
    s: np.ndarray | None = None            # PU: 1 = dương có nhãn
    z: np.ndarray | None = None            # PU-SAR: thuộc tính propensity đã chuẩn hoá
    eta: tuple | None = None               # PU-SAR: (η0, η1) thật — chỉ cho phân tích "propensity đã biết"
    bags: list | None = None               # LLP
    counts: np.ndarray | None = None       # LLP: (G, C)
    y_obs: np.ndarray | None = None        # SSL: −1 = không nhãn
    w: np.ndarray | None = None            # nhãn nhiễu
    T: np.ndarray | None = None            # nhãn nhiễu: T[y, w] đã biết
    S: np.ndarray | None = None            # PLL: (n, C) bool
    naive_pll: np.ndarray | None = None    # PLL: một ứng viên chọn đều (nhãn ngây thơ, rút cùng dãy ngẫu nhiên như thí điểm)
    extra: str = ""
    meta: dict = field(default_factory=dict)


def real_pll_weak(S_tr: np.ndarray, seed: int, code: int) -> WeakData:
    """PLL thật: tập ứng viên có sẵn; nhãn ngây thơ = một ứng viên chọn đều, dãy ngẫu nhiên như thí điểm many-class
    (default_rng(1000·seed + mã + 11))."""
    rng = np.random.default_rng(1000 * seed + code + 11)
    S = np.asarray(S_tr).astype(bool)
    pseudo = np.array([rng.choice(np.flatnonzero(s)) for s in S])
    return WeakData("pll_real", 0.0, S.shape[1], S=S, naive_pll=pseudo, extra=f"avg_candidates={S.sum(1).mean():.2f}")


def weak_rng(seed: int, did: int, setting: str):
    return np.random.default_rng(1000 * seed + did + OFFSET[setting])


def make_weak(setting: str, level, y_tr: np.ndarray, X_tfm_tr: np.ndarray, rng, n_classes: int | None = None) -> WeakData:
    y = np.asarray(y_tr).astype(int)
    n = len(y)
    C = n_classes or int(y.max()) + 1
    if setting == "pu_scar":
        c = float(level)
        s = ((y == 1) & (rng.uniform(size=n) < c)).astype(int)
        if s.sum() < 2:
            s[rng.choice(np.flatnonzero(y == 1), 2, replace=False)] = 1
        return WeakData(setting, c, 2, s=s, extra=f"labeled_pos={int(s.sum())}")
    if setting == "pu_sar":
        from scipy.optimize import brentq
        from scipy.special import expit
        z = np.nan_to_num(np.asarray(X_tfm_tr[:, 0], float))
        z = (z - z.mean()) / (z.std() + 1e-8)
        c = float(level)
        b0 = brentq(lambda b: expit(b + SAR_SLOPE * z[y == 1]).mean() - c, -30, 30)
        e = expit(b0 + SAR_SLOPE * z)
        s = ((y == 1) & (rng.uniform(size=n) < e)).astype(int)
        if s.sum() < 2:
            s[rng.choice(np.flatnonzero(y == 1), 2, replace=False)] = 1
        return WeakData(setting, c, 2, s=s, z=z, eta=(b0, SAR_SLOPE), extra=f"labeled_pos={int(s.sum())};eta0={b0:.3f}",
                        meta={"e": e})
    if setting == "llp":
        from ..weak import make_bags
        bags, counts = make_bags(n, int(level), rng, y, n_classes=C)
        return WeakData(setting, int(level), C, bags=bags, counts=counts, extra=f"n_bags={len(bags)}")
    if setting == "ssl":
        lab = np.zeros(n, bool)
        for k in range(C):
            idx = np.flatnonzero(y == k)
            m = max(2, int(round(level * len(idx))))
            lab[rng.choice(idx, min(m, len(idx)), replace=False)] = True
        y_obs = np.where(lab, y, -1)
        return WeakData(setting, float(level), C, y_obs=y_obs, extra=f"n_labeled={int(lab.sum())}")
    if setting == "noisy":
        r = float(level)
        T = np.array([[1 - r / 2, r / 2], [r, 1 - r]])          # T[y, w]
        w = np.array([rng.choice(2, p=T[k]) for k in y])
        return WeakData(setting, r, 2, w=w, T=T, extra=f"noise_rate={float(np.mean(w != y)):.3f}")
    if setting == "pll":
        S = rng.uniform(size=(n, C)) < float(level)
        S[np.arange(n), y] = True
        pseudo = np.array([rng.choice(np.flatnonzero(s)) for s in S])
        return WeakData(setting, float(level), C, S=S, naive_pll=pseudo, extra=f"avg_candidates={S.sum(1).mean():.2f}")
    raise ValueError(setting)
