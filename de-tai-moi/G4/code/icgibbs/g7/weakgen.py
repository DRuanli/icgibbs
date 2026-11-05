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


VARIANTS = {"llp": ("iid", "feat"), "pll": ("iid", "inst")}   # V2 W3: biến thể nhãn yếu cho Study 3
FEAT_BAGS_PER_CLUSTER = 8      # túi gom theo feature: số túi trung bình mỗi cụm k-means
INST_AUX_FOLDS = 5             # PLL phụ thuộc mẫu: số fold của mô hình phụ (dự đoán ngoài fold)


def _std_features(X) -> np.ndarray:
    """Feature số đã chuẩn hoá cho các bộ sinh phụ thuộc feature: NaN → trung vị cột, rồi z-score."""
    X = np.asarray(X, float).copy()
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    bad = np.isnan(X)
    X[bad] = np.take(med, np.nonzero(bad)[1])
    sd = X.std(0)
    return (X - X.mean(0)) / np.where(sd > 0, sd, 1.0)


def feature_bags(X, bag_size: int, rng, y=None, n_classes: int = 2):
    """LLP túi gom theo feature (V2 W3, V2_PLAN §5): k-means trên feature đã chuẩn hoá với
    k = max(1, round(n / (bag_size · FEAT_BAGS_PER_CLUSTER))) cụm; trong mỗi cụm xáo trộn rồi cắt thành túi cỡ
    bag_size; phần dư (< bag_size điểm) của mỗi cụm nhập vào túi cuối của cụm đó (túi cỡ bag_size..2·bag_size − 1),
    hoặc thành một túi riêng nếu cụm có ít hơn bag_size điểm. Mọi điểm thuộc đúng một túi.
    Trả về (bags, counts) cùng định dạng với weak.make_bags."""
    from sklearn.cluster import KMeans
    n = len(X)
    Z = _std_features(X)
    k = max(1, int(round(n / (bag_size * FEAT_BAGS_PER_CLUSTER))))
    lab = KMeans(n_clusters=k, n_init=4, random_state=int(rng.integers(2**31 - 1))).fit_predict(Z) if k > 1 \
        else np.zeros(n, int)
    bags = []
    for c in range(k):
        idx = rng.permutation(np.flatnonzero(lab == c))
        if len(idx) == 0:
            continue
        m = len(idx) // bag_size
        if m == 0:
            bags.append(idx)
            continue
        chunks = [idx[i * bag_size:(i + 1) * bag_size] for i in range(m)]
        if len(idx) > m * bag_size:
            chunks[-1] = np.concatenate([chunks[-1], idx[m * bag_size:]])
        bags.extend(chunks)
    counts = None if y is None else np.array([np.bincount(y[b], minlength=n_classes) for b in bags])
    return bags, counts


def instance_pll(X, y, q: float, rng, n_classes: int):
    """PLL phụ thuộc mẫu (V2 W3, V2_PLAN §5), theo tinh thần Xu et al. (NeurIPS 2021): distractor đến từ các lớp mà
    một mô hình phụ hay nhầm với lớp đúng.
    - Mô hình phụ: hồi quy logistic đa lớp trên feature đã chuẩn hoá, dự đoán NGOÀI FOLD (INST_AUX_FOLDS fold phân
      tầng) để xác suất lớp sai không bị co về 0 do học thuộc.
    - Lớp sai j của điểm i vào tập ứng viên với xác suất ξ_ij = min(1, α·p_ij); α chọn bằng chia đôi sao cho trung
      bình ξ trên mọi cặp (i, j ≠ y_i) bằng q. Như vậy số ứng viên trung bình bằng 1 + q(C − 1), đúng như PLL
      gieo đều cùng mức q, chỉ khác ở chỗ lớp nào làm distractor.
    Trả về (S, xi): S (n, C) bool, xi (n, C) xác suất vào tập ứng viên (0 ở lớp đúng; để kiểm, không đưa vào phương pháp)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    y = np.asarray(y).astype(int)
    n, C = len(y), int(n_classes)
    Z = _std_features(X)
    P = np.zeros((n, C))
    folds = min(INST_AUX_FOLDS, int(np.bincount(y, minlength=C)[np.bincount(y, minlength=C) > 0].min()))
    folds = max(2, folds)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(rng.integers(2**31 - 1)))
    for tr, te in skf.split(Z, y):
        m = LogisticRegression(max_iter=1000).fit(Z[tr], y[tr])
        P[np.ix_(te, m.classes_.astype(int))] = m.predict_proba(Z[te])
    P = np.clip(P, 1e-12, None)
    wrong = np.ones((n, C), bool)
    wrong[np.arange(n), y] = False
    pw = P[wrong]

    def mean_xi(alpha):
        return float(np.minimum(1.0, alpha * pw).mean())

    lo, hi = 0.0, 1.0
    while mean_xi(hi) < q:
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if mean_xi(mid) < q else (lo, mid)
    xi = np.zeros((n, C))
    xi[wrong] = np.minimum(1.0, hi * pw)
    S = rng.uniform(size=(n, C)) < xi
    S[np.arange(n), y] = True
    return S, xi


def make_weak(setting: str, level, y_tr: np.ndarray, X_tfm_tr: np.ndarray, rng, n_classes: int | None = None,
              variant: str = "iid") -> WeakData:
    y = np.asarray(y_tr).astype(int)
    n = len(y)
    C = n_classes or int(y.max()) + 1
    if variant != "iid":
        if variant not in VARIANTS.get(setting, ()):
            raise ValueError(f"biến thể {variant!r} không dùng được với {setting!r}")
        if variant == "feat":
            bags, counts = feature_bags(X_tfm_tr, int(level), rng, y, n_classes=C)
            sizes = np.array([len(b) for b in bags])
            return WeakData(setting, int(level), C, bags=bags, counts=counts,
                            extra=f"n_bags={len(bags)};variant=feat;bag_size_min={sizes.min()};bag_size_max={sizes.max()}")
        if variant == "inst":
            S, xi = instance_pll(X_tfm_tr, y, float(level), rng, C)
            pseudo = np.array([rng.choice(np.flatnonzero(s)) for s in S])
            return WeakData(setting, float(level), C, S=S, naive_pll=pseudo,
                            extra=f"avg_candidates={S.sum(1).mean():.2f};variant=inst", meta={"xi": xi})
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
