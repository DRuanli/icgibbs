"""
llp.py — lấy mẫu chính xác có ràng buộc số đếm cho cập nhật theo túi (LLP).

Cho một túi m mẫu với ma trận xác suất độc lập P (m, C) và vector số đếm w (C,), lấy mẫu
    y_B ~ Π_i P[i, y_i] · 1[counts(y_B) = w] / Z
một cách chính xác bằng quy hoạch động (DP). Với C = 2 đây là phân phối Poisson-binomial có điều kiện.
Chi phí: O(m · Π_c (w_c + 1)) — O(m²) cho nhị phân.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def _binary_logZ(logP: np.ndarray, k: int) -> np.ndarray:
    """logZ[i, r] = log Σ_{y_{i:}: Σ y = r} Π_{j≥i} P[j, y_j], r = 0..k."""
    m = logP.shape[0]
    logZ = np.full((m + 1, k + 1), -np.inf)
    logZ[m, 0] = 0.0
    for i in range(m - 1, -1, -1):
        stay = logP[i, 0] + logZ[i + 1]
        take = np.full(k + 1, -np.inf)
        take[1:] = logP[i, 1] + logZ[i + 1, :-1]
        logZ[i] = np.logaddexp(stay, take)
    return logZ


def sample_count_constrained(P: np.ndarray, counts, rng: np.random.Generator) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    counts = np.asarray(counts, dtype=int)
    m, C = P.shape
    if counts.sum() != m:
        raise ValueError("tổng số đếm phải bằng kích thước túi")
    with np.errstate(divide="ignore"):
        logP = np.log(P)
    if C == 2:
        k = int(counts[1])
        logZ = _binary_logZ(logP, k)
        if not np.isfinite(logZ[0, k]):
            raise ValueError("không có cách gán nào có xác suất dương")
        y = np.zeros(m, dtype=int)
        r = k
        for i in range(m):
            if r == 0:
                break
            lp1 = logP[i, 1] + logZ[i + 1, r - 1]
            if np.log(rng.uniform()) < lp1 - logZ[i, r]:
                y[i] = 1
                r -= 1
        return y
    return _sample_general(logP, tuple(counts), rng)


def _sample_general(logP, counts, rng):
    m, C = logP.shape
    memo = {}

    def logZ(i, rem):
        if i == m:
            return 0.0 if not any(rem) else -np.inf
        key = (i, rem)
        if key in memo:
            return memo[key]
        terms = []
        for c in range(C):
            if rem[c] > 0:
                r2 = rem[:c] + (rem[c] - 1,) + rem[c + 1:]
                terms.append(logP[i, c] + logZ(i + 1, r2))
        v = logsumexp(terms) if terms else -np.inf
        memo[key] = v
        return v

    rem = tuple(counts)
    if not np.isfinite(logZ(0, rem)):
        raise ValueError("không có cách gán nào có xác suất dương")
    y = np.zeros(m, dtype=int)
    for i in range(m):
        lw = np.full(C, -np.inf)
        for c in range(C):
            if rem[c] > 0:
                r2 = rem[:c] + (rem[c] - 1,) + rem[c + 1:]
                lw[c] = logP[i, c] + logZ(i + 1, r2)
        w = np.exp(lw - logsumexp(lw))
        c = rng.choice(C, p=w / w.sum())
        y[i] = c
        rem = rem[:c] + (rem[c] - 1,) + rem[c + 1:]
    return y


def count_constrained_log_normalizer(P: np.ndarray, counts) -> float:
    """log Z của phân phối có ràng buộc (dùng trong kiểm thử)."""
    P = np.asarray(P, float)
    counts = np.asarray(counts, int)
    with np.errstate(divide="ignore"):
        logP = np.log(P)
    if P.shape[1] == 2:
        return float(_binary_logZ(logP, int(counts[1]))[0, int(counts[1])])
    import itertools
    vals = []
    for y in itertools.product(range(P.shape[1]), repeat=P.shape[0]):
        y = np.array(y)
        if (np.bincount(y, minlength=P.shape[1]) == counts).all():
            vals.append(logP[np.arange(len(y)), y].sum())
    return float(logsumexp(vals))


def topk_assignment(P: np.ndarray, counts) -> np.ndarray:
    """Cách gán có xác suất lớn nhất dưới ràng buộc số đếm (nhị phân: k mẫu có log-odds cao nhất là 1).
    Dùng cho baseline hard-EM. Đa lớp: tham lam theo log-xác suất (xấp xỉ)."""
    P = np.asarray(P, float)
    counts = np.asarray(counts, int)
    m, C = P.shape
    y = np.zeros(m, dtype=int)
    if C == 2:
        k = int(counts[1])
        score = np.log(np.maximum(P[:, 1], 1e-300)) - np.log(np.maximum(P[:, 0], 1e-300))
        y[np.argsort(-score, kind="stable")[:k]] = 1
        return y
    rem = counts.copy()
    order = np.dstack(np.unravel_index(np.argsort(-np.log(np.maximum(P, 1e-300)), axis=None), P.shape))[0]
    done = np.zeros(m, bool)
    for i, c in order:
        if not done[i] and rem[c] > 0:
            y[i] = c
            rem[c] -= 1
            done[i] = True
    return y
