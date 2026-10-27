"""Chia dữ liệu (giao thức Mục 3 và Mục 1.4)."""
from __future__ import annotations

import hashlib

import numpy as np


def stratified_split(y, seed: int, test_size: float = 0.3):
    """Như các script thí điểm: train_test_split(arange(n), test_size, stratify=y, random_state=seed)."""
    from sklearn.model_selection import train_test_split
    return train_test_split(np.arange(len(y)), test_size=test_size, stratify=y, random_state=seed)


def random_split(n: int, seed: int, test_size: float = 0.1):
    """PLL thật (RQ5): 10 lần chia ngẫu nhiên 90/10, không phân tầng, seed 0–9 (giao thức Mục 1.4)."""
    from sklearn.model_selection import train_test_split
    return train_test_split(np.arange(n), test_size=test_size, random_state=seed, shuffle=True)


def nested_subsample(idx_tr, y, m: int, seed: int, did: int):
    """Ablation cỡ dữ liệu (giao thức Mục 7): m chỉ số đầu của một thứ tự phân tầng ngẫu nhiên của train — các cỡ lồng
    nhau (500 ⊂ 1000 ⊂ ...), tỉ lệ lớp lệch tối đa khoảng 1 mẫu mỗi lớp. Test không đổi. m ≥ len(idx_tr): trả nguyên."""
    idx_tr = np.asarray(idx_tr)
    if m >= len(idx_tr):
        return idx_tr
    rng = np.random.default_rng(1000 * seed + did + 31)
    yy = np.asarray(y)[idx_tr]
    rank = np.empty(len(idx_tr))
    for c in np.unique(yy):
        pos = np.flatnonzero(yy == c)
        perm = rng.permutation(len(pos))
        rank[pos[perm]] = (np.arange(len(pos)) + rng.uniform(0, 1, len(pos))) / len(pos)
    return idx_tr[np.argsort(rank, kind="stable")[:m]]


def split_sha256(idx_tr, idx_te) -> str:
    h = hashlib.sha256()
    for a in (idx_tr, idx_te):
        a = np.asarray(a, np.int64)
        h.update(np.asarray([len(a)], np.int64).tobytes())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()
