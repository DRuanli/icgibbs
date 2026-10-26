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


def split_sha256(idx_tr, idx_te) -> str:
    h = hashlib.sha256()
    for a in (idx_tr, idx_te):
        a = np.asarray(a, np.int64)
        h.update(np.asarray([len(a)], np.int64).tobytes())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()
