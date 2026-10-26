"""weakgen của G7 phải cho đúng cùng nhãn yếu như đoạn code sinh nhãn trong các script thí điểm (sao nguyên văn dưới đây)."""
import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import expit

from icgibbs.g7.weakgen import make_weak, weak_rng
from icgibbs.weak import make_bags


def _pilot_binary(setting, level, ytr, Xtr, rng):           # kaggle_g4_pilot.py, run_cell (không cắt train)
    if setting == "pu_scar":
        c = float(level)
        s = ((ytr == 1) & (rng.uniform(size=len(ytr)) < c)).astype(int)
        if s.sum() < 2:
            s[rng.choice(np.flatnonzero(ytr == 1), 2, replace=False)] = 1
        return s
    if setting == "pu_sar":
        z = np.nan_to_num(Xtr[:, 0].astype(float)); z = (z - z.mean()) / (z.std() + 1e-8)
        c = float(level)
        b0 = brentq(lambda b: expit(b + 2.0 * z[ytr == 1]).mean() - c, -30, 30)
        e = expit(b0 + 2.0 * z)
        s = ((ytr == 1) & (rng.uniform(size=len(ytr)) < e)).astype(int)
        if s.sum() < 2:
            s[rng.choice(np.flatnonzero(ytr == 1), 2, replace=False)] = 1
        return s
    bags, counts = make_bags(len(ytr), int(level), rng, ytr)
    return bags, counts


def _pilot_rq4(setting, level, ytr, C, rng):                # kaggle_g4_rq4_pilot.py, run_cell
    n = len(ytr)
    if setting == "ssl":
        lab = np.zeros(n, bool)
        for k in range(C):
            idx = np.flatnonzero(ytr == k)
            m = max(2, int(round(level * len(idx))))
            lab[rng.choice(idx, min(m, len(idx)), replace=False)] = True
        return np.where(lab, ytr, -1)
    if setting == "noisy":
        r = float(level)
        T = np.array([[1 - r / 2, r / 2], [r, 1 - r]])
        return np.array([rng.choice(2, p=T[k]) for k in ytr])
    S = rng.uniform(size=(n, C)) < float(level)
    S[np.arange(n), ytr] = True
    pseudo = np.array([rng.choice(np.flatnonzero(s)) for s in S])
    return S, pseudo


@pytest.mark.parametrize("setting,level", [("pu_scar", 0.1), ("pu_sar", 0.3), ("llp", 8), ("llp", 16)])
def test_binary_settings_match_pilot(setting, level):
    data_rng = np.random.default_rng(0)
    X = data_rng.normal(size=(700, 5)); y = (X[:, 1] + data_rng.normal(size=700) > 0.6).astype(int)
    for seed, did in ((0, 31), (2, 1462)):
        ours = make_weak(setting, level, y, X, weak_rng(seed, did, setting))
        ref = _pilot_binary(setting, level, y, X, np.random.default_rng(1000 * seed + did))
        if setting.startswith("pu"):
            assert np.array_equal(ours.s, ref)
        else:
            assert all(np.array_equal(a, b) for a, b in zip(ours.bags, ref[0])) and np.array_equal(ours.counts, ref[1])


@pytest.mark.parametrize("setting,level,C", [("ssl", 0.05, 2), ("ssl", 0.2, 2), ("noisy", 0.4, 2), ("pll", 0.3, 4),
                                             ("pll", 0.6, 7)])
def test_rq4_settings_match_pilot(setting, level, C):
    data_rng = np.random.default_rng(1)
    y = data_rng.integers(0, C, 600)
    X = data_rng.normal(size=(600, 3))
    for seed, did in ((0, 54), (1, 37)):
        ours = make_weak(setting, level, y, X, weak_rng(seed, did, setting), n_classes=C)
        ref = _pilot_rq4(setting, level, y, C, np.random.default_rng(1000 * seed + did + 7))
        if setting == "ssl":
            assert np.array_equal(ours.y_obs, ref)
        elif setting == "noisy":
            assert np.array_equal(ours.w, ref)
        else:
            assert np.array_equal(ours.S, ref[0]) and np.array_equal(ours.naive_pll, ref[1])


def test_weakdata_has_no_clean_labels():
    y = np.array([0, 1] * 50); X = np.zeros((100, 1))
    for st, lv in (("pu_scar", 0.3), ("llp", 8), ("ssl", 0.2), ("noisy", 0.2), ("pll", 0.3)):
        wd = make_weak(st, lv, y, X, np.random.default_rng(0), n_classes=2)
        for name, val in vars(wd).items():
            assert name not in ("y", "y_tr", "y_true"), name
