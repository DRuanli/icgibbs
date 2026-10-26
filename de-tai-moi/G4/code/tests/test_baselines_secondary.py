"""Kiểm thử baseline phụ còn thiếu sau G6 lượt 3: GLWS cho PU, VPU, ILL (nhiễu, SSL), VIME."""
import itertools
import os
import sys

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from icgibbs.baselines_secondary import (ILLNet, PUNet, VIME, count_posterior, fit_ill, fit_pu2,  # noqa: E402
                                         fit_vime)


def _data(n=600, d=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + 0.5 * X[:, 1] + 0.3 * rng.normal(size=n) > 0.4).astype(int)
    return rng, X, y


def test_count_posterior_exact():
    rng = np.random.default_rng(0)
    for K in (1, 3, 6):
        q = rng.uniform(0.02, 0.98, K)
        for c in range(K + 1):
            num, den = np.zeros(K), 0.0
            for y in itertools.product([0, 1], repeat=K):
                if sum(y) == c:
                    p = np.prod([a if b else 1 - a for a, b in zip(q, y)])
                    num, den = num + p * np.array(y), den + p
            assert np.allclose(count_posterior(q, c), num / den, atol=1e-10)
    q = rng.uniform(size=100)
    assert abs(count_posterior(q, 37).sum() - 37) < 1e-8


@pytest.mark.parametrize("method", ["glws", "vpu"])
def test_pu_baselines_learn_under_scar(method):
    rng, X, y = _data()
    s = ((y == 1) & (rng.uniform(size=len(y)) < 0.3)).astype(int)
    cfg = dict(lr=1e-3, hidden=(64,), epochs=30) if method == "vpu" else dict(lr=1e-3, hidden=(64,), epochs=60)
    m = PUNet(method, prior=float(y.mean()), seed=0, **cfg).fit(X, s)
    assert roc_auc_score(y, m.predict_proba(X)[:, 1]) > 0.85


@pytest.mark.parametrize("noise_model", ["learned", "known"])
def test_ill_noisy_learns(noise_model):
    rng, X, y = _data()
    T = np.array([[0.9, 0.1], [0.2, 0.8]])
    w = np.array([rng.choice(2, p=T[k]) for k in y])
    m = ILLNet("noisy", noise_model=noise_model, epochs=60, seed=0).fit(X, w, T, 2)
    assert roc_auc_score(y, m.predict_proba(X)[:, 1]) > 0.85


def test_ill_ssl_and_vime_learn():
    rng, X, y = _data()
    y_obs = np.where(rng.uniform(size=len(y)) < 0.1, y, -1)
    m = ILLNet("ssl", epochs=60, seed=0).fit(X, y_obs, None, 2)
    assert roc_auc_score(y, m.predict_proba(X)[:, 1]) > 0.85
    v = VIME(seed=0, semi_iters=300).fit(X, y_obs, 2)
    P = v.predict_proba(X)
    assert np.allclose(P.sum(1), 1, atol=1e-5) and roc_auc_score(y, P[:, 1]) > 0.8


def test_tuning_wrappers_run():
    rng, X, y = _data(n=300)
    s = ((y == 1) & (rng.uniform(size=len(y)) < 0.3)).astype(int)
    for name in ("glws", "vpu"):
        m = fit_pu2(name, X, s, prior=float(y.mean()), seed=0, tune=True, n_configs=2)
        assert np.isfinite(m.tune_.best_score)
    T = np.array([[0.9, 0.1], [0.2, 0.8]])
    w = np.array([rng.choice(2, p=T[k]) for k in y])
    assert np.isfinite(fit_ill("noisy", X, w, T, seed=0, tune=True, n_configs=2).tune_.best_score)
    y_obs = np.where(rng.uniform(size=len(y)) < 0.2, y, -1)
    assert np.isfinite(fit_ill("ssl", X, y_obs, None, seed=0, tune=True, n_configs=2).tune_.best_score)
    assert np.isfinite(fit_vime(X, y_obs, seed=0, tune=True, n_configs=2).tune_.best_score)
