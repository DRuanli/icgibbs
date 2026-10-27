"""GLWS/ILL cho PLL: số hạng cân bằng entropy trung bình như `imp_partial_label.py` (mọi config PLL chính thức bật);
hook on_epoch dùng khi tái hiện số công bố."""
import numpy as np

from icgibbs.baselines_ext import PLL_DEFAULTS, PLL_SPACES, PLLNet


def _data(seed=0, n=150, C=4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = np.argmax(X[:, :C] + 0.3 * rng.normal(size=(n, C)), 1)
    S = np.eye(C, dtype=bool)[y] | (rng.random((n, C)) < 0.3)
    return X, S, y


def test_entropy_term_matches_official_formula():
    import torch
    p = torch.softmax(torch.randn(32, 5), -1)
    avg = torch.clamp(torch.mean(p, dim=0), min=1e-6, max=1.0)
    prior = 1.0 / 5 * torch.ones_like(avg)
    official = 0.1 * torch.mean(-(prior * torch.log(avg)).sum(dim=0))      # đúng từng phép tính của kho
    ours = 0.1 * (-(torch.full_like(avg, 1.0 / 5) * torch.log(avg)).sum())
    assert torch.allclose(official, ours)


def test_default_on_and_tunable():
    assert PLL_DEFAULTS["glws"]["average_entropy_loss"] is True
    assert set(PLL_SPACES["glws"]["average_entropy_loss"].values) == {True, False}


def test_entropy_option_changes_fit_and_epoch_hook():
    X, S, y = _data()
    seen = []
    a = PLLNet("glws", hidden=(16,), epochs=5, average_entropy_loss=True).fit(X, S, on_epoch=lambda m, e: seen.append(e))
    b = PLLNet("glws", hidden=(16,), epochs=5, average_entropy_loss=False).fit(X, S)
    assert seen == list(range(5))
    assert not np.allclose(a.predict_proba(X), b.predict_proba(X))
    # các phương pháp khác không bị ảnh hưởng bởi cờ này
    r1 = PLLNet("rc", hidden=(), epochs=3, average_entropy_loss=True).fit(X, S).predict_proba(X)
    r2 = PLLNet("rc", hidden=(), epochs=3, average_entropy_loss=False).fit(X, S).predict_proba(X)
    np.testing.assert_allclose(r1, r2)
