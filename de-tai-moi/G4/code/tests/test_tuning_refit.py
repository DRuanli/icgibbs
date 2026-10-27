"""Quy tắc dự phòng khi fit lại cấu hình tune tốt nhất cho xác suất không hữu hạn (G6 lượt 5: proden_tun, segment,
q = 0,3, seed 2)."""
import numpy as np

from icgibbs.tuning import TuneResult, refit_best


class _M:
    def __init__(self, cfg):
        self.cfg = cfg

    def predict_proba(self, X):
        v = np.nan if self.cfg["a"] in (1, 4) else 0.5
        return np.full((len(X), 2), v)


def _res():
    trials = [dict(a=1, score=0.1), dict(a=2, score=0.2), dict(a=3, score=np.inf), dict(a=5, score=0.15, error="x")]
    return TuneResult(best=dict(a=1), best_score=0.1, trials=trials)


def test_refit_falls_back_to_next_ranked_config():
    res = _res()
    m = refit_best(res, _M, np.zeros((4, 2)))
    assert m.cfg == dict(a=2) and res.refit_rank == 1


def test_refit_keeps_best_when_finite():
    res = TuneResult(best=dict(a=2), best_score=0.1, trials=[dict(a=2, score=0.1), dict(a=6, score=0.3)])
    m = refit_best(res, _M, np.zeros((3, 2)))
    assert m.cfg == dict(a=2) and res.refit_rank == 0


def test_refit_all_fail_returns_last_and_flags():
    res = TuneResult(best=dict(a=1), best_score=0.1, trials=[dict(a=1, score=0.1), dict(a=4, score=0.2)])
    m = refit_best(res, _M, np.zeros((2, 2)))
    assert res.refit_rank == -1 and m.cfg == dict(a=4)


def test_refit_real_baseline_identical_to_plain_refit():
    from icgibbs.baselines_ext import PLLNet, fit_pll
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4))
    y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0).astype(int)
    S = np.eye(3, dtype=bool)[y] | (rng.random((120, 3)) < 0.3)
    m = fit_pll("cc", X, S, seed=0, tune=True, n_configs=3)
    ref = PLLNet("cc", seed=0, **m.tune_.best).fit(X, S)
    assert m.tune_.refit_rank == 0
    np.testing.assert_allclose(m.predict_proba(X), ref.predict_proba(X))
