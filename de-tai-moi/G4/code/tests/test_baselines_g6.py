"""Kiểm thử baseline thêm ở G6 (bỏ qua nếu chưa cài gói tương ứng)."""
import numpy as np
import pytest

from icgibbs.tuning import LogUniform, random_search, sample_configs


def test_random_search_budget_and_default():
    cfgs = sample_configs({"a": LogUniform(1e-2, 1e2)}, n=20, seed=0, include_default={"a": 1.0})
    assert len(cfgs) == 20 and cfgs[0] == {"a": 1.0}
    assert all(1e-2 <= c["a"] <= 1e2 for c in cfgs)
    res = random_search(lambda c: (np.log(c["a"]) - 1.0) ** 2, {"a": LogUniform(1e-2, 1e2)}, n=20, seed=0)
    assert len(res.trials) == 20 and abs(np.log(res.best["a"]) - 1.0) < 1.0


def test_random_search_survives_bad_config():
    def f(c):
        if c["a"] > 10:
            raise ValueError("hỏng")
        return c["a"]
    res = random_search(f, {"a": LogUniform(1e-2, 1e2)}, n=20, seed=1)
    assert np.isfinite(res.best_score) and res.best["a"] <= 10


def _sar_data(n=4000, seed=0):
    from scipy.special import expit
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = (rng.uniform(size=n) < expit(1.5 * X[:, 1] - X[:, 2] + 0.5 * X[:, 3])).astype(int)
    z = rng.normal(size=n)                                   # thuộc tính propensity
    e = expit(-0.5 + 2.0 * z)
    s = ((y == 1) & (rng.uniform(size=n) < e)).astype(int)
    return np.column_stack([X, z]), y, s, e


@pytest.mark.parametrize("separate", [True, False])
def test_sar_em_recovers_classifier_and_propensity(separate):
    """separate=True: thiết lập của bài gốc (thuộc tính propensity tách khỏi thuộc tính phân loại) → gần oracle.
    separate=False: z vừa là feature vừa quyết định propensity (như PU-SAR của dự án) → khó định danh hơn,
    chỉ đòi hơn hẳn bộ ngây thơ và ước lượng propensity tốt."""
    pytest.importorskip("sarpu")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from icgibbs.baselines_sar import SAREM
    Xa, y, s, e = _sar_data()
    tr, te = slice(0, 3000), slice(3000, None)
    m = SAREM().fit(Xa[tr], s[tr], propensity_cols=[5], classification_cols=range(5) if separate else None)
    auc_sar = roc_auc_score(y[te], m.predict_proba(Xa[te])[:, 1])
    auc_naive = roc_auc_score(y[te], LogisticRegression().fit(Xa[tr], s[tr]).predict_proba(Xa[te])[:, 1])
    auc_oracle = roc_auc_score(y[te], LogisticRegression().fit(Xa[tr, :5], y[tr]).predict_proba(Xa[te, :5])[:, 1])
    mse_e = np.mean((m.propensity(Xa[te])[y[te] == 1] - e[te][y[te] == 1]) ** 2)
    assert auc_sar > auc_naive + 0.05, (auc_sar, auc_naive)
    assert mse_e < 0.02, mse_e
    if separate:
        assert auc_sar > auc_oracle - 0.01, (auc_sar, auc_oracle)


def test_sar_em_tuned_uses_only_weak_labels():
    pytest.importorskip("sarpu")
    from icgibbs.baselines_sar import fit_sar_em_tuned
    Xa, y, s, e = _sar_data(n=1500, seed=1)
    m = fit_sar_em_tuned(Xa, s, propensity_cols=[5], n_configs=5, seed=0)
    assert len(m.tune_.trials) == 5 and np.isfinite(m.tune_.best_score)
    p = m.predict_proba(Xa)
    assert p.shape == (1500, 2) and np.allclose(p.sum(1), 1)
