import numpy as np
import pytest

from icgibbs.g7.metrics import binary_metrics, ece, multiclass_metrics, prior_match


def test_binary_metrics_known_values_and_prior_match():
    y = np.array([0, 0, 1, 1]); p = np.array([0.1, 0.4, 0.35, 0.8])
    m = binary_metrics(y, p, prior=0.5)
    assert m["auc"] == pytest.approx(0.75) and m["acc"] == pytest.approx(0.75)
    q = prior_match(p, 0.5)
    assert q.mean() == pytest.approx(0.5, abs=1e-8) and np.argsort(q).tolist() == np.argsort(p).tolist()
    assert set(m) >= {"auc", "pr_auc", "logloss", "brier", "acc", "ece", "logloss_pm", "brier_pm"}


def test_ece_perfect_and_bad():
    y = np.array([0, 1] * 50)
    assert ece(y, np.eye(2)[y]) == pytest.approx(0.0)
    P = np.tile([0.9, 0.1], (100, 1))                    # luôn đoán lớp 0 với độ tin 0,9, đúng 50%
    assert ece(y, P) == pytest.approx(0.4)


def test_multiclass_metrics_consistent_with_sklearn():
    from sklearn.metrics import log_loss
    rng = np.random.default_rng(0)
    P = rng.dirichlet(np.ones(4), 200); y = rng.integers(0, 4, 200)
    m = multiclass_metrics(y, P)
    assert m["logloss"] == pytest.approx(log_loss(y, P, labels=[0, 1, 2, 3]), rel=1e-6)
    assert 0 <= m["brier"] <= 2 and 0.3 < m["auc_ovr"] < 0.7
