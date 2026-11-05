"""V2 W4 (V2/W4_TFM_REPORT.md): bộ bọc TabDPT trong run_g7. Dùng mô hình giả, không cần trọng số hay GPU."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import run_g7  # noqa: E402
from icgibbs.conditional import SklearnTFMConditional  # noqa: E402


class FakeTabDPT:
    """Giống TabDPTClassifier về giao diện: nhãn phải là 0..k−1; trả xác suất theo tần suất lớp trong ngữ cảnh."""

    def __init__(self):
        self.n_fit = 0
        self.calls = []

    def fit(self, X, y):
        assert y.min() == 0 and set(np.unique(y)) == set(range(y.max() + 1)), "TabDPT cần nhãn 0..k−1"
        self.k = int(y.max()) + 1
        self.freq = np.bincount(y, minlength=self.k) / len(y)
        self.n_fit += 1

    def predict_proba(self, X, context_size=None):
        self.calls.append(("single", context_size))
        return np.tile(self.freq, (len(X), 1))

    def ensemble_predict_proba(self, X, n_ensembles=8, context_size=None, seed=None):
        self.calls.append(("ensemble", n_ensembles, seed))
        return np.tile(self.freq, (len(X), 1))


def test_tabdpt_adapter_label_encoding_and_cache():
    fake = FakeTabDPT()
    f = lambda: fake  # noqa: E731
    X = np.random.default_rng(0).normal(size=(12, 3))
    y = np.array([1, 3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 3])          # lớp 0 và 2 vắng mặt
    a = run_g7.TabDPTAdapter(n_ensembles=1, factory=f).fit(X, y)
    assert list(a.classes_) == [1, 3]
    P = a.predict_proba(X[:4])
    assert P.shape == (4, 2) and np.allclose(P[0], [4 / 12, 8 / 12])
    b = run_g7.TabDPTAdapter(n_ensembles=8, factory=f).fit(X, y)    # cùng factory → cùng một mô hình (nạp một lần)
    b.predict_proba(X[:2])
    assert fake.n_fit == 2 and fake.calls == [("single", None), ("ensemble", 8, 0)]


def test_tabdpt_adapter_inside_conditional_maps_columns():
    fake = FakeTabDPT()
    cond = SklearnTFMConditional(lambda: run_g7.TabDPTAdapter(factory=lambda: fake), n_classes=4)
    X = np.random.default_rng(1).normal(size=(10, 2))
    y = np.array([0, 2, 2, 0, 2, 2, 2, 0, 2, 2])
    P = cond.predict_proba(X, y, X[:3])
    assert P.shape == (3, 4)
    assert np.allclose(P.sum(1), 1.0)
    assert P[0, 2] > P[0, 0] > P[0, 1] > 0 and P[0, 1] < 1e-3 and P[0, 3] < 1e-3   # lớp vắng: xác suất sàn > 0


def test_make_estimator_tabdpt_maps_n_estimators(monkeypatch):
    e1 = run_g7.make_estimator("tabdpt", None)
    e8 = run_g7.make_estimator("tabdpt", 8)
    assert isinstance(e1, run_g7.TabDPTAdapter) and e1.n_ensembles == 1 and e8.n_ensembles == 8
