"""S3B (S3/S3B_PLAN.md): baseline CPU và khoá đối số của lượt bổ sung."""
import numpy as np
import pytest

from icgibbs.baselines_cpu import GBDTSelfLLP, GBDTSelfPLL, PLkNN


def _data(n=240, C=3, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, C, n)
    X = rng.normal(size=(n, 4)) + 3.0 * np.eye(C)[y][:, :3].sum(1, keepdims=True) * 0 + 2.5 * np.eye(C)[y] @ rng.normal(size=(C, 4))
    return X, y


def _cands(y, C, q, seed=0):
    rng = np.random.default_rng(seed)
    S = rng.random((len(y), C)) < q
    S[np.arange(len(y)), y] = True
    return S


def test_plknn_and_gbdt_pll_learn_and_respect_shape():
    X, y = _data()
    S = _cands(y, 3, 0.4)
    for m in (PLkNN().fit(X[:180], S[:180]), GBDTSelfPLL(0).fit(X[:180], S[:180])):
        P = m.predict_proba(X[180:])
        assert P.shape == (60, 3) and np.allclose(P.sum(1), 1.0, atol=1e-6) and np.isfinite(P).all()
        assert (P.argmax(1) == y[180:]).mean() > 0.6


def test_gbdt_pll_is_deterministic():
    X, y = _data(seed=1)
    S = _cands(y, 3, 0.5, seed=1)
    a = GBDTSelfPLL(3).fit(X[:150], S[:150]).predict_proba(X[150:])
    b = GBDTSelfPLL(3).fit(X[:150], S[:150]).predict_proba(X[150:])
    assert np.array_equal(a, b)


def test_gbdt_llp_uses_counts():
    rng = np.random.default_rng(0)
    n = 320
    y = rng.integers(0, 2, n)
    X = rng.normal(size=(n, 3)) + 2.0 * y[:, None]
    tr = np.arange(240)
    bags = [tr[i:i + 8] for i in range(0, 240, 8)]
    counts = np.array([[8 - y[b].sum(), y[b].sum()] for b in bags])
    P = GBDTSelfLLP(0).fit(X[:240], bags, counts).predict_proba(X[240:])
    from sklearn.metrics import roc_auc_score
    assert P.shape == (80, 2) and roc_auc_score(y[240:], P[:, 1]) > 0.85
    with pytest.raises(ValueError):
        GBDTSelfLLP(0).fit(X[:240], bags[:-1], counts[:-1])


def _args(extra):
    import run_g7
    base = ["--split", "s3", "--eval-ok", "--seeds", "0,1,2", "--settings", "pll", "--levels", "pll=0.6", "--out", "/dev/null"]
    return run_g7, base + extra


@pytest.mark.parametrize("extra", [
    ["--supp", "--part", "tfm"],                                        # thiếu --only
    ["--supp", "--part", "tfm", "--only", "naive,gibbs_warm"],          # phương pháp ngoài danh sách
    ["--supp", "--part", "v0", "--starts", "warm", "--sweeps", "60"],   # số vòng quét khác 100
    ["--supp", "--part", "v0", "--starts", "warm,base", "--sweeps", "100"],
    ["--part", "tfm", "--only", "naive"],                               # không có --supp: khoá Study 3 giữ nguyên
    ["--part", "baselines"],
])
def test_s3b_argument_lock(extra):
    run_g7, argv = _args(extra)
    with pytest.raises(SystemExit):
        run_g7.main(argv)
