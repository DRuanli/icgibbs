"""Kiểm thử baseline G6 (T3). Phần so với code chính thức chỉ chạy khi có kho (biến môi trường *_REPO)."""
import itertools
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from icgibbs.baselines_ext import (PLLNet, count_dp, fit_llp, fit_nnpu, fit_pll, glws_estep_and_loglik,  # noqa: E402
                                   load_official)
from icgibbs.weak import make_bags  # noqa: E402


def test_count_dp_matches_brute_force():
    rng = np.random.default_rng(0)
    K = 5
    q = rng.uniform(0.05, 0.95, size=(3, K))
    mask = np.ones((3, K), bool)
    mask[2, 3:] = False
    logP = count_dp(torch.tensor(np.log(q)), torch.tensor(np.log(1 - q)), torch.tensor(mask)).numpy()
    for g in range(3):
        k = int(mask[g].sum())
        brute = np.zeros(K + 1)
        for y in itertools.product([0, 1], repeat=k):
            brute[sum(y)] += np.prod([q[g, i] if y[i] else 1 - q[g, i] for i in range(k)])
        assert np.allclose(np.exp(logP[g]), brute, atol=1e-9)


def test_glws_estep_is_exact_posterior():
    rng = np.random.default_rng(1)
    K, c = 6, 2
    q = rng.uniform(0.1, 0.9, size=K)
    probs = torch.tensor(np.column_stack([1 - q, q])[None], dtype=torch.float64)
    pseudo, sup = glws_estep_and_loglik(probs, torch.ones(1, K, dtype=torch.bool), np.array([c]))
    num, Pc, Ple = np.zeros(K), 0.0, 0.0
    for y in itertools.product([0, 1], repeat=K):
        p = np.prod([q[i] if y[i] else 1 - q[i] for i in range(K)])
        if sum(y) <= c:
            Ple += p
        if sum(y) == c:
            Pc += p
            num += p * np.array(y)
    assert np.allclose(pseudo[0, :, 1].numpy(), num / Pc, atol=1e-8)
    assert np.isclose(sup.item(), np.log(Pc / Ple), atol=1e-8)


def test_glws_against_official_code():
    """Bag likelihood P(đếm = c | đếm ≤ c) khớp hàm chính thức. Nhãn giả của hàm chính thức KHÔNG phải hậu nghiệm
    chính xác (tổng theo túi khác c) — phát hiện ở G6, ghi trong docs/baseline_reproduction.md; dự án dùng hậu
    nghiệm chính xác (đúng như bài mô tả), đã kiểm bằng liệt kê ở test_glws_estep_is_exact_posterior."""
    nfa = load_official("GLWS_REPO", os.path.join("src", "core", "nfa.py"), "glws_nfa")
    if nfa is None:
        pytest.skip("chưa có kho GLWS (đặt GLWS_REPO)")
    rng = np.random.default_rng(2)
    off_sums_wrong = 0
    for K, c in [(8, 0), (8, 3), (8, 8), (16, 5), (5, 1)]:
        q = rng.uniform(0.05, 0.95, size=K)
        probs = torch.tensor(np.column_stack([1 - q, q]), dtype=torch.float64)
        em, sp = nfa.create_proportion_graph(torch.log(probs), torch.tensor(c))
        pseudo, sup = glws_estep_and_loglik(probs[None], torch.ones(1, K, dtype=torch.bool), np.array([c]))
        assert np.isclose(float(torch.log(sp)), sup.item(), atol=1e-6), (K, c)
        assert np.isclose(pseudo[0, :, 1].sum().item(), c, atol=1e-6)          # của dự án: tổng đúng bằng c
        off_sums_wrong += not np.isclose(em[:, 1].sum().item(), c, atol=1e-3)
    assert off_sums_wrong >= 1


def test_vectorized_proportion_loss_matches_llp_vat():
    from icgibbs.third_party.llp_vat_losses import ProportionLoss
    rng = np.random.default_rng(3)
    avg = torch.tensor(rng.dirichlet([1, 1], size=6), dtype=torch.float32)
    tgt = torch.tensor(rng.dirichlet([1, 1], size=6), dtype=torch.float32)
    ref = sum(ProportionLoss("ce", 1.0)(avg[j], tgt[j]) for j in range(6)) / 6
    vec = (-tgt * torch.log(avg.clamp(1e-8, 1 - 1e-8))).sum(-1).mean()
    assert torch.allclose(ref, vec, atol=1e-6)


def _llp_data(n=800, seed=0, bag=8):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] + 0.3 * rng.normal(size=n) > 0).astype(int)
    bags, counts = make_bags(n, bag, rng, y)
    return X, y, bags, counts


@pytest.mark.parametrize("name", ["dllp", "dllp_linear", "llp_vat", "easyllp", "glws"])
def test_llp_baselines_learn(name):
    from sklearn.metrics import roc_auc_score
    X, y, bags, counts = _llp_data()
    torch.set_num_threads(1)
    m = fit_llp(name, X, bags, counts, seed=0)
    auc = roc_auc_score(y, m.predict_proba(X)[:, 1])
    assert auc > 0.85, (name, auc)
    assert np.isfinite(m.weak_loss(X, bags, counts))


def test_llp_tuning_runs():
    X, y, bags, counts = _llp_data(n=400)
    m = fit_llp("easyllp", X, bags, counts, seed=0, tune=True, n_configs=3)
    assert len(m.tune_.trials) == 3 and np.isfinite(m.tune_.best_score)


def _pll_data(n=600, C=4, q=0.3, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=3, size=(C, 5))
    y = rng.integers(C, size=n)
    X = centers[y] + rng.normal(size=(n, 5))
    S = rng.uniform(size=(n, C)) < q
    S[np.arange(n), y] = True
    return X, y, S


@pytest.mark.parametrize("name", ["proden", "glws", "rc", "cc"])
def test_pll_baselines_learn(name):
    X, y, S = _pll_data()
    torch.set_num_threads(1)
    m = fit_pll(name, X, S, seed=0)
    acc = float(np.mean(m.predict_proba(X).argmax(1) == y))
    assert acc > 0.85, (name, acc)


def test_proden_official_loss_matches_reimplementation():
    if load_official("PRODEN_REPO", os.path.join("utils", "utils_loss.py"), "x") is None:
        pytest.skip("chưa có kho PRODEN (đặt PRODEN_REPO)")
    X, y, S = _pll_data(n=300)
    torch.set_num_threads(1)
    m1 = PLLNet("proden", epochs=20, seed=0).fit(X, S)
    assert m1.source_ == "official"
    repo = os.environ.pop("PRODEN_REPO")
    try:
        m2 = PLLNet("proden", epochs=20, seed=0).fit(X, S)
        assert m2.source_ == "reimplemented"
    finally:
        os.environ["PRODEN_REPO"] = repo
    assert np.allclose(m1.predict_proba(X), m2.predict_proba(X), atol=1e-5)


def test_nnpu_tuned_runs():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 4))
    y = (X[:, 0] > 0).astype(int)
    s = ((y == 1) & (rng.uniform(size=500) < 0.3)).astype(int)
    m = fit_nnpu(X, s, prior=float(y.mean()), seed=0, tune=True, n_configs=3)
    assert len(m.tune_.trials) == 3


def test_proden_optimizer_choice_and_default_is_official_sgd():
    from icgibbs.baselines_ext import PLLNet, PLL_SPACES, PLL_DEFAULTS
    assert PLLNet("proden").optimizer == "sgd" and PLLNet("rc").optimizer == "adam"
    assert "optimizer" not in PLL_DEFAULTS["proden"] and set(PLL_SPACES["proden"]["optimizer"].values) == {"sgd", "adam"}
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4)); y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0).astype(int)
    S = rng.uniform(size=(300, 3)) < 0.3; S[np.arange(300), y] = True
    for opt in ("sgd", "adam"):
        m = PLLNet("proden", optimizer=opt, epochs=30, seed=0).fit(X, S)
        assert np.isfinite(m.predict_proba(X)).all()
    with pytest.raises(ValueError):
        PLLNet("proden", optimizer="rmsprop", epochs=1).fit(X, S)


def test_label_spreading_default_matches_pilot_config():
    """Cấu hình mặc định của bản tune trùng bản label spreading đã chạy trong thí điểm (kNN, k = 7, α = 0,2)."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.semi_supervised import LabelSpreading
    from icgibbs.baselines_ext import fit_ssl
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5)); y = (X[:, 0] + 0.5 * rng.normal(size=300) > 0).astype(int)
    y_obs = np.where(rng.uniform(size=300) < 0.1, y, -1)
    ours = fit_ssl("label_spreading", X[:200], y_obs[:200], tune=False).predict_proba(X[200:])
    sc = StandardScaler().fit(X[:200])
    ref = LabelSpreading(kernel="knn", n_neighbors=7, alpha=0.2, max_iter=100).fit(sc.transform(X[:200]), y_obs[:200])
    with np.errstate(invalid="ignore", divide="ignore"):
        R = ref.predict_proba(sc.transform(X[200:]))
    ok = np.isfinite(R).all(1)
    assert np.allclose(ours[ok], R[ok], atol=1e-6)


def test_ssl_and_forward_tuning_run_and_choose_from_space():
    from icgibbs.baselines_ext import fit_ssl, fit_forward, forward_val_loss
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 4)); y = (X[:, 0] > 0).astype(int)
    y_obs = np.where(rng.uniform(size=400) < 0.2, y, -1)
    for name in ("label_spreading", "self_training"):
        m = fit_ssl(name, X, y_obs, seed=0, tune=True, n_configs=5)
        P = m.predict_proba(X)
        assert P.shape == (400, 2) and np.allclose(P.sum(1), 1) and np.isfinite(m.tune_.best_score)
    T = np.array([[0.9, 0.1], [0.2, 0.8]])
    w = np.array([rng.choice(2, p=T[k]) for k in y])
    m = fit_forward(X, w, T, seed=0, tune=True, n_configs=3)
    assert np.isfinite(forward_val_loss(m, X, w, T)) and roc_auc(y, m.predict_proba(X)[:, 1]) > 0.8


def roc_auc(y, s):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s)
