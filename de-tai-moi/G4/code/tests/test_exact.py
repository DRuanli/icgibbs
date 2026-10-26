"""Kiểm thử tính đúng (chạy: cd G4/code && python -m pytest -q tests)."""
import itertools

import numpy as np
import pytest
from scipy.special import logsumexp

from icgibbs import (LLP, PUSCAR, SSL, AnalyticCellPPD, DiscreteMixturePPD, InContextGibbs, NoisyLabel,
                     PartialLabel, PerturbedConditional, fit_transductive, sample_count_constrained)
from icgibbs.exact import (detailed_balance_error, exact_predictive, index_of, kernel_bag_block, kernel_block,
                           kernel_single_site, kernel_swap, log_ratio_incompatibility, log_target, stationary,
                           support_states, tv)
from icgibbs.llp import count_constrained_log_normalizer

RNG = np.random.default_rng(0)


def toy(n=8, M=3, seed=0):
    r = np.random.default_rng(seed)
    return r.integers(0, M, size=(n, 1)).astype(float)


def priors(M=3, C=2):
    return [AnalyticCellPPD(C, alpha=1.0), DiscreteMixturePPD.random(M, C, seed=1)]


# ---- Bổ đề 1 -------------------------------------------------------------------------------------
@pytest.mark.parametrize("pidx", [0, 1])
def test_lemma1_full_conditional_equals_ppd(pidx):
    X = toy(7)
    cond = priors()[pidx]
    for y in itertools.product(range(2), repeat=7):
        y = np.array(y)
        for i in range(7):
            lj = np.array([cond.log_joint(X, np.where(np.arange(7) == i, k, y)) for k in range(2)])
            full = np.exp(lj - logsumexp(lj))
            mask = np.arange(7) != i
            ppd = cond.predict_proba(X[mask], y[mask], X[[i]])[0]
            assert np.allclose(full, ppd, atol=1e-12)


# ---- Định lý 1(a): Gibbs một mẫu ------------------------------------------------------------------
def weak_models(n, rng):
    y_true = rng.integers(0, 2, n)
    s = np.where((y_true == 1) & (rng.uniform(size=n) < 0.5), 1, 0)
    yo = np.where(rng.uniform(size=n) < 0.4, y_true, -1)
    S = np.ones((n, 2), bool)
    S[:2] = False
    S[0, y_true[0]] = True
    S[1, y_true[1]] = True
    T = np.array([[0.8, 0.2], [0.3, 0.7]])
    w = np.array([rng.choice(2, p=T[k]) for k in y_true])
    return {"pu": PUSCAR(s, c=0.5), "ssl": SSL(yo, 2), "pll": PartialLabel(S), "noisy": NoisyLabel(w, 2, T=T)}


@pytest.mark.parametrize("pidx", [0, 1])
@pytest.mark.parametrize("wname", ["pu", "ssl", "pll", "noisy"])
def test_single_site_stationary_is_posterior(pidx, wname):
    X = toy(8)
    cond = priors()[pidx]
    weak = weak_models(8, np.random.default_rng(3))[wname]
    states = support_states(weak)
    logpi = log_target(cond.log_joint, X, states, weak)
    P = kernel_single_site(cond, X, states, weak)
    assert np.allclose(P.sum(1), 1)
    assert tv(stationary(P), np.exp(logpi)) < 1e-10
    assert detailed_balance_error(P, np.exp(logpi)) < 1e-12


# ---- Định lý 1(b): MH hoán đổi -------------------------------------------------------------------
@pytest.mark.parametrize("pidx", [0, 1])
def test_swap_detailed_balance_bag6(pidx):
    X = toy(6)
    cond = priors()[pidx]
    weak = LLP([np.arange(6)], [[3, 3]], n=6)
    states = support_states(weak)
    logpi = log_target(cond.log_joint, X, states, weak)
    P = kernel_swap(cond, X, states, weak)
    assert detailed_balance_error(P, np.exp(logpi)) < 1e-12
    assert tv(stationary(P), np.exp(logpi)) < 1e-10


def test_swap_stationary_multi_bag():
    X = toy(9)
    cond = priors()[1]
    weak = LLP([np.arange(0, 4), np.arange(4, 7), np.arange(7, 9)], [[2, 2], [1, 2], [1, 1]], n=9)
    states = support_states(weak)
    logpi = log_target(cond.log_joint, X, states, weak)
    P = kernel_swap(cond, X, states, weak)
    assert tv(stationary(P), np.exp(logpi)) < 1e-10


# ---- Mệnh đề 4: khối ----------------------------------------------------------------------------
def test_block_kinds():
    X = toy(8)
    cond = priors()[1]                          # x mang thông tin: cả hai nguồn sai đều có mặt
    weak = SSL(np.array([1, 0, -1, -1, -1, -1, -1, -1]), 2)
    states = support_states(weak)
    pi = np.exp(log_target(cond.log_joint, X, states, weak))
    free = weak.free_sites()
    blocks = [list(c) for c in itertools.combinations(free, 3)]
    assert tv(stationary(kernel_block(cond, X, states, weak, blocks, kind="exact", logpi=np.log(pi))), pi) < 1e-10
    assert tv(stationary(kernel_block(cond, X, states, weak, blocks, kind="mh")), pi) < 1e-10
    assert tv(stationary(kernel_block(cond, X, states, weak, blocks, kind="approx")), pi) > 1e-4


def test_bag_block_exact_vs_approx():
    X = toy(8)
    cond = priors()[0]
    weak = LLP([np.arange(0, 4), np.arange(4, 8)], [[2, 2], [1, 3]], n=8)
    states = support_states(weak)
    pi = np.exp(log_target(cond.log_joint, X, states, weak))
    Pe = kernel_bag_block(cond, X, states, weak, [[0], [1]], kind="exact", logpi=np.log(pi))
    assert tv(stationary(Pe), pi) < 1e-10
    Pa = kernel_bag_block(cond, X, states, weak, [[0], [1]], kind="approx")
    assert np.allclose(Pa.sum(1), 1)


# ---- DP có ràng buộc số đếm ----------------------------------------------------------------------
@pytest.mark.parametrize("C", [2, 3])
def test_count_constrained_sampler(C):
    rng = np.random.default_rng(0)
    m = 5
    P = rng.dirichlet(np.ones(C), size=m)
    counts = np.bincount(rng.integers(0, C, m), minlength=C)
    configs, logw = [], []
    for y in itertools.product(range(C), repeat=m):
        y = np.array(y)
        if (np.bincount(y, minlength=C) == counts).all():
            configs.append(tuple(y))
            logw.append(np.log(P[np.arange(m), y]).sum())
    logw = np.array(logw)
    assert abs(logsumexp(logw) - count_constrained_log_normalizer(P, counts)) < 1e-10
    target = np.exp(logw - logsumexp(logw))
    draws = [tuple(sample_count_constrained(P, counts, rng)) for _ in range(20000)]
    emp = np.array([draws.count(c) for c in configs]) / len(draws)
    assert tv(emp, target) < 0.03


# ---- Cài đặt thật của bộ lấy mẫu (Monte Carlo) ---------------------------------------------------
def _emp_tv(samples, states, pi):
    idx = index_of(states)
    cnt = np.zeros(len(states))
    for s in samples:
        cnt[idx[tuple(s)]] += 1
    return tv(cnt / cnt.sum(), pi)


@pytest.mark.parametrize("scheme,mh", [("single", False), ("block", True)])
def test_sampler_instance_mc(scheme, mh):
    X = toy(7)
    cond = priors()[1]
    weak = PUSCAR(np.array([1, 1, 0, 0, 0, 0, 0]), c=0.4)
    states = support_states(weak)
    pi = np.exp(log_target(cond.log_joint, X, states, weak))
    g = InContextGibbs(cond, weak, scheme=scheme, n_blocks=2, mh_correct=mh, n_sweeps=6000, burn_in=50, seed=1).run(X)
    assert _emp_tv(g.samples_, states, pi) < 0.04


@pytest.mark.parametrize("scheme", ["swap", "bag-block"])
def test_sampler_llp_mc(scheme):
    X = toy(8)
    cond = priors()[1]
    weak = LLP([np.arange(0, 4), np.arange(4, 8)], [[2, 2], [1, 3]], n=8)
    states = support_states(weak)
    pi = np.exp(log_target(cond.log_joint, X, states, weak))
    g = InContextGibbs(cond, weak, scheme=scheme, n_blocks=2, mh_correct=(scheme == "bag-block"),
                       n_sweeps=5000, burn_in=50, seed=2).run(X)
    assert weak.satisfied(g.y_)
    assert _emp_tv(g.samples_, states, pi) < 0.04


def test_rao_blackwell_and_transductive():
    """Mixture prior: dự đoán quy nạp p̄ (Định lý 1(ii)) khác p(y*|x*,X,W) chính xác; chế độ chuyển nạp khớp."""
    X = toy(7, M=3, seed=4)
    cond = priors()[1]
    yo = np.array([1, 0, 1, -1, -1, -1, -1])
    weak = SSL(yo, 2)
    states = support_states(weak)
    logpi = log_target(cond.log_joint, X, states, weak)
    x_star = np.array([2.0])
    pbar = exact_predictive(cond.log_joint, X, states, logpi, x_star, 2)
    # đúng Bayes: thêm x* như điểm tự do rồi lấy biên
    weak2 = SSL(np.append(yo, -1), 2)
    X2 = np.vstack([X, x_star[None]])
    st2 = support_states(weak2)
    lp2 = log_target(cond.log_joint, X2, st2, weak2)
    exact = np.array([np.exp(lp2[st2[:, -1] == k]).sum() for k in range(2)])
    # mẫu từ thuật toán
    g = InContextGibbs(cond, SSL(yo, 2), scheme="single", n_sweeps=4000, burn_in=50, seed=0).run(X)
    rb = g.predict_proba(X, x_star[None])[0]
    assert abs(rb[1] - pbar[1]) < 0.02
    pt, _ = fit_transductive(cond, SSL(yo, 2), X, x_star[None], scheme="single", n_sweeps=4000, burn_in=50, seed=0)
    assert abs(pt[0, 1] - exact[1]) < 0.02


def test_perturbed_is_incompatible():
    X = toy(6)
    q = PerturbedConditional(AnalyticCellPPD(2), delta=0.5, seed=0)
    y = np.array([0, 1, 0, 1, 1, 0])
    vals = [abs(log_ratio_incompatibility(q, X, y, i, j, 1 - y[i], 1 - y[j]))
            for i in range(6) for j in range(6) if i != j]
    assert max(vals) > 1e-3
    p = AnalyticCellPPD(2)
    vals0 = [abs(log_ratio_incompatibility(p, X, y, i, j, 1 - y[i], 1 - y[j])) for i in range(6) for j in range(6) if i != j]
    assert max(vals0) < 1e-12


def test_block_over_all_keeps_fixed_and_runs():
    """Khối trên mọi mẫu: mẫu có nhãn cố định không đổi; chuỗi chạy được; với mh_correct vẫn chính xác."""
    X = toy(7)
    cond = priors()[1]
    weak = PUSCAR(np.array([1, 1, 0, 0, 0, 0, 0]), c=0.4)
    states = support_states(weak)
    pi = np.exp(log_target(cond.log_joint, X, states, weak))
    g = InContextGibbs(cond, weak, scheme="block", n_blocks=3, block_over="all", mh_correct=True,
                       n_sweeps=6000, burn_in=50, seed=4).run(X)
    assert all(s[0] == 1 and s[1] == 1 for s in g.samples_)
    assert _emp_tv(g.samples_, states, pi) < 0.04
