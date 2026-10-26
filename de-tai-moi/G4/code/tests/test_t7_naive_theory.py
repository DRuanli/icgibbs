"""Kiểm thử cho G6 T7 (PROOFS §7): luật affine của nhãn ngây thơ, chặn hối tiếc AUC, thông tin Fisher."""
import numpy as np
import pytest

import g6_t7_naive_theory as t7


@pytest.mark.parametrize("C,q", [(3, 0.3), (4, 0.6), (7, 0.6)])
def test_pll_affine_law_matches_pilot_generator(C, q):
    rng = np.random.default_rng(0)
    M = t7.pll_affine_mc(C, q, 30000, rng)
    a, b = t7.pll_affine_coeffs(C, q)
    assert abs(np.diag(M).mean() - a) < 0.01
    assert abs(M[~np.eye(C, dtype=bool)].mean() - b) < 0.01
    assert np.abs(M - (b + (a - b) * np.eye(C))).max() < 0.03          # từng phần tử, không chỉ trung bình


def test_pll_coeffs_limits():
    for C in (2, 3, 7):
        a, b = t7.pll_affine_coeffs(C, 0.0)
        assert (a, b) == (1.0, 0.0)
        for q in (0.1, 0.5, 0.9):
            a, b = t7.pll_affine_coeffs(C, q)
            assert a - b > 0 and abs(a + (C - 1) * b - 1) < 1e-12
        a, b = t7.pll_affine_coeffs(C, 1.0)
        assert abs(a - b) < 1e-12                                       # q = 1: tập ứng viên là mọi lớp, mất hết tín hiệu


@pytest.mark.parametrize("k", [4, 16])
def test_llp_affine_law_matches_pilot_generator(k):
    rng = np.random.default_rng(1)
    y = (rng.uniform(size=1000) < 0.35).astype(int)
    ex, mc = t7.llp_affine_exact(y, k), t7.llp_affine_mc(y, k, 400, rng)
    assert abs(ex[1] - mc[1]) < 0.01 and abs(ex[0] - mc[0]) < 0.01
    n = len(y) - len(y) % k
    assert abs((ex[1] - ex[0]) - (1 / k - (k - 1) / (k * (n - 1)))) < 1e-12


def test_auc_regret_bound_never_violated():
    assert t7.auc_bound_check(np.random.default_rng(2), trials=400) <= 1e-12


def test_fisher_pll_limits_and_data_processing():
    rng = np.random.default_rng(3)
    cl, fu, na = t7.fisher_pll(rng, 4, 1e-9, 20000)
    assert abs(fu - cl) / cl < 1e-6 and abs(na - cl) / cl < 1e-6        # q → 0: ứng viên = nhãn thật
    for q in (0.3, 0.6, 0.9):
        cl, fu, na = t7.fisher_pll(rng, 4, q, 20000)
        assert na <= fu + 1e-12 <= cl + 1e-12                          # xử lý dữ liệu: ngây thơ ≤ đầy đủ ≤ sạch


def test_fisher_llp_bag_of_one_is_clean_label():
    rng = np.random.default_rng(4)
    xs = rng.normal(size=200000)
    clean = t7.fisher_binary_channel(xs, 1.0, 0.0)
    full = t7.fisher_llp_bag(np.random.default_rng(5), 1, 20000)
    god, H = t7.godambe_llp_naive(np.random.default_rng(6), 1, 20000, reps_per_bag=4)
    assert abs(full - clean) / clean < 0.05 and abs(H - clean) / clean < 0.05 and abs(god - clean) / clean < 0.1


def test_fisher_data_processing_inequality_random_models():
    """I_w̃ ≤ I_O khi w̃ sinh từ O qua nhân không phụ thuộc θ (PROOFS §7, Mệnh đề 6(a)) — trên mô hình hữu hạn ngẫu nhiên."""
    rng = np.random.default_rng(7)
    for _ in range(200):
        m, r = rng.integers(2, 7), rng.integers(2, 7)
        A, B = rng.normal(size=m), rng.normal(size=m)
        th, h = 0.3, 1e-6
        pO = lambda t: np.exp(A + B * t) / np.exp(A + B * t).sum()
        R = rng.dirichlet(np.ones(r), size=m)                          # R[o, w] không phụ thuộc θ
        I = lambda f: float(np.sum(((f(th + h) - f(th - h)) / (2 * h)) ** 2 / f(th)))
        assert I(lambda t: pO(t) @ R) <= I(pO) + 1e-6
