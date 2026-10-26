import numpy as np

from icgibbs.g7.rhat import label_rhat, split_rhat


def test_split_rhat_mixed_vs_stuck():
    rng = np.random.default_rng(0)
    mixed = rng.normal(size=(2, 400))
    assert abs(split_rhat(mixed) - 1.0) < 0.02
    stuck = np.stack([rng.normal(0, 1, 400), rng.normal(3, 1, 400)])
    assert split_rhat(stuck) > 1.5
    trend = np.stack([np.linspace(0, 5, 400) + rng.normal(size=400)] * 2)   # trôi trong mỗi chuỗi: split-R̂ bắt được
    assert split_rhat(trend) > 1.2


def test_label_rhat_detects_chains_in_different_modes():
    rng = np.random.default_rng(1)
    n_draws, n = 40, 200
    same = [rng.integers(0, 2, (n_draws, n)) for _ in range(2)]
    r1 = label_rhat(*same)
    assert r1["frac_rhat_gt_1_1"] < 0.3 and abs(r1["rhat_global"] - 1) < 0.2
    A = (rng.uniform(size=(n_draws, n)) < 0.9).astype(int)
    B = (rng.uniform(size=(n_draws, n)) < 0.1).astype(int)
    r2 = label_rhat(list(A), list(B))
    assert r2["rhat_median"] > 1.5 and r2["frac_rhat_gt_1_1"] > 0.9
