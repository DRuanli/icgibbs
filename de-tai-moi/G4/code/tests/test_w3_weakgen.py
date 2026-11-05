"""V2 W3: biến thể nhãn yếu cho Study 3 — LLP túi gom theo feature (feat) và PLL phụ thuộc mẫu (inst)."""
import csv
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import run_g7  # noqa: E402
from icgibbs.g7.weakgen import feature_bags, instance_pll, make_weak, weak_rng  # noqa: E402


def _data(n=900, d=5, C=4, seed=0):
    from sklearn.datasets import make_classification
    X, y = make_classification(n, d, n_informative=4, n_redundant=0, n_classes=C, n_clusters_per_class=1,
                               random_state=seed)
    X[::37, 1] = np.nan                                   # có NaN như dữ liệu thật
    return X.astype(np.float32), y


def test_feature_bags_partition_sizes_and_homogeneity():
    X, y = _data()
    bags, counts = feature_bags(X, 8, np.random.default_rng(3), y, n_classes=4)
    allidx = np.concatenate(bags)
    assert np.array_equal(np.sort(allidx), np.arange(len(y)))              # mỗi điểm đúng một túi
    sizes = np.array([len(b) for b in bags])
    assert sizes.max() <= 15 and np.mean(sizes == 8) > 0.8
    assert counts.shape == (len(bags), 4) and np.array_equal(counts.sum(0), np.bincount(y, minlength=4))
    # túi gom theo feature đồng nhất hơn túi ngẫu nhiên: phương sai feature trong túi nhỏ hơn rõ
    Z = np.nan_to_num(X)
    within = np.mean([Z[b].var(0).mean() for b in bags])
    perm = np.random.default_rng(0).permutation(len(y))
    rand = np.mean([Z[perm[i:i + 8]].var(0).mean() for i in range(0, len(y), 8)])
    assert within < 0.6 * rand
    # tất định theo seed
    bags2, _ = feature_bags(X, 8, np.random.default_rng(3), y, n_classes=4)
    assert all(np.array_equal(a, b) for a, b in zip(bags, bags2))


def test_instance_pll_matches_mean_candidates_and_depends_on_instance():
    X, y = _data(C=5)
    q, C = 0.6, 5
    S, xi = instance_pll(X, y, q, np.random.default_rng(1), C)
    n = len(y)
    assert S[np.arange(n), y].all() and np.allclose(xi[np.arange(n), y], 0)
    wrong = np.ones((n, C), bool)
    wrong[np.arange(n), y] = False
    assert abs(xi[wrong].mean() - q) < 1e-6                                  # đúng mức q trung bình
    assert abs(S.sum(1).mean() - (1 + q * (C - 1))) < 0.15                  # số ứng viên trung bình như PLL gieo đều
    # phụ thuộc mẫu: ξ không hằng — có cặp gần như chắc chắn và cặp rất hiếm
    assert np.mean(xi[wrong] > 0.99) > 0.1 and np.mean(xi[wrong] < 0.2) > 0.1
    S2, _ = instance_pll(X, y, q, np.random.default_rng(1), C)
    assert np.array_equal(S, S2)


def test_make_weak_variants_and_errors():
    X, y = _data(n=400, C=3)
    wd = make_weak("llp", 8, y, X, weak_rng(0, 99, "llp"), n_classes=3, variant="feat")
    assert wd.setting == "llp" and "variant=feat" in wd.extra and len(wd.bags) == len(wd.counts)
    wp = make_weak("pll", 0.6, y, X, weak_rng(0, 99, "pll"), n_classes=3, variant="inst")
    assert wp.setting == "pll" and wp.S.shape == (400, 3) and "variant=inst" in wp.extra
    assert all(wp.S[i, wp.naive_pll[i]] for i in range(400))
    with pytest.raises(ValueError):
        make_weak("llp", 8, y, X, weak_rng(0, 99, "llp"), n_classes=3, variant="inst")
    with pytest.raises(ValueError):
        make_weak("pll", 0.6, y, X, weak_rng(0, 99, "pll"), n_classes=3, variant="feat")
    # mặc định iid không đổi so với trước (cùng seed → cùng túi như make_bags)
    a = make_weak("llp", 8, y, X, weak_rng(0, 99, "llp"), n_classes=3)
    b = make_weak("llp", 8, y, X, weak_rng(0, 99, "llp"), n_classes=3, variant="iid")
    assert all(np.array_equal(u, v) for u, v in zip(a.bags, b.bags))


@pytest.mark.parametrize("setting,level,did,variant", [("llp", "8", "1510", "feat"), ("pll", "0.6", "54", "inst")])
def test_runner_dry_writes_variant_label(tmp_path, setting, level, did, variant):
    out, sd = str(tmp_path / "w3.csv"), str(tmp_path / "npz")
    args = ["--part", "v0", "--settings", setting, "--levels", f"{setting}={level}", "--datasets", did,
            "--backend", "dry", "--source", "synthetic", "--seeds", "0", "--weak_variant", variant,
            "--save_dir", sd, "--out", out]
    assert run_g7.main(args) == 0
    rows = list(csv.DictReader(open(out)))
    assert rows and all(r["status"] == "ok" and r["setting"] == f"{setting}_{variant}" for r in rows)
    assert os.listdir(sd)[0].startswith(f"v1_dry_{did}_{setting}_{variant}_")


def test_runner_rejects_bad_variant_combo(tmp_path):
    with pytest.raises(SystemExit):
        run_g7.main(["--part", "v0", "--settings", "pll", "--backend", "dry", "--source", "synthetic", "--seeds", "0",
                     "--weak_variant", "feat", "--out", str(tmp_path / "x.csv")])
