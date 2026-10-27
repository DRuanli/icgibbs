"""Phần ablation của bộ chạy G7 (giao thức Mục 7; deviation 11) và ablation cỡ dữ liệu (--n_train_max), với TFM giả
('dry') trên dữ liệu tổng hợp."""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import run_g7  # noqa: E402
from icgibbs.g7.splits import nested_subsample  # noqa: E402

DRY = ["--backend", "dry", "--source", "synthetic", "--seeds", "0"]


def _rows(path):
    return list(csv.DictReader(open(path)))


def test_nested_subsample_is_nested_stratified_deterministic():
    rng = np.random.default_rng(0)
    y = (rng.random(3000) < 0.1).astype(int)
    itr = rng.permutation(3000)[:2100]
    a = nested_subsample(itr, y, 500, seed=1, did=7)
    b = nested_subsample(itr, y, 1000, seed=1, did=7)
    assert len(a) == 500 and len(b) == 1000 and set(a) <= set(b) <= set(itr)
    assert np.array_equal(a, nested_subsample(itr, y, 500, seed=1, did=7))
    assert abs(y[a].sum() - 500 * y[itr].mean()) <= 1.5
    assert np.array_equal(nested_subsample(itr, y, 5000, seed=1, did=7), itr)


def test_rules_preregistered():
    assert run_g7.ABLATIONS["blocks"]["datasets"] == "small3"
    assert run_g7.ABLATIONS["init"]["levels"] == {"llp": [8], "pll": [0.3, 0.6]}
    from icgibbs.g7.registry import ABL_LARGE_EVAL, ABL_SMALL_EVAL, RHAT_EVAL
    assert ABL_SMALL_EVAL == RHAT_EVAL and ABL_LARGE_EVAL == {"binary": [1489, 44, 38], "multiclass": [182, 1475, 28]}


def test_ablation_parts_dry(tmp_path):
    expect = {"init": {("llp", "gibbs_cold"), ("pll", "gibbs_cold")},
              "blocks": {("llp", "gibbs_warm_K5"), ("llp", "gibbs_warm_K20"), ("pll", "gibbs_warm_K5"),
                         ("pll", "gibbs_warm_K20")},
              "sweeps": {("llp", "gibbs_warm_S15"), ("llp", "gibbs_warm_S60"), ("pll", "gibbs_warm_S15"),
                         ("pll", "gibbs_warm_S60")},
              "trans": {("llp", "gibbs_warm_trans")},
              "prior": {("pu_sar", "gibbs_warm_prior1"), ("pu_sar", "gibbs_warm_prior4")}}
    for name, want in expect.items():
        out = str(tmp_path / f"{name}.csv")
        assert run_g7.main(["--part", "ablation", "--ablation", name, "--datasets", "1510,54", *DRY, "--out", out]) == 0
        rows = _rows(out)
        assert all(r["status"] == "ok" for r in rows), [r["status"] for r in rows if r["status"] != "ok"]
        assert {(r["setting"], r["method"]) for r in rows} == want, name
        if name == "sweeps":
            assert any("sweeps=15;burn_in=5" in r["extra"] for r in rows)
        if name == "trans":
            assert all("mode=transductive" in r["extra"] and float(r["auc"]) > 0.7 for r in rows)
        if name == "init":
            assert {r["level"] for r in rows if r["setting"] == "pll"} == {"0.3", "0.6"}
    # chạy lại: không thêm dòng
    out = str(tmp_path / "trans.csv")
    n = len(_rows(out))
    run_g7.main(["--part", "ablation", "--ablation", "trans", "--datasets", "1510", *DRY, "--out", out])
    assert len(_rows(out)) == n


def test_mh_ablation_dry(tmp_path):
    out = str(tmp_path / "mh.csv")
    assert run_g7.main(["--part", "ablation", "--ablation", "mh", "--datasets", "1510", *DRY, "--out", out]) == 0
    rows = _rows(out)
    assert [(r["setting"], r["method"], r["status"]) for r in rows] == [("llp", "gibbs_warm_mh", "ok")]
    assert "accept=" in rows[0]["extra"]


def test_size_ablation_subsamples_and_skips(tmp_path):
    out = str(tmp_path / "size.csv")
    args = ["--part", "tfm", "--n_train_max", "200", "--datasets", "31", *DRY, "--no-oracle", "--out", out]
    assert run_g7.main(args) == 0
    rows = _rows(out)
    assert rows and all(r["n_train"] == "200" and r["setting"] == "llp" and r["level"] == "8" for r in rows)
    out2 = str(tmp_path / "size_big.csv")
    assert run_g7.main(["--part", "tfm", "--n_train_max", "100000", "--datasets", "31", *DRY, "--out", out2]) == 0
    assert not os.path.exists(out2) or len(_rows(out2)) == 0
    # cùng file, cỡ khác: khoá có n_train nên không bị coi là đã xong
    assert run_g7.main(["--part", "tfm", "--n_train_max", "300", "--datasets", "31", *DRY, "--no-oracle",
                        "--out", out]) == 0
    assert {r["n_train"] for r in _rows(out)} == {"200", "300"}
