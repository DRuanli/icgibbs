"""Bộ chạy G7: chặn tập đánh giá nếu thiếu --eval-ok; chạy đầu-cuối trên dữ liệu tổng hợp với TFM giả ('dry'); chạy tiếp."""
import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import run_g7  # noqa: E402


def test_eval_split_requires_explicit_flag(tmp_path):
    with pytest.raises(SystemExit):
        run_g7.main(["--part", "tfm", "--split", "eval", "--out", str(tmp_path / "x.csv")])


def test_end_to_end_dry_and_resume(tmp_path):
    out = str(tmp_path / "r.csv")
    args = ["--part", "tfm", "--backend", "dry", "--source", "synthetic", "--datasets", "31", "--settings", "llp",
            "--levels", "llp=8", "--seeds", "0", "--out", out]
    assert run_g7.main(args) == 0
    rows = list(csv.DictReader(open(out)))
    got = {r["method"]: r["status"] for r in rows}
    assert got == {m: "ok" for m in ("oracle", "naive", "gibbs_warm", "sem_warm", "hard_em_warm")}
    r = next(r for r in rows if r["method"] == "gibbs_warm")
    assert r["sha256_table"] and r["sha256_split"] and r["weak_hash"] and r["code"] and "flips=" in r["extra"]
    assert float(r["auc"]) > 0.8
    assert run_g7.main(args) == 0                              # chạy lại: không thêm dòng nào
    assert len(list(csv.DictReader(open(out)))) == len(rows)


def test_methods_never_receive_clean_train_labels():
    from icgibbs.g7.methods import Cell
    assert not {"y", "ytr", "y_train", "y_true"} & set(Cell.__dataclass_fields__)


def test_rhat_part_dry(tmp_path):
    out = str(tmp_path / "rh.csv")
    assert run_g7.main(["--part", "rhat", "--backend", "dry", "--source", "synthetic", "--datasets", "31",
                        "--settings", "llp", "--levels", "llp=8", "--seeds", "0", "--out", out]) == 0
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 1 and rows[0]["status"] == "ok" and "rhat_median=" in rows[0]["extra"]


def test_real_pll_part_requires_eval_flags_and_runs(tmp_path):
    import numpy as np
    from icgibbs.g7.registry import REAL_PLL
    rng = np.random.default_rng(0)
    for name, (_, sha) in REAL_PLL.items():                 # dữ liệu giả, cùng định dạng tệp của việc F7
        n, d, Q = 300, 5, 6
        X = rng.normal(size=(n, d)).astype(np.float32); y = (X[:, 0] > 0).astype(int) + 2 * (X[:, 1] > 0)
        S = rng.uniform(size=(n, Q)) < 0.2; S[np.arange(n), y] = True
        np.savez_compressed(tmp_path / f"{name}.npz", X=X, S=S, y=y.astype(np.int16), sha256=sha)
    out = str(tmp_path / "rq5.csv")
    base = ["--part", "tfm", "--backend", "dry", "--settings", "pll_real", "--rq5_dir", str(tmp_path),
            "--rq5_seeds", "0", "--out", out]
    with pytest.raises(SystemExit):
        run_g7.main(base)                                    # thiếu --split eval --eval-ok
    assert run_g7.main(base + ["--split", "eval", "--eval-ok"]) == 0
    rows = list(csv.DictReader(open(out)))
    ok = [r for r in rows if r["status"] == "ok"]
    assert {r["dataset"] for r in ok} == set(REAL_PLL) and {r["method"] for r in ok} >= {"naive", "gibbs_warm"}
    assert all(r["n_classes"] == "6" for r in ok) and all(r["n_test"] == "30" for r in ok)
