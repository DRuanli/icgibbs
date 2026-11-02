"""Nghiên cứu 2: danh sách S2 rời NC1, khớp bảng chọn tất định; bộ chạy chặn S2 khi thiếu --eval-ok và chỉ cho
phần tfm/baselines; chạy đầu-cuối trên dữ liệu tổng hợp với ID của S2."""
import csv
import os
import sys

import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
import run_g7  # noqa: E402
from icgibbs.g7 import registry as R  # noqa: E402


def test_s2_disjoint_from_nc1():
    nc1 = set(R.EVAL_BINARY) | set(R.EVAL_MULTICLASS) | set(R.DEV_BINARY) | set(R.DEV_MULTICLASS)
    s2 = set(R.S2_BINARY) | set(R.S2_MULTICLASS) | {d for d, _, _ in R.S2_RESERVE_MULTICLASS}
    assert not nc1 & s2
    assert len(R.S2_BINARY) >= 10 and len(R.S2_MULTICLASS) >= 10
    assert not set(R.S2_BINARY) & set(R.S2_MULTICLASS)


def test_s2_registry_matches_selection_table():
    base = os.path.join(HERE, "..", "..", "..", "S2")
    if not os.path.exists(os.path.join(base, "s2_final.csv")):  # bản code tải lên Kaggle không kèm thư mục S2
        pytest.skip("không có S2/s2_final.csv")
    sel = list(csv.DictReader(open(os.path.join(base, "s2_selected.csv"))))
    reserve = [int(r["did"]) for r in sel if r["group"] == "multiclass" and r["role"] == "dự phòng"]
    assert reserve == [d for d, _, _ in R.S2_RESERVE_MULTICLASS]
    fin = list(csv.DictReader(open(os.path.join(base, "s2_final.csv"))))
    kept = {g: {int(r["did"]) for r in fin if r["group"] == g and r["kept"] == "True"} for g in ("binary", "multiclass")}
    assert kept["binary"] == set(R.S2_BINARY)
    assert kept["multiclass"] == set(R.S2_MULTICLASS)
    sub = {int(r["did"]): int(float(r["subsample"])) for r in fin if r["subsample"]}
    assert sub == R.S2_SUBSAMPLE


def test_subsample_table_stratified_and_deterministic():
    import numpy as np
    import pandas as pd
    from icgibbs.g7.data import Table, subsample_table
    rng = np.random.default_rng(0)
    y = rng.choice(4, 3000, p=[0.5, 0.3, 0.15, 0.05])
    t = Table(1, "x", 1, None, pd.DataFrame({"a": rng.normal(size=3000)}), y, [0, 1, 2, 3], 3000, 0, 0)
    a, b = subsample_table(t, 1000, 0), subsample_table(t, 1000, 0)
    assert len(a.y) == 1000 and np.array_equal(a.y, b.y) and a.X.equals(b.X)
    assert np.allclose(np.bincount(a.y) / 1000, np.bincount(y) / 3000, atol=0.01)
    assert subsample_table(t, 5000, 0) is t


def test_s2_requires_flag_and_restricts_parts(tmp_path):
    out = str(tmp_path / "x.csv")
    with pytest.raises(SystemExit):
        run_g7.main(["--part", "tfm", "--split", "s2", "--out", out])
    for extra in (["--part", "rhat"], ["--part", "ablation", "--ablation", "init"], ["--part", "tfm", "--n_train_max", "500"]):
        with pytest.raises(SystemExit):
            run_g7.main(extra + ["--split", "s2", "--eval-ok", "--out", out])


def test_s2_end_to_end_dry(tmp_path):
    out = str(tmp_path / "s2.csv")
    did = next(iter(R.S2_BINARY))
    args = ["--part", "tfm", "--split", "s2", "--eval-ok", "--backend", "dry", "--source", "synthetic",
            "--datasets", str(did), "--settings", "llp", "--levels", "llp=8", "--seeds", "0", "--out", out]
    assert run_g7.main(args) == 0
    rows = list(csv.DictReader(open(out)))
    assert {r["method"] for r in rows} == {"oracle", "naive", "gibbs_warm", "sem_warm", "hard_em_warm"}
    assert all(r["status"] == "ok" and int(r["dataset_id"]) == did for r in rows)
