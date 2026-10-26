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
