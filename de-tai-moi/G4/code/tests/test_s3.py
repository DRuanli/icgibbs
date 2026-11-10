"""Study 3 bản gọn (S3/STUDY3_PLAN.md): danh sách dataset và khoá cấu hình của --split s3 (không gọi mạng)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import run_g7  # noqa: E402
from icgibbs.g7 import registry as R  # noqa: E402
import s3_dataset_census as S3  # noqa: E402


def test_s3_lists_frozen_and_disjoint():
    assert len(R.S3_BINARY) == 12 and len(R.S3_MULTICLASS) == 8
    ex = S3.load_excluded()
    assert not (set(R.S3_BINARY) | set(R.S3_MULTICLASS)) & set(ex)        # không dataset đã dùng, không dataset TabDPT đã học
    assert set(R.S3_SUBSAMPLE) <= set(R.S3_BINARY) | set(R.S3_MULTICLASS)
    assert len(R.S3_TABPFN_V2) == 11 and 1038 not in R.S3_TABPFN_V2 and 4538 in R.S3_TABPFN_V2


BASE = ["--split", "s3", "--eval-ok", "--part", "v0", "--backend", "dry", "--starts", "warm,base,cold",
        "--settings", "llp", "--levels", "llp=8", "--seeds", "0,1,2"]


@pytest.mark.parametrize("bad", [
    ["--sweeps", "10"], ["--weak_variant", "feat"], ["--levels", "llp=32"], ["--seeds", "0"],
    ["--starts", "warm"], ["--source", "synthetic"], ["--part", "tfm"],
])
def test_s3_rejects_any_config_change(tmp_path, bad):
    with pytest.raises(SystemExit):
        run_g7.main(BASE + bad + ["--out", str(tmp_path / "x.csv")])


def test_s3_needs_eval_ok(tmp_path):
    args = [x for x in BASE if x != "--eval-ok"]
    with pytest.raises(SystemExit):
        run_g7.main(args + ["--out", str(tmp_path / "x.csv")])
