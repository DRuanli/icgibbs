"""
g6_dataset_hashes.py — G6, việc T5: mã băm và thống kê của mọi dataset (tập đánh giá và tập phát triển) sau đúng
cách chuẩn hoá mà bộ chạy G7 dùng (icgibbs/g7/data.py). KHÔNG chạy phương pháp nào; chỉ tải, chuẩn hoá, băm.

Ghi docs/dataset_hashes.csv: ID, tên, phiên bản OpenML, md5 của OpenML, n gốc, số hàng thiếu target, số hàng trùng
đã loại, n, d, số lớp, tỉ lệ lớp nhỏ nhất, SHA-256 của bảng (X_tfm, y), SHA-256 của 3 lần chia (seed 0, 1, 2).
Cần Internet tới OpenML. Chạy:  python g6_dataset_hashes.py --out dataset_hashes.csv
"""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from icgibbs.g7.data import encode_tfm, load_openml, subsample_table, table_sha256
from icgibbs.g7.registry import (DEV_BINARY, DEV_MULTICLASS, EVAL_BINARY, EVAL_MULTICLASS, S2_BINARY, S2_MULTICLASS,
                                 S2_RESERVE_MULTICLASS, S2_SUBSAMPLE)
from icgibbs.g7.splits import split_sha256, stratified_split

FIELDS = ["split_set", "task", "did", "name", "version_expected", "version", "md5_openml", "n_raw",
          "n_missing_target", "n_duplicates_removed", "n", "d", "n_classes", "min_class_rate", "sha256_table",
          "sha256_split_s0", "sha256_split_s1", "sha256_split_s2", "pass_criteria", "status"]


def s2_todo():
    """S2 (S2_PLAN §3.4): tập chọn và mọi dự phòng, để bổ sung theo quy tắc mà không cần tải lại."""
    return ([("s2", "binary", d, n, v) for d, (n, v) in S2_BINARY.items()] +
            [("s2", "multiclass", d, n, v) for d, (n, v) in S2_MULTICLASS.items()] +
            [("s2_reserve", "multiclass", d, n, v) for d, n, v in S2_RESERVE_MULTICLASS])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset_hashes.csv")
    ap.add_argument("--split", choices=["nc1", "s2", "s3"], default="nc1")
    a = ap.parse_args()
    from icgibbs.g7.registry import S3_BINARY, S3_MULTICLASS, S3_SUBSAMPLE
    s3 = [("s3", "binary", d, n, v) for d, (n, v) in S3_BINARY.items()] + \
         [("s3", "multiclass", d, n, v) for d, (n, v) in S3_MULTICLASS.items()]
    todo = s3 if a.split == "s3" else s2_todo() if a.split == "s2" else [("eval", "binary", d, n, v) for d, (n, v) in EVAL_BINARY.items()] + \
           [("eval", "multiclass", d, n, v) for d, (n, v) in EVAL_MULTICLASS.items()] + \
           [("dev", "binary", d, n, None) for d, n in DEV_BINARY.items()] + \
           [("dev", "multiclass", d, n, None) for d, n in DEV_MULTICLASS.items()]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for sset, task, did, name, ver in todo:
            row = dict(split_set=sset, task=task, did=did, name=name, version_expected=ver)
            try:
                t = load_openml(did, binary=(task == "binary"))
                if sset == "s2" and did in S2_SUBSAMPLE:        # như bộ chạy (S2_PLAN §3.4)
                    t = subsample_table(t, S2_SUBSAMPLE[did], seed=0)
                if sset == "s3" and did in S3_SUBSAMPLE:        # như bộ chạy (STUDY3_PLAN §2)
                    t = subsample_table(t, S3_SUBSAMPLE[did], seed=0)
                X = encode_tfm(t.X)
                rates = np.bincount(t.y) / len(t.y)
                row.update(version=t.version, md5_openml=t.md5_openml, n_raw=t.n_raw,
                           n_missing_target=t.n_missing_target, n_duplicates_removed=t.n_duplicates_removed,
                           n=len(t.y), d=X.shape[1], n_classes=t.n_classes, min_class_rate=round(float(rates.min()), 4),
                           sha256_table=table_sha256(X, t.y), status="ok")
                for s in (0, 1, 2):
                    row[f"sha256_split_s{s}"] = split_sha256(*stratified_split(t.y, s))
                if ver is not None and t.version != ver:
                    row["status"] = f"ok; phiên bản {t.version} khác DATASET_CARDS ({ver})"
                # tiêu chí sau loại trùng (NC1 §1.1; S2_PLAN §3.4): n ≥ 500; nhị phân: lớp nhỏ nhất ≥ 10%
                ok = len(t.y) >= 500 and (task != "binary" or rates.min() >= 0.10) and \
                    (task != "multiclass" or 3 <= t.n_classes <= 10 or sset == "s2_reserve")
                if sset == "s3":                                # bậc 3 của S2_PLAN §3.2 (STUDY3_PLAN §2)
                    ok = len(t.y) >= 500 and X.shape[1] <= 2000 and (task != "binary" or rates.min() >= 0.05) and \
                        (task != "multiclass" or 3 <= t.n_classes <= 20)
                row["pass_criteria"] = bool(ok)
            except Exception as e:  # ghi lỗi, làm tiếp dataset khác
                row["status"] = f"error: {e!r}"[:200]
            w.writerow(row)
            fh.flush()
            print({k: row.get(k) for k in ("split_set", "did", "name", "n", "n_duplicates_removed", "status")}, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
