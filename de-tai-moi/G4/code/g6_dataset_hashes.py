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

from icgibbs.g7.data import encode_tfm, load_openml, table_sha256
from icgibbs.g7.registry import DEV_BINARY, DEV_MULTICLASS, EVAL_BINARY, EVAL_MULTICLASS
from icgibbs.g7.splits import split_sha256, stratified_split

FIELDS = ["split_set", "task", "did", "name", "version_expected", "version", "md5_openml", "n_raw",
          "n_missing_target", "n_duplicates_removed", "n", "d", "n_classes", "min_class_rate", "sha256_table",
          "sha256_split_s0", "sha256_split_s1", "sha256_split_s2", "status"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset_hashes.csv")
    a = ap.parse_args()
    todo = [("eval", "binary", d, n, v) for d, (n, v) in EVAL_BINARY.items()] + \
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
            except Exception as e:  # ghi lỗi, làm tiếp dataset khác
                row["status"] = f"error: {e!r}"[:200]
            w.writerow(row)
            fh.flush()
            print({k: row.get(k) for k in ("split_set", "did", "name", "n", "n_duplicates_removed", "status")}, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
