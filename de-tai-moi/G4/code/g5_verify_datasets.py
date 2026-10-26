"""
g5_verify_datasets.py — xác minh danh sách dataset ĐÁNH GIÁ dự kiến của G5 (chỉ đọc metadata và dữ liệu,
KHÔNG chạy phương pháp nào, nên không vi phạm nguyên tắc "không chạm tập đánh giá trước khi đóng băng").

Với mỗi ID: tải từ OpenML, kiểm tên, n, d, số lớp, tỉ lệ lớp nhỏ nhất, tỉ lệ thiếu dữ liệu, số hàng trùng lặp
(L1.4), và đánh dấu dataset có thoả tiêu chí chọn đã viết trong G5/EXPERIMENT_PROTOCOL.md §1 hay không.
Chạy được trên CPU (Kaggle có Internet, hoặc máy thầy):
    python g5_verify_datasets.py --out g5_dataset_check.csv
"""
import argparse

import numpy as np
import pandas as pd

BINARY = {3: "kr-vs-kp", 29: "credit-approval", 44: "spambase", 50: "tic-tac-toe", 1049: "pc4",
          1489: "phoneme", 1494: "qsar-biodeg", 6332: "cylinder-bands", 23381: "dresses-sales", 40701: "churn"}
BINARY_RESERVE = {38: "sick", 1487: "ozone-level-8hr", 40983: "wilt", 4534: "PhishingWebsites"}
MULTI = {11: "balance-scale", 28: "optdigits", 46: "splice", 182: "satimage", 188: "eucalyptus",
         458: "analcatdata_authorship", 469: "analcatdata_dmft", 1475: "first-order-theorem-proving",
         1497: "wall-robot-navigation", 1501: "semeion", 40966: "MiceProtein", 40975: "car",
         40982: "steel-plates-fault", 12: "mfeat-factors"}
DEV = {31: "credit-g", 37: "diabetes", 1464: "blood-transfusion-service-center", 1462: "banknote-authentication",
       1480: "ilpd", 1510: "wdbc", 54: "vehicle", 36: "segment", 23: "cmc"}


def check(did, expected, group):
    from sklearn.datasets import fetch_openml
    b = fetch_openml(data_id=did, as_frame=True, parser="auto")
    X, y = b.data, b.target
    name = b.details.get("name", "")
    cnt = y.astype(str).value_counts()
    dup_rows = int(pd.concat([X, y.rename("__y__")], axis=1).duplicated().sum())
    dup_X_conflict = int(X.duplicated(keep=False).sum() - pd.concat([X, y.rename("__y__")], axis=1).duplicated(keep=False).sum())
    row = dict(group=group, id=did, expected=expected, name=name, name_ok=expected.lower() in name.lower(),
               n=len(X), d=X.shape[1], n_classes=len(cnt), minority=round(float(cnt.min() / cnt.sum()), 4),
               missing=round(float(X.isna().mean().mean()), 4), n_categorical=int((X.dtypes == "category").sum()),
               duplicate_rows=dup_rows, same_X_different_y=max(dup_X_conflict, 0),
               license=b.details.get("licence", ""), version=b.details.get("version", ""))
    if group == "binary" or group == "binary_reserve":
        row["meets_criteria"] = (row["n_classes"] == 2 and 500 <= row["n"] <= 10000 and row["d"] <= 500
                                 and row["minority"] >= 0.10)
    elif group == "multi":
        row["meets_criteria"] = (3 <= row["n_classes"] <= 10 and 500 <= row["n"] <= 10000 and row["d"] <= 500)
    else:
        row["meets_criteria"] = None
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="g5_dataset_check.csv")
    a = ap.parse_args()
    rows = []
    for group, pool in (("binary", BINARY), ("binary_reserve", BINARY_RESERVE), ("multi", MULTI), ("dev", DEV)):
        for did, exp in pool.items():
            try:
                rows.append(check(did, exp, group))
                r = rows[-1]
                print(f"{group:15s} {did:6d} {r['name'][:30]:30s} n={r['n']:6d} d={r['d']:4d} C={r['n_classes']:2d} "
                      f"min={r['minority']:.3f} miss={r['missing']:.3f} dup={r['duplicate_rows']} ok={r['meets_criteria']}",
                      flush=True)
            except Exception as e:
                rows.append(dict(group=group, id=did, expected=exp, error=repr(e)[:200]))
                print(f"[lỗi] {did} {exp}: {e!r}", flush=True)
            pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
