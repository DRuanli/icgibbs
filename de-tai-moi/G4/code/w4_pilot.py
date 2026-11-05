"""
w4_pilot.py — V2 lượt V1C phần A (V2/PHASE1B_PLAN.md §1A, ghi trước): thời gian, bộ nhớ GPU đỉnh và tính tất định của
một lượt gọi TFM (fit trên ngữ cảnh + predict_proba trên truy vấn) như trong chuỗi Gibbs, cho các backend W4.

Mỗi ô (backend, n_ctx, C): 1 lượt khởi động, `--reps` lượt đo với cấu hình trong chuỗi (n_estimators = 1); hai lượt
gọi giống hệt nhau để đo chênh tuyệt đối lớn nhất; ở ô (2.000, 10) thêm 2 lượt với cấu hình dự đoán cuối.
Dữ liệu tổng hợp (make_classification, d = 20, truy vấn 200 điểm), không chạm dataset thật.
Chạy:  python w4_pilot.py --backends tabpfn_v2,tabicl_v2,tabpfn_v3,tabpfn_v3_5,tabdpt --out v1c_w4_timing.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FIELDS = ["backend", "n_ctx", "n_classes", "d", "n_query", "kind", "rep", "seconds", "max_abs_diff", "peak_gpu_mb",
          "status"]
GRID = "500x2,2000x2,500x10,2000x10,2000x26"
FINAL_EST = {"tabicl_v2": 2, "tabdpt": 8}          # như run_g7: TabPFN dùng mặc định của gói (None)
MAX_CLASSES = {"tabpfn_v2": 10, "tabpfn_v2_5": 10}


def _data(n_ctx, C, d, n_query, seed=0):
    from sklearn.datasets import make_classification
    X, y = make_classification(n_ctx + n_query, d, n_informative=min(d, 10), n_redundant=0, n_classes=C,
                               n_clusters_per_class=1, random_state=seed)
    return X[:n_ctx].astype(np.float32), y[:n_ctx], X[n_ctx:].astype(np.float32)


def _peak_reset():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _peak_mb():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return round(torch.cuda.max_memory_allocated() / 2**20, 1)
    except Exception:
        pass
    return ""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="tabpfn_v2,tabicl_v2,tabpfn_v3,tabpfn_v3_5,tabdpt")
    ap.add_argument("--grid", default=GRID)
    ap.add_argument("--d", type=int, default=20)
    ap.add_argument("--n_query", type=int, default=200)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from icgibbs.conditional import SklearnTFMConditional
    from run_g7 import make_estimator
    done = set()
    if os.path.exists(a.out):
        for r in csv.DictReader(open(a.out)):
            if r["status"] in ("ok", "unsupported"):
                done.add((r["backend"], r["n_ctx"], r["n_classes"]))
    fh = open(a.out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if os.path.getsize(a.out) == 0:
        w.writeheader()
    grid = [tuple(int(v) for v in g.split("x")) for g in a.grid.split(",")]
    for be in a.backends.split(","):
        for n_ctx, C in grid:
            if (be, str(n_ctx), str(C)) in done:
                continue
            base = dict(backend=be, n_ctx=n_ctx, n_classes=C, d=a.d, n_query=a.n_query)
            if C > MAX_CLASSES.get(be, 10**6):
                w.writerow(dict(base, kind="chain", rep=0, status="unsupported"))
                fh.flush()
                continue
            Xc, yc, Xq = _data(n_ctx, C, a.d, a.n_query)
            rows = []
            try:
                cond = SklearnTFMConditional(lambda: make_estimator(be, 1), C)
                t0 = time.time()
                P_prev = cond.predict_proba(Xc, yc, Xq)                     # khởi động (tải trọng số, biên dịch)
                rows.append(dict(base, kind="warmup", rep=0, seconds=round(time.time() - t0, 3)))
                for rep in range(1, a.reps + 1):
                    _peak_reset()
                    t0 = time.time()
                    P = cond.predict_proba(Xc, yc, Xq)
                    sec = time.time() - t0
                    rows.append(dict(base, kind="chain", rep=rep, seconds=round(sec, 4),
                                     max_abs_diff=float(np.max(np.abs(P - P_prev))), peak_gpu_mb=_peak_mb()))
                    P_prev = P
                if (n_ctx, C) == (2000, 10):
                    fe = FINAL_EST.get(be)
                    fin = SklearnTFMConditional(lambda: make_estimator(be, fe), C)
                    for rep in (1, 2):
                        _peak_reset()
                        t0 = time.time()
                        fin.predict_proba(Xc, yc, Xq)
                        rows.append(dict(base, kind="final", rep=rep, seconds=round(time.time() - t0, 4),
                                         peak_gpu_mb=_peak_mb()))
                for r in rows:
                    r["status"] = "ok"
            except Exception as e:
                rows.append(dict(base, kind="chain", rep=-1, status=f"error: {e!r}"[:200]))
            for r in rows:
                w.writerow(r)
            fh.flush()
            print(be, n_ctx, C, [(r["kind"], r.get("seconds")) for r in rows][:3], rows[-1]["status"], flush=True)
    fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
