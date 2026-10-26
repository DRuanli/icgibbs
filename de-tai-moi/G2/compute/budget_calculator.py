"""
budget_calculator.py — Tính ngân sách GPU cho giao thức của đề tài A từ số đo thật
==============================================================================

Đầu vào: CSV do kaggle_g2_benchmark.py tạo ra. Thời gian mỗi forward được nội suy song tuyến tính trên
(log n_train, log n_features) từ số đo, với cùng backend, cùng n_estimators, số lớp gần nhất.

Số forward cho một lần chạy (một dataset × một loại nhãn yếu × một mức × một seed):
    forwards_gibbs = chains × (burn_in + samples) × K          (mỗi vòng quét Gibbs = K forward theo khối)
    forwards_final = chains × samples                          (dự đoán test với mỗi mẫu nhãn ẩn đã lấy)

Chạy:
    python budget_calculator.py g2_benchmark.csv --backend tabpfn_v2 --gibbs_n_estimators 1 --final_n_estimators default
    python budget_calculator.py g2_benchmark.csv --scenario pilot
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

FULL = dict(
    datasets=[(1000, 10)] * 4 + [(2000, 20)] * 4 + [(5000, 30)] * 4,   # (n_train, n_features), 12 dataset
    n_classes=2, weak_types=5, levels=3, seeds=5,
    chains=2, burn_in=20, samples=30, k_blocks=10,
    baseline_gpu_hours=75.0,       # baseline học sâu (ước tính; thay bằng số đo ở G6)
    safety_factor=2.0,
    quota_hours_per_week=30.0,     # quota GPU Kaggle mỗi tuần (kiểm tra trang Kaggle hiện hành)
)
PILOT = dict(FULL, datasets=[(1000, 10)] * 4 + [(2000, 20)] * 3 + [(5000, 30)] * 3,  # 10 dataset
             weak_types=2, levels=2, seeds=3, baseline_gpu_hours=15.0)
SCENARIOS = {"full": FULL, "pilot": PILOT}


def interp(df, n, m, c, col):
    sub = df[df.n_classes == min(df.n_classes.unique(), key=lambda v: abs(v - c))]
    ms = sorted(sub.n_features.unique())
    ns = sorted(sub.n_train.unique())
    grid = sub.pivot_table(index="n_features", columns="n_train", values=col).loc[ms, ns].to_numpy()
    lg = np.log(grid)

    def idx(vals, x):
        lv = np.log(vals)
        i = int(np.clip(np.searchsorted(lv, math.log(x)) - 1, 0, len(vals) - 2))
        t = (math.log(x) - lv[i]) / (lv[i + 1] - lv[i])          # t ngoài [0,1] = ngoại suy tuyến tính log-log
        return i, t

    if len(ms) == 1:
        j, u = idx(ns, n)
        return float(math.exp(lg[0, j] * (1 - u) + lg[0, j + 1] * u))
    i, t = idx(ms, m)
    j, u = idx(ns, n)
    v = (lg[i, j] * (1 - t) * (1 - u) + lg[i + 1, j] * t * (1 - u)
         + lg[i, j + 1] * (1 - t) * u + lg[i + 1, j + 1] * t * u)
    return float(math.exp(v))


def budget(df_all, backend, gibbs_est, final_est, p, verbose=True):
    sel = lambda e: df_all[(df_all.status == "ok") & (df_all.backend == backend)
                           & (df_all.n_estimators.astype(str) == e)]
    dg, dfin = sel(gibbs_est), sel(final_est)
    if dg.empty or dfin.empty:
        raise SystemExit("CSV không có dòng ok cho cấu hình này")
    runs_per_dataset = p["weak_types"] * p["levels"] * p["seeds"]
    total_s, rows = 0.0, []
    for n, m in p["datasets"]:
        tb = interp(dg, n, m, p["n_classes"], "block_median_s")
        tf = interp(dfin, n, m, p["n_classes"], "final_median_s")
        per_run = p["chains"] * (p["burn_in"] + p["samples"]) * p["k_blocks"] * tb + p["chains"] * p["samples"] * tf
        total_s += per_run * runs_per_dataset
        rows.append((n, m, tb, tf, per_run / 60))
    method_h = total_s / 3600
    total_h = (method_h + p["baseline_gpu_hours"]) * p["safety_factor"]
    weeks = total_h / p["quota_hours_per_week"]
    if verbose:
        print(f"\n[{backend}] Gibbs n_estimators={gibbs_est}, dự đoán cuối n_estimators={final_est}")
        print(f"{'n':>6} {'m':>4} {'s/khối':>8} {'s/final':>8} {'phút/lần chạy':>14}")
        for r in sorted(set(rows)):
            print(f"{r[0]:6d} {r[1]:4d} {r[2]:8.3f} {r[3]:8.3f} {r[4]:14.2f}")
        print(f"Số lần chạy: {len(p['datasets']) * runs_per_dataset} | GPU-giờ phương pháp: {method_h:.1f} | "
              f"tổng (baseline {p['baseline_gpu_hours']}h, ×{p['safety_factor']}): {total_h:.1f} giờ | "
              f"≈ {weeks:.1f} tuần với {p['quota_hours_per_week']} giờ/tuần")
    return method_h, total_h, weeks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--scenario", default="full", choices=list(SCENARIOS))
    ap.add_argument("--backend", default=None)
    ap.add_argument("--gibbs_n_estimators", default="1")
    ap.add_argument("--final_n_estimators", default="default")
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    p = SCENARIOS[args.scenario]
    print(f"Kịch bản: {args.scenario} | thiết bị: {df.device.iloc[0]}")
    backends = [args.backend] if args.backend else sorted(df.backend.unique())
    for b in backends:
        for g, f in [("1", "1"), ("1", "default"), ("default", "default")]:
            if args.backend and (g, f) != (args.gibbs_n_estimators, args.final_n_estimators):
                continue
            budget(df, b, g, f, p)


if __name__ == "__main__":
    main()
