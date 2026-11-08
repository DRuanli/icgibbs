"""
v1d_choose_q.py — V2 lượt V1D-2 (V2/V1D_PLAN.md §2, ghi trước): chọn q* cho PLL phụ thuộc mẫu theo TabICL v2.
q* = q LỚN NHẤT trong {0,2; 0,3; 0,4} sao cho trung bình theo dataset (54, 36, 23; mỗi dataset trung bình 3 seed) của
[A1_inst(q) − A1 iid q 0,6 của V1] ≥ −0,10; không mức nào đạt → 0,2. A1 = accuracy của trung bình dự đoán 3 chuỗi.
In q* (ví dụ "0.3") hoặc chuỗi rỗng nếu chưa đủ 27 ô hiệu chỉnh + 9 ô tham chiếu.
Chạy:  python v1d_choose_q.py --calib_dir v1d_npz --ref_dir v1_npz [--verbose]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

QS, DATASETS, SEEDS, THRESH = (0.2, 0.3, 0.4), (54, 36, 23), (0, 1, 2), -0.10


def a1_acc(path):
    z = np.load(path)
    P = np.mean([np.asarray(z[f"P_{s}"], float) for s in ("warm", "base", "cold")], axis=0)
    return float(np.mean(P.argmax(1) == np.asarray(z["yte"])))


def choose(calib_dir, ref_dir, verbose=False):
    ref, cal = {}, {}
    for d in DATASETS:
        for s in SEEDS:
            p = os.path.join(ref_dir, f"v1_tabicl_v2_{d}_pll_0.6_s{s}.npz")
            if os.path.exists(p):
                ref[(d, s)] = a1_acc(p)
            for q in QS:
                p = os.path.join(calib_dir, f"v1_tabicl_v2_{d}_pll_inst_{q}_s{s}.npz")
                if os.path.exists(p):
                    cal[(q, d, s)] = a1_acc(p)
    if len(ref) < 9 or len(cal) < 27:
        if verbose:
            print(f"chưa đủ: tham chiếu {len(ref)}/9, hiệu chỉnh {len(cal)}/27", file=sys.stderr)
        return None, {}
    gap = {q: float(np.mean([np.mean([cal[(q, d, s)] for s in SEEDS]) - np.mean([ref[(d, s)] for s in SEEDS])
                             for d in DATASETS])) for q in QS}
    ok = [q for q in QS if gap[q] >= THRESH]
    return (max(ok) if ok else 0.2), gap


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib_dir", required=True)
    ap.add_argument("--ref_dir", required=True)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    q, gap = choose(a.calib_dir, a.ref_dir, a.verbose)
    if a.verbose and gap:
        print({k: round(v, 4) for k, v in gap.items()}, file=sys.stderr)
    print("" if q is None else str(q))
    return 0


if __name__ == "__main__":
    sys.exit(main())
