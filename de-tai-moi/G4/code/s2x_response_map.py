"""
s2x_response_map.py — S2_PLAN §5.1 (THĂM DÒ, chỉ trên dataset cũ của NC1 = tập phát triển của S2): đo "ánh xạ phản
ứng" Φ(r) của một vòng quét Gibbs theo khối cho PLL, với từng TFM.

Với một cách gán nhãn z trên train có tỉ lệ đúng r (gieo có kiểm soát: z_i = y_i với xác suất ρ, ngược lại chọn đều
một nhãn SAI trong tập ứng viên S_i; ρ = "u" nghĩa là chọn đều trong S_i), tính
    Φ(z) = trung bình_i  p_TFM(y_i | x_i, ngữ cảnh) · 1[y_i ∈ S_i] / Σ_{c ∈ S_i} p_TFM(c | x_i, ngữ cảnh),
tức xác suất một bước cập nhật khối (đúng như bước Gibbs: K = 10 khối trên mọi mẫu, ngữ cảnh = các khối còn lại, tối
đa 2.000 điểm, TFM đúng cấu hình của chuỗi) gán đúng nhãn cho mẫu i. Bản "Jacobi": mọi khối dùng cùng z.
Điểm bất động Φ(r) = r ổn định ở r thấp ⇔ chuỗi kẹt ở trạng thái gần ngẫu nhiên (kiểu sụp của NC1, G8_REPORT §5).
Nhãn thật chỉ dùng để gieo z và để đo — đây là phân tích cơ chế, không phải phương pháp.

Chạy:  python s2x_response_map.py --backends tabicl_v2 --out s2x_tabicl.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from icgibbs.g7.data import encode_tfm, load_openml  # noqa: E402
from icgibbs.g7.splits import stratified_split  # noqa: E402
from icgibbs.g7.weakgen import make_weak, weak_rng  # noqa: E402

# 4 dataset sụp ở NC1 (≥ 8 lớp) và 3 dataset khoẻ (≤ 7 lớp), cùng nhóm đa lớp của tập đánh giá NC1
DEFAULT_DATASETS = "12,1501,40966,28,188,40982,182"
FIELDS = ["backend", "dataset_id", "dataset", "q", "seed", "n_classes", "n_train", "avg_candidates", "rho", "r_actual",
          "phi", "phi_sd_blocks", "seconds", "status"]


def planted_labels(y, S, rho, rng):
    """z_i = y_i với xác suất rho, ngược lại chọn đều một nhãn sai trong S_i (nếu S_i chỉ có y_i thì z_i = y_i).
    rho = None: chọn đều trong S_i."""
    z = np.empty_like(y)
    for i in range(len(y)):
        cand = np.flatnonzero(S[i])
        if rho is None:
            z[i] = rng.choice(cand)
            continue
        wrong = cand[cand != y[i]]
        z[i] = y[i] if (len(wrong) == 0 or rng.uniform() < rho) else rng.choice(wrong)
    return z


def response(cond, X, z, y, S, n_blocks=10, max_context=2000, rng=None):
    n = len(y)
    blocks = np.array_split(rng.permutation(n), n_blocks)
    phi = np.empty(n)
    per_block = []
    for blk in blocks:
        mask = np.ones(n, bool)
        mask[blk] = False
        ctx = np.flatnonzero(mask)
        if len(ctx) > max_context:
            ctx = np.sort(rng.choice(ctx, max_context, replace=False))
        P = cond.predict_proba(X[ctx], z[ctx], X[blk]) * S[blk]
        P = P / P.sum(1, keepdims=True)
        phi[blk] = P[np.arange(len(blk)), y[blk]]
        per_block.append(phi[blk].mean())
    return float(phi.mean()), float(np.std(per_block))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="tabpfn_v2,tabicl_v2")
    ap.add_argument("--datasets", default=DEFAULT_DATASETS)
    ap.add_argument("--q", default="0.3,0.6")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--rho", default="u,0.1,0.2,0.3,0.45,0.6,0.8", help="'u' = chọn đều trong tập ứng viên")
    ap.add_argument("--source", choices=["openml", "synthetic"], default="openml")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from icgibbs.conditional import SklearnTFMConditional
    from run_g7 import make_estimator, synthetic_table
    done = set()
    if os.path.exists(a.out):
        for r in csv.DictReader(open(a.out)):
            if r["status"] == "ok":
                done.add((r["backend"], r["dataset_id"], r["q"], r["seed"], r["rho"]))
    fh = open(a.out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if os.path.getsize(a.out) == 0:
        w.writeheader()
    rhos = [None if x == "u" else float(x) for x in a.rho.split(",")]
    for did in [int(x) for x in a.datasets.split(",")]:
        try:
            t = synthetic_table(did, False, (600, 8, 6)) if a.source == "synthetic" else load_openml(did, binary=False)
        except Exception as e:
            print(f"[lỗi tải] {did}: {e!r}", flush=True)
            continue
        X_all = encode_tfm(t.X)
        for seed in [int(s) for s in a.seeds.split(",")]:
            itr, _ = stratified_split(t.y, seed)
            Xtr, ytr = X_all[itr], t.y[itr]
            for q in [float(x) for x in a.q.split(",")]:
                wd = make_weak("pll", q, ytr, Xtr, weak_rng(seed, did, "pll"), n_classes=t.n_classes)  # đúng nhãn yếu G7
                S = np.asarray(wd.S, bool)
                for be in a.backends.split(","):
                    cond = SklearnTFMConditional(lambda: make_estimator(be, 1), t.n_classes)
                    for rho in rhos:
                        key = (be, str(did), str(q), str(seed), "u" if rho is None else str(rho))
                        if key in done:
                            continue
                        rng = np.random.default_rng([seed, did, int(q * 100), 7 if rho is None else int(rho * 100)])
                        t0 = time.time()
                        row = dict(backend=be, dataset_id=did, dataset=t.name, q=q, seed=seed, n_classes=t.n_classes,
                                   n_train=len(itr), avg_candidates=round(float(S.sum(1).mean()), 3),
                                   rho="u" if rho is None else rho)
                        try:
                            z = planted_labels(ytr, S, rho, rng)
                            phi, sd = response(cond, Xtr, z, ytr, S, rng=rng)
                            row.update(r_actual=round(float((z == ytr).mean()), 4), phi=round(phi, 4),
                                       phi_sd_blocks=round(sd, 4), status="ok")
                        except Exception as e:
                            row.update(status=f"error: {e!r}"[:200])
                        row["seconds"] = round(time.time() - t0, 1)
                        w.writerow(row)
                        fh.flush()
                        print(row, flush=True)
    fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
