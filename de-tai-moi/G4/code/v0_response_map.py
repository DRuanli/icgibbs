"""
v0_response_map.py — V2 giai đoạn 0 (THĂM DÒ, dataset của NC1; V2/PHASE0_PREDICTIONS.md §3): ánh xạ phản ứng Φ(r)
của một vòng quét Gibbs theo khối, mở rộng S2X (s2x_response_map.py, giữ nguyên không sửa):

  --setting pll --plant uniform   như S2X (nhãn sai chọn đều trong tập ứng viên); dùng để thêm ρ cao (0,85; 0,90; 0,95)
  --setting pll --plant struct    lỗi CÓ CẤU TRÚC (W1): số nhãn sai = round((1 − ρ)·n_tự_do); điểm sai chọn không hoàn lại
                                  với xác suất ∝ 1 − P0_i(y_i); nhãn sai chọn ∝ P0_i trên các ứng viên sai, với P0 = khởi
                                  động ấm của chuỗi (TFM ngây thơ tính chéo 5 fold, giới hạn trong tập ứng viên)
  --setting llp --plant uniform   LLP túi cỡ --bag: trong mỗi túi, m = min(#dương, #âm), s ~ Binomial(m, 1 − ρ), đổi nhãn
                                  s cặp (dương, âm) chọn ngẫu nhiên ⇒ số đếm giữ nguyên. Φ = xác suất một vòng quét
                                  bag-block (K = 10 nhóm túi, ngữ cảnh = các nhóm còn lại, tối đa 2.000 điểm) gán đúng nhãn,
                                  tính CHÍNH XÁC từ xác suất biên của phân phối Bernoulli có điều kiện số đếm (DP).

Bản "Jacobi": mọi khối dùng cùng cách gán z. Nhãn thật chỉ dùng để gieo z và để đo — phân tích cơ chế.
Chạy:  python v0_response_map.py --setting pll --plant struct --backends tabpfn_v2 --out v0x_struct_tabpfn.csv
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
from s2x_response_map import planted_labels, response  # noqa: E402

FIELDS = ["backend", "dataset_id", "dataset", "setting", "level", "plant", "seed", "n_classes", "n_train", "rho",
          "r_actual", "phi", "phi_sd_blocks", "seconds", "status"]
DEFAULTS = {"pll": "12,1501,40966,28,188,40982", "llp": "50,6332"}


def planted_struct(y, S, rho, P0, rng):
    """Lỗi có cấu trúc: điểm khó (P0 gán ít xác suất cho nhãn thật) bị sai trước; nhãn sai theo P0."""
    S = np.asarray(S, bool)
    z = np.asarray(y).copy()
    if rho is None:
        return planted_labels(y, S, None, rng)
    P = np.clip(np.asarray(P0, float), 1e-9, None) * S
    P = P / P.sum(1, keepdims=True)
    free = np.flatnonzero(S.sum(1) >= 2)
    m = int(round((1.0 - rho) * len(free)))
    if m <= 0:
        return z
    w = 1.0 - P[free, np.asarray(y)[free]] + 1e-6
    wrong = rng.choice(free, size=m, replace=False, p=w / w.sum())
    for i in wrong:
        cand = np.flatnonzero(S[i])
        cand = cand[cand != y[i]]
        p = P[i, cand] + 1e-9
        z[i] = rng.choice(cand, p=p / p.sum())
    return z


def planted_llp(y, bags, rho, rng):
    """Đổi nhãn khác lớp trong cùng túi, giữ số đếm. rho = None: hoán vị ngẫu nhiên trong túi (khởi động lạnh)."""
    z = np.asarray(y).copy()
    for b in bags:
        b = np.asarray(b)
        if rho is None:
            z[b] = rng.permutation(z[b])
            continue
        pos, neg = b[z[b] == 1], b[z[b] == 0]
        m = min(len(pos), len(neg))
        s = rng.binomial(m, 1.0 - rho) if m else 0
        if s:
            p = rng.choice(pos, s, replace=False)
            q = rng.choice(neg, s, replace=False)
            z[p], z[q] = 0, 1
    return z


def count_marginals(P, k):
    """P(y_i = 1 | Σ y = k) cho Bernoulli độc lập p_i = P[i, 1], chính xác (O(m²) mỗi túi)."""
    from icgibbs.llp import _binary_logZ
    P = np.clip(np.asarray(P, float), 1e-12, None)
    P = P / P.sum(1, keepdims=True)
    m = len(P)
    if k == 0:
        return np.zeros(m)
    if k == m:
        return np.ones(m)
    logZ = _binary_logZ(np.log(P), k)[0, k]
    out = np.empty(m)
    for i in range(m):
        rest = np.delete(np.arange(m), i)
        lz = _binary_logZ(np.log(P[rest]), k - 1)[0, k - 1]
        out[i] = np.exp(np.log(P[i, 1]) + lz - logZ)
    return out


def response_llp(cond, X, z, y, bags, n_blocks=10, max_context=2000, rng=None):
    G = len(bags)
    groups = np.array_split(rng.permutation(G), min(n_blocks, G))
    n = len(y)
    phi = np.full(n, np.nan)
    per_block = []
    for grp in groups:
        sites = np.concatenate([np.asarray(bags[g]) for g in grp]).astype(int)
        mask = np.ones(n, bool)
        mask[sites] = False
        ctx = np.flatnonzero(mask)
        if len(ctx) > max_context:
            ctx = np.sort(rng.choice(ctx, max_context, replace=False))
        P = cond.predict_proba(X[ctx], z[ctx], X[sites])
        row = {s: r for r, s in enumerate(sites)}
        for g in grp:
            b = np.asarray(bags[g])
            pi = count_marginals(P[[row[s] for s in b]], int(z[b].sum()))
            phi[b] = np.where(y[b] == 1, pi, 1.0 - pi)
        per_block.append(np.nanmean(phi[sites]))
    return float(np.nanmean(phi)), float(np.std(per_block))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=["pll", "llp"], default="pll")
    ap.add_argument("--plant", choices=["uniform", "struct"], default="uniform")
    ap.add_argument("--backends", default="tabpfn_v2")
    ap.add_argument("--datasets", default="")
    ap.add_argument("--levels", default="", help="PLL: q (mặc định 0.6); LLP: cỡ túi (mặc định 8)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--rho", default="0.85,0.9,0.95", help="'u' = chọn đều trong tập ứng viên / hoán vị trong túi")
    ap.add_argument("--source", choices=["openml", "synthetic"], default="openml")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.setting == "llp" and a.plant != "uniform":
        ap.error("LLP chỉ có --plant uniform ở giai đoạn 0")
    from icgibbs.baselines import warm_start_cross
    from icgibbs.conditional import SklearnTFMConditional
    from run_g7 import make_estimator, synthetic_table
    done = set()
    if os.path.exists(a.out):
        for r in csv.DictReader(open(a.out)):
            if r["status"] == "ok":
                done.add((r["backend"], r["dataset_id"], r["level"], r["seed"], r["rho"]))
    fh = open(a.out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if os.path.getsize(a.out) == 0:
        w.writeheader()
    rhos = [None if x == "u" else float(x) for x in a.rho.split(",")]
    levels = [float(x) if a.setting == "pll" else int(x) for x in (a.levels or ("0.6" if a.setting == "pll" else "8")).split(",")]
    binary = a.setting == "llp"
    for did in [int(x) for x in (a.datasets or DEFAULTS[a.setting]).split(",")]:
        try:
            t = synthetic_table(did, binary, None if binary else (600, 8, 6)) if a.source == "synthetic" \
                else load_openml(did, binary=binary)
        except Exception as e:
            print(f"[lỗi tải] {did}: {e!r}", flush=True)
            continue
        X_all = encode_tfm(t.X)
        for seed in [int(s) for s in a.seeds.split(",")]:
            itr, _ = stratified_split(t.y, seed)
            Xtr, ytr = X_all[itr], t.y[itr]
            for lev in levels:
                wd = make_weak(a.setting, lev, ytr, Xtr, weak_rng(seed, did, a.setting), n_classes=t.n_classes)
                for be in a.backends.split(","):
                    cond = SklearnTFMConditional(lambda: make_estimator(be, 1), t.n_classes)
                    P0 = None
                    for rho in rhos:
                        key = (be, str(did), str(lev), str(seed), "u" if rho is None else str(rho))
                        if key in done:
                            continue
                        rng = np.random.default_rng([seed, did, int(float(lev) * 100), 7 if rho is None else int(rho * 100),
                                                     11 if a.plant == "struct" else 0])
                        t0 = time.time()
                        row = dict(backend=be, dataset_id=did, dataset=t.name, setting=a.setting, level=lev,
                                   plant=a.plant, seed=seed, n_classes=t.n_classes, n_train=len(itr),
                                   rho="u" if rho is None else rho)
                        try:
                            if a.setting == "llp":
                                z = planted_llp(ytr, wd.bags, rho, rng)
                                phi, sd = response_llp(cond, Xtr, z, ytr, wd.bags, rng=rng)
                            else:
                                S = np.asarray(wd.S, bool)
                                if a.plant == "struct":
                                    if P0 is None:          # khởi động ấm của chuỗi (seed như methods.warm_start)
                                        P0 = warm_start_cross(cond, Xtr, wd.naive_pll, n_folds=5, seed=seed)
                                    z = planted_struct(ytr, S, rho, P0, rng)
                                else:
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
