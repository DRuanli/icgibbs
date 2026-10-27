"""
g6_reproduce_pll_uci.py — G6, việc T3: tái hiện số công bố của PRODEN, RC, CC trên các bộ UCI dạng bảng của chính bài
gốc, bằng bản cài của dự án (icgibbs.baselines_ext.PLLNet). Không dùng dataset nào của tập phát triển hay tập đánh giá.
==============================================================================================================

PRODEN (Lv et al., ICML 2020), Bảng 1 — nhãn ứng viên "binomial", q ∈ {0,1; 0,7}:
  • Sinh nhãn: hàm `partialize` CHÍNH THỨC của kho PRODEN (nạp lúc chạy từ PRODEN_REPO, commit 637b531; kho không có
    tệp giấy phép nên không chép vào kho này): mỗi lớp vào tập ứng viên với xác suất q; tập chỉ có nhãn đúng thì thêm
    một lớp ngẫu nhiên cho tới khi có ≥ 2.
  • Mô hình tuyến tính, chuẩn hoá z-score, SGD momentum 0,9, 500 epoch; lr, weight decay, cỡ batch theo
    `parameter_settings.txt` của kho. Báo cáo: trung bình accuracy test của 10 epoch cuối (như bài).
  • Chia: bài ghi "five-fold cross-validation" nhưng bảng số mẫu (vd. Yeast 1335/149) là chia 90/10. Dùng 5 lần chia
    ngẫu nhiên 90/10 (seed 0–4), khớp số mẫu trong bảng của bài.
RC, CC (Feng et al., NeurIPS 2020), Bảng 3 — mô hình tuyến tính, tập ứng viên "uniform":
  • Sinh nhãn: tập ứng viên lấy đều trong mọi tập con chứa nhãn đúng, trừ tập toàn bộ lớp (2^(k−1) − 1 tập).
  • Adam, full batch, 2000 epoch (phụ lục của bài cho UCI); lr và weight decay chọn trong {1e-6, …, 1e-1} theo accuracy
    trên 10% train làm validation (như bài; validation dùng nhãn đúng — quy trình của bài, KHÔNG phải của dự án). Chọn
    một lần trên lần chia đầu rồi dùng cho mọi lần chia. 10 lần chia ngẫu nhiên 90/10. Báo cáo trung bình 10 epoch cuối.
Tiêu chí (G6_PLAN §3, ghi trước): đạt nếu |tái hiện − công bố| ≤ max(1 điểm %, độ lệch chuẩn công bố).

Chạy trên Kaggle (CPU, Internet: On):  PRODEN_REPO=/kaggle/working/PRODEN python g6_reproduce_pll_uci.py --out ...
Chạy thử không cần Internet:            python g6_reproduce_pll_uci.py --selftest --out /tmp/x.csv
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# OpenML: id, tên, (n, d, số lớp) để kiểm; không bộ nào thuộc tập phát triển hay tập đánh giá (registry.py)
DATASETS = {"yeast": (181, (1484, 8, 10)), "texture": (40499, (5500, 40, 11)), "dermatology": (35, (366, 34, 6)),
            "synthetic_control": (377, (600, 60, 6)), "har": (1478, (10299, 561, 6))}
# PRODEN, parameter_settings.txt của kho (mô hình tuyến tính): lr, weight decay, cỡ batch
PRODEN_HP = {"yeast": (1e-1, 1e-3, 100), "texture": (1e-1, 1e-5, 100), "dermatology": (1e-2, 1e-3, 100),
             "synthetic_control": (1e-1, 1e-3, 100)}
# Số công bố (accuracy %, trung bình ± độ lệch chuẩn)
TARGETS = {
    ("proden", "binomial_0.1"): {"yeast": (59.05, 4.66), "texture": (99.64, 0.16), "dermatology": (96.16, 3.26),
                                 "synthetic_control": (98.33, 1.32)},
    ("proden", "binomial_0.7"): {"yeast": (55.15, 3.87), "texture": (99.33, 0.32), "dermatology": (95.34, 3.15),
                                 "synthetic_control": (95.00, 4.75)},
    ("rc", "uniform"): {"texture": (99.24, 0.14), "yeast": (59.89, 1.27), "dermatology": (99.41, 1.00),
                        "har": (98.03, 0.09)},
    ("cc", "uniform"): {"texture": (98.02, 2.91), "yeast": (59.97, 1.57), "dermatology": (99.73, 0.85),
                        "har": (98.10, 0.18)},
}
GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]


def load(name, selftest=False):
    if selftest:
        from sklearn.datasets import make_classification
        n, d, k = (400, 8, 5)
        X, y = make_classification(n, d, n_informative=6, n_classes=k, n_clusters_per_class=1, random_state=1)
        return X.astype(float), y
    from sklearn.datasets import fetch_openml
    import pandas as pd
    did, (n, d, k) = DATASETS[name]
    b = fetch_openml(data_id=did, as_frame=True, parser="auto")
    X = b.data.copy()
    for c in X.columns:                                   # feature phân loại → mã số (Dermatology có vài cột như vậy)
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes.replace(-1, np.nan)
    y = pd.Series(b.target).astype("category").cat.codes.to_numpy()
    got = (len(y), X.shape[1], len(np.unique(y)))
    if got != (n, d, k):
        raise ValueError(f"{name} (OpenML {did}): kích thước {got} khác kỳ vọng {(n, d, k)}")
    return X.to_numpy(float), y


def split90(n, seed):
    rng = np.random.default_rng(seed)
    p = rng.permutation(n)
    k = int(round(0.9 * n))
    return p[:k], p[k:]


def impute(Xtr, Xte):
    mu = np.nan_to_num(np.nanmean(Xtr, 0))
    return np.where(np.isnan(Xtr), mu, Xtr), np.where(np.isnan(Xte), mu, Xte)


def proden_candidates(y, k, q, seed):
    """Hàm `partialize` chính thức (binomial). Không có kho thì dùng bản viết lại cùng quy tắc (chỉ cho --selftest)."""
    from icgibbs.baselines_ext import load_official
    np.random.seed(seed)
    mod = load_official("PRODEN_REPO", os.path.join("utils", "utils_algo.py"), "proden_utils_algo")
    if mod is not None:
        import torch
        Y = mod.binarize_class(np.asarray(y))
        if Y.shape[1] < k:
            Y = torch.cat([Y, torch.zeros(len(y), k - Y.shape[1])], 1)
        newY, _ = mod.partialize(Y, torch.as_tensor(y), "binomial", q)
        return newY.numpy() > 0, "official"
    S = np.eye(k, dtype=bool)[y]
    for i in range(len(y)):
        S[i, np.random.binomial(1, q, k) == 1] = True
        while S[i].sum() == 1:
            S[i, np.random.randint(0, k)] = True
    return S, "reimplemented"


def uniform_candidates(y, k, seed):
    """Đều trên 2^(k−1) − 1 tập con chứa nhãn đúng, trừ tập toàn bộ lớp (Feng et al. 2020, Eq. 5)."""
    rng = np.random.default_rng(seed)
    S = np.zeros((len(y), k), bool)
    for i, t in enumerate(y):
        while True:
            row = rng.random(k) < 0.5
            row[t] = True
            if not row.all():
                break
        S[i] = row
    return S


def train_eval(method, Xtr, S, Xte, yte, lr, wd, bs, epochs, opt, seed):
    from icgibbs.baselines_ext import PLLNet
    accs = []

    def hook(m, ep):
        if ep >= epochs - 10:
            accs.append(float(np.mean(m.predict_proba(Xte).argmax(1) == yte)))
    PLLNet(method, lr=lr, weight_decay=wd, hidden=(), epochs=epochs, batch_size=bs, seed=seed,
           optimizer=opt).fit(Xtr, S, on_epoch=hook)
    return 100 * float(np.mean(accs)), 100 * accs[-1]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true", help="dữ liệu tổng hợp, ít epoch — chỉ kiểm đường chạy")
    ap.add_argument("--only", default="proden,rc,cc")
    a = ap.parse_args(argv)
    only = set(a.only.split(","))
    E_PRODEN, E_RC, n_proden, n_rc = (500, 2000, 5, 10) if not a.selftest else (12, 15, 2, 2)
    fields = ["method", "candidates", "dataset", "trial", "acc_last10", "acc_final", "lr", "wd", "batch", "epochs",
              "avg_candidates", "source", "seconds"]
    fh = open(a.out, "w", newline="")
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    names = list(DATASETS) if not a.selftest else ["yeast"]
    for name in names:
        try:
            X, y = load(name, a.selftest)
        except Exception as e:
            print(f"[lỗi tải] {name}: {e!r}", flush=True)
            continue
        k = int(y.max()) + 1
        print(f"\n=== {name}: n={len(y)}, d={X.shape[1]}, k={k}", flush=True)
        # PRODEN, binomial
        if "proden" in only and (a.selftest or name in PRODEN_HP):
            lr, wd, bs = PRODEN_HP.get(name, (1e-1, 1e-3, 100))
            for q in (0.1, 0.7):
                for t in range(n_proden):
                    tr, te = split90(len(y), t)
                    Xtr, Xte = impute(X[tr], X[te])
                    S, src = proden_candidates(y[tr], k, q, seed=t)
                    t0 = time.time()
                    acc, fin = train_eval("proden", Xtr, S, Xte, y[te], lr, wd, bs, E_PRODEN, "sgd", t)
                    w.writerow(dict(method="proden", candidates=f"binomial_{q}", dataset=name, trial=t, acc_last10=round(acc, 3),
                                    acc_final=round(fin, 3), lr=lr, wd=wd, batch=bs, epochs=E_PRODEN,
                                    avg_candidates=round(S.sum(1).mean(), 3), source=src, seconds=round(time.time() - t0, 1)))
                    fh.flush()
                    print(f"  proden q={q} t{t}: {acc:.2f}", flush=True)
        # RC, CC, uniform
        for m in ("rc", "cc"):
            if m not in only or not (a.selftest or name in TARGETS[(m, "uniform")]):
                continue
            tr, te = split90(len(y), 0)
            S0 = uniform_candidates(y[tr], k, seed=0)
            ntr = len(tr)
            p = np.random.default_rng(99).permutation(ntr)
            vi, ti = p[: int(round(0.1 * ntr))], p[int(round(0.1 * ntr)):]
            Xa, Xv = impute(X[tr][ti], X[tr][vi])
            best = (-1, None)
            for lr, wd in itertools.product(GRID, GRID):
                acc, _ = train_eval(m, Xa, S0[ti], Xv, y[tr][vi], lr, wd, len(ti), E_RC, "adam", 0)
                if acc > best[0]:
                    best = (acc, (lr, wd))
            lr, wd = best[1]
            print(f"  {m}: chọn lr={lr:g}, wd={wd:g} (val {best[0]:.2f})", flush=True)
            for t in range(n_rc):
                tr, te = split90(len(y), t)
                Xtr, Xte = impute(X[tr], X[te])
                S = uniform_candidates(y[tr], k, seed=t)
                t0 = time.time()
                acc, fin = train_eval(m, Xtr, S, Xte, y[te], lr, wd, len(tr), E_RC, "adam", t)
                w.writerow(dict(method=m, candidates="uniform", dataset=name, trial=t, acc_last10=round(acc, 3),
                                acc_final=round(fin, 3), lr=lr, wd=wd, batch=len(tr), epochs=E_RC,
                                avg_candidates=round(S.sum(1).mean(), 3), source="project", seconds=round(time.time() - t0, 1)))
                fh.flush()
                print(f"  {m} t{t}: {acc:.2f}", flush=True)
    fh.close()
    summarize(a.out)
    return 0


def summarize(path):
    import pandas as pd
    d = pd.read_csv(path)
    if d.empty:
        return
    print("\n=== So với số công bố (tiêu chí: |chênh| ≤ max(1 điểm %, độ lệch chuẩn công bố)) ===")
    for (m, c, ds), g in d.groupby(["method", "candidates", "dataset"]):
        pub = TARGETS.get((m, c), {}).get(ds)
        mu, sd = g.acc_last10.mean(), g.acc_last10.std()
        if pub is None:
            print(f"  {m:6s} {c:13s} {ds:18s} {mu:6.2f} ± {sd:4.2f}  (không có số công bố)")
            continue
        diff = mu - pub[0]
        ok = abs(diff) <= max(1.0, pub[1])
        print(f"  {m:6s} {c:13s} {ds:18s} {mu:6.2f} ± {sd:4.2f} | công bố {pub[0]:.2f} ± {pub[1]:.2f} | chênh {diff:+.2f}"
              f" → {'ĐẠT' if ok else 'LỆCH'}")


if __name__ == "__main__":
    sys.exit(main())
