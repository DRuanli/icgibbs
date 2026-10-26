"""
in_prior_check.py — E6: kiểm tra "ví dụ đồ chơi có đáp án dự đoán được" với một PFN THẬT (mini-TabPFN của khóa
học, huấn luyện trên prior SCM) và dữ liệu sinh từ CHÍNH prior đó.

Vì dữ liệu nằm trong prior và q ≈ PPD, Mệnh đề 3 dự đoán: trung bình theo prior, In-Context Gibbs (chuyển nạp:
tối ưu; quy nạp: gần tối ưu) có log loss thấp hơn mọi bộ dự đoán khác dùng cùng thông tin (naive, hard-EM, SEM).
Đây là kiểm tra hướng đi trên CPU trước khi tốn GPU; không phải kết quả thực nghiệm chính.

    OMP_NUM_THREADS=1 python in_prior_check.py --n_datasets 40 --out ../../results/rq1/e6_in_prior.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, "/home/claude/mini_tabpfn")
from icgibbs import LLP, PUSCAR, InContextGibbs, SklearnTFMConditional, make_bags  # noqa: E402
from icgibbs.baselines import hard_em_tfm, naive_tfm_llp, naive_tfm_pu  # noqa: E402


def sample_binary(rng, n, max_features=10):
    from prior import sample_dataset, sample_num_features
    while True:
        m = sample_num_features(rng, max_features)
        X, y, _ = sample_dataset(n, m, "clf", rng)
        cls, cnt = np.unique(y, return_counts=True)
        if len(cls) == 2 and cnt.min() >= 0.15 * n:
            return X.astype(np.float32), (y == cls[np.argmin(cnt)]).astype(int)


def metrics(y, p):
    from sklearn.metrics import log_loss, roc_auc_score
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return roc_auc_score(y, p), log_loss(y, p, labels=[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_datasets", type=int, default=40)
    ap.add_argument("--n_train", type=int, default=80)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--ckpt", default="/home/claude/runs/clf.pt")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "results", "rq1", "e6_in_prior.csv"))
    a = ap.parse_args()
    from inference import MiniTabPFNClassifier
    est1 = MiniTabPFNClassifier(a.ckpt, n_estimators=1, preprocess=("none",), temperature=1.0, device="cpu")
    est4 = MiniTabPFNClassifier(a.ckpt, n_estimators=4, preprocess=("none", "quantile"), temperature=1.0, device="cpu")
    cond = SklearnTFMConditional(lambda: est1, 2)
    cond_final = SklearnTFMConditional(lambda: est4, 2)
    rows = []
    rng = np.random.default_rng(2026)
    for d in range(a.n_datasets):
        X, y = sample_binary(rng, a.n_train + a.n_test)
        Xtr, ytr, Xte, yte = X[: a.n_train], y[: a.n_train], X[a.n_train:], y[a.n_train:]
        if len(np.unique(ytr)) < 2:
            continue
        res = {}
        t0 = time.time()
        res["oracle"] = cond_final.predict_proba(Xtr, ytr, Xte)[:, 1]
        # ---- PU-SCAR, c = 0.3 đã biết ----
        c = 0.3
        s = ((ytr == 1) & (rng.uniform(size=len(ytr)) < c)).astype(int)
        if s.sum() >= 1:
            _, res["pu/naive"] = naive_tfm_pu(cond_final, Xtr, s, Xte, c)
            p, _ = hard_em_tfm(cond, PUSCAR(s, c=c), Xtr, Xte, n_iter=10, n_blocks=10, cond_final=cond_final, seed=d)
            res["pu/hard_em"] = p[:, 1]
            g = InContextGibbs(cond, PUSCAR(s, c=c), n_blocks=10, n_sweeps=30, burn_in=10, seed=d).run(Xtr)
            res["pu/gibbs_block"] = g.predict_proba(Xtr, Xte, cond=cond_final, n_use=10)[:, 1]
            res["pu/sem"] = g.predict_proba(Xtr, Xte, cond=cond_final, last_only=True)[:, 1]
            g = InContextGibbs(cond, PUSCAR(s, c=c), scheme="single", n_sweeps=15, burn_in=5, seed=d).run(Xtr)
            res["pu/gibbs_single"] = g.predict_proba(Xtr, Xte, cond=cond_final, n_use=10)[:, 1]
        # ---- LLP, túi cỡ 8 ----
        bags, counts = make_bags(len(ytr), 8, rng, ytr)
        res["llp/naive"] = naive_tfm_llp(cond_final, Xtr, LLP(bags, counts, len(ytr)), Xte, rng)[:, 1]
        p, _ = hard_em_tfm(cond, LLP(bags, counts, len(ytr)), Xtr, Xte, n_iter=10, n_blocks=10, cond_final=cond_final, seed=d)
        res["llp/hard_em"] = p[:, 1]
        g = InContextGibbs(cond, LLP(bags, counts, len(ytr)), n_blocks=10, n_sweeps=30, burn_in=10, seed=d).run(Xtr)
        res["llp/gibbs_block"] = g.predict_proba(Xtr, Xte, cond=cond_final, n_use=10)[:, 1]
        res["llp/sem"] = g.predict_proba(Xtr, Xte, cond=cond_final, last_only=True)[:, 1]
        g = InContextGibbs(cond, LLP(bags, counts, len(ytr)), scheme="swap", n_sweeps=15, burn_in=5, seed=d).run(Xtr)
        res["llp/gibbs_swap"] = g.predict_proba(Xtr, Xte, cond=cond_final, n_use=10)[:, 1]
        for k, p in res.items():
            auc, ll = metrics(yte, p)
            rows.append(dict(dataset=d, method=k, auc=auc, logloss=ll, d=X.shape[1], pos=float(y.mean())))
        print(f"dataset {d}: {time.time() - t0:.1f}s  " + "  ".join(
            f"{k}={metrics(yte, p)[1]:.3f}" for k, p in res.items()), flush=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
