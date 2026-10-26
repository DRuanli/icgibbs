"""
warm_start_check.py — kiểm tra nhanh trên CPU giả thuyết rút ra từ thí điểm v1: chuỗi khởi tạo từ ℓ (ngẫu nhiên)
kẹt ở mode xấu khi nhãn yếu ít thông tin (LLP túi lớn, PU c nhỏ, PU-SAR propensity mạnh); khởi động ấm sửa được.

TFM thay thế: mini-TabPFN của khóa học (không tải được TabPFN ở đây). Dữ liệu: breast_cancer của scikit-learn
(= wdbc, một dataset PHÁT TRIỂN), lớp dương = ác tính. Chỉ là kiểm tra hướng đi, không thay thí điểm GPU.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, "/home/claude/mini_tabpfn")
from icgibbs import LLP, PUSAR, PUSCAR, InContextGibbs, SklearnTFMConditional, make_bags  # noqa: E402
from icgibbs.baselines import DLLP, naive_tfm_pu, warm_start_llp, warm_start_pu  # noqa: E402


def main(out):
    from inference import MiniTabPFNClassifier
    ck = "/home/claude/runs/clf.pt"
    est1 = MiniTabPFNClassifier(ck, n_estimators=1, preprocess=("quantile",), max_features=10, temperature=1.0, device="cpu")
    est4 = MiniTabPFNClassifier(ck, n_estimators=4, preprocess=("none", "quantile"), max_features=10, temperature=1.0, device="cpu")
    cond = SklearnTFMConditional(lambda: est1, 2)
    fin = SklearnTFMConditional(lambda: est4, 2)
    X, y = load_breast_cancer(return_X_y=True)
    y = 1 - y                                          # dương = ác tính (lớp thiểu số), như wdbc trong thí điểm
    X = X.astype(np.float32)
    rows = []

    def rec(setting, seed, method, p, t0):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        rows.append(dict(setting=setting, seed=seed, method=method, auc=roc_auc_score(yte, p),
                         logloss=log_loss(yte, p, labels=[0, 1]), seconds=round(time.time() - t0, 1)))
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)

    for seed in range(3):
        rng = np.random.default_rng(seed)
        itr, ite = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=seed)
        Xtr, Xte, ytr, yte = X[itr], X[ite], y[itr], y[ite]
        t0 = time.time()
        rec("all", seed, "oracle", fin.predict_proba(Xtr, ytr, Xte)[:, 1], t0)
        kw = dict(n_blocks=10, n_sweeps=30, burn_in=10, seed=seed)
        # ---- LLP túi 32 và 8 ----
        for bag in (32, 8):
            bags, counts = make_bags(len(ytr), bag, rng, ytr)
            st = f"llp{bag}"
            t0 = time.time()
            P0 = warm_start_llp(Xtr, bags, counts, seed=seed)
            rec(st, seed, "dllp_linear(init)", DLLP(epochs=100, hidden=(), seed=seed).fit(Xtr, bags, counts).predict_proba(Xte)[:, 1], t0)
            for init_name, init in (("lik", "lik"), ("warm", P0)):
                t0 = time.time()
                g = InContextGibbs(cond, LLP(bags, counts, len(ytr)), init=init, **kw).run(Xtr)
                rec(st, seed, f"gibbs_{init_name}", g.predict_proba(Xtr, Xte, cond=fin, n_use=10)[:, 1], t0)
        # ---- PU-SCAR c = 0.1 ----
        c = 0.1
        s = ((ytr == 1) & (rng.uniform(size=len(ytr)) < c)).astype(int)
        t0 = time.time()
        rec("pu_scar0.1", seed, "naive", naive_tfm_pu(fin, Xtr, s, Xte, c)[1], t0)
        P0 = warm_start_pu(cond, Xtr, s, c, seed=seed)
        for init_name, init in (("lik", "lik"), ("warm", P0)):
            t0 = time.time()
            g = InContextGibbs(cond, PUSCAR(s, c=c), init=init, **kw).run(Xtr)
            rec("pu_scar0.1", seed, f"gibbs_{init_name}", g.predict_proba(Xtr, Xte, cond=fin, n_use=10)[:, 1], t0)
        # ---- PU-SAR, propensity đã biết, độ dốc 2 theo cột 0 ----
        z = (Xtr[:, 0] - Xtr[:, 0].mean()) / Xtr[:, 0].std()
        b0 = brentq(lambda b: expit(b + 2 * z[ytr == 1]).mean() - 0.3, -30, 30)
        e = expit(b0 + 2 * z)
        s = ((ytr == 1) & (rng.uniform(size=len(ytr)) < e)).astype(int)
        t0 = time.time()
        rec("pu_sar_known", seed, "naive", naive_tfm_pu(fin, Xtr, s, Xte, 0.3)[1], t0)
        P0 = warm_start_pu(cond, Xtr, s, e, seed=seed)
        for init_name, init in (("lik", "lik"), ("warm", P0)):
            t0 = time.time()
            g = InContextGibbs(cond, PUSAR(s, Z=z[:, None], eta=[b0, 2.0]), init=init, **kw).run(Xtr)
            rec("pu_sar_known", seed, f"gibbs_{init_name}", g.predict_proba(Xtr, Xte, cond=fin, n_use=10)[:, 1], t0)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "results", "pilot_v1_followup", "warm_start_check.csv"))
