"""Metric của G7 (giao thức Mục 5): chính theo bối cảnh; phụ: log loss, Brier, accuracy tại ngưỡng 0,5 (nhị phân) hoặc
argmax (đa lớp), PR-AUC (nhị phân, dùng cho PU), ECE (top-label, 15 khoảng đều); log loss và Brier khớp tiên nghiệm
(PU-SCAR, GATE_REVIEW_G4 §3c Q1). Không dùng nhãn test để hiệu chỉnh: khớp tiên nghiệm dùng tỉ lệ lớp đã biết."""
from __future__ import annotations

import numpy as np

EPS = 1e-6


def prior_match(p, prior):
    """Dịch logit một hằng số để trung bình xác suất dự đoán bằng `prior` (như kaggle_g4_pilot.prior_match)."""
    from scipy.optimize import brentq
    from scipy.special import expit, logit
    z = logit(np.clip(np.asarray(p, float), EPS, 1 - EPS))
    f = lambda b: expit(z + b).mean() - prior
    if f(-40) > 0 or f(40) < 0:
        return np.asarray(p, float)
    return expit(z + brentq(f, -40, 40))


def ece(y, P, n_bins=15):
    """Expected calibration error, top-label, khoảng đều trên [0, 1]."""
    P = np.asarray(P, float)
    conf, pred = P.max(1), P.argmax(1)
    hit = (pred == np.asarray(y)).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(out)


def binary_metrics(y, p, prior=None):
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    out = dict(auc=float(roc_auc_score(y, p)), pr_auc=float(average_precision_score(y, p)),
               logloss=float(log_loss(y, p, labels=[0, 1])), brier=float(brier_score_loss(y, p)),
               acc=float(np.mean((p > 0.5) == y)), ece=ece(y, np.column_stack([1 - p, p])))
    if prior is not None:
        q = np.clip(prior_match(p, prior), EPS, 1 - EPS)
        out.update(logloss_pm=float(log_loss(y, q, labels=[0, 1])), brier_pm=float(brier_score_loss(y, q)))
    return out


def multiclass_metrics(y, P):
    from sklearn.metrics import log_loss, roc_auc_score
    y = np.asarray(y).astype(int)
    P = np.clip(np.asarray(P, float), EPS, None)
    P = P / P.sum(1, keepdims=True)
    C = P.shape[1]
    onehot = np.eye(C)[y]
    out = dict(acc=float(np.mean(P.argmax(1) == y)), logloss=float(log_loss(y, P, labels=list(range(C)))),
               brier=float(np.mean(((P - onehot) ** 2).sum(1))), ece=ece(y, P))
    try:
        out["auc_ovr"] = float(roc_auc_score(y, P, multi_class="ovr", labels=list(range(C))))
    except ValueError:                                  # lớp vắng mặt trong test
        out["auc_ovr"] = float("nan")
    return out


PRIMARY = {"llp": "auc", "pu_sar": "auc", "pu_scar": "logloss_pm", "ssl": "auc", "noisy": "auc", "pll": "acc"}
