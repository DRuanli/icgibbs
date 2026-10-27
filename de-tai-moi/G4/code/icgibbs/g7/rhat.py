"""Chẩn đoán hội tụ cho G7 (giao thức Mục 5; G7_RUNNER_DESIGN R5): split-R̂ (Gelman et al. 2013; Vehtari et al. 2021,
bản không xếp hạng) trên trạng thái nhãn sau burn-in của 2 chuỗi độc lập.

Với nhãn rời rạc, mỗi mẫu train i cho một dãy chỉ báo I[y_i^(t) = k_i], với k_i là nhãn xuất hiện nhiều nhất ở mẫu i
trên cả hai chuỗi (nhị phân: tương đương I[y_i = 1] về R̂). Báo cáo trung vị, phân vị 90 và tỉ lệ mẫu có R̂ > 1,1, chỉ
trên các mẫu có phương sai dương; cùng R̂ của một đại lượng toàn cục: tỉ lệ mẫu đang mang nhãn k_i (cả nhị phân lẫn đa
lớp; không dùng tỉ lệ lớp 1 vì với LLP đại lượng này bị số đếm của túi cố định).

Hiệu chuẩn (G6 lượt 5): với 20 lần rút mỗi chuỗi và rút độc lập (trộn hoàn hảo), tỉ lệ mẫu có R̂ > 1,1 chỉ khoảng
0,4–4,5% và phân vị 90 khoảng 1,01–1,06 (mô phỏng Bernoulli p ∈ [0,02; 0,5]); giá trị cao hơn rõ là dấu hiệu trộn chưa
đủ trong 30 vòng quét."""
from __future__ import annotations

import numpy as np


def split_rhat(chains: np.ndarray) -> float:
    """chains: (m, n) — m chuỗi, n lần rút. Tách mỗi chuỗi làm đôi rồi tính R̂ cổ điển trên 2m nửa chuỗi."""
    x = np.asarray(chains, float)
    m, n = x.shape
    h = n // 2
    if h < 2:
        return float("nan")
    parts = np.concatenate([x[:, :h], x[:, n - h:]], 0)            # (2m, h)
    W = parts.var(1, ddof=1).mean()
    B = h * parts.mean(1).var(ddof=1)
    if W <= 0:
        return 1.0 if B <= 0 else float("inf")
    var_plus = (h - 1) / h * W + B / h
    return float(np.sqrt(var_plus / W))


def label_rhat(samples_a: list, samples_b: list) -> dict:
    A, B = np.asarray(samples_a), np.asarray(samples_b)              # (n_draws, n_train)
    n = min(len(A), len(B))
    A, B = A[-n:], B[-n:]
    allv = np.concatenate([A, B], 0)
    k = np.array([np.bincount(col).argmax() for col in allv.T])
    IA, IB = (A == k).astype(float), (B == k).astype(float)
    r = []
    for i in range(IA.shape[1]):
        c = np.stack([IA[:, i], IB[:, i]])
        if c.var() > 0:
            r.append(split_rhat(c))
    r = np.asarray([v for v in r if np.isfinite(v)])
    g = split_rhat(np.stack([IA.mean(1), IB.mean(1)]))
    return dict(rhat_median=float(np.median(r)) if len(r) else 1.0,
                rhat_p90=float(np.quantile(r, 0.9)) if len(r) else 1.0,
                frac_rhat_gt_1_1=float(np.mean(r > 1.1)) if len(r) else 0.0,
                n_varying=int(len(r)), rhat_global=g)
