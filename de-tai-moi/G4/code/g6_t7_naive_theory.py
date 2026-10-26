"""
g6_t7_naive_theory.py — G6, việc T7: kiểm chứng số cho Mệnh đề 5 và Quan sát 6 (PROOFS.md §7).

Mệnh đề 5 (bộ ngây thơ có nhất quán về xếp hạng không). Với cơ chế quan sát KHÔNG phụ thuộc mẫu dùng trong giao thức,
nhãn mà TFM ngây thơ học (w̃) thoả P(w̃ = k | x) = b + a·p(y = k | x), a > 0, nên thứ tự (nhị phân) và argmax (đa lớp)
của mục tiêu ngây thơ trùng với của posterior thật. Script kiểm:
  (1) luật affine bằng Monte Carlo, dùng ĐÚNG thủ tục sinh nhãn yếu và nhãn ngây thơ của các script thí điểm;
  (2) chặn hối tiếc AUC: AUC* − AUC(s) ≤ E|s − η̃| / (a·p(1 − p)), trên các bài toán rời rạc ngẫu nhiên (tính chính xác).

Quan sát 6 (thông tin). Thông tin Fisher (theo tham số θ của p(y|x)) của dữ liệu mà bộ ngây thơ dùng so với của quan sát
yếu đầy đủ, trên mô hình logistic một tham số. PU-SCAR và nhiễu phụ thuộc lớp: bằng nhau (nhãn ngây thơ CHÍNH LÀ quan
sát). PLL: ≤ (bất đẳng thức xử lý dữ liệu, PROOFS §7). LLP: so khả năng hợp lý từng điểm (composite) của bộ ngây thơ —
kể cả khi nó biết đúng kênh — với khả năng hợp lý của số đếm túi.

Chạy:  python g6_t7_naive_theory.py            (in bảng, khoảng 1–2 phút trên CPU)
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
from scipy.special import expit

from icgibbs.weak import LLP, make_bags


# ================================================================================================
# (1) Luật affine của nhãn ngây thơ
# ================================================================================================
def pll_affine_coeffs(C: int, q: float):
    """Ứng viên sinh như thí điểm: lớp thật luôn có, mỗi lớp sai vào tập độc lập với xác suất q; bộ ngây thơ chọn đều
    một ứng viên. Khi đó P(w̃ = k | y) = b + (a − b)·1[k = y] với a = E[1/(1 + B)], B ~ Bin(C − 1, q)."""
    a = 1.0 if q == 0 else (1 - (1 - q) ** C) / (C * q)
    b = (1 - a) / (C - 1)
    return a, b                                     # độ dốc a − b > 0 khi q < 1


def pll_affine_mc(C, q, n, rng):
    """Monte Carlo đúng như kaggle_g4_rq4_pilot.py: S = U < q, S[y] = True, pseudo = rng.choice(flatnonzero(S))."""
    y = rng.integers(C, size=n)
    S = rng.uniform(size=(n, C)) < q
    S[np.arange(n), y] = True
    w = np.array([rng.choice(np.flatnonzero(s)) for s in S])
    M = np.zeros((C, C))
    for k in range(C):
        M[k] = np.bincount(w[y == k], minlength=C) / max(1, (y == k).sum())
    return M


def llp_affine_exact(y, k):
    """Túi chia ngẫu nhiên cỡ k (make_bags) từ quần thể hữu hạn n điểm có N₊ dương; bộ ngây thơ gán nhãn bằng hoán vị
    ngẫu nhiên khớp số đếm (LLP.init_labels). Khi đó P(w̃_i = 1 | y_i) = (y_i + (k − 1)(N₊ − y_i)/(n − 1)) / k."""
    y = y[: len(y) - len(y) % k]                            # như llp_affine_mc: chỉ dùng phần chia hết cho k
    n, npos = len(y), int(y.sum())
    return {yy: (yy + (k - 1) * (npos - yy) / (n - 1)) / k for yy in (0, 1)}


def llp_affine_mc(y, k, reps, rng):
    n = len(y)
    hit = np.zeros(2)
    tot = np.zeros(2)
    for _ in range(reps):
        bags, counts = make_bags(n - n % k, k, rng, y[: n - n % k])
        w = LLP(bags, counts, n - n % k).init_labels(rng)
        yy = y[: n - n % k]
        for c in (0, 1):
            hit[c] += w[yy == c].sum()
            tot[c] += (yy == c).sum()
    return {c: hit[c] / tot[c] for c in (0, 1)}


# ================================================================================================
# (2) Chặn hối tiếc AUC
# ================================================================================================
def auc_exact(score, eta, px):
    """AUC chính xác (tính hoà = 1/2) khi x rời rạc với khối lượng px, P(y=1|x) = eta."""
    pos, neg = px * eta, px * (1 - eta)
    gt = (score[:, None] > score[None, :]).astype(float) + 0.5 * (score[:, None] == score[None, :])
    return float(pos @ gt @ neg) / (pos.sum() * neg.sum())


def auc_bound_check(rng, trials=2000, m=12):
    worst = -np.inf
    for _ in range(trials):
        px = rng.dirichlet(np.ones(m))
        eta = rng.uniform(size=m) ** rng.choice([0.5, 1, 3])
        a, b = rng.uniform(0.02, 1), rng.uniform(0, 1)
        b = min(b, 1 - a)                                   # η̃ = b + a·η ∈ [0, 1]
        eta_t = b + a * eta
        s = np.clip(eta_t + rng.normal(0, rng.choice([0.01, 0.05, 0.2]), m), 0, 1)
        if rng.uniform() < 0.2:
            s = np.round(s, 1)                              # có hoà
        p = float(px @ eta)
        regret = auc_exact(eta, eta, px) - auc_exact(s, eta, px)
        bound = float(px @ np.abs(s - eta_t)) / (a * p * (1 - p))
        worst = max(worst, regret - bound)
        assert abs(auc_exact(eta_t, eta, px) - auc_exact(eta, eta, px)) < 1e-12   # affine tăng giữ AUC
    return worst                                            # ≤ 0 nếu chặn luôn đúng


# ================================================================================================
# (3) Thông tin Fisher trên mô hình logistic một tham số
# ================================================================================================
THETA = 2.0


def _bin_model(xs):
    eta = expit(THETA * xs)
    return eta, xs * eta * (1 - eta)                       # η, ∂η/∂θ


def fisher_binary_channel(xs, a, b):
    """Quan sát một điểm có P(w=1|x) = b + a·η(x): I = E[(a η')² / (m(1 − m))]."""
    eta, d = _bin_model(xs)
    m = b + a * eta
    return float(np.mean((a * d) ** 2 / (m * (1 - m))))


def _poisson_binomial(eta):
    P = np.zeros(len(eta) + 1)
    P[0] = 1.0
    for e in eta:
        P[1:] = P[1:] * (1 - e) + P[:-1] * e
        P[0] *= 1 - e
    return P


def fisher_llp_bag(rng, k, n_bags):
    """Thông tin Fisher của MỘT túi (x_B, số đếm c) — khả năng hợp lý đầy đủ mà Gibbs nhắm tới."""
    tot = 0.0
    for _ in range(n_bags):
        xs = rng.normal(size=k)
        eta, d = _bin_model(xs)
        P = _poisson_binomial(eta)
        dP = np.zeros(k + 1)
        for i in range(k):
            Pm = _poisson_binomial(np.delete(eta, i))       # ∂P(c)/∂η_i = P₋ᵢ(c − 1) − P₋ᵢ(c)
            g = np.zeros(k + 1)
            g[1:] += Pm
            g[:-1] -= Pm
            dP += d[i] * g
        ok = P > 1e-300
        tot += float(np.sum(dP[ok] ** 2 / P[ok]))
    return tot / n_bags


def godambe_llp_naive(rng, k, n_bags, reps_per_bag=20):
    """Bộ ngây thơ: hợp lý từng điểm Π_i P(w̃_i | x_i) với kênh ĐÚNG m(x) = (η(x) + (k − 1)π)/k (π = 1/2 không phụ
    thuộc θ vì x đối xứng). Thông tin Godambe của một túi: H²/J, H = Σ_i E[s_i²], J = E[(Σ_i s_i)²]."""
    pi = 0.5
    H, J = 0.0, 0.0
    for _ in range(n_bags):
        xs = rng.normal(size=k)
        eta, d = _bin_model(xs)
        m = (eta + (k - 1) * pi) / k
        g = (d / k) / (m * (1 - m))                          # s_i = (w̃_i − m_i)·g_i
        H += float(np.sum((d / k) ** 2 / (m * (1 - m))))
        for _ in range(reps_per_bag):
            y = (rng.uniform(size=k) < eta).astype(int)
            w = rng.permutation(np.repeat([0, 1], [k - y.sum(), y.sum()]))
            J += float(np.sum((w - m) * g)) ** 2 / reps_per_bag
    return (H / n_bags) ** 2 / (J / n_bags), H / n_bags


def _pll_model(x, C):
    W = np.stack([np.cos(2 * np.pi * np.arange(C) / C), np.sin(2 * np.pi * np.arange(C) / C)], 1)
    z = THETA * (x @ W.T)
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    v = x @ W.T                                             # ∂z_k/∂θ
    u = v - (p * v).sum(1, keepdims=True)                   # ∂ log p_k / ∂θ
    return p, u


def fisher_pll(rng, C, q, n_x):
    """(clean, full, naive): thông tin Fisher mỗi mẫu của nhãn sạch, của tập ứng viên S, và của nhãn ngây thơ w̃."""
    x = rng.normal(size=(n_x, 2))
    p, u = _pll_model(x, C)
    clean = float(np.mean((p * u ** 2).sum(1)))
    full = np.zeros(n_x)
    for mask in itertools.product([0, 1], repeat=C):
        Sm = np.array(mask, bool)
        if not Sm.any():
            continue
        prob_S_given_y = q ** (Sm.sum() - 1) * (1 - q) ** (C - Sm.sum())
        num = (p[:, Sm] * u[:, Sm]).sum(1)
        den = p[:, Sm].sum(1)
        full += prob_S_given_y * den * (num / den) ** 2      # P(S|x) · score²
    a, b = pll_affine_coeffs(C, q)
    mt = b + (a - b) * p
    naive = float(np.mean((((a - b) * p * u) ** 2 / mt).sum(1)))
    return clean, float(full.mean()), naive


# ================================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    nb = 400 if a.quick else 3000

    print("== (1) Luật affine của nhãn ngây thơ (Monte Carlo theo đúng thủ tục thí điểm) ==")
    for C, q in ((4, 0.3), (4, 0.6), (7, 0.6), (3, 0.6)):
        M = pll_affine_mc(C, q, 40000 if not a.quick else 8000, rng)
        aa, bb = pll_affine_coeffs(C, q)
        off = M[~np.eye(C, dtype=bool)]
        print(f"PLL C={C} q={q}: a lý thuyết {aa:.4f} | MC đường chéo {np.diag(M).mean():.4f}; "
              f"b lý thuyết {bb:.4f} | MC ngoài chéo {off.mean():.4f}; độ dốc a−b = {aa - bb:.4f}")
    y = (rng.uniform(size=1400) < 0.35).astype(int)
    for k in (8, 16, 32):
        ex, mc = llp_affine_exact(y, k), llp_affine_mc(y, k, 60 if a.quick else 300, rng)
        print(f"LLP k={k}: P(w̃=1|y=1) lý thuyết {ex[1]:.4f} | MC {mc[1]:.4f};  P(w̃=1|y=0) {ex[0]:.4f} | {mc[0]:.4f};"
              f" độ dốc {ex[1] - ex[0]:.4f} (= 1/k − (k−1)/(k(n−1)))")

    print("\n== (2) Chặn hối tiếc AUC: max(hối tiếc − chặn) phải ≤ 0 ==")
    print(f"max(hối tiếc − chặn) = {auc_bound_check(rng, 300 if a.quick else 2000):.3e}")

    print("\n== (3) Thông tin Fisher mỗi mẫu (mô hình logistic một tham số, θ = 2, x ~ N(0, I)) ==")
    xs = rng.normal(size=200000)
    clean = fisher_binary_channel(xs, 1.0, 0.0)
    print(f"nhãn sạch: {clean:.4f}")
    for c in (0.1, 0.3):
        v = fisher_binary_channel(xs, c, 0.0)
        print(f"PU-SCAR c={c}: quan sát đầy đủ {v:.4f} = ngây thơ {v:.4f} (tỉ số 1); so với nhãn sạch {v / clean:.3f}")
    for r in (0.2, 0.4):
        v = fisher_binary_channel(xs, 1 - 1.5 * r, r / 2)
        print(f"nhiễu r={r}: quan sát đầy đủ {v:.4f} = ngây thơ (tỉ số 1); so với nhãn sạch {v / clean:.3f}")
    print("LLP (mỗi mẫu): k | đầy đủ (số đếm túi) | ngây thơ Godambe | tỉ số ngây thơ/đầy đủ | k·tỉ số")
    for k in (2, 4, 8, 16, 32):
        full = fisher_llp_bag(rng, k, nb if k <= 16 else nb // 3) / k
        god, H = godambe_llp_naive(rng, k, nb if k <= 16 else nb // 3)
        god, H = god / k, H / k
        print(f"  {k:>2d} | {full:.4f} ({full / clean:.3f} sạch) | {god:.5f} | {god / full:.3f} | {k * god / full:.2f}"
              f"   [nếu coi các cặp độc lập (H): {H:.5f}, tỉ số {H / full:.3f}]")
    print("PLL (mỗi mẫu): C q | sạch | đầy đủ (tập S) | ngây thơ | tỉ số ngây thơ/đầy đủ")
    for C, q in ((3, 0.3), (3, 0.6), (4, 0.3), (4, 0.6), (7, 0.3), (7, 0.6)):
        cl, fu, na = fisher_pll(rng, C, q, 20000 if a.quick else 100000)
        print(f"  {C} {q} | {cl:.4f} | {fu:.4f} | {na:.4f} | {na / fu:.3f}")


if __name__ == "__main__":
    main()
