"""
exact.py — công cụ kiểm chứng CHÍNH XÁC trên bài toán nhỏ (n ≤ 14): liệt kê mọi cách gán nhãn, dựng ma trận
chuyển của từng loại cập nhật, tính phân phối dừng và so với posterior đúng π.

Dùng cho RQ1 và kiểm chứng Định lý 1, Định lý 2, Mệnh đề 4. Không dùng trong thực nghiệm chính.
"""
from __future__ import annotations

import itertools
from typing import Callable, Sequence

import numpy as np
from scipy.special import logsumexp

from .conditional import ConditionalModel
from .llp import count_constrained_log_normalizer
from .weak import WeakModel


# --------------------------------------------------------------------------------------------------
# Không gian trạng thái và posterior chính xác
# --------------------------------------------------------------------------------------------------
def support_states(weak: WeakModel) -> np.ndarray:
    """Mọi y có ℓ(y) > 0 (và thoả ràng buộc túi với LLP)."""
    C, n = weak.n_classes, weak.n_total
    if not weak.is_group:
        L = weak.lik()
        sup = [np.flatnonzero(L[i] > 0) for i in range(n)]
        return np.array(list(itertools.product(*sup)), dtype=int)
    per_bag = []
    for b, cnt in zip(weak.bags, weak.counts):
        opts = [np.array(a) for a in itertools.product(range(C), repeat=len(b))
                if (np.bincount(np.array(a, dtype=int), minlength=C) == cnt).all()]
        per_bag.append(opts)
    free = weak.free_sites()
    states = []
    for combo in itertools.product(*per_bag, *([range(C)] * len(free))):
        y = np.zeros(n, dtype=int)
        for b, a in zip(weak.bags, combo[: len(weak.bags)]):
            y[b] = a
        y[free] = combo[len(weak.bags):]
        states.append(y)
    return np.array(states, dtype=int)


def index_of(states: np.ndarray) -> dict:
    return {tuple(s): k for k, s in enumerate(states)}


def log_target(log_joint: Callable, X, states, weak: WeakModel) -> np.ndarray:
    """log π(y) chuẩn hoá = log p(y | X) + Σ log ℓ_i(y_i) − log Z, trên các trạng thái đã liệt kê."""
    L = weak.lik()
    with np.errstate(divide="ignore"):
        logL = np.log(L)
    lj = np.array([log_joint(X, s) + logL[np.arange(len(s)), s].sum() for s in states])
    return lj - logsumexp(lj)


def exact_predictive(log_joint: Callable, X, states, logpi, x_star, n_classes) -> np.ndarray:
    """Posterior predictive "quy nạp" p̄(y_*|x_*) = Σ_y π(y) p(y_* | x_*, X, y) — tính bằng log_joint."""
    out = np.zeros(n_classes)
    Xa = np.vstack([X, x_star[None]])
    for s, lp in zip(states, logpi):
        lj = np.array([log_joint(Xa, np.append(s, k)) for k in range(n_classes)])
        out += np.exp(lp) * np.exp(lj - logsumexp(lj))
    return out


# --------------------------------------------------------------------------------------------------
# Ma trận chuyển
# --------------------------------------------------------------------------------------------------
def _cond_row(cond, X, y, exclude, query):
    mask = np.ones(len(y), bool)
    mask[exclude] = False
    return cond.predict_proba(X[mask], y[mask], X[query])


def kernel_single_site(cond: ConditionalModel, X, states, weak: WeakModel, index=None) -> np.ndarray:
    """Gibbs một mẫu, chọn ngẫu nhiên đều một mẫu tự do mỗi bước (random scan)."""
    index = index or index_of(states)
    L = weak.lik()
    free = weak.free_sites()
    S = len(states)
    P = np.zeros((S, S))
    for a, y in enumerate(states):
        for i in free:
            w = _cond_row(cond, X, y, [i], [i])[0] * L[i]
            w = w / w.sum()
            for k in np.flatnonzero(w > 0):
                z = y.copy()
                z[i] = k
                P[a, index[tuple(z)]] += w[k] / len(free)
    return P


def _block_configs(states, index, y, b):
    """Các trạng thái trùng y ngoài khối b (kèm chỉ số)."""
    out = []
    mask = np.ones(len(y), bool)
    mask[b] = False
    key = y[mask]
    for z in states:                                  # đủ nhanh với S ≤ vài nghìn
        if np.array_equal(z[mask], key):
            out.append(z)
    return out


def kernel_block(cond: ConditionalModel, X, states, weak: WeakModel, blocks: Sequence, kind: str = "approx",
                 logpi=None, index=None) -> np.ndarray:
    """Cập nhật khối cấp mẫu, chọn đều một khối trong `blocks` mỗi bước.
    kind='approx' : Π_i q(y_i | x_i, D_{-b}) ℓ_i(y_i)  (thuật toán thực tế, Mệnh đề 4)
    kind='mh'     : đề xuất 'approx' + chấp nhận MH, tỉ số π qua bổ đề Brook với chính q (Mệnh đề 4b)
    kind='exact'  : Gibbs khối từ π (chuẩn so sánh; cần logpi)"""
    index = index or index_of(states)
    L = weak.lik()
    S = len(states)
    P = np.zeros((S, S))
    with np.errstate(divide="ignore"):
        logL = np.log(L)
    sup = [np.flatnonzero(L[i] > 0) for i in range(L.shape[0])]
    for a, y in enumerate(states):
        for b in blocks:
            b = np.asarray(b)
            cand = []
            for vals in itertools.product(*[sup[i] for i in b]):
                z = y.copy()
                z[b] = vals
                cand.append(z)
            if kind == "exact":
                lp = np.array([logpi[index[tuple(z)]] for z in cand])
                w = np.exp(lp - logsumexp(lp))
                for z, wz in zip(cand, w):
                    P[a, index[tuple(z)]] += wz / len(blocks)
                continue
            W = _cond_row(cond, X, y, b, b) * L[b]
            W = W / W.sum(1, keepdims=True)
            for z in cand:
                qz = float(np.prod(W[np.arange(len(b)), z[b]]))
                if qz == 0:
                    continue
                if kind == "approx":
                    P[a, index[tuple(z)]] += qz / len(blocks)
                    continue
                changed = b[z[b] != y[b]]
                if len(changed) == 0:
                    P[a, a] += qz / len(blocks)
                    continue
                # tỉ số Brook với q
                cur = y.copy()
                lr = 0.0
                for k in changed:
                    p = _cond_row(cond, X, cur, [k], [k])[0]
                    lr += np.log(p[z[k]]) - np.log(p[cur[k]])
                    cur[k] = z[k]
                lr += (logL[changed, z[changed]] - logL[changed, y[changed]]).sum()
                qy = float(np.prod(W[np.arange(len(b)), y[b]]))
                lr += np.log(qy) - np.log(qz)
                alpha = min(1.0, float(np.exp(min(lr, 0.0))))
                P[a, index[tuple(z)]] += qz * alpha / len(blocks)
                P[a, a] += qz * (1 - alpha) / len(blocks)
    return P


def kernel_swap(cond: ConditionalModel, X, states, weak: WeakModel, index=None, lazy: float = 0.0) -> np.ndarray:
    """MH hoán đổi cho LLP: chọn đều một túi, chọn đều một cặp có thứ tự (i, j) với y_i ≠ y_j, đổi chỗ,
    chấp nhận với min(1, R), R tính bằng 2 lần gọi q (bổ đề Brook). lazy: xác suất đứng yên thêm."""
    index = index or index_of(states)
    S = len(states)
    G = len(weak.bags)
    P = np.zeros((S, S))
    for a, y in enumerate(states):
        for b in weak.bags:
            yb = y[b]
            pairs = np.argwhere(yb[:, None] != yb[None, :])
            if len(pairs) == 0:
                P[a, a] += 1.0 / G
                continue
            for u, v in pairs:
                i, j = b[u], b[v]
                yi, yj = y[i], y[j]
                p1 = _cond_row(cond, X, y, [i], [i])[0]
                zz = y.copy()
                zz[i] = yj
                p2 = _cond_row(cond, X, zz, [j], [j])[0]
                R = (p1[yj] / p1[yi]) * (p2[yi] / p2[yj])
                alpha = min(1.0, R)
                z = y.copy()
                z[i], z[j] = yj, yi
                P[a, index[tuple(z)]] += alpha / (G * len(pairs))
                P[a, a] += (1 - alpha) / (G * len(pairs))
    return lazy * np.eye(S) + (1 - lazy) * P


def kernel_bag_block(cond: ConditionalModel, X, states, weak: WeakModel, groups: Sequence, kind: str = "approx",
                     logpi=None, index=None) -> np.ndarray:
    """Cập nhật theo nhóm túi cho LLP (chọn đều một nhóm trong `groups`, mỗi nhóm là danh sách chỉ số túi).
    kind='approx' : mỗi túi ~ Π_i q(y_i | x_i, D_{-nhóm}) có điều kiện theo số đếm
    kind='exact'  : Gibbs khối từ π"""
    index = index or index_of(states)
    S = len(states)
    P = np.zeros((S, S))
    for a, y in enumerate(states):
        for grp in groups:
            sites = np.concatenate([weak.bags[g] for g in grp])
            cand = _block_configs(states, index, y, sites)
            if kind == "exact":
                lp = np.array([logpi[index[tuple(z)]] for z in cand])
                w = np.exp(lp - logsumexp(lp))
            else:
                Q = _cond_row(cond, X, y, sites, sites)
                row = {s: r for r, s in enumerate(sites)}
                logZ = {g: count_constrained_log_normalizer(Q[[row[s] for s in weak.bags[g]]], weak.counts[g])
                        for g in grp}
                w = []
                for z in cand:
                    lw = 0.0
                    for g in grp:
                        rows = [row[s] for s in weak.bags[g]]
                        lw += np.log(Q[rows, z[weak.bags[g]]]).sum() - logZ[g]
                    w.append(np.exp(lw))
                w = np.array(w)
            for z, wz in zip(cand, w):
                P[a, index[tuple(z)]] += wz / len(groups)
    return P


# --------------------------------------------------------------------------------------------------
# Phân tích chuỗi
# --------------------------------------------------------------------------------------------------
def stationary(P: np.ndarray) -> np.ndarray:
    S = len(P)
    A = np.vstack([P.T - np.eye(S), np.ones((1, S))])
    b = np.zeros(S + 1)
    b[-1] = 1.0
    pi = np.linalg.lstsq(A, b, rcond=None)[0]
    pi = np.maximum(pi, 0)
    return pi / pi.sum()


def tv(p, q) -> float:
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def detailed_balance_error(P, pi) -> float:
    F = pi[:, None] * P
    return float(np.abs(F - F.T).max())


def dobrushin(P: np.ndarray, chunk: int = 256) -> float:
    """δ(P) = max_{x,y} TV(P(x,·), P(y,·)) = 1 − min_{x,y} Σ_z min(P(x,z), P(y,z))."""
    S = len(P)
    best = 1.0
    for s in range(0, S, chunk):
        ov = np.minimum(P[s:s + chunk, None, :], P[None, :, :]).sum(-1)
        best = min(best, float(ov.min()))
    return 1.0 - best


def ergodicity_profile(P: np.ndarray, pi: np.ndarray, tol: float = 1e-10, tmax: int = 20000):
    """Trả về (d, delta): d[t] = max_x TV(P^t(x,·), π), delta[t] = δ(P^t), t = 0..T (dừng khi delta < tol)."""
    S = len(P)
    M = np.eye(S)
    d, delta = [], []
    for _ in range(tmax):
        d.append(float(0.5 * np.abs(M - pi[None]).sum(1).max()))
        delta.append(dobrushin(M))
        if delta[-1] < tol:
            break
        M = M @ P
    return np.array(d), np.array(delta)


def perturbation_bound(delta_profile: np.ndarray, eps_bar: float) -> float:
    """Định lý 2 (dạng Dobrushin): TV(π̃, π) ≤ (Σ_t δ(P^t)) · ε̄."""
    return float(delta_profile.sum() * eps_bar)


def mitrophanov_constants(d_profile: np.ndarray, rho_grid=None):
    """Chọn (C, ρ) sao cho 2 d(t) ≤ C ρ^t với mọi t đã tính (chuẩn ℓ1 như Mitrophanov 2005), và trả về
    hệ số K(C, ρ) = n̂ + C ρ^{n̂}/(1 − ρ), n̂ = max(0, ⌈log_ρ(1/C)⌉), nhỏ nhất trên lưới ρ."""
    t = np.arange(len(d_profile))
    d1 = 2 * np.maximum(d_profile, 1e-300)
    pos = d_profile > 1e-12
    rate = np.exp(np.polyfit(t[pos][-max(3, pos.sum() // 3):], np.log(d1[pos][-max(3, pos.sum() // 3):]), 1)[0])
    rho_grid = rho_grid if rho_grid is not None else np.linspace(min(max(rate, 1e-3), 0.999), 0.9999, 400)
    best = (np.inf, None, None)
    for rho in rho_grid:
        C = float(np.max(d1[pos] / rho ** t[pos])) if pos.any() else 1.0
        nhat = max(0, int(np.ceil(np.log(1.0 / C) / np.log(rho)))) if C > 1 else 0
        K = nhat + C * rho ** nhat / (1 - rho)
        if K < best[0]:
            best = (K, C, rho)
    return {"K": best[0], "C": best[1], "rho": best[2]}


def kernel_row_tv(P_tilde: np.ndarray, P: np.ndarray) -> np.ndarray:
    """TV(P̃(x,·), P(x,·)) cho từng trạng thái x."""
    return 0.5 * np.abs(P_tilde - P).sum(1)


def site_eps_kappa(cond_q: ConditionalModel, cond_p: ConditionalModel, X, states, weak: WeakModel):
    """ε = max TV(q(·|x_i,D_{-i}), p(·|x_i,D_{-i})) và κ = max (max_y ℓ_i)/(Σ_y p ℓ_i) trên mọi trạng thái và mẫu tự do."""
    L = weak.lik()
    eps, kap = 0.0, 0.0
    for y in states:
        for i in weak.free_sites():
            q = _cond_row(cond_q, X, y, [i], [i])[0]
            p = _cond_row(cond_p, X, y, [i], [i])[0]
            eps = max(eps, tv(q, p))
            kap = max(kap, L[i].max() / float((p * L[i]).sum()))
    return eps, kap


def log_ratio_incompatibility(cond: ConditionalModel, X, y, i, j, a, b) -> float:
    """Độ không tương thích: tỉ số p(y'|X)/p(y|X) tính qua hai đường Brook (đổi i trước, hoặc j trước).
    Với q tương thích (tồn tại phân phối đồng thời) hai giá trị bằng nhau; trả về chênh lệch log."""
    def path(first, second, v1, v2):
        z = y.copy()
        p = _cond_row(cond, X, z, [first], [first])[0]
        r = np.log(p[v1]) - np.log(p[z[first]])
        z[first] = v1
        p = _cond_row(cond, X, z, [second], [second])[0]
        r += np.log(p[v2]) - np.log(p[z[second]])
        return r
    return float(path(i, j, a, b) - path(j, i, b, a))
