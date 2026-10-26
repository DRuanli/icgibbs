"""
g6_review_checks.py — kiểm lại độc lập (không dùng script của người kiểm duyệt) các phát hiện F1, F2, F3, F5, F6 của
lượt kiểm duyệt độc lập PROOFS.md (26/10/2026). Chạy: python experiments/g6_review_checks.py  (khoảng 1 phút, CPU)
"""
import itertools
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from icgibbs import exact as E  # noqa: E402
from icgibbs.conditional import AnalyticCellPPD, DiscreteMixturePPD  # noqa: E402
from icgibbs.weak import LLP, SSL  # noqa: E402


def f5_psw_reversible(rng):
    out = []
    for trial in range(3):
        n = 4
        cond = DiscreteMixturePPD.random(n_cells=3, n_classes=2, seed=40 + trial)
        X = rng.integers(0, 3, (n, 1)).astype(float)
        weak = SSL(-np.ones(n, int), 2)
        states = E.support_states(weak); idx = E.index_of(states)
        lp = E.log_target(cond.log_joint, X, states, weak); pi = np.exp(lp - lp.max()); pi /= pi.sum()
        S = len(states); Pi = []
        for i in range(n):
            P = np.zeros((S, S))
            for a, y in enumerate(states):
                w = E._cond_row(cond, X, y, [i], [i])[0]; w = w / w.sum()
                for k in range(2):
                    z = y.copy(); z[i] = k; P[a, idx[tuple(z)]] += w[k]
            Pi.append(P)
        Psw = sum(np.linalg.multi_dot([Pi[j] for j in s]) for s in itertools.permutations(range(n))) / math.factorial(n)
        out.append((E.detailed_balance_error(Psw, pi), E.detailed_balance_error(np.linalg.multi_dot(Pi), pi)))
    return out


def f6_first_t(rng):
    res = {}
    for n in range(2, 7):
        cond = DiscreteMixturePPD.random(n_cells=3, n_classes=2, seed=7)
        X = rng.integers(0, 3, (n, 1)).astype(float); weak = SSL(-np.ones(n, int), 2)
        states = E.support_states(weak)
        P = E.kernel_single_site(cond, X, states, weak); Q = np.eye(len(states))
        for t in range(1, 3 * n):
            Q = Q @ P
            if E.dobrushin(Q) < 1 - 1e-12:
                break
        res[n] = t
    return res


def f3_identity(rng, trials=200):
    worst, ok = 0.0, 0
    for _ in range(trials):
        S = int(rng.integers(2, 6)); P = rng.dirichlet(np.ones(S), S)
        Pt = 0.8 * P + 0.2 * rng.dirichlet(np.ones(S), S)
        pi, pit = E.stationary(P), E.stationary(Pt)
        Z = np.linalg.inv(np.eye(S) - P + np.outer(np.ones(S), pi)) - np.outer(np.ones(S), pi)
        nu = pit @ (Pt - P)
        worst = max(worst, np.abs((pit - pi) - nu @ Z).max())
        tau1Z = 0.5 * max(np.abs(Z[i] - Z[j]).sum() for i in range(S) for j in range(S))
        prof, Q = [], np.eye(S)
        for _ in range(4000):
            d = E.dobrushin(Q); prof.append(d); Q = Q @ P
            if d < 1e-13:
                break
        ok += tau1Z <= sum(prof) + 1e-9
    return worst, ok, trials


def f2_transductive():
    diffs = []
    for seed in range(100, 130):
        cond = DiscreteMixturePPD.random(n_cells=3, n_classes=2, seed=seed)
        X = np.array([[0], [1], [2], [0], [1], [2]], float); s = np.array([1, 0, 0, 0, 1, 0]); c = 0.5
        lik = lambda y: np.prod([(c if si else 1 - c) if yi else (1.0 if si == 0 else 0.0) for yi, si in zip(y, s)])
        n1 = d1 = n2 = d2 = 0.0
        for y in itertools.product([0, 1], repeat=6):
            L = lik(y)
            if L == 0:
                continue
            for a in (0, 1):
                pj = math.exp(cond.log_joint(np.vstack([X, [[0]]]), np.array(list(y) + [a]))) * L
                d1 += pj; n1 += pj * a
                for b in (0, 1):
                    pj2 = math.exp(cond.log_joint(np.vstack([X, [[0]], [[2]]]), np.array(list(y) + [a, b]))) * L
                    d2 += pj2; n2 += pj2 * a
        diffs.append(abs(n1 / d1 - n2 / d2))
    return max(diffs), float(np.median(diffs))


def f1_bag_block(rng):
    out = []
    for name, cond in (("ô", AnalyticCellPPD(2, alpha=1.0)), ("hỗn hợp", DiscreteMixturePPD.random(3, 2, seed=11))):
        X = rng.integers(0, 3, (12, 1)).astype(float); y0 = rng.integers(0, 2, 12)
        bags = [np.arange(0, 4), np.arange(4, 8), np.arange(8, 12)]
        weak = LLP(bags, np.array([[4 - y0[b].sum(), y0[b].sum()] for b in bags]), 12)
        states = E.support_states(weak); idx = E.index_of(states)
        lp = E.log_target(cond.log_joint, X, states, weak); pi = np.exp(lp - lp.max()); pi /= pi.sum()
        for groups in ([[0], [1], [2]], [[0, 1, 2]]):
            Pa = E.kernel_bag_block(cond, X, states, weak, groups, kind="approx", index=idx)
            out.append((name, len(groups), E.tv(E.stationary(Pa), pi)))
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(5)
    print("F5 (P_sw khả nghịch): sai số cân bằng chi tiết (P_sw, quét cố định):", [f"{a:.1e}/{b:.1e}" for a, b in f5_psw_reversible(rng)])
    print("F6 (bước đầu tiên có δ<1, |F| = 2..6):", f6_first_t(rng))
    w, ok, n = f3_identity(rng)
    print(f"F3 (π̃ − π = νZ; τ1(Z) ≤ τ(P)): sai số lớn nhất {w:.1e}; {ok}/{n}")
    mx, med = f2_transductive()
    print(f"F2 (chuyển nạp 2 điểm test so với f*): chênh lớn nhất {mx:.3f}, trung vị {med:.3f}")
    print("F1 (bag-block xấp xỉ, LLP 3 túi × 4): ", [(a, f"{b} nhóm", round(c, 3)) for a, b, c in f1_bag_block(rng)])
