"""
rq1_exact.py — kiểm chứng số cho RQ1 (Định lý 1, Định lý 2, Mệnh đề 4) trên prior giải tích. Chạy trên CPU.

    python rq1_exact.py e1      # Định lý 1: phân phối dừng của từng loại cập nhật = π (chính xác)
    python rq1_exact.py e1mc    # Định lý 1: bộ lấy mẫu thật, TV(tần suất, π) theo số mẫu
    python rq1_exact.py e2      # Định lý 1(ii): dự đoán quy nạp vs chuyển nạp vs Bayes chính xác
    python rq1_exact.py e3      # Định lý 2: q bị nhiễu (không tương thích) — TV thật vs chặn
    python rq1_exact.py e4      # Định lý 2: q = PFN nhỏ học được (cần results/pfn/*.pt)
    python rq1_exact.py e5a     # Mệnh đề 4: cập nhật khối, bài toán nhỏ (cả hai nguồn sai)
    python rq1_exact.py e5b     # Mệnh đề 4: chuỗi gộp một ô, n lớn (nguồn sai (i))
Kết quả: ../../results/rq1/*.csv
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import betaln, gammaln, logsumexp
from scipy.stats import binom, hypergeom

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from icgibbs import (LLP, PUSCAR, SSL, AnalyticCellPPD, DiscreteMixturePPD, InContextGibbs,  # noqa: E402
                     NoisyLabel, PartialLabel, PerturbedConditional)
from icgibbs.conditional import ConditionalModel  # noqa: E402
from icgibbs.exact import (detailed_balance_error, ergodicity_profile, exact_predictive, index_of,  # noqa: E402
                           kernel_block, kernel_row_tv, kernel_single_site, kernel_swap, log_target,
                           mitrophanov_constants, stationary, support_states, tv)

OUT = os.path.join(HERE, "..", "..", "results", "rq1")
os.makedirs(OUT, exist_ok=True)
M, C = 3, 2


def cells(n, seed):
    return np.random.default_rng(seed).integers(0, M, size=(n, 1)).astype(float)


def prior(name):
    return AnalyticCellPPD(C, 1.0) if name == "cell" else DiscreteMixturePPD.random(M, C, seed=1)


def save(df, name):
    path = os.path.join(OUT, name)
    df.to_csv(path, index=False)
    print(df.to_string(index=False))
    print("→", path)


# ==================================================================================================
def problems(n=10, seed=0):
    r = np.random.default_rng(seed)
    S = np.ones((n, 2), bool)
    S[:3] = False
    S[0, 1] = S[1, 0] = S[2, 1] = True
    T = np.array([[0.8, 0.2], [0.3, 0.7]])
    return {
        "PU-SCAR (c=0.5)": (PUSCAR(np.r_[1, 1, 1, np.zeros(n - 3, int)], c=0.5), "single"),
        "SSL": (SSL(np.r_[1, 0, 1, -np.ones(n - 3, int)], 2), "single"),
        "PLL": (PartialLabel(S), "single"),
        "Noisy (T known)": (NoisyLabel(r.integers(0, 2, 8), 2, T=T), "single"),
        "LLP (3 bags of 4)": (LLP([np.arange(0, 4), np.arange(4, 8), np.arange(8, 12)], [[2, 2], [1, 3], [3, 1]], n=12), "swap"),
        "SSL, block-MH (m=3)": (SSL(np.r_[1, 0, 1, -np.ones(n - 3, int)], 2), "block-mh"),
    }


def e1():
    rows = []
    for pname in ("cell", "mixture"):
        cond = prior(pname)
        for wname, (weak, scheme) in problems().items():
            X = cells(weak.n_total, seed=7)
            states = support_states(weak)
            logpi = log_target(cond.log_joint, X, states, weak)
            pi = np.exp(logpi)
            t0 = time.time()
            if scheme == "single":
                P = kernel_single_site(cond, X, states, weak)
            elif scheme == "swap":
                P = kernel_swap(cond, X, states, weak)
            else:
                blocks = [list(b) for b in itertools.combinations(weak.free_sites(), 3)]
                P = kernel_block(cond, X, states, weak, blocks, kind="mh")
            rows.append(dict(prior=pname, weak=wname, scheme=scheme, n=weak.n_total, n_states=len(states),
                             tv_stationary_vs_pi=tv(stationary(P), pi),
                             detailed_balance_err=detailed_balance_error(P, pi),
                             row_sum_err=float(np.abs(P.sum(1) - 1).max()), seconds=round(time.time() - t0, 2)))
    # PLL thật sự cần ≥ 3 lớp: thêm một bài C = 3, tập ứng viên cỡ 2
    r = np.random.default_rng(5)
    for pname, cond in (("cell", AnalyticCellPPD(3, 1.0)), ("mixture", DiscreteMixturePPD.random(M, 3, seed=2))):
        ytrue = r.integers(0, 3, 7)
        S = np.zeros((7, 3), bool)
        S[np.arange(7), ytrue] = True
        S[np.arange(7), (ytrue + r.integers(1, 3, 7)) % 3] = True
        weak = PartialLabel(S)
        X = cells(7, seed=7)
        states = support_states(weak)
        pi = np.exp(log_target(cond.log_joint, X, states, weak))
        P = kernel_single_site(cond, X, states, weak)
        rows.append(dict(prior=pname, weak="PLL (C=3, |S_i|=2)", scheme="single", n=7, n_states=len(states),
                         tv_stationary_vs_pi=tv(stationary(P), pi), detailed_balance_err=detailed_balance_error(P, pi),
                         row_sum_err=float(np.abs(P.sum(1) - 1).max()), seconds=0.0))
    save(pd.DataFrame(rows), "e1_exact_stationary.csv")


# ==================================================================================================
def iid_floor(pi, N, reps=50, seed=0):
    r = np.random.default_rng(seed)
    return float(np.mean([tv(r.multinomial(N, pi) / N, pi) for _ in range(reps)]))


def e1mc(max_sweeps=100_000):
    checkpoints = [100, 300, 1000, 3000, 10_000, 30_000, 100_000]
    configs = [
        ("cell", "PU-SCAR single-site", lambda: PUSCAR(np.r_[1, 1, np.zeros(6, int)], c=0.5), dict(scheme="single"), 8),
        ("mixture", "LLP swap", lambda: LLP([np.arange(0, 4), np.arange(4, 8)], [[2, 2], [1, 3]], n=8), dict(scheme="swap"), 8),
        ("mixture", "SSL block + MH (K=2)", lambda: SSL(np.r_[1, 0, 1, -np.ones(5, int)], 2), dict(scheme="block", n_blocks=2, mh_correct=True), 8),
        ("mixture", "LLP bag-block + MH (K=2)", lambda: LLP([np.arange(0, 4), np.arange(4, 8)], [[2, 2], [1, 3]], n=8), dict(scheme="bag-block", n_blocks=2, mh_correct=True), 8),
    ]
    rows = []
    for pname, label, mk, kw, n in configs:
        cond = prior(pname)
        X = cells(n, seed=11)
        weak = mk()
        states = support_states(weak)
        pi = np.exp(log_target(cond.log_joint, X, states, weak))
        idx = index_of(states)
        t0 = time.time()
        g = InContextGibbs(cond, mk(), n_sweeps=max_sweeps + 100, burn_in=100, seed=3, **kw).run(X)
        ids = np.array([idx[tuple(s)] for s in g.samples_])
        for N in checkpoints:
            if N > len(ids):
                break
            emp = np.bincount(ids[:N], minlength=len(states)) / N
            rows.append(dict(prior=pname, config=label, n_states=len(states), samples=N, tv=tv(emp, pi),
                             iid_floor=iid_floor(pi, N), seconds=round(time.time() - t0, 1)))
        print(rows[-1], flush=True)
    save(pd.DataFrame(rows), "e1mc_convergence.csv")


# ==================================================================================================
def e2():
    """Dự đoán tại x* (mỗi ô): p̄ quy nạp = Σ_y π(y) p(y*|x*,X,y) và Bayes chính xác p(y*|x*,X,W)."""
    rows = []
    for pname in ("cell", "mixture"):
        cond = prior(pname)
        for wname, mk in [("SSL", lambda m: SSL(np.r_[1, 0, 1, -np.ones(5, int)], 2)),
                          ("PU-SCAR", lambda m: PUSCAR(np.r_[1, 1, np.zeros(6, int)], c=0.5))]:
            X = cells(8, seed=5)
            weak = mk(0)
            states = support_states(weak)
            logpi = log_target(cond.log_joint, X, states, weak)
            for c_star in range(M):
                x_star = np.array([float(c_star)])
                pbar = exact_predictive(cond.log_joint, X, states, logpi, x_star, C)[1]
                X2 = np.vstack([X, x_star[None]])
                weak2 = mk(0).extend(1)                                # x* là điểm tự do (ℓ ≡ 1), không chịu cơ chế quan sát
                st2 = support_states(weak2)
                lp2 = log_target(cond.log_joint, X2, st2, weak2)
                exact = float(np.exp(lp2[st2[:, -1] == 1]).sum() / np.exp(lp2).sum())
                rows.append(dict(prior=pname, weak=wname, x_star_cell=c_star, inductive_pbar=pbar,
                                 exact_bayes=exact, abs_diff=abs(pbar - exact)))
    save(pd.DataFrame(rows), "e2_inductive_vs_exact.csv")


# ==================================================================================================
class TableConditional(ConditionalModel):
    """Tra bảng các hàng q đã tính trước theo lô (cho PFN); khoá = (ô ngữ cảnh, nhãn ngữ cảnh, ô truy vấn)."""

    def __init__(self, base, n_classes=2):
        super().__init__(n_classes, floor=0.0)
        self.base, self.table = base, {}

    @staticmethod
    def key(cc, yc, cq):
        return (tuple(int(v) for v in cc), tuple(int(v) for v in yc), int(cq))

    def prefetch(self, requests):
        todo = {}
        for cc, yc, cq in requests:
            k = self.key(cc, yc, cq)
            if k not in self.table:
                todo.setdefault(len(cc), {})[k] = (cc, yc, cq)
        for n, d in todo.items():
            keys = list(d)
            cc = np.array([d[k][0] for k in keys], dtype=int).reshape(len(keys), n)
            yc = np.array([d[k][1] for k in keys], dtype=int).reshape(len(keys), n)
            cq = np.array([d[k][2] for k in keys], dtype=int)
            P = self.base.batch_predict(cc, yc, cq)
            self.table.update(zip(keys, P))

    def _predict(self, X_ctx, y_ctx, X_query):
        out = []
        for cq in X_query[:, 0]:
            k = self.key(X_ctx[:, 0], y_ctx, cq)
            if k not in self.table:
                self.prefetch([(X_ctx[:, 0].astype(int), y_ctx, int(cq))])
            out.append(self.table[k])
        return np.array(out)


def single_requests(X, states, sites):
    req = []
    for y in states:
        for i in sites:
            mask = np.arange(len(y)) != i
            req.append((X[mask, 0].astype(int), y[mask], int(X[i, 0])))
    return req


def swap_requests(X, states, weak):
    req = []
    for y in states:
        for b in weak.bags:
            for u, v in itertools.product(range(len(b)), repeat=2):
                i, j = b[u], b[v]
                if y[i] == y[j]:
                    continue
                m = np.arange(len(y)) != i
                req.append((X[m, 0].astype(int), y[m], int(X[i, 0])))
                z = y.copy()
                z[i] = y[j]
                m = np.arange(len(y)) != j
                req.append((X[m, 0].astype(int), z[m], int(X[j, 0])))
    return req


def thm2_setup(kind):
    if kind == "single":
        weak = PUSCAR(np.r_[1, 1, np.zeros(8, int)], c=0.5)
        X = cells(10, seed=13)
    else:
        weak = LLP([np.arange(0, 4), np.arange(4, 8), np.arange(8, 12)], [[2, 2], [1, 3], [3, 1]], n=12)
        X = cells(12, seed=13)
    p = AnalyticCellPPD(C, 1.0)
    states = support_states(weak)
    pi = np.exp(log_target(p.log_joint, X, states, weak))
    build = (lambda cond: kernel_single_site(cond, X, states, weak)) if kind == "single" else \
            (lambda cond: kernel_swap(cond, X, states, weak))
    P = build(p)
    d, delta = ergodicity_profile(P, pi, tol=1e-9)
    tau = float(delta.sum())
    mit = mitrophanov_constants(d)
    return dict(weak=weak, X=X, p=p, states=states, pi=pi, build=build, P=P, tau=tau, mit=mit, t_steps=len(delta))


def site_errors(q, p, X, states, weak, kind):
    """ε_TV = max TV(q,p), ε_∞ = max |log q − log p| trên mọi hàng mà thuật toán dùng; κ (Định lý 2)."""
    L = weak.lik()
    eps, einf, kap = 0.0, 0.0, 0.0
    rows = single_requests(X, states, weak.free_sites()) if kind == "single" else swap_requests(X, states, weak)
    for cc, yc, cq in rows:
        Xc = cc[:, None].astype(float)
        qq = q.predict_proba(Xc, yc, np.array([[float(cq)]]))[0]
        pp = p.predict_proba(Xc, yc, np.array([[float(cq)]]))[0]
        eps = max(eps, tv(qq, pp))
        einf = max(einf, float(np.abs(np.log(qq) - np.log(pp)).max()))
    if kind == "single":
        for y in states:
            for i in weak.free_sites():
                m = np.arange(len(y)) != i
                pp = p.predict_proba(X[m], y[m], X[[i]])[0]
                kap = max(kap, L[i].max() / float((pp * L[i]).sum()))
    return eps, einf, kap


def thm2_row(label, q, S, kind):
    Pq = S["build"](q)
    pit = stationary(Pq)
    rowtv = kernel_row_tv(Pq, S["P"])
    eps, einf, kap = site_errors(q, S["p"], S["X"], S["states"], S["weak"], kind)
    actual = tv(pit, S["pi"])
    eps_bar, eps_max = float(pit @ rowtv), float(rowtv.max())
    r = dict(q=label, kernel=kind, n_states=len(S["states"]), eps_tv=eps, eps_inf=einf, kappa=kap,
             kernel_err_avg=eps_bar, kernel_err_max=eps_max, tau=S["tau"], tv_actual=actual,
             bound_thm2_avg=S["tau"] * eps_bar, bound_thm2_max=S["tau"] * eps_max,
             bound_mitrophanov=S["mit"]["K"] * eps_max,
             bound_from_eps=S["tau"] * (2 * kap * eps if kind == "single" else 4 * einf))
    r["looseness_avg"] = r["bound_thm2_avg"] / max(actual, 1e-300)
    # Hệ quả: sai số dự đoán Rao–Blackwell tại mỗi ô
    X, states, weak, p = S["X"], S["states"], S["weak"], S["p"]
    errs = []
    for c_star in range(M):
        xs = np.array([[float(c_star)]])
        e_q = sum(w * q.predict_proba(X, y, xs)[0, 1] for w, y in zip(pit, states))
        e_p = sum(w * p.predict_proba(X, y, xs)[0, 1] for w, y in zip(S["pi"], states))
        errs.append(abs(e_q - e_p))
    r["pred_err_max"] = max(errs)
    r["pred_bound_eps_plus_tv"] = eps + actual
    return r


def e3():
    rows = []
    for kind in ("single", "swap"):
        S = thm2_setup(kind)
        print(kind, "tau =", S["tau"], "mitrophanov", S["mit"], flush=True)
        for delta in (0.01, 0.03, 0.1, 0.3, 1.0):
            for seed in (0, 1, 2):
                q = PerturbedConditional(S["p"], delta=delta, seed=seed)
                r = thm2_row(f"perturbed δ={delta}", q, S, kind)
                r.update(delta=delta, seed=seed)
                rows.append(r)
                print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    save(pd.DataFrame(rows), "e3_thm2_perturbed.csv")


def e4():
    sys.path.insert(0, HERE)
    import torch
    from cell_pfn import CellPFN, CellPFNConditional
    pfn_dir = os.path.join(HERE, "..", "..", "results", "pfn")
    ckpts = sorted([f for f in os.listdir(pfn_dir) if f.endswith(".pt")],
                   key=lambda f: 10 ** 9 if "final" in f else int(f.split("step")[1].split(".")[0]))
    rows = []
    for kind in ("single", "swap"):
        S = thm2_setup(kind)
        req = single_requests(S["X"], S["states"], S["weak"].free_sites()) if kind == "single" else \
            swap_requests(S["X"], S["states"], S["weak"])
        for f in ckpts:
            model = CellPFN(M, C)
            model.load_state_dict(torch.load(os.path.join(pfn_dir, f)))
            q = TableConditional(CellPFNConditional(model))
            q.prefetch(req)
            r = thm2_row(f.replace(".pt", ""), q, S, kind)
            rows.append(r)
            print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    save(pd.DataFrame(rows), "e4_thm2_learned_pfn.csv")


# ==================================================================================================
def e5a():
    rows = []
    for pname in ("cell", "mixture"):
        cond = prior(pname)
        weak = SSL(np.r_[1, 0, -np.ones(8, int)], 2)
        X = cells(10, seed=17)
        states = support_states(weak)
        logpi = log_target(cond.log_joint, X, states, weak)
        pi = np.exp(logpi)
        free = weak.free_sites()
        pred_exact = [exact_predictive(cond.log_joint, X, states, logpi, np.array([float(c)]), C)[1] for c in range(M)]
        for m in (1, 2, 4, 8):
            blocks = [list(b) for b in itertools.combinations(free, m)]
            for kind in ("approx", "mh"):
                if kind == "mh" and m == 1:
                    continue
                t0 = time.time()
                P = kernel_block(cond, X, states, weak, blocks, kind=kind)
                pb = stationary(P)
                lpb = np.log(np.maximum(pb, 1e-300))
                pred = [exact_predictive(cond.log_joint, X, states, lpb, np.array([float(c)]), C)[1] for c in range(M)]
                rows.append(dict(prior=pname, n=10, n_free=len(free), block_size=m, K=len(free) // m, kind=kind,
                                 tv=tv(pb, pi), pred_err_max=float(np.max(np.abs(np.array(pred) - pred_exact))),
                                 seconds=round(time.time() - t0, 1)))
                print(rows[-1], flush=True)
    save(pd.DataFrame(rows), "e5a_block_small.csv")


def e5a2():
    """Như e5a nhưng khối là tập con ngẫu nhiên của MỌI mẫu (mẫu có nhãn trong khối bị loại khỏi ngữ cảnh,
    không bị lấy mẫu lại) — tách nguồn lệch thành phần khỏi hai nguồn sai của Nhận xét 1.2."""
    rows = []
    for pname in ("cell", "mixture"):
        cond = prior(pname)
        weak = SSL(np.r_[1, 0, -np.ones(8, int)], 2)
        X = cells(10, seed=17)
        states = support_states(weak)
        logpi = log_target(cond.log_joint, X, states, weak)
        pi = np.exp(logpi)
        pred_exact = [exact_predictive(cond.log_joint, X, states, logpi, np.array([float(c)]), C)[1] for c in range(M)]
        for m in (2, 4, 8):
            blocks = [list(b) for b in itertools.combinations(range(10), m)]
            t0 = time.time()
            P = kernel_block(cond, X, states, weak, blocks, kind="approx")
            pb = stationary(P)
            lpb = np.log(np.maximum(pb, 1e-300))
            pred = [exact_predictive(cond.log_joint, X, states, lpb, np.array([float(c)]), C)[1] for c in range(M)]
            rows.append(dict(prior=pname, n=10, block_size=m, K=10 // m, kind="approx, khối trên mọi mẫu",
                             tv=tv(pb, pi), pred_err_max=float(np.max(np.abs(np.array(pred) - pred_exact))),
                             seconds=round(time.time() - t0, 1)))
            print(rows[-1], flush=True)
    save(pd.DataFrame(rows), "e5a2_block_small_all.csv")


def lumped(n_lab_pos, n_lab, U, m, alpha=1.0, c=None):
    """Một ô, nhãn nhị phân. Trạng thái k = số dương trong U mẫu tự do.
    SSL (c=None): n_lab mẫu có nhãn, n_lab_pos dương. PU (c): n_lab mẫu dương có nhãn (n_lab_pos = n_lab),
    mẫu tự do có ℓ = (1, 1−c). Trả về (π chính xác, phân phối dừng của chuỗi khối cỡ m, dự đoán)."""
    n = n_lab + U
    k = np.arange(U + 1)
    lw = gammaln(U + 1) - gammaln(k + 1) - gammaln(U - k + 1) + betaln(n_lab_pos + k + alpha, n - n_lab_pos - k + alpha)
    if c is not None:
        lw = lw + k * np.log(1 - c)
    pi = np.exp(lw - logsumexp(lw))
    P = np.zeros((U + 1, U + 1))
    j = np.arange(m + 1)
    for kk in range(U + 1):
        hj = hypergeom.pmf(j, U, kk, m)
        for jj in j[hj > 0]:
            K_ctx = n_lab_pos + kk - jj
            ph = (K_ctx + alpha) / (n - m + 2 * alpha)
            r = ph if c is None else ph * (1 - c) / (ph * (1 - c) + 1 - ph)
            P[kk, kk - jj + j] += hj[jj] * binom.pmf(j, m, r)
    pb = stationary(P)
    pred = lambda w: float(w @ ((n_lab_pos + k + alpha) / (n + 2 * alpha)))
    return pi, pb, pred(pi), pred(pb)


def lumped_all(n_lab_pos, n_lab, U, Mb, alpha=1.0, c=None):
    """Như lumped(), nhưng khối là một tập con ngẫu nhiên cỡ Mb của TẤT CẢ n mẫu (kể cả mẫu có nhãn cố định):
    mẫu có nhãn trong khối bị bỏ khỏi ngữ cảnh nhưng không bị lấy mẫu lại. Ngữ cảnh khi đó là một mẫu con
    đều của toàn bộ dữ liệu ⇒ không lệch thành phần (xem RQ1_REPORT §5)."""
    n = n_lab + U
    k = np.arange(U + 1)
    lw = gammaln(U + 1) - gammaln(k + 1) - gammaln(U - k + 1) + betaln(n_lab_pos + k + alpha, n - n_lab_pos - k + alpha)
    if c is not None:
        lw = lw + k * np.log(1 - c)
    pi = np.exp(lw - logsumexp(lw))
    P = np.zeros((U + 1, U + 1))
    ls = np.arange(min(n_lab, Mb) + 1)
    pl = hypergeom.pmf(ls, n, n_lab, Mb)                      # số mẫu có nhãn trong khối
    for kk in range(U + 1):
        for l in ls[pl > 1e-14]:
            u = Mb - l
            if u > U or u < 0:
                continue
            j = np.arange(u + 1)
            hj = hypergeom.pmf(j, U, kk, u)
            ok = hj > 1e-16
            if not ok.any():
                continue
            j = j[ok]
            hj = hj[ok]
            pos_lab_in_block = 0 if c is not None else None
            # số dương có nhãn trong khối: PU ⇒ mọi mẫu có nhãn là dương; SSL ⇒ siêu bội
            if c is not None:
                lab_pos_ctx = np.array([n_lab_pos - l])
                w_lab = np.array([1.0])
            else:
                a = np.arange(min(l, n_lab_pos) + 1)
                w_lab = hypergeom.pmf(a, n_lab, n_lab_pos, l)
                lab_pos_ctx = n_lab_pos - a
            for lp, wl in zip(lab_pos_ctx, w_lab):
                if wl < 1e-16:
                    continue
                K_ctx = lp + kk - j
                ph = (K_ctx + alpha) / (n - Mb + 2 * alpha)
                r = ph if c is None else ph * (1 - c) / (ph * (1 - c) + 1 - ph)
                jp = np.arange(u + 1)
                B = binom.pmf(jp[None, :], u, r[:, None])          # (len j, u+1)
                tgt = kk - j[:, None] + jp[None, :]
                np.add.at(P[kk], tgt.ravel(), (pl[l] * wl * hj[:, None] * B).ravel())
    pb = stationary(P)
    pred = lambda w: float(w @ ((n_lab_pos + k + alpha) / (n + 2 * alpha)))
    return pi, pb, pred(pi), pred(pb)


def e5c():
    """So sánh hai cách chia khối: chỉ trên mẫu tự do ('free') và trên mọi mẫu ('all')."""
    rows = []
    for setting in ("PU", "SSL-asym"):
        for n in (50, 100, 200, 500, 1000):
            if setting == "PU":
                n_lab = max(1, int(round(0.12 * n)))
                n_lab_pos, U, c = n_lab, n - n_lab, 0.3
            else:                                          # 10% có nhãn, 30% trong số đó dương
                n_lab = max(3, n // 10)
                n_lab_pos, U, c = max(1, int(round(0.3 * n_lab))), n - n_lab, None
            for K in (20, 10, 5):
                t0 = time.time()
                pi, pb_f, pr, pr_f = lumped(n_lab_pos, n_lab, U, max(1, U // K), c=c)
                _, pb_a, _, pr_a = lumped_all(n_lab_pos, n_lab, U, max(1, n // K), c=c)
                rows.append(dict(setting=setting, n=n, K=K, tv_free=tv(pi, pb_f), tv_all=tv(pi, pb_a),
                                 pred_exact=pr, pred_err_free=abs(pr - pr_f), pred_err_all=abs(pr - pr_a),
                                 seconds=round(time.time() - t0, 1)))
                print(rows[-1], flush=True)
    save(pd.DataFrame(rows), "e5c_block_composition.csv")


def e5b():
    rows = []
    for setting in ("SSL", "PU"):
        for n in (20, 50, 100, 200, 500, 1000, 2000):
            if setting == "SSL":                     # 10% có nhãn, một nửa dương
                n_lab = max(2, n // 10)
                n_lab_pos, U, c = n_lab // 2, n - n_lab, None
            else:                                    # π_p = 0.4, c = 0.3 ⇒ 12% là dương có nhãn
                n_lab = max(1, int(round(0.12 * n)))
                n_lab_pos, U, c = n_lab, n - n_lab, 0.3
            for rule, m in [("m=2", 2), ("m=10", 10), ("K=20", max(1, U // 20)), ("K=10", max(1, U // 10)),
                            ("K=5", max(1, U // 5))]:
                if m > U:
                    continue
                t0 = time.time()
                pi, pb, pr, prb = lumped(n_lab_pos, n_lab, U, m, c=c)
                rows.append(dict(setting=setting, n=n, U=U, rule=rule, block_size=m, tv=tv(pi, pb),
                                 pred_exact=pr, pred_block=prb, pred_err=abs(pr - prb),
                                 sd_exact=float(np.sqrt(pi @ (np.arange(U + 1) ** 2) - (pi @ np.arange(U + 1)) ** 2)),
                                 sd_block=float(np.sqrt(pb @ (np.arange(U + 1) ** 2) - (pb @ np.arange(U + 1)) ** 2)),
                                 seconds=round(time.time() - t0, 1)))
                print(rows[-1], flush=True)
    save(pd.DataFrame(rows), "e5b_block_lumped.csv")


if __name__ == "__main__":
    {"e1": e1, "e1mc": e1mc, "e2": e2, "e3": e3, "e4": e4, "e5a": e5a, "e5b": e5b, "e5c": e5c, "e5a2": e5a2}[sys.argv[1]]()
