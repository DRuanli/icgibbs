"""
r1a_block_bias.py — V2 W2 / R1a (05/11/2026, CPU): độ lệch của cập nhật khối KHÔNG SỬA, trên bài giải được chính xác.
Kiểm hướng chứng minh R2 (REVIEW_RESPONSE_PLAN §3.R2): total correlation của một khối dưới prior d chiều ≈ d/(4(K−1)²),
KHÔNG phụ thuộc n; và dùng Định lý 2 với nhân khối chính xác làm tham chiếu để có chặn KHÁC RỖNG (Mệnh đề 3(b) hiện rỗng).

    python r1a_block_bias.py tc       # E7a: KL(p(y_B | D_−B) ‖ Π_i p(y_i | D_−B)) chính xác, Dirichlet–multinomial, C lớp
    python r1a_block_bias.py thm2     # E7b: chuỗi gộp một ô (SSL, PU), khối trên mọi mẫu: TV(π̃, π) thật, ε̄, τ(P), chặn
    python r1a_block_bias.py decomp   # E7d (thăm dò): chặn Định lý 2 lỏng ở đâu — ε̄ so với khuyết một bước, τ₁(Z) so với khuếch đại thật
    python r1a_block_bias.py moments  # E7c (thăm dò): π̃ lệch π ở đâu — độ rộng hay vị trí — và vì sao TV không giảm theo n
Kết quả: ../../results/r1a/*.csv
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import betaln, gammaln, logsumexp
from scipy.stats import betabinom, binom, hypergeom

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from icgibbs.exact import stationary, tv  # noqa: E402
from rq1_exact import lumped_all  # noqa: E402

OUT = os.path.join(HERE, "..", "..", "results", "r1a")
os.makedirs(OUT, exist_ok=True)


def save(df, name):
    path = os.path.join(OUT, name)
    df.to_csv(path, index=False)
    print(df.to_string(index=False))
    print("→", path)


# ==================================================================================================
# E7a — total correlation của một khối
# ==================================================================================================
def block_kl(alpha_post: np.ndarray, m: int) -> float:
    """KL(DirMult(m; α) ‖ Mult(m; α/A)) = KL giữa phân phối chung của m nhãn khối cho ngữ cảnh (Dirichlet–multinomial
    trên dãy) và tích các biên. Cả hai trao đổi được nên KL trên dãy = KL trên vector đếm; hệ số đa thức triệt tiêu.
    log DM(k) − log Mult(k) = lnΓ(A) − lnΓ(A+m) + Σ_c [lnΓ(α_c+k_c) − lnΓ(α_c) − k_c ln p̂_c], và biên của k_c dưới DM
    là Beta-binomial(m; α_c, A−α_c) ⇒ kỳ vọng tách theo lớp, tính chính xác."""
    a = np.asarray(alpha_post, float)
    A = a.sum()
    p = a / A
    kl = gammaln(A) - gammaln(A + m)
    k = np.arange(m + 1)
    for ac, pc in zip(a, p):
        w = betabinom.pmf(k, m, ac, A - ac)
        kl += float(w @ (gammaln(ac + k) - gammaln(ac) - k * np.log(pc)))
    return float(kl)


def block_tv_binary(alpha_post, m: int) -> float:
    a1, a0 = float(alpha_post[1]), float(alpha_post[0])
    k = np.arange(m + 1)
    return tv(betabinom.pmf(k, m, a1, a0), binom.pmf(k, m, a1 / (a0 + a1)))


def e7a():
    rows = []
    for C in (2, 3, 5, 10):
        props = np.r_[0.5, np.full(C - 1, 0.5 / (C - 1))] if C > 2 else np.array([0.6, 0.4])
        for n in (100, 300, 1000, 3000, 10000):
            for K in (5, 10, 20, 40):
                m = n // K
                N = n - m                                   # ngữ cảnh = các khối còn lại (khối trên mọi mẫu)
                counts = np.round(props * N)
                alpha = counts + 1.0                        # prior Dirichlet(1, …, 1)
                kl = block_kl(alpha, m)
                theory = (C - 1) / (4 * (K - 1) ** 2)
                r = dict(C=C, d=C - 1, n=n, K=K, m=m, tc_exact=kl, tc_theory=theory, ratio=kl / theory,
                         pinsker_tv=float(np.sqrt(kl / 2)))
                if C == 2:
                    r["tv_exact"] = block_tv_binary(alpha, m)
                rows.append(r)
    save(pd.DataFrame(rows), "e7a_block_tc.csv")


# ==================================================================================================
# E7b — Định lý 2 với nhân khối chính xác làm tham chiếu (chuỗi gộp một ô, khối trên mọi mẫu)
# ==================================================================================================
def lumped_all_exact(n_lab_pos, n_lab, U, Mb, alpha=1.0, c=None):
    """Khối Gibbs CHÍNH XÁC tham chiếu cho Mệnh đề 3(b): cùng cách chọn khối (tập con ngẫu nhiên cỡ Mb của mọi mẫu) với
    nhân xấp xỉ, nhưng các mẫu tự do trong khối được rút CÙNG NHAU từ phân phối có điều kiện đầy đủ p(y_{B∩F} | mọi nhãn
    còn lại) — ngữ cảnh gồm MỌI mẫu ngoài B∩F, kể cả mẫu có nhãn nằm trong khối (Beta-binomial; PU: nghiêng theo
    ℓ = (1, 1−c)). Bất biến với π (cột check_exact_invariant)."""
    n = n_lab + U
    k = np.arange(U + 1)
    lw = gammaln(U + 1) - gammaln(k + 1) - gammaln(U - k + 1) + betaln(n_lab_pos + k + alpha, n - n_lab_pos - k + alpha)
    if c is not None:
        lw = lw + k * np.log(1 - c)
    pi = np.exp(lw - logsumexp(lw))
    P = np.zeros((U + 1, U + 1))
    ls = np.arange(min(n_lab, Mb) + 1)
    pl = hypergeom.pmf(ls, n, n_lab, Mb)
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
            j, hj = j[ok], hj[ok]
            N_ctx = n - u                                              # mọi mẫu trừ các mẫu tự do của khối
            K_ctx = n_lab_pos + kk - j                                 # mọi dương có nhãn + dương tự do ngoài khối
            jp = np.arange(u + 1)
            lB = betabinom.logpmf(jp[None, :], u, (K_ctx + alpha)[:, None], (N_ctx - K_ctx + alpha)[:, None])
            if c is not None:
                lB = lB + jp[None, :] * np.log(1 - c)
            B = np.exp(lB - logsumexp(lB, axis=1, keepdims=True))
            tgt = kk - j[:, None] + jp[None, :]
            np.add.at(P[kk], tgt.ravel(), (pl[l] * hj[:, None] * B).ravel())
    return pi, P


def lumped_all_kernel(n_lab_pos, n_lab, U, Mb, alpha=1.0, c=None):
    """Nhân XẤP XỈ (tích các biên) — cùng công thức với rq1_exact.lumped_all, trả về ma trận để tính ε̄."""
    n = n_lab + U
    P = np.zeros((U + 1, U + 1))
    ls = np.arange(min(n_lab, Mb) + 1)
    pl = hypergeom.pmf(ls, n, n_lab, Mb)
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
            j, hj = j[ok], hj[ok]
            if c is not None:
                lab_pos_ctx, w_lab = np.array([n_lab_pos - l]), np.array([1.0])
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
                B = binom.pmf(jp[None, :], u, r[:, None])
                tgt = kk - j[:, None] + jp[None, :]
                np.add.at(P[kk], tgt.ravel(), (pl[l] * wl * hj[:, None] * B).ravel())
    return P


def dobrushin(P):
    S = len(P)
    best = 0.0
    for x in range(S):
        best = max(best, float(0.5 * np.abs(P[x][None, :] - P).sum(1).max()))
    return best


def tau1_Z(P, pi):
    """τ₁(Z) = max_{x,y} ½‖Z(x,·) − Z(y,·)‖₁ với Z = (I − P + 1π)⁻¹ (ma trận cơ bản): chặn chặt hơn TV(π̃, π) ≤ τ₁(Z) ε̄
    (Seneta 1988; Cho & Meyer 2001; phụ lục A.3 của bài)."""
    S = len(P)
    Z = np.linalg.inv(np.eye(S) - P + np.outer(np.ones(S), pi))
    return dobrushin(Z)


def tau(P, tol=1e-9, max_t=5000):
    """τ(P) = Σ_{t≥0} δ(P^t), δ = hệ số Dobrushin (cực đại theo cặp trạng thái, chính xác)."""
    total, Pt = 1.0, P.copy()                                    # δ(P^0) = 1
    for t in range(1, max_t):
        d = dobrushin(Pt)
        total += d
        if d < tol:
            return total, t
        Pt = Pt @ P
    return total, max_t


def setting_params(setting, n):
    if setting == "PU":                                          # π_p = 0,4, c = 0,3 ⇒ 12% dương có nhãn (như E5)
        n_lab = max(1, int(round(0.12 * n)))
        return n_lab, n_lab, n - n_lab, 0.3
    n_lab = max(3, n // 10)                                      # SSL-asym: 10% có nhãn, 30% trong số đó dương
    return max(1, int(round(0.3 * n_lab))), n_lab, n - n_lab, None


def e7b():
    """Lưới: n ≤ 200 cho cả hai bối cảnh (đủ rẻ để tính τ(P) chính xác), rồi n = 500 (chỉ τ₁(Z)). n = 1000 bỏ: nhân
    xấp xỉ một mình đã mất 7–20 phút/ô (E5c), và kết luận (chặn rỗng vì τ) đã rõ ở n nhỏ. Ghi CSV sau mỗi ô."""
    rows = []
    grid = [(s, n) for n in (50, 100, 200) for s in ("SSL-asym", "PU")] + [("SSL-asym", 500), ("PU", 500)]
    for setting, n in grid:
        for K in (5, 10, 20):
            t0 = time.time()
            n_lab_pos, n_lab, U, c = setting_params(setting, n)
            Mb = max(1, n // K)
            pi, Pex = lumped_all_exact(n_lab_pos, n_lab, U, Mb, c=c)
            Pap = lumped_all_kernel(n_lab_pos, n_lab, U, Mb, c=c)
            pi_ex = stationary(Pex)
            pi_ap = stationary(Pap)
            if n <= 200:                                                 # đối chiếu với nhân đã công bố (E5c)
                _, pb_ref, _, _ = lumped_all(n_lab_pos, n_lab, U, Mb, c=c)
                same = tv(pi_ap, pb_ref)
            else:
                same = float("nan")
            eps = float(sum(pi_ap[k] * tv(Pap[k], Pex[k]) for k in range(U + 1)))
            ta, t_used = tau(Pex) if U <= 200 else (float("nan"), 0)        # τ(P) đắt; τ₁(Z) đủ cho n lớn
            tz = tau1_Z(Pex, pi)
            actual = tv(pi_ap, pi)
            k = np.arange(U + 1)
            pred = lambda w: float(w @ ((n_lab_pos + k + 1.0) / (n + 2.0)))
            rows.append(dict(setting=setting, n=n, K=K, block=Mb, tv_actual=actual,
                             pred_err=abs(pred(pi_ap) - pred(pi)),
                             check_exact_invariant=tv(pi_ex, pi), check_same_as_e5c=same,
                             eps_bar=eps, tau=ta, t_steps=t_used, bound_tau=ta * eps, tau1_Z=tz, bound_Z=tz * eps,
                             looseness_Z=tz * eps / actual, seconds=round(time.time() - t0, 1)))
            print({kk: (round(v, 6) if isinstance(v, float) else v) for kk, v in rows[-1].items()}, flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(OUT, "e7b_thm2_block_bound.csv"), index=False)
    save(pd.DataFrame(rows), "e7b_thm2_block_bound.csv")


def e7c():
    """So hai mô-men đầu của π̃ (nhân khối xấp xỉ) và π (hậu nghiệm chính xác) trên số dương tự do k, cùng nhân của E7b.
    Mục đích: tách TV(π̃, π) thành phần do lệch ĐỘ RỘNG (sd) và phần do lệch VỊ TRÍ (trung bình). Phần sau là thứ duy
    nhất tác động lên dự đoán (xác suất dương của một điểm mới tuyến tính theo k)."""
    rows = []
    for setting in ("SSL-asym", "PU"):
        for n in (50, 100, 200):
            for K in (5, 10, 20):
                n_lab_pos, n_lab, U, c = setting_params(setting, n)
                pi, Pex = lumped_all_exact(n_lab_pos, n_lab, U, max(1, n // K), c=c)
                pt = stationary(lumped_all_kernel(n_lab_pos, n_lab, U, max(1, n // K), c=c))
                k = np.arange(U + 1)
                m0, m1 = float(pi @ k), float(pt @ k)
                s0, s1 = float(np.sqrt(pi @ (k - m0) ** 2)), float(np.sqrt(pt @ (k - m1) ** 2))
                # TV giữa hai phân phối chỉ lệch độ rộng (cùng hình dạng π, co giãn quanh m0) và chỉ lệch vị trí
                rows.append(dict(setting=setting, n=n, K=K, U=U, mean_pi=m0, mean_tilde=m1, sd_pi=s0, sd_tilde=s1,
                                 sd_ratio_minus_1=s1 / s0 - 1, mean_shift_counts=m1 - m0, mean_shift_in_sd=(m1 - m0) / s0,
                                 tv=tv(pt, pi), pred_err=abs(m1 - m0) / (n + 2.0),
                                 K_times_sd_excess=K * (s1 / s0 - 1), K_times_mean_shift=K * (m1 - m0)))
    save(pd.DataFrame(rows), "e7c_moments.csv")


def e7d():
    """π̃ − π = w Z với w = π̃(P̃ − P) (khuyết một bước của π̃ dưới nhân chính xác, tổng bằng 0) và Z = (I − P + 1π)⁻¹.
    Vì vậy TV(π̃, π) = ½‖wZ‖₁ ≤ τ₁(Z)·½‖w‖₁ ≤ τ₁(Z)·ε̄. Tách độ lỏng của chặn Định lý 2 thành hai thừa số:
      triệt_tiêu  = ε̄ / ½‖w‖₁    (sai số theo hàng triệt tiêu khi lấy trung bình theo π̃),
      khuếch_đại  = τ₁(Z) / (TV / ½‖w‖₁)   (τ₁(Z) tính trường hợp xấu nhất; khuếch đại thật nhỏ hơn).
    Chặn τ₁(Z)·½‖w‖₁ hợp lệ nhưng cần biết π̃ (hậu kiểm, không phải chặn tiên nghiệm). P chính xác thuận nghịch nên
    phân tích phổ được: relax2 = 1/(1 − λ₂) tính theo bước khối."""
    rows = []
    for setting in ("SSL-asym", "PU"):
        for n in (50, 100, 200):
            for K in (5, 10, 20):
                n_lab_pos, n_lab, U, c = setting_params(setting, n)
                Mb = max(1, n // K)
                pi, P = lumped_all_exact(n_lab_pos, n_lab, U, Mb, c=c)
                Pt = lumped_all_kernel(n_lab_pos, n_lab, U, Mb, c=c)
                pt = stationary(Pt)
                w = pt @ (Pt - P)
                defect = 0.5 * float(np.abs(w).sum())
                eps = float(sum(pt[k] * tv(Pt[k], P[k]) for k in range(U + 1)))
                tz = tau1_Z(P, pi)
                d = np.sqrt(pi)
                S = (d[:, None] * P) / d[None, :]
                lam = np.sort(np.linalg.eigvalsh((S + S.T) / 2))[::-1]
                actual = tv(pt, pi)
                rows.append(dict(setting=setting, n=n, K=K, tv_actual=actual, eps_bar=eps, defect=defect,
                                 cancellation=eps / defect, amplification_actual=actual / defect, tau1_Z=tz,
                                 relax2=1 / (1 - lam[1]), bound_Z_eps=tz * eps, bound_Z_defect=tz * defect,
                                 looseness_eps=tz * eps / actual, looseness_defect=tz * defect / actual))
    save(pd.DataFrame(rows), "e7d_bound_decomposition.csv")


if __name__ == "__main__":
    {"tc": e7a, "thm2": e7b, "moments": e7c, "decomp": e7d}[sys.argv[1]]()
