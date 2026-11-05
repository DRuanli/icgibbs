"""
r1a_block_bias.py — V2 W2 / R1a (05/11/2026, CPU): độ lệch của cập nhật khối KHÔNG SỬA, trên bài giải được chính xác.
Kiểm hướng chứng minh R2 (REVIEW_RESPONSE_PLAN §3.R2): total correlation của một khối dưới prior d chiều ≈ d/(4(K−1)²),
KHÔNG phụ thuộc n; và dùng Định lý 2 với nhân khối chính xác làm tham chiếu để có chặn KHÁC RỖNG (Mệnh đề 3(b) hiện rỗng).

    python r1a_block_bias.py tc       # E7a: KL(p(y_B | D_−B) ‖ Π_i p(y_i | D_−B)) chính xác, Dirichlet–multinomial, C lớp
    python r1a_block_bias.py thm2     # E7b: chuỗi gộp một ô (SSL, PU), khối trên mọi mẫu: TV(π̃, π) thật, ε̄, τ(P), chặn
    python r1a_block_bias.py decomp   # E7d (thăm dò): chặn Định lý 2 lỏng ở đâu — ε̄ so với khuyết một bước, τ₁(Z) so với khuếch đại thật
    python r1a_block_bias.py affine   # E7f (R2): trung bình dừng dạng đóng của chuỗi gộp SSL (drift affine), mọi n
    python r1a_block_bias.py llp      # E7e: LLP, bag-block không sửa, prior hai loại feature (giải chính xác trên histogram túi)
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
from rq1_exact import lumped, lumped_all  # noqa: E402

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


# ==================================================================================================
# E7e — LLP: bag-block không sửa trên bài giải được chính xác có feature
# ==================================================================================================
def _compositions(total, parts):
    """Mọi vector nguyên không âm dài `parts` có tổng `total`."""
    if parts == 1:
        return [(total,)]
    out = []
    for first in range(total + 1):
        out += [(first,) + rest for rest in _compositions(total - first, parts - 1)]
    return out


def llp_chain(B, M, s0, s1, c, alpha=1.0):
    """Mô hình: hai loại feature x ∈ {0, 1}, p_x ~ Beta(α, α) độc lập; B túi giống nhau, mỗi túi s0 điểm loại 0, s1 điểm
    loại 1, đúng c dương. Cấu hình của một túi tóm bằng k = số dương loại 0 (k ∈ [max(0, c − s1), min(s0, c)]),
    trọng số w(k) = C(s0, k)·C(s1, c − k). Do các túi giống nhau, chuỗi gộp được trên histogram h (số túi theo từng k).
    Khối = M túi chọn đều (quét khối ngẫu nhiên, như E7b). Nhân XẤP XỈ: mỗi túi trong khối rút độc lập theo quy hoạch
    động có ràng buộc số đếm với xác suất biên p̂_x của ngữ cảnh (mọi túi ngoài khối) — đúng như bag-block của bộ lấy
    mẫu. Nhân CHÍNH XÁC: rút chung mọi túi của khối từ p(k_khối | phần còn lại). Trả về (states, π, P_xấp_xỉ, P_chính_xác,
    hàm dự đoán)."""
    ks = np.arange(max(0, c - s1), min(s0, c) + 1)
    R = len(ks)
    lw = gammaln(s0 + 1) - gammaln(ks + 1) - gammaln(s0 - ks + 1) + gammaln(s1 + 1) - gammaln(c - ks + 1) \
        - gammaln(s1 - c + ks + 1)
    states = _compositions(B, R)
    index = {h: i for i, h in enumerate(states)}
    H = np.array(states, float)
    N0, N1 = B * s0, B * s1
    K0 = H @ ks
    K1 = B * c - K0
    lmult = gammaln(B + 1) - gammaln(H + 1).sum(1)
    lpi = lmult + H @ lw + betaln(K0 + alpha, N0 - K0 + alpha) + betaln(K1 + alpha, N1 - K1 + alpha)
    pi = np.exp(lpi - logsumexp(lpi))
    blocks = _compositions(M, R)
    J = np.array(blocks, float)
    SJ = J @ ks
    lmultJ = gammaln(M + 1) - gammaln(J + 1).sum(1)
    S = len(states)
    Pap, Pex = np.zeros((S, S)), np.zeros((S, S))
    for a, h in enumerate(states):
        h = np.array(h)
        for j in blocks:
            j = np.array(j)
            if np.any(j > h):
                continue
            lh = (gammaln(h + 1) - gammaln(j + 1) - gammaln(h - j + 1)).sum() - (gammaln(B + 1) - gammaln(M + 1) - gammaln(B - M + 1))
            ph = np.exp(lh)                                            # siêu bội đa biến: histogram của khối
            if ph < 1e-15:
                continue
            ctx = h - j
            K0c, N0c = float(ctx @ ks), (B - M) * s0
            K1c, N1c = (B - M) * c - K0c, (B - M) * s1
            p0, p1 = (K0c + alpha) / (N0c + 2 * alpha), (K1c + alpha) / (N1c + 2 * alpha)
            lq = lw + ks * np.log(p0) + (s0 - ks) * np.log1p(-p0) + (c - ks) * np.log(p1) + (s1 - c + ks) * np.log1p(-p1)
            lq = lq - logsumexp(lq)
            l_ap = lmultJ + J @ lq                                     # đa thức: các túi độc lập
            l_ex = lmultJ + J @ lw + betaln(K0c + SJ + alpha, N0c + M * s0 - K0c - SJ + alpha) \
                + betaln(K1c + M * c - SJ + alpha, N1c + M * s1 - K1c - M * c + SJ + alpha)
            w_ap, w_ex = np.exp(l_ap - logsumexp(l_ap)), np.exp(l_ex - logsumexp(l_ex))
            tgt = [index[tuple((ctx + jj).astype(int))] for jj in J.astype(int)]
            np.add.at(Pap[a], tgt, ph * w_ap)
            np.add.at(Pex[a], tgt, ph * w_ex)
    pred = lambda w: (float(w @ ((K0 + alpha) / (N0 + 2 * alpha))), float(w @ ((K1 + alpha) / (N1 + 2 * alpha))))
    return H, pi, Pap, Pex, pred, ks


def e7e():
    """Prior Beta(5, 5) cho mỗi p_x: với Beta(1, 1), hậu nghiệm của mô hình gộp hai loại dồn về cấu hình thuần (mọi
    dương cùng một loại) hoặc hai mode đối xứng — không đại diện (đã thử, xem W1_R1A_REPORT §5.5).
    Lưới: R = 2 (túi 4: (3, 1), c = 1) với B ∈ {20, …, 320} túi; R = 3 (túi 8: (6, 2) và (4, 4), c = 2) với
    B ∈ {20, 40, 60}. K ∈ {5, 10, 20} khối (M = B/K túi mỗi khối)."""
    rows = []
    alpha = 5.0
    grid = [("bag4_31_c1", 3, 1, 1, B) for B in (20, 40, 80, 160, 320)] \
        + [("bag8_62_c2", 6, 2, 2, B) for B in (20, 40, 60)] \
        + [("bag8_44_c2", 4, 4, 2, B) for B in (20, 40, 60)]
    for name, s0, s1, c, B in grid:
        for K in (5, 10, 20):
            M = B // K
            t0 = time.time()
            H, pi, Pap, Pex, pred, ks = llp_chain(B, M, s0, s1, c, alpha=alpha)
            pt, pe = stationary(Pap), stationary(Pex)
            eps = float(sum(pt[a] * tv(Pap[a], Pex[a]) for a in range(len(pi))))
            tz = tau1_Z(Pex, pi)
            actual = tv(pt, pi)
            w = pt @ (Pap - Pex)
            defect = 0.5 * float(np.abs(w).sum())
            (e0, e1), (a0, a1) = pred(pi), pred(pt)
            K0 = H @ ks
            if len(ks) == 2:                                           # số mode của π theo h₁ (R = 2)
                modes = int(np.sum((pi[1:-1] > pi[:-2]) & (pi[1:-1] > pi[2:])) + (pi[0] > pi[1]) + (pi[-1] > pi[-2]))
            else:
                modes = np.nan
            rows.append(dict(setting=name, alpha=alpha, bag_size=s0 + s1, B=B, n=B * (s0 + s1), K=K, bags_per_block=M,
                             states=len(pi),
                             tv_actual=actual, eps_bar=eps, defect=defect, tau1_Z=tz, bound_Z=tz * eps,
                             looseness_Z=tz * eps / actual, tv_over_eps=actual / eps,
                             check_exact_invariant=tv(pe, pi), pred_err_type0=abs(a0 - e0), pred_err_type1=abs(a1 - e1),
                             sd_K0_pi=float(np.sqrt(pi @ (K0 - pi @ K0) ** 2)), sd_K0_tilde=float(np.sqrt(pt @ (K0 - pt @ K0) ** 2)),
                             modes_pi=modes, seconds=round(time.time() - t0, 1)))
            print({kk: (round(v, 6) if isinstance(v, float) else v) for kk, v in rows[-1].items()}, flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(OUT, "e7e_llp_bag_block.csv"), index=False)
    save(pd.DataFrame(rows), "e7e_llp_bag_block.csv")


# ==================================================================================================
# E7f — R2 ở mức dự đoán: drift affine ⇒ trung bình dừng dạng đóng (W1_R1A_REPORT §5.7)
# ==================================================================================================
def ssl_affine_means(n_lab_pos, n_lab, U, M, alpha=1.0, design="all"):
    """Mô hình gộp Beta–Bernoulli, SSL (mẫu tự do không bị nghiêng). Với cả nhân chính xác (rút chung các mẫu tự do của
    khối, biết MỌI nhãn còn lại) và nhân không sửa (rút độc lập theo dự đoán của ngữ cảnh = mọi mẫu ngoài khối), kỳ vọng
    có điều kiện của số đếm sau một bước là hàm affine của k: E[k' − k | k] = A − a·k. Vì E_π[k' − k] = 0 nên trung bình
    dừng của k là đúng A/a (không xấp xỉ). design='free': khối gồm M mẫu tự do; 'all': khối là tập con đều cỡ M của mọi
    mẫu (mặc định của bộ lấy mẫu), l = số mẫu có nhãn trong khối ~ siêu bội. Trả về (E_π k, E_π̃ k)."""
    n = n_lab + U
    L = n_lab_pos
    if design == "free":
        ls, wl = np.array([0]), np.array([1.0])
    else:
        ls = np.arange(min(n_lab, M) + 1)
        wl = hypergeom.pmf(ls, n, n_lab, M)
    u = (M - ls).astype(float)
    ok = (u >= 0) & (u <= U) & (wl > 0)
    u, ls, wl = u[ok], ls[ok], wl[ok] / wl[ok].sum()
    den_ex = n - u + 2 * alpha
    A = float(wl @ (u * (L + alpha) / den_ex))
    a = float(wl @ (u / U - u * (1 - u / U) / den_ex))
    if design == "free":
        At, at = A, a                                                  # cùng ngữ cảnh n − M, cùng trung bình
    else:
        den_ap = n - M + 2 * alpha
        At = float(wl @ (u * (L - ls * L / n_lab + alpha) / den_ap))   # E[a | l] = l·L/n_lab
        at = float(wl @ (u / U - u * (1 - u / U) / den_ap))
    return A / a, At / at


def e7f():
    """(1) Kiểm dạng đóng trên các ô của E7c (so với trung bình tính từ ma trận chuyển, cùng nhân của E7b).
    (2) Tính sai lệch dự đoán dạng đóng cho n tới 10⁶ (không cần ma trận) để lấy thang theo n và K."""
    rows = []
    for setting in ("SSL-asym",):
        for n in (50, 100, 200):
            for K in (5, 10, 20):
                n_lab_pos, n_lab, U, _ = setting_params(setting, n)
                M = max(1, n // K)
                m_ex, m_ap = ssl_affine_means(n_lab_pos, n_lab, U, M, design="all")
                pi, Pex = lumped_all_exact(n_lab_pos, n_lab, U, M)
                pt = stationary(lumped_all_kernel(n_lab_pos, n_lab, U, M))
                k = np.arange(U + 1)
                mf_ex, mf_ap = ssl_affine_means(n_lab_pos, n_lab, U, max(1, U // K), design="free")
                _, pb_f, _, _ = lumped(n_lab_pos, n_lab, U, max(1, U // K))
                rows.append(dict(check="matrix", design="all", n=n, K=K, mean_pi_matrix=float(pi @ k), mean_pi_closed=m_ex,
                                 mean_tilde_matrix=float(pt @ k), mean_tilde_closed=m_ap,
                                 free_mean_tilde_matrix=float(pb_f @ k), free_mean_tilde_closed=mf_ap,
                                 free_mean_pi_closed=mf_ex))
    chk = pd.DataFrame(rows)
    save(chk, "e7f_affine_check.csv")
    rows = []
    for n in (100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000):
        for K in (5, 10, 20, 40):
            for lab_frac, pos_frac in ((0.1, 0.3), (0.1, 0.5), (0.3, 0.3), (0.02, 0.3)):
                n_lab = max(3, int(round(lab_frac * n)))
                n_lab_pos = max(1, int(round(pos_frac * n_lab)))
                U = n - n_lab
                M = max(1, n // K)
                m_ex, m_ap = ssl_affine_means(n_lab_pos, n_lab, U, M, design="all")
                rows.append(dict(n=n, K=K, lab_frac=lab_frac, pos_frac=pos_frac, mean_pi=m_ex, mean_tilde=m_ap,
                                 shift_counts=m_ap - m_ex, pred_err=abs(m_ap - m_ex) / (n + 2.0),
                                 n_times_pred_err=abs(m_ap - m_ex) * n / (n + 2.0)))
    t = pd.DataFrame(rows)
    save(t, "e7f_affine_closed_form.csv")
    # Dạng đóng gọn (W1_REVIEW_INDEPENDENT_02, Phụ lục A): E_π̃ k = U(L + α′)/(n_lab + 2α′), α′ = αK(n−1)/((K−1)n);
    # E_π k = U(L + α)/(n_lab + 2α). Kiểm với ssl_affine_means trên nhiều α, λ, p, K, n (C = 2).
    rows = []
    for alpha in (0.5, 1.0, 3.0):
        for n, lam, pp, K in [(100, 0.1, 0.3, 10), (1000, 0.05, 0.2, 10), (5000, 0.2, 0.8, 20), (400, 0.3, 0.1, 5),
                              (100000, 0.1, 0.3, 10)]:
            nl = int(round(lam * n)); L = int(round(pp * nl)); U = n - nl; M = n // K
            m_ex, m_ap = ssl_affine_means(L, nl, U, M, alpha=alpha, design="all")
            a2 = alpha * K * (n - 1) / ((K - 1) * n)
            f_ex, f_ap = U * (L + alpha) / (nl + 2 * alpha), U * (L + a2) / (nl + 2 * a2)
            asym = alpha * (1 - nl / n) * (1 - 2 * L / nl) / ((nl / n) * (K - 1) * n)
            rows.append(dict(alpha=alpha, n=n, lam=nl / n, p=L / nl, K=K, diff_pi=abs(m_ex - f_ex), diff_tilde=abs(m_ap - f_ap),
                             pred_bias_exact=(m_ap - m_ex) / (n + 2 * alpha), pred_bias_asym=asym))
    save(pd.DataFrame(rows), "e7f_alpha_prime_check.csv")
    for (lf, pf), g in t[t.n >= 1000].groupby(["lab_frac", "pos_frac"]):
        X = np.column_stack([np.ones(len(g)), np.log(g.n), np.log(g.K)])
        bcoef = np.linalg.lstsq(X, np.log(g.pred_err), rcond=None)[0]
        print(f"lab_frac={lf}, pos_frac={pf}: sai lệch dự đoán ∝ n^{bcoef[1]:.3f} · K^{bcoef[2]:.3f}")


if __name__ == "__main__":
    {"tc": e7a, "thm2": e7b, "moments": e7c, "decomp": e7d, "llp": e7e, "affine": e7f}[sys.argv[1]]()
