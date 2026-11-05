"""
w1_drift_check.py — V2 W1 (05/11/2026, CPU): kiểm bằng số Mệnh đề drift (V2/W1_R1A_REPORT.md §2) với một "TFM" có
ánh xạ phản ứng BIẾT TRƯỚC, chạy bằng ĐÚNG bộ lấy mẫu của đề tài (InContextGibbs, khối hệ thống K = 10, khối trên mọi mẫu).

Xác suất có điều kiện giả: với truy vấn i và ngữ cảnh D (các điểm ngoài khối), gọi r(D) là tỉ lệ nhãn đúng trong các điểm
tự do của D. Sau khi nhân với ℓ_i (tập ứng viên) và chuẩn hoá, xác suất rút đúng nhãn của i bằng
    c_i = clip(Φ(r(D)) + h_i, 0,01, 0,99),     h_i ~ Đều(−h, h) cố định theo điểm (không đồng nhất ⇒ δ > 0),
phần còn lại chia đều cho các ứng viên sai. Nhãn thật chỉ dùng để dựng "TFM" giả này và để đo — đây là kiểm chứng
một mệnh đề xác suất, không phải phương pháp.

    python w1_drift_check.py [--h 0.1]            → ../../results/w1/w1_drift_runs{tag}.csv, w1_drift_summary{tag}.csv
    python w1_drift_check.py theory [--h 0.1]     → w1_theory_check{tag}.csv, w1_hitting{tag}.csv
    python w1_drift_check.py partition [--h 0.1]  → w1_partition_bound{tag}.csv
  tag = "" khi h = 0,1 (lượt đầu, có clip ở ánh xạ 0,9 và hai điểm hút) và "_h002" khi h = 0,02 (KHÔNG clip: Φ + h ∈
  [0,16; 0,97] nên (H_δ) đúng với δ = 0 cho mọi trạng thái và mọi khối — W1_REVIEW_INDEPENDENT_01, MAJOR-1).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from icgibbs.conditional import FunctionConditional  # noqa: E402
from icgibbs.sampler import InContextGibbs  # noqa: E402
from icgibbs.weak import PartialLabel  # noqa: E402

OUT = os.path.join(HERE, "..", "..", "results", "w1")
os.makedirs(OUT, exist_ok=True)

MAPS = {   # gain g(r) = Φ(r) − r
    "single_0.9": lambda r: 0.5 * (0.9 - r),
    "single_0.6": lambda r: 0.4 * (0.6 - r),
    "bistable_0.2_0.5_0.95": lambda r: -2.5 * (r - 0.2) * (r - 0.5) * (r - 0.95),
}
ATTRACTORS = {"single_0.9": [0.9], "single_0.6": [0.6], "bistable_0.2_0.5_0.95": [0.2, 0.95]}


def problem(n, C=10, q=0.6, h=0.1, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, C, n)
    S = rng.uniform(size=(n, C)) < q
    S[np.arange(n), y] = True
    het = rng.uniform(-h, h, n)
    return y, S, het


def make_cond(y, S, het, gain, C):
    free = S.sum(1) >= 2

    def fn(Xc, yc, Xq):
        ic, iq = Xc[:, 0].astype(int), Xq[:, 0].astype(int)
        f = free[ic]
        r = float(np.mean(yc[f] == y[ic][f])) if f.any() else 0.0
        c = np.clip(r + gain(r) + het[iq], 0.01, 0.99)
        P = np.full((len(iq), C), 1e-9)
        k = S[iq].sum(1) - 1                                     # số ứng viên sai
        P[S[iq]] = np.repeat((1 - c) / np.maximum(k, 1), S[iq].sum(1))
        P[np.arange(len(iq)), y[iq]] = c
        return P
    return FunctionConditional(fn, C, floor=0.0), free


def planted(y, S, r0, rng):
    z = y.copy()
    for i in range(len(y)):
        wrong = np.flatnonzero(S[i] & (np.arange(S.shape[1]) != y[i]))
        if len(wrong) and rng.uniform() >= r0:
            z[i] = rng.choice(wrong)
    return z


def block_delta(het, free, n, K=10, draws=2000, seed=0):
    """δ do không đồng nhất: phân vị 99 của |trung bình h trên các điểm tự do của một khối ngẫu nhiên cỡ n/K|."""
    rng = np.random.default_rng(seed)
    m = n // K
    v = [abs(het[b][free[b]].mean()) for b in (rng.choice(n, m, replace=False) for _ in range(draws)) if free[b].any()]
    return float(np.quantile(v, 0.99))


def _tag(h):
    return "" if abs(h - 0.1) < 1e-12 else "_h" + f"{h:.2f}".replace(".", "").lstrip("0").rjust(3, "0")


def main(h=0.1):
    rows = []
    C, K, T = 10, 10, 60
    for n in (500, 3000):
        y, S, het = problem(n, C, h=h)
        X = np.arange(n, dtype=float)[:, None]
        for name, gain in MAPS.items():
            cond, free = make_cond(y, S, het, gain, C)
            delta = block_delta(het, free, n, K)
            for r0 in (0.1, 0.3, 0.45, 0.55, 0.7, 0.95):
                for rep in range(20):
                    rng = np.random.default_rng([n, rep, int(r0 * 100)])
                    z0 = planted(y, S, r0, rng)
                    mon = lambda yy: float(np.mean(yy[free] == y[free]))
                    g = InContextGibbs(cond, PartialLabel(S), n_blocks=K, n_sweeps=T, burn_in=T, seed=int(rng.integers(1 << 30)),
                                       init=z0, monitor=mon).run(X)
                    tr = np.r_[mon(z0), g.trace_.monitor]
                    rows.append(dict(n=n, map_name=name, delta_block_q99=delta, r0=r0, rep=rep, r_start=tr[0],
                                     r_final=tr[-1], r_min_after10=tr[10:].min(), r_max_after10=tr[10:].max(),
                                     trace="/".join(f"{v:.3f}" for v in tr)))
            print(n, name, "δ̂ =", round(delta, 4), flush=True)
    d = pd.DataFrame(rows)
    clip = [float(np.mean((MAPS[m](r) + r + het > 0.99) | (MAPS[m](r) + r + het < 0.01))) for m in MAPS for r in np.linspace(0, 1, 101)]
    print("tỉ lệ (điểm, r) bị clip, lớn nhất theo ánh xạ và r:", max(clip), flush=True)
    d.to_csv(os.path.join(OUT, f"w1_drift_runs{_tag(h)}.csv"), index=False)

    def basin(r, name):
        a = ATTRACTORS[name]
        return int(np.argmin([abs(r - x) for x in a]))
    d["basin"] = [basin(r, m) for r, m in zip(d.r_final, d.map_name)]
    s = d.groupby(["n", "map_name", "r0"]).agg(delta=("delta_block_q99", "first"), r_start=("r_start", "mean"),
                                          r_final_mean=("r_final", "mean"), r_final_sd=("r_final", "std"),
                                          frac_upper_basin=("basin", "mean"),
                                          band_lo=("r_min_after10", "min"), band_hi=("r_max_after10", "max")).reset_index()
    s.to_csv(os.path.join(OUT, f"w1_drift_summary{_tag(h)}.csv"), index=False)
    pd.set_option("display.width", 200)
    print(s.round(4).to_string(index=False))


def _ode_time(g, r0, b, delta, grid=4000):
    """∫_{r0}^{b} dr / (|g(r)| − δ): số vòng quét của phương trình trôi (phần chính của Mệnh đề 4(a) ghép đoạn)."""
    r = np.linspace(r0, b, grid)
    v = np.abs(g(r)) - delta
    if np.any(v <= 0):
        return float("inf")
    return float(np.trapezoid(1.0 / v, r)) * np.sign(b - r0)


def _jacobi_iters(g, r0, b, delta, max_it=500):
    """Số lần lặp r ← r + g(r) ∓ δ (một vòng quét kiểu Jacobi, phía bất lợi của δ) để tới b."""
    r, up = float(r0), b > r0
    for it in range(1, max_it):
        r = r + g(r) - (delta if up else -delta)
        if (up and r >= b) or ((not up) and r <= b):
            return it
    return max_it


def _gs_sweeps(g, r0, b, delta, K=10, max_it=500):
    """Quét HỆ THỐNG theo khối (cái bộ lấy mẫu làm), dạng tất định: trong một vòng quét, khối k thay phần cũ (đúng với
    tỉ lệ r_đầu-vòng) bằng Φ(r hiện tại) ∓ δ: r ← r + (1/K)(Φ(r) ∓ δ − r_đầu-vòng), K lần. Điểm bất động trùng với Φ."""
    r, up = float(r0), b > r0
    for it in range(1, max_it):
        r_start = r
        for _ in range(K):
            r = r + (r + g(r) - (delta if up else -delta) - r_start) / K
        if (up and r >= b) or ((not up) and r <= b):
            return it
    return max_it


def delta_ctx(L, n_F, rho, K, N, eps):
    """Số hạng ngữ cảnh của Bổ đề 1 (bản đã sửa, W1_R1A_REPORT §2.2): K·L·m·x_N/(n_F(n_F − m)),
    x_N = √(m ln(2N/ε)/2), m = n/K. Đổi Φ(r_{−B}) sang Φ(R) trên biến cố xác suất ≥ 1 − ε."""
    n = n_F / rho
    m = n / K
    x = np.sqrt(m * np.log(2 * N / eps) / 2)
    return float(K * L * m * x / (n_F * (n_F - m)))


def theory(eta=0.1, burn=20, eps=0.01, K=10, T_sweeps=60, h_max=0.1):
    tag = _tag(h_max)
    """So với Mệnh đề 4 bản đã sửa. Trong TFM giả: δ (phần không đồng nhất ngoài h_i) = 0, chỉ trừ ảnh hưởng của clip
    (≤ 0,001); h_i có trung bình ≈ 0 nên gộp vào Φ; δ' = δ + δ_ctx. δ̂_blk (phân vị 99 của trung bình h trên một khối)
    giữ lại để tham khảo: nó là thang NHIỄU do h, không phải δ của (H)."""
    d = pd.read_csv(os.path.join(OUT, f"w1_drift_runs{tag}.csv"))
    tr = [np.array([float(v) for v in t.split("/")]) for t in d.trace]
    rows, hit = [], []
    grid = np.linspace(0, 1, 20001)
    c_w = (1 + (1 + 2 * h_max) ** 2) / 2
    for (n, name), g in d.groupby(["n", "map_name"]):
        gain, delta_blk = MAPS[name], float(g.delta_block_q99.iloc[0])
        y, S, _ = problem(n, h=h_max)
        free = S.sum(1) >= 2
        n_F, rho = int(free.sum()), float(free.mean())
        L = float(np.max(np.abs(np.gradient(grid + gain(grid), grid))))       # hằng số Lipschitz của Φ
        delta = delta_ctx(L, n_F, rho, K, T_sweeps * K, eps)
        T = [tr[i] for i in g.index]
        late = np.concatenate([t[burn:] for t in T])
        grid = np.linspace(0, 1, 20001)
        inside = grid[np.abs(gain(grid)) <= delta]                    # tập G_δ' = {r : |g(r)| ≤ δ'}
        per_start = g.assign(m=[t[burn:].mean() for t in T]).groupby("r0").m.mean()
        row = dict(n=n, map_name=name, n_F=n_F, L=L, delta_prime=delta, delta_blk_q99=delta_blk,
                   sd_noise_sweep=float(np.sqrt(0.25 / n_F)),
                   start_spread_late_mean=float(per_start.max() - per_start.min()))
        for a in ATTRACTORS[name]:
            near = late[np.abs(late - a) < 0.25]
            loc = inside[np.abs(inside - a) < 0.25]
            slope = (gain(a + 1e-4) - gain(a - 1e-4)) / 2e-4                # g'(r*) = Φ'(r*) − 1
            sd_jacobi = np.sqrt(a * (1 - a) / (n * (1 - (1 + slope) ** 2)))   # AR(1) r ← Φ(r) + nhiễu nhị thức
            sd_obs = float(np.mean([t[burn:][np.abs(t[burn:] - a) < 0.25].std() for t in T
                                    if np.sum(np.abs(t[burn:] - a) < 0.25) > 5]))
            rows.append(dict(row, attractor=a, G_delta_lo=float(loc.min()), G_delta_hi=float(loc.max()),
                             late_mean=float(near.mean()), late_q005=float(np.quantile(near, 0.005)),
                             late_q995=float(np.quantile(near, 0.995)), late_min=float(near.min()), late_max=float(near.max()),
                             sd_time_obs=sd_obs, sd_time_jacobi_pred=float(sd_jacobi), sd_obs_x_sqrt_n=sd_obs * np.sqrt(n)))
        for r0, h in g.groupby("r0"):
            a = min(ATTRACTORS[name], key=lambda x: abs(x - r0)) if name.startswith("single") else None
            if a is None:                                                # hai điểm hút: lưu vực theo phía của u = 0,5
                a = 0.95 if r0 > 0.5 else 0.2
            b = a - eta if r0 < a - eta else (a + eta if r0 > a + eta else None)
            if b is None:
                continue
            obs = []
            for i in h.index:
                t = tr[i]
                idx = np.flatnonzero(t >= b) if b > r0 else np.flatnonzero(t <= b)
                obs.append(int(idx[0]) if len(idx) else np.nan)
            gam = float(np.min(np.abs(gain(np.linspace(min(r0, b), max(r0, b), 2001)))))
            dist = abs(b - r0)
            T_a = 2 * dist / (gam - delta) if gam > delta else np.inf       # Mệnh đề 4(a) với λ = dist, γ hằng
            fail_a = float(np.exp(-rho * n_F * dist ** 2 / (c_w * T_a)) + eps) if np.isfinite(T_a) else np.nan
            hit.append(dict(n=n, map_name=name, r0=r0, target=b, delta_prime=delta, gain_at_r0=float(gain(r0)),
                            hit_obs_median=float(np.nanmedian(obs)), hit_obs_max=float(np.nanmax(obs)),
                            frac_hit=float(np.mean(np.isfinite(obs))), ode_sweeps=abs(_ode_time(gain, r0, b, delta)),
                            jacobi_sweeps=_jacobi_iters(gain, r0, b, delta), gs_sweeps=_gs_sweeps(gain, r0, b, delta),
                            T_bound_const=T_a, fail_bound_const=fail_a))
    t1, t2 = pd.DataFrame(rows), pd.DataFrame(hit)
    t1.to_csv(os.path.join(OUT, f"w1_theory_check{tag}.csv"), index=False)
    t2.to_csv(os.path.join(OUT, f"w1_hitting{tag}.csv"), index=False)
    pd.set_option("display.width", 250)
    print(t1.round(4).to_string(index=False), "\n")
    print(t2.round(3).to_string(index=False))


def partition_bound(total_fail=0.05, H=60, K=10, h_max=0.1, burn=20):
    tag = _tag(h_max)
    """Dải BẤT BIẾN của Mệnh đề 4′(b′) (W1_R1A_REPORT §2.6; không phải dải "từ mọi r_0" của (c′), vốn cần thêm γ′ > 0 và
    2⌈1/γ′⌉p) cho hai ánh xạ tuyến tính Φ = a + b·r: chọn p mỗi vòng = total_fail/(2H), chia đều cho hai số hạng của
    p(ν, λ), tính ν, λ, rồi dải [r* − (bν + λ)/(1 − b), r* + (bν + λ)/(1 − b)]. c dùng a_m = (n_F − ⌈n/K⌉)/n_F."""
    d = pd.read_csv(os.path.join(OUT, f"w1_drift_runs{tag}.csv"))
    rows = []
    for (n, name), g in d.groupby(["n", "map_name"]):
        if not name.startswith("single"):
            continue
        _, S, _ = problem(n, h=h_max)
        free = S.sum(1) >= 2
        n_F, rho = int(free.sum()), float(free.mean())
        bslope = 1 + (MAPS[name](0.5) - MAPS[name](0.4)) / 0.1          # Φ' = 1 + g'
        a0 = MAPS[name](0.0)                                             # Φ(0) = a
        r_star = a0 / (1 - bslope)
        a_m = (n_F - int(np.ceil(n / K))) / n_F
        c = a_m ** 2 * min(2 / 9, 1 / (18 * h_max ** 2))
        p = total_fail / (2 * H)
        lam = np.sqrt(np.log(2 / p) / (2 * n_F))
        eps = np.sqrt(np.log(6 * K / p) / (c * rho * n_F))
        half = (bslope * eps + lam) / (1 - bslope)
        late = np.concatenate([np.array([float(v) for v in t.split("/")])[burn:] for t in g.trace])
        rows.append(dict(n=n, map_name=name, n_F=n_F, c=c, p_per_sweep=p, eps=eps, lam=lam, r_star=r_star,
                         band_lo=max(0.0, r_star - half), band_hi=min(1.0, r_star + half),
                         obs_q005=float(np.quantile(late, 0.005)), obs_q995=float(np.quantile(late, 0.995)),
                         n_F_for_halfwidth_0_05=float((half * np.sqrt(n_F) / 0.05) ** 2)))   # ε, λ ∝ 1/√n_F khi p cố định
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, f"w1_partition_bound{tag}.csv"), index=False)
    pd.set_option("display.width", 250)
    print(t.round(4).to_string(index=False))


def _solve_tail(tail, target, lo=1e-6, hi=1.0):
    """Nghiệm x nhỏ nhất với tail(x) ≤ target (tail giảm theo x), tìm nhị phân."""
    if tail(hi) > target:
        return np.inf
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if tail(mid) > target else (lo, mid)
    return hi


def refined_band(n_F, rho, K, h_max, a, b, p_sweep, method):
    """Nửa dải bất biến của Mệnh đề 4′(b′) cho Φ = a + b·r (δ = 0, không điểm xấu, ngữ cảnh đầy đủ), theo ba cách lấy
    hằng số của Bổ đề 3 (W1_R1A_REPORT §2.7):
      'hoeffding3' — bản gốc: chia ν đều cho ba mảnh, cộng trên K khối cho cả ba;
      'split'      — Hoeffding, chia xác suất thất bại đều cho các mảnh (không phải chia ν tối ưu), Ville cho mảnh Z;
      'bernstein'  — như 'split' nhưng Bernstein/Freedman với phương sai một phía (phía sàn): v = max c(1−c) trên
                     c ∈ [Φ(lo − ν) − h_max, 1] và max R(1−R) trên [lo, 1], lặp tới điểm bất động (W1_R1A_REPORT §2.7)."""
    n = n_F / rho
    a_m = (n_F - np.ceil(n / K)) / n_F
    r_star = a / (1 - b)
    q = p_sweep / 4                                                     # chia p cho λ, Z, o, h
    if method == "hoeffding3":
        c = a_m ** 2 * min(2 / 9, 1 / (18 * h_max ** 2))
        lam = np.sqrt(np.log(2 / p_sweep) / (2 * n_F))
        nu = np.sqrt(np.log(6 * K / p_sweep) / (c * rho * n_F))
        return (b * nu + lam) / (1 - b), nu, lam
    v = 0.25
    for _ in range(50):
        if method == "split":
            tz = lambda x: np.exp(-2 * a_m ** 2 * n_F * x ** 2)
            to = lambda x: K * np.exp(-2 * a_m ** 2 * rho * n_F * x ** 2)
            tl = lambda x: np.exp(-2 * n_F * x ** 2)
        else:
            vz = v                                                       # Σ c(1−c)/n_F ≤ v; phương sai tổng thể của o ≤ ρ·v
            tz = lambda x: np.exp(-(x * a_m * n_F) ** 2 / (2 * (n_F * vz + x * a_m * n_F / 3)))
            to = lambda x: K * np.exp(-(x * a_m * n_F) ** 2 / (2 * (n_F * vz + x * a_m * n_F / 3)))
            tl = lambda x: np.exp(-n_F * x ** 2 / (2 * (vz + x / 3)))
        th = lambda x: K * np.exp(-a_m ** 2 * rho * n_F * x ** 2 / (2 * h_max ** 2))
        nu = _solve_tail(tz, q) + _solve_tail(to, q) + _solve_tail(th, q)
        lam = _solve_tail(tl, q)
        half = (b * nu + lam) / (1 - b)
        if method == "split":
            break
        # Phía sàn (IR2-MAJOR-1): biến cố sàn chỉ cho r_k ≥ lo − ν và R_s ≥ lo. Theo Jensen, Σ c_i(1−c_i) ≤ f·c̄(1−c̄)
        # với c̄ ∈ [Φ(lo − ν) − h_max, 1]; phương sai tổng thể của o là ρ·R_s(1−R_s) với R_s ∈ [lo, 1].
        lo = max(0.0, r_star - half)
        c_lo = max(0.0, a + b * (lo - nu) - h_max)
        cc = np.linspace(c_lo, 1.0, 401)
        rr = np.linspace(lo, 1.0, 401)
        v_new = float(max(np.max(rr * (1 - rr)), np.max(cc * (1 - cc))))
        if abs(v_new - v) < 1e-6:
            break
        v = v_new
    return half, nu, lam


def refined_table(total_fail=0.05, H=60, K=10, h_max=0.02):
    rows = []
    for name, (a, b) in {"single_0.9": (0.45, 0.5), "single_0.6": (0.24, 0.6)}.items():
        for n_F in (500, 3000, 10000, 30000):
            for method in ("hoeffding3", "split", "bernstein"):
                half, nu, lam = refined_band(n_F, 1.0, K, h_max, a, b, total_fail / (2 * H), method)
                rows.append(dict(map_name=name, n_F=n_F, method=method, nu=nu, lam=lam, halfwidth=half))
    t = pd.DataFrame(rows)
    need = []
    for (name, method), g in t.groupby(["map_name", "method"]):
        a, b = {"single_0.9": (0.45, 0.5), "single_0.6": (0.24, 0.6)}[name]
        grid = np.unique(np.round(np.geomspace(300, 3e5, 400)))
        ok = [nf for nf in grid if refined_band(nf, 1.0, K, h_max, a, b, total_fail / (2 * H), method)[0] <= 0.05]
        need.append(dict(map_name=name, method=method, n_F_for_halfwidth_0_05=float(ok[0]) if ok else np.nan))
    t = t.merge(pd.DataFrame(need), on=["map_name", "method"])
    t.to_csv(os.path.join(OUT, "w1_partition_bound_refined.csv"), index=False)
    pd.set_option("display.width", 200)
    print(t.round(4).to_string(index=False))


def sweep_moments(S, n, K, c0, b):
    """Mệnh đề 6 (W1_R1A_REPORT §2.8). Lịch quét thật, ρ = 1, h = 0, Φ(r) = c0 + b·r không bị cắt, K khối bằng nhau
    cỡ m = n/K. Cho S = n·R_s, trả về E[R_{s+1}] và Var[R_{s+1}] CHÍNH XÁC. Cách tính: Z = (N_1..N_K, O_1..O_K), với
    O_j là số đúng cũ của khối j (siêu bội đa biến) và N_k là số đúng mới. N_k = m·c0 + g_k·Z + ε_k là affine theo các
    biến trước nó cộng một nhiễu có kỳ vọng có điều kiện 0 và phương sai có điều kiện m·c_k(1 − c_k), bậc hai theo Z.
    Vì vậy vector trung bình và ma trận hiệp phương sai lan truyền chính xác qua từng khối."""
    m = n // K
    assert m * K == n
    p = S / n
    d = 2 * K
    mu = np.zeros(d)
    Sig = np.zeros((d, d))
    mu[K:] = m * p
    Sig[K:, K:] = -m * m * p * (1 - p) / (n - 1)
    Sig[np.arange(K, d), np.arange(K, d)] = m * p * (1 - p) * (n - m) / (n - 1)
    for k in range(K):
        w = np.zeros(d)
        w[:k] = 1.0
        w[K + k + 1:] = 1.0
        g = m * b / (n - m) * w
        mu_r, var_r = w @ mu / (n - m), w @ Sig @ w / (n - m) ** 2
        Ec = c0 + b * mu_r
        Veps = m * (Ec - (b * b * var_r + Ec * Ec))
        mu[k] = m * c0 + g @ mu
        row = g @ Sig
        Sig[k, :], Sig[:, k] = row, row
        Sig[k, k] = g @ Sig @ g + Veps
    one = np.zeros(d)
    one[:K] = 1.0 / n
    return float(one @ mu), float(one @ Sig @ one)


def sweep_stationary(n, K, c0, b):
    """E[R'|R] affine, Var[R'|R] bậc hai theo R (kiểm lại ở điểm thứ tư) ⇒ trung bình và phương sai dừng giải đúng:
    μ = α0/(1 − α1), σ² = (q0 + q1 μ + q2 μ²)/(1 − α1² − q2)."""
    R = np.array([0.2, 0.5, 0.8])
    EV = np.array([sweep_moments(r * n, n, K, c0, b) for r in R])
    a1, a0 = np.polyfit(R, EV[:, 0], 1)
    q2, q1, q0 = np.polyfit(R, EV[:, 1], 2)
    e4, v4 = sweep_moments(0.37 * n, n, K, c0, b)
    assert abs(e4 - (a0 + a1 * 0.37)) < 1e-9 and abs(v4 - (q0 + q1 * 0.37 + q2 * 0.37 ** 2)) < 1e-9
    mu = a0 / (1 - a1)
    var = (q0 + q1 * mu + q2 * mu * mu) / (1 - a1 ** 2 - q2)
    return dict(alpha0=a0, alpha1=a1, q0=q0, q1=q1, q2=q2, mean=mu, sd=float(np.sqrt(var)))


def second_order(h=0.02, burn=20, K=10):
    d = pd.read_csv(os.path.join(OUT, f"w1_drift_runs{_tag(h)}.csv"))
    rows = []
    for (n, name), g in d.groupby(["n", "map_name"]):
        if not name.startswith("single"):
            continue
        b = 1 + (MAPS[name](0.5) - MAPS[name](0.4)) / 0.1
        c0 = MAPS[name](0.0)
        st = sweep_stationary(n, K, c0, b)
        T = [np.array([float(v) for v in t.split("/")]) for t in g.trace]
        late = np.concatenate([t[burn:] for t in T])
        L = len(T[0]) - burn
        r = c0 / (1 - b)
        rows.append(dict(n=n, map_name=name, mean_exact=st["mean"], mean_obs=float(late.mean()), sd_exact=st["sd"],
                         sd_obs_pooled=float(late.std()), ratio=float(late.std()) / st["sd"],
                         sd_jacobi=float(np.sqrt(r * (1 - r) / (n * (1 - b * b)))), lag1_exact=st["alpha1"],
                         lag1_exact_finiteT=st["alpha1"] - (1 + 4 * st["alpha1"]) / L,   # chệch của ước lượng với chuỗi dài L
                         lag1_obs=float(np.mean([np.corrcoef(t[burn:-1], t[burn + 1:])[0, 1] for t in T])),
                         band99_gauss_lo=st["mean"] - 2.576 * st["sd"], band99_gauss_hi=st["mean"] + 2.576 * st["sd"],
                         band99_obs_lo=float(np.quantile(late, 0.005)), band99_obs_hi=float(np.quantile(late, 0.995))))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, f"w1_second_order{_tag(h)}.csv"), index=False)
    pd.set_option("display.width", 250)
    print(t.round(5).to_string(index=False))


def context_cap(n=3000, h=0.02, reps=10, K=10, T=60, burn=20):
    """Kiểm Nhận xét 7 / Bổ đề 3 bản có max_context (W1_R1A_REPORT §2.6): cùng TFM giả (không clip), nhưng ngữ cảnh
    mỗi khối là mẫu con đều cỡ M_c của [n]∖B (đúng như sampler._ctx). Đo độ rộng dải (sd theo thời gian) và lưu vực
    sát u. Lý thuyết: chặn (mỗi khối, xấu nhất) bão hoà ở cỡ M_c; trung bình qua K khối thì phương sai thêm vào chỉ
    cỡ Φ′²·r(1−r)/(K·M_c), nên hiệu ứng thật nhỏ hơn chặn."""
    C = 10
    y, S, het = problem(n, C, h=h)
    X = np.arange(n, dtype=float)[:, None]
    rows = []
    for name in ("single_0.9", "bistable_0.2_0.5_0.95"):
        cond, free = make_cond(y, S, het, MAPS[name], C)
        mon = lambda yy, free=free: float(np.mean(yy[free] == y[free]))
        for mc in (100, 300, 1000, None):
            for r0 in (0.1, 0.45, 0.55, 0.95):
                for rep in range(reps):
                    rng = np.random.default_rng([n, rep, int(r0 * 100), 7])
                    z0 = planted(y, S, r0, rng)
                    g = InContextGibbs(cond, PartialLabel(S), n_blocks=K, n_sweeps=T, burn_in=T, max_context=mc,
                                       seed=int(rng.integers(1 << 30)), init=z0, monitor=mon).run(X)
                    tr = np.r_[mon(z0), g.trace_.monitor]
                    rows.append(dict(map_name=name, max_context=mc if mc else n, r0=r0, rep=rep, r_final=tr[-1],
                                     sd_time=float(tr[burn:].std()), mean_late=float(tr[burn:].mean()),
                                     upper=int(tr[-1] > 0.5)))
            print(name, mc, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, f"w1_context_cap_runs{_tag(h)}.csv"), index=False)
    s = d.groupby(["map_name", "max_context", "r0"]).agg(mean_late=("mean_late", "mean"), sd_time=("sd_time", "mean"),
                                                         frac_upper=("upper", "mean")).reset_index()
    s.to_csv(os.path.join(OUT, f"w1_context_cap{_tag(h)}.csv"), index=False)
    pd.set_option("display.width", 200)
    print(s.round(4).to_string(index=False))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "theory", "partition", "context", "refined", "second"])
    ap.add_argument("--h", type=float, default=0.1)
    a = ap.parse_args()
    {"run": lambda: main(h=a.h), "theory": lambda: theory(h_max=a.h), "partition": lambda: partition_bound(h_max=a.h),
     "context": lambda: context_cap(h=a.h), "refined": lambda: refined_table(h_max=a.h),
     "second": lambda: second_order(h=a.h)}[a.cmd]()
