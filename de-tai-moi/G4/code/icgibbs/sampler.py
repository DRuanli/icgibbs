"""
sampler.py — In-Context Gibbs: suy luận Bayes trên nhãn ẩn, dùng một TFM có sẵn làm xác suất có điều kiện đầy đủ.

Mục tiêu:  π(y) = p(y | X, W) ∝ p(y | X) · ℓ(y).
Bổ đề 1:   p(y_i | y_{-i}, X) = PPD(y_i | x_i, D_{-i})  ⇒  q(y_i | x_i, D_{-i}) thay được cho xác suất có điều kiện đầy đủ.

Sơ đồ cập nhật (scheme)
  Cấp mẫu (PU, PLL, nhãn nhiễu, SSL)
    'single'     Gibbs một mẫu, thứ tự ngẫu nhiên mỗi vòng quét. Chính xác khi q = PPD (Định lý 1a). n forward/vòng.
    'block'      chia các mẫu tự do thành K khối ngẫu nhiên; cập nhật cả khối từ Π_i q(y_i | x_i, D_{-b}) ℓ_i(y_i).
                 Xấp xỉ (Mệnh đề 4). K forward/vòng. mh_correct=True: dùng phân phối khối làm đề xuất độc lập và
                 sửa bằng Metropolis–Hastings, tỉ số π tính qua bổ đề Brook ⇒ chính xác lại (Mệnh đề 4b),
                 thêm 1 forward cho mỗi nhãn bị đổi.
  Cấp nhóm (LLP)
    'swap'       MH hoán đổi hai nhãn khác nhau trong một túi (Định lý 1b). 2 forward/đề xuất.
    'bag-block'  chia các túi thành K nhóm; cập nhật cả nhóm, mỗi túi lấy mẫu chính xác có ràng buộc số đếm (DP).
                 K forward/vòng. mh_correct như trên.

mode='argmax' biến vòng lặp thành hard-EM (gán nhãn có xác suất lớn nhất, có ràng buộc) — baseline của RQ3.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .conditional import ConditionalModel
from .llp import sample_count_constrained, topk_assignment
from .weak import WeakModel


@dataclass
class Trace:
    n_calls: list = field(default_factory=list)      # số forward cộng dồn sau mỗi vòng quét
    flips: list = field(default_factory=list)        # tỉ lệ nhãn thay đổi trong vòng quét
    accept: list = field(default_factory=list)       # tỉ lệ chấp nhận MH trong vòng quét (nếu có)
    params: list = field(default_factory=list)       # tham số phiền nhiễu sau mỗi vòng quét
    seconds: list = field(default_factory=list)
    monitor: list = field(default_factory=list)      # giá trị monitor(y) sau mỗi vòng quét (nếu có)


class InContextGibbs:
    def __init__(self, cond: ConditionalModel, weak: WeakModel, scheme: str = "auto", n_blocks: int = 10,
                 n_sweeps: int = 30, burn_in: int | None = None, keep_every: int = 1, mh_correct: bool = False,
                 swaps_per_sweep: int | None = None, max_context: int | None = None, mode: str = "sample",
                 init: str = "lik", seed: int = 0, track=None, block_over: str = "all", monitor=None):
        if scheme == "auto":
            scheme = "bag-block" if weak.is_group else "block"
        valid = ("swap", "bag-block") if weak.is_group else ("single", "block")
        if scheme not in valid:
            raise ValueError(f"scheme {scheme!r} không hợp với mô hình quan sát này; chọn một trong {valid}")
        if mode not in ("sample", "argmax"):
            raise ValueError(mode)
        self.cond, self.weak, self.scheme = cond, weak, scheme
        self.n_blocks, self.n_sweeps = int(n_blocks), int(n_sweeps)
        self.burn_in = self.n_sweeps // 3 if burn_in is None else int(burn_in)
        self.keep_every, self.mh_correct = int(keep_every), bool(mh_correct)
        self.swaps_per_sweep, self.max_context, self.mode, self.init = swaps_per_sweep, max_context, mode, init
        self.rng = np.random.default_rng(seed)
        self.track = None if track is None else np.asarray(track, dtype=int)
        if block_over not in ("all", "free"):
            raise ValueError(block_over)
        self.monitor = monitor                # hàm(y) -> số, gọi sau mỗi vòng quét (chẩn đoán; ví dụ độ chính xác so với
                                              # nhãn thật trên dữ liệu phát triển). Không ảnh hưởng thuật toán.
        self.block_over = block_over          # 'all': khối là tập con ngẫu nhiên của MỌI mẫu (ngữ cảnh không lệch
                                              # thành phần); 'free': chỉ chia các mẫu tự do (lệch — RQ1_REPORT §5)

    # ------------------------------------------------------------------------------------------
    def _ctx(self, exclude: np.ndarray, n_total: int) -> np.ndarray:
        mask = np.ones(n_total, bool)
        mask[exclude] = False
        ctx = np.flatnonzero(mask)
        if self.max_context is not None and len(ctx) > self.max_context:
            ctx = np.sort(self.rng.choice(ctx, self.max_context, replace=False))
        return ctx

    def _q(self, X, y, exclude, query):
        ctx = self._ctx(exclude, len(y))
        return self.cond.predict_proba(X[ctx], y[ctx], X[query])

    def _draw(self, W):
        W = W / W.sum(1, keepdims=True)
        if self.mode == "argmax":
            return W.argmax(1)
        u = self.rng.uniform(size=(len(W), 1))
        return np.minimum((W.cumsum(1) < u).sum(1), W.shape[1] - 1)

    def _log_prior_ratio_brook(self, X, y_old, y_new, changed) -> float:
        """log p(y_new | X) − log p(y_old | X) qua bổ đề Brook: đổi từng nhãn một, mỗi bước một forward."""
        z = y_old.copy()
        total = 0.0
        for k in changed:
            p = self._q(X, z, np.array([k]), np.array([k]))[0]
            total += np.log(p[y_new[k]]) - np.log(p[z[k]])
            z[k] = y_new[k]
        return total

    def _record(self, sites, W, sweep):
        if self.track is None or sweep < self.burn_in:
            return
        hit = np.isin(sites, self.track)
        if hit.any():
            for s, w in zip(sites[hit], W[hit]):
                self._rb_sum[self._track_pos[s]] += w / w.sum()
                self._rb_cnt[self._track_pos[s]] += 1

    # ------------------------------------------------------------------------------------------
    def run(self, X: np.ndarray) -> "InContextGibbs":
        X = np.asarray(X)
        weak, rng = self.weak, self.rng
        n_total = weak.n_total
        if len(X) != n_total:
            raise ValueError(f"X có {len(X)} hàng, mô hình quan sát có {n_total}")
        C = weak.n_classes
        if isinstance(self.init, str):                      # 'lik': y_i ~ ℓ_i chuẩn hoá (LLP: hoán vị ngẫu nhiên)
            y = weak.init_labels(rng)
        else:
            init = np.asarray(self.init)
            if init.ndim == 2:                              # ma trận xác suất P0: khởi động ấm, y_i ~ P0_i · ℓ_i
                y = weak.init_labels(rng, P0=init)
            else:                                           # vector nhãn cho sẵn
                y = init.astype(int).copy()
        self.samples_, self.trace_ = [], Trace()
        if self.track is not None:
            self._track_pos = {int(s): k for k, s in enumerate(self.track)}
            self._rb_sum = np.zeros((len(self.track), C))
            self._rb_cnt = np.zeros(len(self.track))
        t0 = time.perf_counter()
        for sweep in range(self.n_sweeps):
            y_before = y.copy()
            acc = []
            if weak.is_group:
                y = self._sweep_group(X, y, sweep, acc)
            else:
                y = self._sweep_instance(X, y, sweep, acc)
            if self.mode == "sample":
                weak.update_params(X, y, rng)
            tr = self.trace_
            tr.n_calls.append(self.cond.n_calls)
            tr.flips.append(float(np.mean(y != y_before)))
            tr.accept.append(float(np.mean(acc)) if acc else np.nan)
            tr.params.append(weak.params())
            tr.seconds.append(time.perf_counter() - t0)
            if self.monitor is not None:
                tr.monitor.append(float(self.monitor(y)))
            if sweep >= self.burn_in and (sweep - self.burn_in) % self.keep_every == 0:
                self.samples_.append(y.copy())
                if self.track is not None and self.mh_correct:      # xác suất đề xuất không phải điều kiện đầy đủ
                    for s in self.track:
                        self._rb_sum[self._track_pos[int(s)]] += np.eye(C)[y[s]]
                        self._rb_cnt[self._track_pos[int(s)]] += 1
        self.y_ = y
        return self

    # ------------------------------------------------------------------------------------------
    def _sweep_instance(self, X, y, sweep, acc):
        L = self.weak.lik()
        free = self.weak.free_sites()
        if self.scheme == "single":
            for i in self.rng.permutation(free):
                site = np.array([i])
                W = self._q(X, y, site, site) * L[site]
                self._record(site, W, sweep)
                y[i] = self._draw(W)[0]
            return y
        pool = np.arange(len(y)) if self.block_over == "all" else free
        blocks = np.array_split(self.rng.permutation(pool), min(self.n_blocks, max(len(pool), 1)))
        is_free = np.zeros(len(y), bool)
        is_free[free] = True
        for blk in blocks:
            b = np.sort(blk[is_free[blk]])               # chỉ cập nhật mẫu tự do; cả khối bị loại khỏi ngữ cảnh
            if len(b) == 0:
                continue
            W = self._q(X, y, blk, b) * L[b]
            Wn = W / W.sum(1, keepdims=True)
            y_prop = self._draw(Wn)
            if not self.mh_correct or self.mode == "argmax":
                self._record(b, Wn, sweep)
                y[b] = y_prop
                continue
            y_new = y.copy()
            y_new[b] = y_prop
            changed = b[y_prop != y[b]]
            if len(changed) == 0:
                acc.append(1.0)
                continue
            log_r = self._log_prior_ratio_brook(X, y, y_new, changed)
            log_r += np.sum(np.log(L[changed, y_new[changed]]) - np.log(L[changed, y[changed]]))
            log_r += np.sum(np.log(Wn[np.isin(b, changed), y[changed]]) - np.log(Wn[np.isin(b, changed), y_new[changed]]))
            ok = np.log(self.rng.uniform()) < min(0.0, log_r)
            acc.append(float(ok))
            if ok:
                y = y_new
        return y

    def _sweep_group(self, X, y, sweep, acc):
        weak = self.weak
        bags, counts = weak.bags, weak.counts
        free = weak.free_sites()
        L = weak.lik()
        if self.scheme == "swap":
            n_bagged = sum(len(b) for b in bags)
            n_prop = self.swaps_per_sweep or max(1, n_bagged // 2)
            for _ in range(n_prop):
                g = self.rng.integers(len(bags))
                b = bags[g]
                yb = y[b]
                pairs = np.argwhere(yb[:, None] != yb[None, :])        # cặp có thứ tự (i, j), y_i ≠ y_j
                if len(pairs) == 0:
                    continue
                a, c = pairs[self.rng.integers(len(pairs))]
                i, j = b[a], b[c]
                yi, yj = y[i], y[j]
                p1 = self._q(X, y, np.array([i]), np.array([i]))[0]
                z = y.copy()
                z[i] = yj
                p2 = self._q(X, z, np.array([j]), np.array([j]))[0]
                log_r = (np.log(p1[yj]) - np.log(p1[yi]) + np.log(p2[yi]) - np.log(p2[yj])
                         + np.log(L[i, yj] * L[j, yi]) - np.log(L[i, yi] * L[j, yj]))
                if self.mode == "argmax":
                    ok = log_r > 0
                else:
                    ok = np.log(self.rng.uniform()) < min(0.0, log_r)
                acc.append(float(ok))
                if ok:
                    y[i], y[j] = yj, yi
            for i in self.rng.permutation(free):                          # điểm tự do: Gibbs một mẫu
                site = np.array([i])
                W = self._q(X, y, site, site) * L[site]
                self._record(site, W, sweep)
                y[i] = self._draw(W)[0]
            return y
        # 'bag-block'
        K = min(self.n_blocks, len(bags))
        bag_groups = np.array_split(self.rng.permutation(len(bags)), K)
        free_groups = np.array_split(self.rng.permutation(free), K)
        for bg, fg in zip(bag_groups, free_groups):
            sites = np.concatenate([bags[g] for g in bg] + [fg]).astype(int)
            if len(sites) == 0:
                continue
            P = self._q(X, y, sites, sites)
            row = {s: r for r, s in enumerate(sites)}
            y_new = y.copy()
            for g in bg:
                rows = [row[s] for s in bags[g]]
                Pg = P[rows] * L[bags[g]]
                if self.mode == "argmax":
                    y_new[bags[g]] = topk_assignment(Pg, counts[g])
                else:
                    y_new[bags[g]] = sample_count_constrained(Pg, counts[g], self.rng)
            if len(fg):
                rows = [row[s] for s in fg]
                W = P[rows] * L[fg]
                if not self.mh_correct:
                    self._record(fg, W, sweep)
                y_new[fg] = self._draw(W)
            if not self.mh_correct or self.mode == "argmax":
                y = y_new
                continue
            changed = sites[y_new[sites] != y[sites]]
            if len(changed) == 0:
                acc.append(1.0)
                continue
            log_r = self._log_prior_ratio_brook(X, y, y_new, changed)
            Lr = L[changed]
            log_r += np.sum(np.log(Lr[np.arange(len(changed)), y_new[changed]]) - np.log(Lr[np.arange(len(changed)), y[changed]]))
            rows = np.array([row[s] for s in changed])
            Wq = P[rows] * Lr
            Wq = Wq / Wq.sum(1, keepdims=True)            # các hằng chuẩn hoá (theo túi hoặc theo mẫu) triệt tiêu
            log_r += np.sum(np.log(Wq[np.arange(len(changed)), y[changed]]) - np.log(Wq[np.arange(len(changed)), y_new[changed]]))
            ok = np.log(self.rng.uniform()) < min(0.0, log_r)
            acc.append(float(ok))
            if ok:
                y = y_new
        return y

    # ------------------------------------------------------------------------------------------
    def predict_proba(self, X_train, X_test, cond: ConditionalModel | None = None, n_use: int | None = None,
                      last_only: bool = False) -> np.ndarray:
        """Ước lượng Rao–Blackwell (1/S) Σ_s q(y_* | x_*, (X, y^(s))) — Định lý 1(ii).
        cond: TFM dùng cho bước dự đoán cuối (ví dụ cùng TFM nhưng n_estimators lớn hơn).
        last_only=True: chỉ dùng mẫu cuối (= stochastic EM; ablation của RQ3)."""
        cond = cond or self.cond
        n = self.weak.n
        X_train = np.asarray(X_train)[:n]
        S = [self.y_] if (last_only or not self.samples_) else self.samples_
        if n_use is not None and len(S) > n_use:
            idx = np.unique(np.linspace(0, len(S) - 1, n_use).round().astype(int))
            S = [S[k] for k in idx]
        P = np.zeros((len(X_test), self.weak.n_classes))
        for ys in S:
            ctx = np.arange(n)
            if self.max_context is not None and n > self.max_context:
                ctx = np.sort(self.rng.choice(n, self.max_context, replace=False))
            P += cond.predict_proba(X_train[ctx], ys[:n][ctx], X_test)
        return P / len(S)

    def transductive_proba(self) -> np.ndarray:
        """Trung bình Rao–Blackwell của xác suất có điều kiện tại các điểm được theo dõi (track)."""
        if self.track is None:
            raise ValueError("cần track=...")
        return self._rb_sum / np.maximum(self._rb_cnt, 1)[:, None]


def fit_transductive(cond, weak, X_train, X_test, **kw):
    """Chế độ chuyển nạp: đưa X_test vào chuỗi như các điểm tự do (ℓ ≡ 1). Với scheme='single' và q = PPD,
    kết quả hội tụ về đúng p(y_* | x_*, X, W, X_test) — kể cả khi feature mang thông tin về φ."""
    m = len(X_test)
    weak.extend(m)
    track = np.arange(weak.n, weak.n + m)
    g = InContextGibbs(cond, weak, track=track, **kw).run(np.vstack([X_train, X_test]))
    return g.transductive_proba(), g
