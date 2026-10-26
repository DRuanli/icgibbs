"""
weak.py — mô hình quan sát yếu p(W | y, X) dưới dạng hàm hợp lý ℓ.

Cấp mẫu: ℓ_i(y) = p(w_i | y_i = y, x_i), trả về bởi lik() dưới dạng ma trận (n, C).
Cấp nhóm (LLP): ràng buộc số đếm theo túi; lik() bằng 1 cho mọi mẫu, ràng buộc do bộ lấy mẫu xử lý.

Tham số phiền nhiễu (c của PU-SCAR, η của PU-SAR, T của nhãn nhiễu) có thể biết trước hoặc được lấy mẫu
trong cùng chuỗi qua update_params(); các bước này không cần TFM. Điều kiện: tham số phiền nhiễu độc lập
tiên nghiệm với (φ, X, y), nên Bổ đề 1 vẫn áp dụng cho bước cập nhật y.

Mọi mô hình đều hỗ trợ extend(m): thêm m "điểm tự do" (ℓ ≡ 1, không thuộc túi nào) — dùng cho chế độ
chuyển nạp (đưa các điểm test vào chuỗi) và SSL.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit


class WeakModel:
    is_group = False

    def __init__(self, n: int, n_classes: int):
        self.n, self.n_classes = int(n), int(n_classes)
        self.n_extra = 0

    # ---- giao diện -----------------------------------------------------------------------------
    def lik(self) -> np.ndarray:
        """ℓ (n_total, C) với tham số phiền nhiễu hiện tại."""
        raise NotImplementedError

    def update_params(self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> None:
        """Một bước Gibbs/MH cho tham số phiền nhiễu (mặc định: không có)."""

    def params(self) -> dict:
        return {}

    # ---- tiện ích chung -------------------------------------------------------------------------
    @property
    def n_total(self) -> int:
        return self.n + self.n_extra

    def extend(self, m: int) -> "WeakModel":
        self.n_extra += int(m)
        return self

    def _extra_rows(self) -> np.ndarray:
        return np.ones((self.n_extra, self.n_classes))

    def free_sites(self) -> np.ndarray:
        """Các mẫu có ít nhất 2 nhãn khả dĩ (các mẫu còn lại cố định, không cần cập nhật)."""
        return np.flatnonzero((self.lik() > 0).sum(1) >= 2)

    def init_labels(self, rng: np.random.Generator, P0: np.ndarray | None = None) -> np.ndarray:
        """Khởi tạo hợp lệ: y_i ~ (P0_i ·) ℓ_i chuẩn hoá. P0 là dự đoán ban đầu tuỳ chọn."""
        L = self.lik()
        W = L if P0 is None else L * P0
        W = W / W.sum(1, keepdims=True)
        return np.array([rng.choice(self.n_classes, p=w) for w in W])


# --------------------------------------------------------------------------------------------------
class SSL(WeakModel):
    """Bán giám sát: y_obs[i] = -1 nếu không nhãn."""

    def __init__(self, y_obs, n_classes: int):
        y_obs = np.asarray(y_obs, dtype=int)
        super().__init__(len(y_obs), n_classes)
        self.y_obs = y_obs

    def lik(self):
        L = np.ones((self.n, self.n_classes))
        lab = self.y_obs >= 0
        L[lab] = 0.0
        L[np.flatnonzero(lab), self.y_obs[lab]] = 1.0
        return np.vstack([L, self._extra_rows()])


class PartialLabel(WeakModel):
    """Nhãn một phần: S (n, C) bool, tập ứng viên chứa nhãn đúng.
    Giả định (chuẩn trong PLL Bayes): p(S | y, x) như nhau với mọi y ∈ S, nên ℓ_i(y) ∝ 1[y ∈ S_i]."""

    def __init__(self, S):
        S = np.asarray(S, dtype=bool)
        if not S.any(1).all():
            raise ValueError("mỗi mẫu cần ít nhất một ứng viên")
        super().__init__(S.shape[0], S.shape[1])
        self.S = S

    def lik(self):
        return np.vstack([self.S.astype(float), self._extra_rows()])


class NoisyLabel(WeakModel):
    """Nhãn nhiễu w_i với ma trận chuyển T[y, w] = p(w | y) (nhiễu phụ thuộc lớp).
    T=None: T có prior Dirichlet theo hàng (đường chéo `prior_diag`, ngoài chéo `prior_off`), cập nhật liên hợp."""

    def __init__(self, w, n_classes: int, T=None, prior_diag: float = 10.0, prior_off: float = 1.0):
        w = np.asarray(w, dtype=int)
        super().__init__(len(w), n_classes)
        self.w = w
        self.learn_T = T is None
        C = n_classes
        self.beta = np.full((C, C), prior_off) + np.eye(C) * (prior_diag - prior_off)
        self.T = (self.beta / self.beta.sum(1, keepdims=True)) if T is None else np.asarray(T, float)

    def lik(self):
        return np.vstack([self.T[:, self.w].T, self._extra_rows()])

    def update_params(self, X, y, rng):
        if not self.learn_T:
            return
        C = self.n_classes
        cnt = np.zeros((C, C))
        np.add.at(cnt, (y[: self.n], self.w), 1.0)
        self.T = np.vstack([rng.dirichlet(self.beta[k] + cnt[k]) for k in range(C)])

    def params(self):
        return {"T": self.T.copy()}


class PUSCAR(WeakModel):
    """PU, SCAR: s_i = 1 với xác suất c nếu y_i = 1, và s_i = 0 nếu y_i = 0. Lớp dương là 1.
    ℓ_i = (0, c) nếu s_i = 1; (1, 1 − c) nếu s_i = 0 (hằng số c ở hàng s=1 không ảnh hưởng cập nhật y).
    c=None: c ~ Beta(a, b), cập nhật liên hợp c | y, s ~ Beta(a + L, b + P − L), L = #(s=1), P = #(y=1).
    Lưu ý: PU-SCAR không định danh được (c và tỉ lệ lớp đánh đổi nhau); khi học c, posterior của c dựa
    vào prior của TFM về tỉ lệ lớp. Khi so sánh với nnPU/uPU (được cho biết tỉ lệ lớp), dùng c đã biết."""

    def __init__(self, s, c: float | None = None, prior=(1.0, 1.0)):
        s = np.asarray(s, dtype=int)
        super().__init__(len(s), 2)
        self.s = s
        self.learn_c = c is None
        self.prior = prior
        self.c = prior[0] / (prior[0] + prior[1]) if c is None else float(c)

    def lik(self):
        L = np.empty((self.n, 2))
        L[:, 0] = np.where(self.s == 1, 0.0, 1.0)
        L[:, 1] = np.where(self.s == 1, self.c, 1.0 - self.c)
        return np.vstack([L, self._extra_rows()])

    def update_params(self, X, y, rng):
        if self.learn_c:
            Lc = int(self.s.sum())
            P = int((y[: self.n] == 1).sum())
            self.c = float(rng.beta(self.prior[0] + Lc, self.prior[1] + P - Lc))

    def params(self):
        return {"c": self.c}


class PUSAR(WeakModel):
    """PU, SAR: s_i ~ Bernoulli(e(z_i)) nếu y_i = 1, e(z) = σ(η_0 + z^T η_{1:}). Z là các thuộc tính propensity.
    η ~ N(0, prior_scale² I); cập nhật bằng random-walk MH (n_mh bước mỗi vòng quét).
    eta=... (đã biết) thì không cập nhật."""

    def __init__(self, s, Z, eta=None, prior_scale: float = 1.0, n_mh: int = 5, step: float | None = None):
        s = np.asarray(s, dtype=int)
        super().__init__(len(s), 2)
        self.s, self.Z = s, np.column_stack([np.ones(len(s)), np.asarray(Z, float)])
        self.learn_eta = eta is None
        self.prior_scale, self.n_mh = prior_scale, n_mh
        self.eta = np.zeros(self.Z.shape[1]) if eta is None else np.asarray(eta, float)
        self.step = step
        self.n_accept = 0
        self.n_prop = 0

    def e(self):
        return np.clip(expit(self.Z @ self.eta), 1e-6, 1 - 1e-6)

    def lik(self):
        e = self.e()
        L = np.empty((self.n, 2))
        L[:, 0] = np.where(self.s == 1, 0.0, 1.0)
        L[:, 1] = np.where(self.s == 1, e, 1.0 - e)
        return np.vstack([L, self._extra_rows()])

    def _logpost(self, eta, pos):
        e = np.clip(expit(self.Z[pos] @ eta), 1e-12, 1 - 1e-12)
        s = self.s[pos]
        return float(np.sum(s * np.log(e) + (1 - s) * np.log1p(-e)) - 0.5 * np.sum(eta ** 2) / self.prior_scale ** 2)

    def update_params(self, X, y, rng):
        if not self.learn_eta:
            return
        pos = np.flatnonzero(y[: self.n] == 1)
        step = self.step or 2.0 / np.sqrt(max(len(pos), 1) * self.Z.shape[1])
        cur = self._logpost(self.eta, pos)
        for _ in range(self.n_mh):
            prop = self.eta + step * rng.standard_normal(self.eta.shape)
            new = self._logpost(prop, pos)
            self.n_prop += 1
            if np.log(rng.uniform()) < new - cur:
                self.eta, cur = prop, new
                self.n_accept += 1

    def params(self):
        return {"eta": self.eta.copy()}


# --------------------------------------------------------------------------------------------------
class LLP(WeakModel):
    """Learning from label proportions: các túi rời nhau `bags` (list mảng chỉ số), counts (G, C) là số đếm
    nhãn chính xác trong từng túi. Mẫu không thuộc túi nào là điểm tự do (ℓ ≡ 1)."""

    is_group = True

    def __init__(self, bags, counts, n: int, n_classes: int = 2):
        super().__init__(n, n_classes)
        self.bags = [np.asarray(b, dtype=int) for b in bags]
        self.counts = np.asarray(counts, dtype=int).reshape(len(self.bags), n_classes)
        for b, c in zip(self.bags, self.counts):
            if len(b) != c.sum():
                raise ValueError("số đếm của túi không khớp kích thước túi")
        member = np.full(n, -1)
        for g, b in enumerate(self.bags):
            if (member[b] >= 0).any():
                raise ValueError("các túi phải rời nhau")
            member[b] = g
        self._member = member

    @property
    def bag_of(self) -> np.ndarray:
        return np.concatenate([self._member, np.full(self.n_extra, -1)])

    def lik(self):
        return np.ones((self.n_total, self.n_classes))

    def free_sites(self):
        return np.flatnonzero(self.bag_of < 0)

    def init_labels(self, rng, P0=None):
        """P0=None: hoán vị ngẫu nhiên khớp số đếm. P0 (n_total, C): mỗi túi lấy mẫu chính xác từ
        Π_i P0[i, y_i] có điều kiện số đếm (khởi động ấm)."""
        from .llp import sample_count_constrained
        y = np.zeros(self.n_total, dtype=int)
        for b, c in zip(self.bags, self.counts):
            if P0 is None:
                y[b] = rng.permutation(np.repeat(np.arange(self.n_classes), c))
            else:
                y[b] = sample_count_constrained(np.clip(P0[b], 1e-6, None), c, rng)
        free = self.free_sites()
        if len(free):
            W = np.ones((len(free), self.n_classes)) if P0 is None else P0[free]
            W = W / W.sum(1, keepdims=True)
            y[free] = [rng.choice(self.n_classes, p=w) for w in W]
        return y

    def satisfied(self, y) -> bool:
        return all((np.bincount(y[b], minlength=self.n_classes) == c).all() for b, c in zip(self.bags, self.counts))


def make_bags(n: int, bag_size: int, rng: np.random.Generator, y=None, n_classes: int = 2):
    """Chia ngẫu nhiên {0..n-1} thành các túi cỡ bag_size (túi cuối có thể nhỏ hơn). Trả về (bags, counts)."""
    perm = rng.permutation(n)
    bags = [perm[i:i + bag_size] for i in range(0, n, bag_size)]
    counts = None if y is None else np.array([np.bincount(y[b], minlength=n_classes) for b in bags])
    return bags, counts
