"""
conditional.py — mô hình xác suất có điều kiện q(y | x, D) dùng bên trong In-Context Gibbs.

Mọi lớp con trả về ma trận (n_query, C): mỗi hàng dương (sau `floor`) và có tổng 1 (giả định A2).
Nhãn là số nguyên 0..C-1.

Các lớp
  ConditionalModel       giao diện chung, đếm số lần gọi (= số forward của TFM)
  AnalyticCellPPD        prior "ô–Dirichlet": PPD dạng đóng, x KHÔNG mang thông tin về tham số của y
  DiscreteMixturePPD     prior hỗn hợp hữu hạn: PPD chính xác, x CÓ mang thông tin (kiểm tra nguồn sai thứ hai
                         của cập nhật khối và sự khác nhau giữa dự đoán quy nạp và chuyển nạp)
  PerturbedConditional   PPD bị nhiễu có kiểm soát -> các xác suất có điều kiện không tương thích (Định lý 2)
  FunctionConditional    bọc một hàm bất kỳ (ví dụ một PFN nhỏ tự huấn luyện)
  SklearnTFMConditional  bọc TabPFN / TabICL / bất kỳ classifier kiểu scikit-learn
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from scipy.special import gammaln, logsumexp


class ConditionalModel(ABC):
    """q(y_* | x_*, D), với D = (X_ctx, y_ctx)."""

    def __init__(self, n_classes: int, floor: float = 1e-6):
        self.n_classes = int(n_classes)
        self.floor = float(floor)
        self.n_calls = 0

    @abstractmethod
    def _predict(self, X_ctx: np.ndarray, y_ctx: np.ndarray, X_query: np.ndarray) -> np.ndarray:
        ...

    def predict_proba(self, X_ctx, y_ctx, X_query) -> np.ndarray:
        self.n_calls += 1
        X_ctx = np.asarray(X_ctx)
        X_query = np.asarray(X_query)
        y_ctx = np.asarray(y_ctx, dtype=int)
        P = np.asarray(self._predict(X_ctx, y_ctx, X_query), dtype=float)
        if P.shape != (len(X_query), self.n_classes):
            raise ValueError(f"shape {P.shape}, cần {(len(X_query), self.n_classes)}")
        if self.floor > 0:
            P = np.maximum(P, self.floor)
        return P / P.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------------------------------
# Prior giải tích 1: ô–Dirichlet
# --------------------------------------------------------------------------------------------------
class AnalyticCellPPD(ConditionalModel):
    """x[:, 0] là chỉ số ô c ∈ {0..M-1}; θ_c ~ Dirichlet(α·1_C) độc lập; y | x=c ~ Categorical(θ_c).
    Phân phối của x không phụ thuộc θ, nên
        p(y = k | x = c, D) = (n_{c,k} + α) / (n_c + C α).
    """

    def __init__(self, n_classes: int, alpha: float = 1.0):
        super().__init__(n_classes, floor=0.0)
        self.alpha = float(alpha)

    def _predict(self, X_ctx, y_ctx, X_query):
        C, a = self.n_classes, self.alpha
        cells_ctx = X_ctx[:, 0].astype(int) if len(X_ctx) else np.zeros(0, dtype=int)
        out = np.empty((len(X_query), C))
        for r, c in enumerate(X_query[:, 0].astype(int)):
            counts = np.bincount(y_ctx[cells_ctx == c], minlength=C).astype(float)
            out[r] = (counts + a) / (counts.sum() + C * a)
        return out

    def log_joint(self, X: np.ndarray, y: np.ndarray) -> float:
        """log p(y_{1:n} | X) = Σ_ô log Dirichlet-multinomial (theo thứ tự)."""
        C, a = self.n_classes, self.alpha
        cells = X[:, 0].astype(int)
        total = 0.0
        for c in np.unique(cells):
            cnt = np.bincount(np.asarray(y)[cells == c], minlength=C)
            total += gammaln(C * a) - gammaln(cnt.sum() + C * a) + np.sum(gammaln(cnt + a) - gammaln(a))
        return float(total)


# --------------------------------------------------------------------------------------------------
# Prior giải tích 2: hỗn hợp hữu hạn (x mang thông tin về φ)
# --------------------------------------------------------------------------------------------------
class DiscreteMixturePPD(ConditionalModel):
    """φ ∈ {0..H-1} với trọng số w; p(x=c | φ=h) = A[h,c]; p(y=k | x=c, φ=h) = B[h,c,k].

    PPD chính xác (điều kiện hoá cả trên x_* vì x_* mang thông tin về φ):
        p(y_*=k | x_*=c, D) ∝ Σ_h w_h L_h(D) A[h,c] B[h,c,k],   L_h(D) = Π_j A[h,x_j] B[h,x_j,y_j].
    Có thể điều kiện hoá thêm trên các feature KHÔNG nhãn X_unl (dùng để tính posterior chuyển nạp).
    """

    def __init__(self, w, A, B):
        w, A, B = np.asarray(w, float), np.asarray(A, float), np.asarray(B, float)
        super().__init__(B.shape[2], floor=0.0)
        self.logw, self.logA, self.logB = np.log(w / w.sum()), np.log(A), np.log(B)

    @classmethod
    def random(cls, n_cells: int, n_classes: int = 2, n_components: int = 3, conc_x: float = 0.7,
               conc_y: float = 0.5, seed: int = 0):
        rng = np.random.default_rng(seed)
        w = rng.dirichlet(np.ones(n_components))
        A = rng.dirichlet(conc_x * np.ones(n_cells), size=n_components)
        B = rng.dirichlet(conc_y * np.ones(n_classes), size=(n_components, n_cells))
        A = np.maximum(A, 1e-3); A /= A.sum(1, keepdims=True)
        B = np.maximum(B, 1e-3); B /= B.sum(2, keepdims=True)
        return cls(w, A, B)

    def _log_post_h(self, X_ctx, y_ctx, X_unl=None):
        lp = self.logw.copy()
        if len(X_ctx):
            c = X_ctx[:, 0].astype(int)
            lp = lp + self.logA[:, c].sum(1) + self.logB[:, c, y_ctx].sum(1)
        if X_unl is not None and len(X_unl):
            lp = lp + self.logA[:, X_unl[:, 0].astype(int)].sum(1)
        return lp

    def _predict(self, X_ctx, y_ctx, X_query):
        lp = self._log_post_h(X_ctx, y_ctx)
        out = np.empty((len(X_query), self.n_classes))
        for r, c in enumerate(X_query[:, 0].astype(int)):
            lj = lp[:, None] + self.logA[:, c, None] + self.logB[:, c, :]       # (H, C)
            v = logsumexp(lj, axis=0)
            out[r] = np.exp(v - logsumexp(v))
        return out

    def predict_with_unlabeled(self, X_ctx, y_ctx, X_unl, X_query):
        """p(y_* | x_*, D, X_unl): PPD khi ngữ cảnh có thêm các hàng chỉ có feature."""
        lp = self._log_post_h(X_ctx, y_ctx, X_unl)
        out = np.empty((len(X_query), self.n_classes))
        for r, c in enumerate(X_query[:, 0].astype(int)):
            lj = lp[:, None] + self.logA[:, c, None] + self.logB[:, c, :]
            v = logsumexp(lj, axis=0)
            out[r] = np.exp(v - logsumexp(v))
        return out

    def log_joint(self, X, y) -> float:
        """log p(y | X) = log Σ_h w_h Π A B − log Σ_h w_h Π A."""
        c = X[:, 0].astype(int)
        la = self.logw + self.logA[:, c].sum(1)
        return float(logsumexp(la + self.logB[:, c, np.asarray(y)].sum(1)) - logsumexp(la))


# --------------------------------------------------------------------------------------------------
# PPD bị nhiễu (không tương thích) — kiểm chứng Định lý 2
# --------------------------------------------------------------------------------------------------
class PerturbedConditional(ConditionalModel):
    """q(y | x, D) ∝ p(y | x, D) · exp(δ g(c, n_c(D), n(D), y)), với g ∈ [-1,1] giả ngẫu nhiên, tất định theo
    (seed, ô của x, vector đếm nhãn của ô đó trong D, vector đếm nhãn của toàn bộ D). q là hàm đối xứng của D.

    Vì sao cần cả n(D): nếu nhiễu chỉ phụ thuộc (ô, số đếm trong ô) thì q vẫn là PPD của MỘT prior trao đổi
    được khác (với nhãn nhị phân, mọi xác suất có điều kiện chỉ phụ thuộc số đếm đều tương thích: f(k+1)/f(k)
    = q(1|k)/q(0|k) xác định f). Thêm phụ thuộc vào số đếm toàn cục với g ngẫu nhiên làm các xác suất có điều
    kiện không tương thích (kiểm tra bằng log_ratio_incompatibility)."""

    def __init__(self, base: AnalyticCellPPD, delta: float, seed: int = 0):
        super().__init__(base.n_classes, floor=0.0)
        self.base, self.delta, self.seed = base, float(delta), int(seed)

    def _g(self, c: int, cnt, tot) -> np.ndarray:
        rng = np.random.default_rng([self.seed, int(c)] + [int(v) for v in cnt] + [int(v) for v in tot])
        return rng.uniform(-1.0, 1.0, size=self.n_classes)

    def _predict(self, X_ctx, y_ctx, X_query):
        P = self.base._predict(X_ctx, y_ctx, X_query)
        cells_ctx = X_ctx[:, 0].astype(int) if len(X_ctx) else np.zeros(0, dtype=int)
        out = np.empty_like(P)
        for r, c in enumerate(X_query[:, 0].astype(int)):
            cnt = np.bincount(y_ctx[cells_ctx == c], minlength=self.n_classes)
            tot = np.bincount(y_ctx, minlength=self.n_classes)
            w = P[r] * np.exp(self.delta * self._g(c, cnt, tot))
            out[r] = w / w.sum()
        return out


class FunctionConditional(ConditionalModel):
    """Bọc fn(X_ctx, y_ctx, X_query) -> (n_query, C)."""

    def __init__(self, fn: Callable, n_classes: int, floor: float = 1e-6):
        super().__init__(n_classes, floor)
        self.fn = fn

    def _predict(self, X_ctx, y_ctx, X_query):
        return self.fn(X_ctx, y_ctx, X_query)


# --------------------------------------------------------------------------------------------------
# TFM thật
# --------------------------------------------------------------------------------------------------
class SklearnTFMConditional(ConditionalModel):
    """Bọc một TFM kiểu scikit-learn (TabPFNClassifier, TabICLClassifier, ...).

    factory() trả về một estimator MỚI mỗi lần gọi (giống cách đo ở G2). Mỗi lần gọi = fit trên ngữ cảnh
    + predict_proba trên truy vấn = một forward của TFM.
    - Lớp vắng mặt trong ngữ cảnh nhận xác suất `floor` rồi chuẩn hoá lại.
    - Ngữ cảnh chỉ có một lớp: TFM không fit được; trả về xác suất làm trơn kiểu Laplace
      (n_k + 1)/(n + C) cho mọi truy vấn — ghi rõ trong method spec, hiếm khi xảy ra.
    """

    def __init__(self, factory: Callable[[], object], n_classes: int, floor: float = 1e-6):
        super().__init__(n_classes, floor)
        self.factory = factory

    def _predict(self, X_ctx, y_ctx, X_query):
        C = self.n_classes
        present = np.unique(y_ctx)
        if len(present) < 2:
            cnt = np.bincount(y_ctx, minlength=C).astype(float)
            return np.tile((cnt + 1.0) / (cnt.sum() + C), (len(X_query), 1))
        est = self.factory()
        est.fit(X_ctx, y_ctx)
        P = est.predict_proba(X_query)
        out = np.zeros((len(X_query), C))
        for j, c in enumerate(np.asarray(est.classes_).astype(int)):
            out[:, c] = P[:, j]
        return out
