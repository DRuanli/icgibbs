"""
baselines_sar.py — SAR-EM (Bekker, Robberechts & Davis, ECML-PKDD 2019) cho bối cảnh PU-SAR.

Dùng **code chính thức** của tác giả (gói `sarpu`, github.com/ML-KULeuven/SAR-PU, giấy phép MIT), thuật toán
`pu_learn_sar_em` giữ nguyên. Cài trên Kaggle:
    pip install "git+https://github.com/ML-KULeuven/SAR-PU.git@6e4fc3d8c84ac3512669a4e36ffcb5086f5b42a7#subdirectory=sarpu"
Điểm duy nhất phải thích ứng: lớp LogisticRegressionPU của gói gọi `LogisticRegression(multi_class=...)`, tham số này
đã bị bỏ khỏi scikit-learn mới. Vì `pu_learn_sar_em` cho phép truyền mô hình vào, ta truyền một lớp tương đương
(cùng cách đánh trọng số theo propensity của `BasePU`, cùng solver liblinear như mặc định thời bài báo) thay vì sửa
code của họ.

Giao diện: fit(X, s, propensity_cols) với s = 1 nếu là dương có nhãn; predict_proba(X) -> p(y=1|x) dạng (n, 2);
weak_loss(X, s) = −log-likelihood trung bình của nhãn yếu theo mô hình SAR (dùng để tune, không dùng nhãn sạch).
"""
from __future__ import annotations

import numpy as np

from .tuning import LogUniform, random_search, refit_best, weak_validation_split

SAR_EM_DEFAULT = {"C_f": 1.0, "C_e": 1.0}
SAR_EM_SPACE = {"C_f": LogUniform(1e-3, 1e3), "C_e": LogUniform(1e-3, 1e3)}


def _lr_pu(C: float, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sarpu.PUmodels import BasePU

    class LogisticRegressionPUCompat(LogisticRegression, BasePU):
        """Như sarpu.PUmodels.LogisticRegressionPU, bỏ tham số multi_class (không còn trong scikit-learn mới)."""

        def fit(self, x, s, e=None, sample_weight=None):
            if e is None:
                return super().fit(x, s, sample_weight)
            Xp, Yp, Wp = self._make_propensity_weighted_data(x, s, e, sample_weight)
            return super().fit(Xp, Yp, Wp)

    return LogisticRegressionPUCompat(C=C, solver="liblinear", random_state=seed, max_iter=1000)


class SAREM:
    def __init__(self, C_f: float = 1.0, C_e: float = 1.0, max_its: int = 500, seed: int = 0):
        self.C_f, self.C_e, self.max_its, self.seed = C_f, C_e, max_its, seed

    def fit(self, X, s, propensity_cols, classification_cols=None):
        """classification_cols=None: bộ phân loại dùng mọi cột (kể cả cột propensity, như trong PU-SAR của dự án,
        nơi z là một feature thật của dữ liệu)."""
        from sarpu.pu_learning import pu_learn_sar_em
        X = np.asarray(X, float)
        s = np.asarray(s, int)
        self.propensity_cols = np.asarray(propensity_cols, int)
        cls = None
        if classification_cols is not None:
            cls = np.zeros(X.shape[1], bool)
            cls[np.asarray(classification_cols, int)] = True
        with np.errstate(divide="ignore", invalid="ignore"):   # log(0) trong log-likelihood của gói gốc
            self.f_, self.e_, self.info_ = pu_learn_sar_em(
                X, s, self.propensity_cols, classification_attributes=cls,
                classification_model=_lr_pu(self.C_f, self.seed),
                propensity_model=_lr_pu(self.C_e, self.seed),
                max_its=self.max_its)
        return self

    def predict_proba(self, X):
        p = np.clip(self.f_.predict_proba(np.asarray(X, float)), 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])

    def propensity(self, X):
        return np.clip(self.e_.predict_proba(np.asarray(X, float)), 1e-6, 1 - 1e-6)

    def weak_loss(self, X, s):
        """−log p(s | x) theo mô hình SAR: p(s=1|x) = f(x)·e(x)."""
        p1 = self.predict_proba(X)[:, 1] * self.propensity(X)
        p1 = np.clip(p1, 1e-9, 1 - 1e-9)
        s = np.asarray(s, int)
        return float(-np.mean(s * np.log(p1) + (1 - s) * np.log(1 - p1)))


def fit_sar_em_tuned(X, s, propensity_cols, n_configs: int = 20, seed: int = 0, max_its: int = 500):
    """Tune theo giao thức: 20 cấu hình, validation 20% (phân tầng theo s), chấm bằng weak_loss; fit lại trên toàn bộ."""
    X = np.asarray(X, float)
    s = np.asarray(s, int)
    itr, iva = weak_validation_split(len(s), seed, 0.2, stratify=s)

    def score(cfg):
        m = SAREM(cfg["C_f"], cfg["C_e"], max_its=max_its, seed=seed).fit(X[itr], s[itr], propensity_cols)
        return m.weak_loss(X[iva], s[iva])

    res = random_search(score, SAR_EM_SPACE, n=n_configs, seed=seed, default=SAR_EM_DEFAULT)
    model = refit_best(res, lambda cfg: SAREM(cfg["C_f"], cfg["C_e"], max_its=max_its, seed=seed)
                       .fit(X, s, propensity_cols), X)
    model.tune_ = res
    return model
