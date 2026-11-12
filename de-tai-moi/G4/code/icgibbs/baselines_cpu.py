"""Baseline không dùng mạng nơ-ron, chạy bằng CPU — lượt bổ sung S3B (S3/S3B_PLAN.md, ghi trước 12/11/2026).

Cả ba đều là cài đặt riêng, cấu hình cố định (không tune), chỉ dùng nhãn yếu:
  PLkNN        PLL: bỏ phiếu k láng giềng gần nhất trên tập ứng viên (Hüllermeier & Beringer, 2006), k = 10.
  GBDTSelfPLL  PLL: cây tăng cường (HistGradientBoosting) tự huấn luyện, nhãn giả bị ràng trong tập ứng viên,
               dự đoán ngoài fold ở bước gán nhãn.
  GBDTSelfLLP  LLP nhị phân: điểm số đầu là hồi quy tỉ lệ dương của túi; sau đó như trên, nhãn giả của mỗi túi là
               c_g điểm có điểm số cao nhất (đúng số đếm của túi).
"""
from __future__ import annotations

import numpy as np

N_ROUNDS = 3      # số vòng gán lại nhãn giả
N_FOLDS = 2       # dự đoán ngoài fold khi gán nhãn
K_NN = 10


def _gbdt(seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, early_stopping=False, random_state=seed)


def _proba(model, X, n_classes):
    P = np.full((len(X), n_classes), 0.0)
    P[:, np.asarray(model.classes_, int)] = model.predict_proba(X)
    return P


def _oof(X, y, n_classes, seed):
    """Xác suất ngoài fold của GBDT huấn luyện trên nhãn giả y."""
    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(X)) % N_FOLDS
    P = np.zeros((len(X), n_classes))
    for f in range(N_FOLDS):
        tr, te = fold != f, fold == f
        if len(np.unique(y[tr])) < 2:
            P[te, int(y[tr][0])] = 1.0
            continue
        P[te] = _proba(_gbdt(seed + f).fit(X[tr], y[tr]), X[te], n_classes)
    return P


class PLkNN:
    def __init__(self, k: int = K_NN):
        self.k = k

    def fit(self, X, S):
        from sklearn.neighbors import NearestNeighbors
        self.S_ = np.asarray(S, float)
        self.nn_ = NearestNeighbors(n_neighbors=min(self.k, len(X))).fit(np.asarray(X, float))
        return self

    def predict_proba(self, X):
        idx = self.nn_.kneighbors(np.asarray(X, float), return_distance=False)
        votes = self.S_[idx].sum(1) + 1e-6
        return votes / votes.sum(1, keepdims=True)


class GBDTSelfPLL:
    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, X, S):
        X, S = np.asarray(X, float), np.asarray(S, bool)
        n, C = S.shape
        rng = np.random.default_rng(self.seed)
        y = np.array([rng.choice(np.flatnonzero(s)) for s in S])          # nhãn giả đầu: một ứng viên ngẫu nhiên
        for r in range(N_ROUNDS):
            P = _oof(X, y, C, self.seed + 10 * r) + 1e-9
            y = np.where(S, P, -1.0).argmax(1)
        self.n_classes_ = C
        self.model_ = _gbdt(self.seed).fit(X, y) if len(np.unique(y)) > 1 else None
        self.const_ = int(y[0])
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        if self.model_ is None:
            P = np.zeros((len(X), self.n_classes_))
            P[:, self.const_] = 1.0
            return P
        return _proba(self.model_, X, self.n_classes_)


class GBDTSelfLLP:
    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, X, bags, counts):
        X = np.asarray(X, float)
        counts = np.asarray(counts, int).reshape(len(bags), -1)
        if counts.shape[1] != 2:
            raise ValueError("GBDTSelfLLP chỉ cho LLP nhị phân")
        pos = counts[:, 1]
        from sklearn.ensemble import HistGradientBoostingRegressor
        prop = np.zeros(len(X))
        for b, c in zip(bags, pos):
            prop[np.asarray(b, int)] = c / max(len(b), 1)
        rng = np.random.default_rng(self.seed)
        fold = rng.permutation(len(X)) % N_FOLDS                             # điểm số đầu: hồi quy tỉ lệ của túi, ngoài fold
        score = np.zeros(len(X))
        for f in range(N_FOLDS):
            reg = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, early_stopping=False,
                                                random_state=self.seed + f).fit(X[fold != f], prop[fold != f])
            score[fold == f] = reg.predict(X[fold == f])
        covered = np.zeros(len(X), bool)
        for b in bags:
            covered[np.asarray(b, int)] = True
        if not covered.all():
            raise ValueError("mọi điểm huấn luyện phải thuộc một túi")

        def assign(sc):
            y = np.zeros(len(X), int)
            for b, c in zip(bags, pos):
                b = np.asarray(b, int)
                if c > 0:
                    y[b[np.argsort(-sc[b], kind="stable")[:c]]] = 1
            return y

        y = assign(score)
        for r in range(N_ROUNDS):
            y = assign(_oof(X, y, 2, self.seed + 10 * r)[:, 1])
        self.model_ = _gbdt(self.seed).fit(X, y) if len(np.unique(y)) > 1 else None
        self.const_ = int(y[0])
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        if self.model_ is None:
            P = np.zeros((len(X), 2))
            P[:, self.const_] = 1.0
            return P
        return _proba(self.model_, X, 2)
