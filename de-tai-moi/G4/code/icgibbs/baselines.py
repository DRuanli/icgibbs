"""
baselines.py — các phương pháp so sánh dùng trong thí điểm G4.

TFM (cùng TFM với In-Context Gibbs, để so sánh công bằng):
  naive_tfm_*      bỏ qua mô hình quan sát: PU → coi không nhãn là âm (Elkan–Noto 2008: điểm xếp hạng giữ thứ tự
                   dưới SCAR); LLP → một lần gán ngẫu nhiên khớp số đếm; SSL → chỉ phần có nhãn; nhiễu → dùng w;
                   PLL → chọn ngẫu nhiên trong tập ứng viên.
  hard_em_tfm      cùng vòng lặp với In-Context Gibbs nhưng gán nhãn có xác suất lớn nhất (mode='argmax');
                   dự đoán bằng trạng thái cuối. Đây là cách hiện thực "EM với posterior từng mẫu độc lập" của RQ3
                   cho một TFM (TFM không huấn luyện được trên nhãn mềm).
  (stochastic EM = In-Context Gibbs nhưng dự đoán chỉ bằng mẫu cuối: InContextGibbs.predict_proba(last_only=True))

Peer-review, cài lại (thí điểm; G5 quyết định dùng cài đặt chính thức và ngân sách tune):
  NNPU   Kiryo et al., NeurIPS 2017 (nnPU) và du Plessis et al., ICML 2015 (uPU, nn=False). Loss sigmoid.
  DLLP   Ardehaly & Culotta, ICDM 2017. Cross-entropy giữa tỉ lệ túi thật và trung bình dự đoán trong túi.
"""
from __future__ import annotations

import numpy as np

from .conditional import ConditionalModel
from .sampler import InContextGibbs
from .weak import LLP, PUSCAR, WeakModel


# --------------------------------------------------------------------------------------------------
# TFM ngây thơ
# --------------------------------------------------------------------------------------------------
def naive_tfm_pu(cond: ConditionalModel, X, s, X_test, c: float | None = None):
    """Trả về (điểm xếp hạng p(s=1|x), xác suất p(y=1|x) đã hiệu chỉnh Elkan–Noto nếu biết c)."""
    P = cond.predict_proba(X, s, X_test)[:, 1]
    prob = None if c is None else np.clip(P / c, 0.0, 1.0)
    return P, prob


def naive_tfm_llp(cond: ConditionalModel, X, weak: LLP, X_test, rng, n_draws: int = 1):
    P = np.zeros((len(X_test), weak.n_classes))
    for _ in range(n_draws):
        y = weak.init_labels(rng)
        P += cond.predict_proba(X, y, X_test)
    return P / n_draws


# --------------------------------------------------------------------------------------------------
# Hard-EM với TFM
# --------------------------------------------------------------------------------------------------
def hard_em_tfm(cond: ConditionalModel, weak: WeakModel, X, X_test, n_iter: int = 10, n_blocks: int = 10,
                init=None, seed: int = 0, cond_final: ConditionalModel | None = None):
    if init is None and isinstance(weak, PUSCAR):
        init = weak.s.copy()                                 # bắt đầu từ lời giải ngây thơ
    g = InContextGibbs(cond, weak, mode="argmax", n_sweeps=n_iter, burn_in=n_iter, n_blocks=n_blocks,
                       init="lik" if init is None else init, seed=seed).run(X)
    return g.predict_proba(X, X_test, cond=cond_final, last_only=True), g


# --------------------------------------------------------------------------------------------------
# Mạng nơ-ron cho baseline peer-review
# --------------------------------------------------------------------------------------------------
def _mlp(d_in, d_out, hidden=(64, 64)):
    import torch.nn as nn
    layers, d = [], d_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


class _Standardizer:
    def fit(self, X):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        return self

    def __call__(self, X):
        return ((X - self.mu) / self.sd).astype(np.float32)


class NNPU:
    """nnPU (Kiryo et al., 2017); nn=False cho uPU (du Plessis et al., 2015).
    Thiết lập một tập huấn luyện (censoring): P = các mẫu s=1, U = toàn bộ tập huấn luyện (mẫu từ phân phối biên),
    prior = π_p = P(y=1) của tập huấn luyện (giả định đã biết, như trong bài gốc)."""

    def __init__(self, prior: float, nn_correction: bool = True, beta: float = 0.0, gamma: float = 1.0,
                 epochs: int = 200, lr: float = 1e-3, weight_decay: float = 1e-4, batch_size: int = 256,
                 hidden=(64, 64), seed: int = 0):
        self.prior, self.nn, self.beta, self.gamma = prior, nn_correction, beta, gamma
        self.epochs, self.lr, self.wd, self.bs, self.hidden, self.seed = epochs, lr, weight_decay, batch_size, hidden, seed

    def fit(self, X, s):
        import torch
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.std = _Standardizer().fit(X)
        Xs = torch.tensor(self.std(X))
        s = np.asarray(s)
        P_idx, U_idx = np.flatnonzero(s == 1), np.arange(len(s))
        self.net = _mlp(X.shape[1], 1, self.hidden)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        sig = lambda z: torch.sigmoid(-z)                          # loss sigmoid ℓ(z, +1) = σ(−z)
        n_batches = max(1, int(np.ceil(len(U_idx) / self.bs)))
        for _ in range(self.epochs):
            Up = rng.permutation(U_idx)
            Pp = rng.permutation(P_idx)
            for b in range(n_batches):
                u = Up[b * self.bs:(b + 1) * self.bs]
                p = Pp[(b * len(Pp)) // n_batches:((b + 1) * len(Pp)) // n_batches] if len(Pp) >= n_batches else Pp
                gp, gu = self.net(Xs[p]).squeeze(-1), self.net(Xs[u]).squeeze(-1)
                r_p_pos = sig(gp).mean()
                r_p_neg = sig(-gp).mean()
                r_u_neg = sig(-gu).mean()
                neg = r_u_neg - self.prior * r_p_neg
                if self.nn and neg.item() < -self.beta:
                    loss = -self.gamma * neg
                else:
                    loss = self.prior * r_p_pos + neg
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    def decision_function(self, X):
        import torch
        with torch.no_grad():
            return self.net(torch.tensor(self.std(X))).squeeze(-1).numpy()

    def predict_proba(self, X):
        z = self.decision_function(X)
        p = 1 / (1 + np.exp(-z))
        return np.column_stack([1 - p, p])


class DLLP:
    """DLLP (Ardehaly & Culotta, 2017): min Σ_túi CE(tỉ lệ thật, trung bình softmax trong túi)."""

    def __init__(self, n_classes: int = 2, epochs: int = 200, lr: float = 1e-3, weight_decay: float = 1e-4,
                 bags_per_batch: int = 8, hidden=(64, 64), seed: int = 0):
        self.C, self.epochs, self.lr, self.wd, self.bpb, self.hidden, self.seed = (
            n_classes, epochs, lr, weight_decay, bags_per_batch, hidden, seed)

    def fit(self, X, bags, counts):
        import torch
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.std = _Standardizer().fit(X)
        Xs = torch.tensor(self.std(X))
        props = torch.tensor(np.asarray(counts, float) / np.asarray(counts, float).sum(1, keepdims=True), dtype=torch.float32)
        self.net = _mlp(X.shape[1], self.C, self.hidden)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        G = len(bags)
        for _ in range(self.epochs):
            order = rng.permutation(G)
            for k in range(0, G, self.bpb):
                loss = 0.0
                for g in order[k:k + self.bpb]:
                    pbar = torch.softmax(self.net(Xs[bags[g]]), -1).mean(0).clamp_min(1e-7)
                    loss = loss - (props[g] * torch.log(pbar)).sum()
                opt.zero_grad()
                (loss / len(order[k:k + self.bpb])).backward()
                opt.step()
        return self

    def predict_proba(self, X):
        import torch
        with torch.no_grad():
            return torch.softmax(self.net(torch.tensor(self.std(X))), -1).numpy()


# --------------------------------------------------------------------------------------------------
# Khởi động ấm (warm start) cho chuỗi — không đổi phân phối mục tiêu, chỉ đổi điểm xuất phát
# --------------------------------------------------------------------------------------------------
def warm_start_pu(cond: ConditionalModel, X, s, e, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """P0 (n, 2) cho PU: điểm của TFM ngây thơ p̂(s=1|x) tính chéo theo fold (không dự đoán trên chính ngữ cảnh),
    hiệu chỉnh Elkan–Noto theo propensity: p̂(y=1|x) ≈ min(1, p̂(s=1|x) / e(x)). e: vô hướng (SCAR) hoặc vector (SAR).
    Tốn n_folds forward."""
    X, s = np.asarray(X), np.asarray(s, dtype=int)
    n = len(s)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), n_folds)
    ps = np.zeros(n)
    for f in folds:
        mask = np.ones(n, bool)
        mask[f] = False
        ps[f] = cond.predict_proba(X[mask], s[mask], X[f])[:, 1]
    p1 = np.clip(ps / np.maximum(np.broadcast_to(e, (n,)), 1e-3), 1e-3, 1 - 1e-3)
    return np.column_stack([1 - p1, p1])


def warm_start_llp(X, bags, counts, n_classes: int = 2, epochs: int = 100, seed: int = 0) -> np.ndarray:
    """P0 (n, C) cho LLP: DLLP với mô hình tuyến tính (rẻ, CPU), dùng làm điểm xuất phát cho chuỗi."""
    m = DLLP(n_classes=n_classes, epochs=epochs, hidden=(), seed=seed).fit(np.asarray(X, np.float32), bags, counts)
    return m.predict_proba(np.asarray(X, np.float32))


def warm_start_cross(cond: ConditionalModel, X, y_pseudo, fit_mask=None, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Khởi động ấm chung cho nhãn yếu cấp mẫu: dự đoán của TFM tính chéo theo fold, huấn luyện trên nhãn giả
    y_pseudo (SSL: chỉ phần có nhãn; nhiễu: nhãn quan sát; PLL: một ứng viên ngẫu nhiên). Trả về P0 (n, C);
    init_labels nhân P0 với ℓ nên các ràng buộc của nhãn yếu luôn được giữ."""
    X, y_pseudo = np.asarray(X), np.asarray(y_pseudo, dtype=int)
    n = len(X)
    fit_mask = np.ones(n, bool) if fit_mask is None else np.asarray(fit_mask, bool)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), n_folds)
    P = np.zeros((n, cond.n_classes))
    for f in folds:
        tr = fit_mask.copy()
        tr[f] = False
        P[f] = cond.predict_proba(X[tr], y_pseudo[tr], X[f])
    return P


class ForwardCorrection:
    """Forward loss correction (Patrini et al., CVPR 2017) với ma trận nhiễu T đã biết: CE(Tᵀ softmax(f(x)), w)."""

    def __init__(self, T, epochs: int = 200, lr: float = 1e-3, weight_decay: float = 1e-4, batch_size: int = 128,
                 hidden=(64, 64), seed: int = 0):
        self.T = np.asarray(T, float)
        self.epochs, self.lr, self.wd, self.bs, self.hidden, self.seed = epochs, lr, weight_decay, batch_size, hidden, seed

    def fit(self, X, w):
        import torch
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.std = _Standardizer().fit(X)
        Xs, w = torch.tensor(self.std(X)), torch.tensor(np.asarray(w, dtype=int))
        T = torch.tensor(self.T, dtype=torch.float32)
        C = self.T.shape[0]
        self.net = _mlp(X.shape[1], C, self.hidden)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        for _ in range(self.epochs):
            for b in np.array_split(rng.permutation(len(w)), max(1, len(w) // self.bs)):
                p = torch.softmax(self.net(Xs[b]), -1) @ T
                loss = -torch.log(p.gather(1, w[b, None]).clamp_min(1e-7)).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, X):
        import torch
        with torch.no_grad():
            return torch.softmax(self.net(torch.tensor(self.std(X))), -1).numpy()


class PRODEN:
    """PRODEN (Lv et al., ICML 2020): học nhãn một phần bằng trọng số ứng viên cập nhật dần theo dự đoán của mô hình."""

    def __init__(self, epochs: int = 200, lr: float = 1e-3, weight_decay: float = 1e-4, batch_size: int = 128,
                 hidden=(64, 64), seed: int = 0):
        self.epochs, self.lr, self.wd, self.bs, self.hidden, self.seed = epochs, lr, weight_decay, batch_size, hidden, seed

    def fit(self, X, S):
        import torch
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.std = _Standardizer().fit(X)
        Xs = torch.tensor(self.std(X))
        S = torch.tensor(np.asarray(S, float), dtype=torch.float32)
        W = S / S.sum(1, keepdim=True)
        self.net = _mlp(X.shape[1], S.shape[1], self.hidden)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        for _ in range(self.epochs):
            for b in np.array_split(rng.permutation(len(S)), max(1, len(S) // self.bs)):
                logp = torch.log_softmax(self.net(Xs[b]), -1)
                loss = -(W[b] * logp).sum(1).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    p = torch.softmax(self.net(Xs[b]), -1) * S[b]
                    W[b] = p / p.sum(1, keepdim=True).clamp_min(1e-12)
        return self

    def predict_proba(self, X):
        import torch
        with torch.no_grad():
            return torch.softmax(self.net(torch.tensor(self.std(X))), -1).numpy()
