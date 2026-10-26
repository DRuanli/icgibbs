"""
Baseline peer-review còn thiếu cho các bối cảnh PHỤ (G6, sau lượt 3 — xem docs/primary_comparators.md):

  • GLWS cho PU (Chen et al., ICML 2024) — theo `pos_ulb/imp_pos_ulb.py` của kho chính thức: CE trên mẫu dương có
    nhãn + CE với nhãn giả của "túi" không nhãn có số dương = tiên nghiệm × cỡ batch. E-step dùng hậu nghiệm CHÍNH XÁC
    (như GLWS cho LLP, giao thức §11 deviation 5).
  • VPU (Chen et al., NeurIPS 2020) — theo `vpu.py` của kho chính thức (MIT): loss biến phân + điều chuẩn MixUp,
    Adam betas (0,5; 0,99), lr giảm một nửa mỗi 20 epoch, 30 vòng lặp mỗi epoch.
  • ILL cho nhãn nhiễu và SSL (Chen et al., NeurIPS 2024) — theo `noisy_label/imp_noisy_label.py` và
    `semisup/imp_semisup.py` của kho chính thức.
  • VIME (Yoon et al., NeurIPS 2020) — cài lại theo bài (kho chính thức TF1, không có tệp giấy phép nên không chép).

Khác biệt bắt buộc khi chuyển sang dữ liệu bảng, ghi ở docs/baseline_reproduction.md: backbone MLP thay cho WRN/ResNet;
kho GLWS/ILL dùng hai phép tăng cường (yếu/mạnh) cho ảnh — dữ liệu bảng không có phép tăng cường chuẩn nên hai "góc
nhìn" là cùng một x. Kho GLWS/ILL không có tệp giấy phép: chỉ đọc để cài lại, không chép.
"""
from __future__ import annotations

import math

import numpy as np

from .baselines_ext import _Std, _device, _mlp, _torch, pu_risk
from .tuning import Choice, LogUniform, random_search, weak_validation_split


def _batches(rng, n, bs):
    return np.array_split(rng.permutation(n), max(1, int(math.ceil(n / bs))))


def count_posterior(q1, c):
    """P(y_i = 1 | Σ_j y_j = c) chính xác cho y_j ~ Bernoulli(q1_j) độc lập — forward–backward O(K²) bằng numpy.
    Các hàng của bảng tiến/lùi được chuẩn hoá riêng; hệ số chuẩn hoá triệt tiêu trong tỉ số cho từng i."""
    q = np.clip(np.asarray(q1, np.float64), 1e-12, 1 - 1e-12)
    K = len(q)
    c = int(min(max(c, 0), K))
    Fw = np.zeros((K + 1, K + 1))          # Fw[i] = luật số dương của q[:i]
    Bw = np.zeros((K + 2, K + 1))          # Bw[i] = luật số dương của q[i-1:] (chỉ số 1..K), Bw[K+1] = δ0
    Fw[0, 0] = 1.0
    for i in range(1, K + 1):
        r = Fw[i - 1] * (1 - q[i - 1])
        r[1:] += Fw[i - 1, :-1] * q[i - 1]
        Fw[i] = r / r.sum()
    Bw[K + 1, 0] = 1.0
    for i in range(K, 0, -1):
        r = Bw[i + 1] * (1 - q[i - 1])
        r[1:] += Bw[i + 1, :-1] * q[i - 1]
        Bw[i] = r / r.sum()
    Fm, Bm = Fw[:-1], Bw[2:]                                       # mẫu i (0-chỉ số): F = Fw[i], B = Bw[i+2]
    A = (Fm[:, :c] * Bm[:, c - 1::-1][:, :c]).sum(1) if c >= 1 else np.zeros(K)   # Σ_j F(j) B(c−1−j)
    Z = (Fm[:, :c + 1] * Bm[:, c::-1][:, :c + 1]).sum(1)                           # Σ_j F(j) B(c−j)
    num = q * A
    den = num + (1 - q) * Z
    return np.where(den > 0, num / np.where(den > 0, den, 1), q)


# =====================================================================================================
# PU: GLWS, VPU
# =====================================================================================================
class PUNet:
    """method ∈ {glws, vpu}. s = 1: dương có nhãn. Như cả hai kho chính thức, tập "không nhãn" là toàn bộ dữ liệu
    (kể cả mẫu dương có nhãn: include_lb_to_ulb=True của GLWS, x_loader của VPU) — một mẫu từ p(x) dưới SCAR."""

    def __init__(self, method="vpu", prior=None, lr=3e-5, weight_decay=0.0, hidden=(300,) * 6, epochs=50,
                 batch_size=500, iters_per_epoch=30, mix_alpha=0.3, mix_weight=0.03, uratio=2, seed=0):
        self.method, self.prior, self.lr, self.wd, self.hidden = method, prior, lr, weight_decay, tuple(hidden)
        self.epochs, self.bs, self.iters, self.seed = epochs, batch_size, iters_per_epoch, seed
        self.mix_alpha, self.mix_weight, self.uratio = mix_alpha, mix_weight, uratio

    def fit(self, X, s):
        torch = _torch()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        dev = _device()
        s = np.asarray(s).astype(int)
        self.std = _Std().fit(X)
        Xa = torch.tensor(self.std(X), device=dev)
        Xp = Xa[torch.as_tensor(np.flatnonzero(s == 1), device=dev)]
        self.net = _mlp(X.shape[1], 2, self.hidden).to(dev)
        n, npos = len(Xa), len(Xp)
        if self.method == "vpu":
            lr = self.lr
            opt = torch.optim.Adam(self.net.parameters(), lr=lr, betas=(0.5, 0.99))
            bs = min(self.bs, n)
            for ep in range(self.epochs):
                if ep % 20 == 19:                                  # như vpu.py: lr giảm một nửa, tạo lại Adam
                    lr /= 2
                    opt = torch.optim.Adam(self.net.parameters(), lr=lr, betas=(0.5, 0.99))
                for _ in range(self.iters):
                    xb = Xa[torch.as_tensor(rng.choice(n, bs, replace=False), device=dev)]
                    pb = Xp[torch.as_tensor(rng.choice(npos, bs, replace=npos < bs), device=dev)]
                    out = torch.log_softmax(self.net(torch.cat([pb, xb])), 1)[:, 1]
                    lp, lx = out[:bs], out[bs:]
                    var_loss = torch.logsumexp(lx, 0) - math.log(len(lx)) - lp.mean()
                    with torch.no_grad():
                        tx = lx.exp()
                        perm = torch.randperm(bs)
                        lam = torch.distributions.Beta(self.mix_alpha, self.mix_alpha).sample()
                        mx = lam * xb + (1 - lam) * pb[perm]
                        my = lam * tx + (1 - lam) * torch.ones_like(tx)
                    reg = ((torch.log(my) - torch.log_softmax(self.net(mx), 1)[:, 1]) ** 2).mean()
                    loss = var_loss + self.mix_weight * reg
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
        elif self.method == "glws":
            if self.prior is None:
                raise ValueError("GLWS cho PU cần tiên nghiệm lớp")
            opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
            bs_p = max(1, min(self.bs, npos))
            for _ in range(self.epochs):
                for b in _batches(rng, n, self.uratio * self.bs):
                    xu = Xa[torch.as_tensor(b, device=dev)]
                    xp = Xp[torch.as_tensor(rng.choice(npos, bs_p, replace=npos < bs_p), device=dev)]
                    logits = self.net(torch.cat([xp, xu]))
                    lsm = torch.log_softmax(logits, 1)
                    sup = -lsm[:bs_p, 1].mean()
                    lu = lsm[bs_p:]
                    c = int(round(self.prior * len(b)))           # như kho: int(class_prior * num_ulb)
                    p1 = torch.as_tensor(count_posterior(lu[:, 1].detach().exp().cpu().numpy(), c),
                                         dtype=lu.dtype, device=dev)   # hậu nghiệm chính xác khi biết đếm = c
                    unsup = -(p1 * lu[:, 1] + (1 - p1) * lu[:, 0]).mean()
                    loss = sup + unsup
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
        else:
            raise ValueError(self.method)
        return self

    def predict_proba(self, X):
        torch = _torch()
        with torch.no_grad():
            dev = next(self.net.parameters()).device
            return torch.softmax(self.net(torch.tensor(self.std(X), device=dev)), -1).cpu().numpy()

    def decision_function(self, X):
        P = np.clip(self.predict_proba(X), 1e-12, 1)
        return np.log(P[:, 1]) - np.log(P[:, 0])


PU2_DEFAULTS = {
    # VPU: đúng run.py của kho (lr 3e-5, batch 500, 50 epoch × 30 vòng lặp, α = 0,3, λ = 0,03; mạng 6 lớp ẩn 300
    # như model_vec.py cho dữ liệu vector)
    "vpu": dict(lr=3e-5, hidden=(300,) * 6, epochs=50, batch_size=500, mix_alpha=0.3, mix_weight=0.03),
    # GLWS: không có cấu hình cho dữ liệu bảng trong kho → backbone MLP và lịch như các baseline MLP khác của dự án
    "glws": dict(lr=1e-3, weight_decay=0.0, hidden=(100,), epochs=200, batch_size=64, uratio=2),
}
PU2_SPACES = {
    "vpu": {"lr": LogUniform(1e-5, 1e-3), "mix_weight": Choice([0.01, 0.03, 0.1, 0.3]),
            "mix_alpha": Choice([0.1, 0.3, 1.0]), "hidden": Choice([(64,), (128,), (128, 128)])},
    "glws": {"lr": LogUniform(1e-4, 1e-2), "weight_decay": LogUniform(1e-6, 1e-2),
             "hidden": Choice([(64,), (128,), (128, 128)]), "epochs": Choice([100, 200, 400])},
}


def fit_pu2(name, X, s, prior, seed=0, tune=False, n_configs=20):
    mk = lambda cfg: PUNet(name, prior=prior, seed=seed, **cfg)
    if not tune:
        return mk(PU2_DEFAULTS[name]).fit(X, s)
    s = np.asarray(s)
    itr, iva = weak_validation_split(len(s), seed, 0.2, stratify=s)
    base = {k: v for k, v in PU2_DEFAULTS[name].items() if k not in PU2_SPACES[name]}

    def score(cfg):
        m = mk({**base, **cfg}).fit(X[itr], s[itr])
        return pu_risk(m.decision_function(X[iva]), s[iva], prior)

    res = random_search(score, PU2_SPACES[name], n=n_configs, seed=seed,
                        default={k: PU2_DEFAULTS[name][k] for k in PU2_SPACES[name]})
    m = mk({**base, **res.best}).fit(X, s)
    m.tune_ = res
    return m


# =====================================================================================================
# ILL: nhãn nhiễu và SSL
# =====================================================================================================
class ILLNet:
    """method ∈ {noisy, ssl}.
    noisy: noise_loss (CE trên phân phối nhãn nhiễu dự đoán) + em_loss (hậu nghiệm của nhãn thật khi biết nhãn nhiễu)
           + con_loss + (tuỳ chọn) cân bằng entropy trung bình. Ma trận nhiễu: học được như kho (khởi tạo đơn vị,
           chuẩn hoá hàng và cột) hoặc dùng T đã biết (`noise_model="known"`).
    ssl:   CE trên phần có nhãn + CE với nhãn giả (căn phân phối bằng EMA 0,999, giữ top-3 lớp) + cân bằng entropy."""

    def __init__(self, method="noisy", lr=1e-3, weight_decay=1e-5, hidden=(128, 128), epochs=200, batch_size=128,
                 noise_model="learned", noise_matrix_scale=1.0, average_entropy_loss=True, uratio=7, seed=0):
        self.method, self.lr, self.wd, self.hidden, self.epochs, self.bs = method, lr, weight_decay, tuple(hidden), \
            epochs, batch_size
        self.noise_model, self.scale, self.ent, self.uratio, self.seed = noise_model, noise_matrix_scale, \
            average_entropy_loss, uratio, seed

    def fit(self, X, y_weak, T=None, n_classes=None):
        torch = _torch()
        F = torch.nn.functional
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        dev = _device()
        y_weak = np.asarray(y_weak).astype(int)
        C = n_classes or int(y_weak.max()) + 1
        self.C = C
        self.std = _Std().fit(X)
        Xa = torch.tensor(self.std(X), device=dev)
        self.net = _mlp(X.shape[1], C, self.hidden).to(dev)
        params = list(self.net.parameters())
        if self.method == "noisy" and self.noise_model == "learned":
            W = torch.nn.Parameter(torch.eye(C, device=dev))             # NoiseMatrixLayer: khởi tạo đơn vị
            params.append(W)
        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=self.wd)
        n = len(Xa)
        yt = torch.as_tensor(y_weak, device=dev)

        def balance(p):
            if not self.ent:
                return 0.0
            avg = p.mean(0).clamp(1e-6, 1.0)
            return 0.1 * (-(torch.full_like(avg, 1.0 / C) * torch.log(avg)).sum())

        if self.method == "noisy":
            Tk = None if T is None else torch.tensor(np.asarray(T, np.float32), device=dev)
            for _ in range(self.epochs):
                for b in _batches(rng, n, self.bs):
                    bt = torch.as_tensor(b, device=dev)
                    logits = self.net(Xa[bt])
                    p = logits.softmax(-1)
                    w = yt[bt]
                    if self.noise_model == "learned":
                        N = F.normalize(F.normalize(W, dim=0), dim=1) * self.scale
                        Trow = N.softmax(-1)                                    # T[y, ŵ] dùng cho E-step
                        Tobs = N.softmax(0)                                     # như kho: chuẩn hoá theo cột
                    else:
                        if Tk is None:
                            raise ValueError("noise_model='known' cần T")
                        Trow = Tobs = Tk
                    col = Trow[:, w].detach().T                                 # (B, C): p(ŵ = w_i | y = k)
                    with torch.no_grad():
                        em_y = p.detach() * col
                        em_y = em_y / em_y.sum(1, keepdim=True).clamp_min(1e-12)
                    em_p = p * col
                    em_p = em_p / em_p.sum(1, keepdim=True).clamp_min(1e-12)
                    noisy_p = p @ Tobs
                    noisy_p = noisy_p / noisy_p.sum(-1, keepdim=True)
                    noise_loss = -torch.log(noisy_p.gather(1, w[:, None]).clamp_min(1e-12)).mean()
                    em_loss = -(em_y * torch.log(em_p.clamp_min(1e-12))).sum(1).mean()
                    con_loss = -(p.detach() * torch.log_softmax(logits, -1)).sum(1).mean()   # hai góc nhìn trùng nhau
                    loss = noise_loss + em_loss + con_loss + balance(p)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
        elif self.method == "ssl":
            lab = np.flatnonzero(y_weak >= 0)
            if len(lab) == 0:
                raise ValueError("SSL cần ít nhất một mẫu có nhãn")
            p_hat = torch.full((C,), 1.0 / C, device=dev)
            k = min(3, C)
            bs_l = min(self.bs // self.uratio if self.uratio > 1 else self.bs, len(lab))
            bs_l = max(1, bs_l)
            for _ in range(self.epochs):
                for b in _batches(rng, n, self.bs):
                    bl = torch.as_tensor(rng.choice(lab, bs_l, replace=len(lab) < bs_l), device=dev)
                    bu = torch.as_tensor(b, device=dev)
                    out = self.net(torch.cat([Xa[bl], Xa[bu]]))
                    ll, lu = out[:bs_l], out[bs_l:]
                    sup = F.cross_entropy(ll, yt[bl])
                    with torch.no_grad():
                        pu = lu.softmax(-1)
                        p_hat = 0.999 * p_hat + 0.001 * pu.mean(0)
                        q = pu / p_hat
                        q = q / q.sum(-1, keepdim=True)
                        vals, idx = torch.topk(q, k=k, dim=1)
                        pseudo = torch.zeros_like(q).scatter_(1, idx, vals)
                        pseudo = pseudo / pseudo.sum(1, keepdim=True)
                    unsup = -(pseudo * torch.log_softmax(lu, -1)).sum(1).mean()
                    loss = sup + unsup + balance(lu.softmax(-1))
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
        else:
            raise ValueError(self.method)
        return self

    def predict_proba(self, X):
        torch = _torch()
        with torch.no_grad():
            dev = next(self.net.parameters()).device
            return torch.softmax(self.net(torch.tensor(self.std(X), device=dev)), -1).cpu().numpy()


ILL_DEFAULT = dict(lr=1e-3, weight_decay=1e-5, hidden=(128, 128), epochs=200, batch_size=128)   # "như GLWS"
ILL_SPACE = {"lr": LogUniform(1e-4, 1e-2), "weight_decay": LogUniform(1e-6, 1e-2),
             "hidden": Choice([(64,), (128,), (128, 128)]), "epochs": Choice([100, 200, 400])}
ILL_NOISY_EXTRA = {"noise_model": Choice(["learned", "known"]), "average_entropy_loss": Choice([True, False])}
ILL_SSL_EXTRA = {"average_entropy_loss": Choice([True, False])}


def fit_ill(setting, X, y_weak, T=None, seed=0, tune=False, n_configs=20, n_classes=None):
    """setting ∈ {noisy, ssl}. Chọn cấu hình: nhiễu — log loss hiệu chỉnh tiến với T đã biết trên nhãn nhiễu của
    validation; SSL — log loss trên phần có nhãn của validation (docs/tuning_spaces.md)."""
    from .baselines_ext import _ssl_val_logloss, forward_val_loss
    y_weak = np.asarray(y_weak)
    C = n_classes or int(y_weak.max()) + 1
    mk = lambda cfg: ILLNet(setting, seed=seed, **cfg)
    default = dict(ILL_DEFAULT, **({"noise_model": "learned", "average_entropy_loss": True} if setting == "noisy"
                                   else {"average_entropy_loss": True}))
    if not tune:
        return mk(default).fit(X, y_weak, T, C)
    itr, iva = weak_validation_split(len(y_weak), seed, 0.2, stratify=y_weak)
    space = dict(ILL_SPACE, **(ILL_NOISY_EXTRA if setting == "noisy" else ILL_SSL_EXTRA))

    def score(cfg):
        m = mk({**ILL_DEFAULT, **cfg}).fit(X[itr], y_weak[itr], T, C)
        if setting == "noisy":
            return forward_val_loss(m, X[iva], y_weak[iva], T)
        return _ssl_val_logloss(m, X[iva], y_weak[iva])

    res = random_search(score, space, n=n_configs, seed=seed, default={k: default[k] for k in space})
    m = mk({**ILL_DEFAULT, **res.best}).fit(X, y_weak, T, C)
    m.tune_ = res
    return m


# =====================================================================================================
# VIME (cài lại theo bài: Yoon, Zhang, Jordon & van der Schaar, NeurIPS 2020)
# =====================================================================================================
class VIME:
    """Tự giám sát: encoder học dự đoán mặt nạ và khôi phục feature từ bản bị che (loss = BCE mặt nạ + α·MSE).
    Bán giám sát: bộ dự đoán trên encoder cố định, loss = CE có nhãn + β·phương sai dự đoán qua K bản che của mẫu
    không nhãn. Dữ liệu chuẩn hoá về [0, 1] như kho chính thức; encoder một lớp ẩn cỡ d; K = 3; dừng sớm theo
    loss trên 10% phần có nhãn như kho."""

    def __init__(self, p_m=0.3, alpha=2.0, beta=1.0, K=3, hidden=(100, 100), self_epochs=10, semi_iters=1000,
                 batch_size=128, lr=1e-3, seed=0):
        self.p_m, self.alpha, self.beta, self.K, self.hidden = p_m, alpha, beta, K, tuple(hidden)
        self.self_epochs, self.semi_iters, self.bs, self.lr, self.seed = self_epochs, semi_iters, batch_size, lr, seed

    def _corrupt(self, x, rng_t):
        torch = _torch()
        m = (torch.rand(x.shape, generator=rng_t) < self.p_m).float()
        idx = torch.rand(x.shape, generator=rng_t).argsort(0)            # mỗi cột hoán vị độc lập: mẫu từ biên
        xbar = torch.gather(x, 0, idx)
        return m * xbar + (1 - m) * x, m

    def fit(self, X, y_obs, n_classes=None):
        torch = _torch()
        F = torch.nn.functional
        torch.manual_seed(self.seed)
        g = torch.Generator().manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        y_obs = np.asarray(y_obs).astype(int)
        C = n_classes or int(y_obs.max()) + 1
        self.C = C
        X = np.nan_to_num(np.asarray(X, np.float64))
        self.lo, self.hi = X.min(0), X.max(0)
        Xa = torch.tensor(self._scale(X))
        n, d = Xa.shape
        # --- tự giám sát
        self.enc = torch.nn.Sequential(torch.nn.Linear(d, d), torch.nn.ReLU())
        head_m, head_r = torch.nn.Linear(d, d), torch.nn.Linear(d, d)
        opt = torch.optim.RMSprop(list(self.enc.parameters()) + list(head_m.parameters()) + list(head_r.parameters()),
                                  lr=self.lr)
        for _ in range(self.self_epochs):
            for b in _batches(rng, n, self.bs):
                x = Xa[torch.as_tensor(b)]
                xt, m = self._corrupt(x, g)
                h = self.enc(xt)
                loss = F.binary_cross_entropy_with_logits(head_m(h), m) + \
                    self.alpha * F.mse_loss(torch.sigmoid(head_r(h)), x)
                opt.zero_grad()
                loss.backward()
                opt.step()
        for p in self.enc.parameters():
            p.requires_grad_(False)
        # --- bán giám sát
        lab = np.flatnonzero(y_obs >= 0)
        rng.shuffle(lab)
        n_val = max(1, int(round(0.1 * len(lab)))) if len(lab) >= 10 else 0
        val, tr = lab[:n_val], lab[n_val:]
        if len(tr) == 0:
            tr = lab
        self.pred = _mlp(d, C, self.hidden)
        opt = torch.optim.Adam(self.pred.parameters(), lr=self.lr)
        yt = torch.as_tensor(y_obs)
        best, best_state, bad = np.inf, None, 0
        for it in range(self.semi_iters):
            bl = torch.as_tensor(rng.choice(tr, min(self.bs, len(tr)), replace=False))
            bu = torch.as_tensor(rng.choice(n, min(self.bs, n), replace=False))
            sup = F.cross_entropy(self.pred(self.enc(Xa[bl])), yt[bl])
            xu = Xa[bu]
            outs = torch.stack([self.pred(self.enc(self._corrupt(xu, g)[0])).softmax(-1) for _ in range(self.K)], 0)
            unsup = outs.var(0, unbiased=False).sum(-1).mean()
            loss = sup + self.beta * unsup
            opt.zero_grad()
            loss.backward()
            opt.step()
            if n_val and it % 10 == 9:                                   # dừng sớm theo loss trên phần có nhãn
                with torch.no_grad():
                    v = F.cross_entropy(self.pred(self.enc(Xa[torch.as_tensor(val)])), yt[torch.as_tensor(val)]).item()
                if v < best - 1e-6:
                    best, bad = v, 0
                    best_state = {k: t.clone() for k, t in self.pred.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 10:
                        break
        if best_state is not None:
            self.pred.load_state_dict(best_state)
        return self

    def _scale(self, X):
        return np.clip((np.nan_to_num(np.asarray(X, np.float64)) - self.lo) / (self.hi - self.lo + 1e-8), -1, 2) \
            .astype(np.float32)

    def predict_proba(self, X):
        torch = _torch()
        with torch.no_grad():
            return self.pred(self.enc(torch.tensor(self._scale(X)))).softmax(-1).numpy()


VIME_DEFAULT = dict(p_m=0.3, alpha=2.0, beta=1.0, hidden=(100, 100))      # như kho chính thức
VIME_SPACE = {"p_m": Choice([0.1, 0.3, 0.5]), "alpha": Choice([0.5, 1.0, 2.0]), "beta": Choice([0.1, 1.0, 10.0]),
              "hidden": Choice([(64,), (128,)])}


def fit_vime(X, y_obs, seed=0, tune=False, n_configs=20, n_classes=None):
    from .baselines_ext import _ssl_val_logloss
    y_obs = np.asarray(y_obs)
    C = n_classes or int(y_obs.max()) + 1
    if not tune:
        return VIME(seed=seed, **VIME_DEFAULT).fit(X, y_obs, C)
    itr, iva = weak_validation_split(len(y_obs), seed, 0.2, stratify=y_obs)

    def score(cfg):
        return _ssl_val_logloss(VIME(seed=seed, **cfg).fit(X[itr], y_obs[itr], C), X[iva], y_obs[iva])

    res = random_search(score, VIME_SPACE, n=n_configs, seed=seed, default=VIME_DEFAULT)
    m = VIME(seed=seed, **res.best).fit(X, y_obs, C)
    m.tune_ = res
    return m
