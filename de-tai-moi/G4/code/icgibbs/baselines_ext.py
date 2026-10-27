"""
baselines_ext.py — baseline peer-review bổ sung ở G6 (việc T3), giao diện thống nhất, có tune theo giao thức.

Nguồn của từng phương pháp (chi tiết: de-tai-moi/G6/G6_PLAN.md Mục 1, docs/baseline_reproduction.md):
  LLP
    DLLP     Ardehaly & Culotta (ICDMW 2017): CE giữa tỉ lệ túi và trung bình softmax — dùng ProportionLoss
             của kho LLP-VAT (MIT, sao chép trong icgibbs/third_party).
    LLP-VAT  Tsai & Lin (ACML 2020): ProportionLoss + VATLoss của kho chính thức (MIT, sao chép nguyên văn), trọng
             số nhất quán tăng dần theo sigmoid ramp-up (Laine & Aila 2017) như main.py của kho.
    EasyLLP  Busa-Fekete et al. (NeurIPS 2023): không có code chính thức → cài theo Eq. 3 của bài:
             ℓ̃(x, α) = (k(α−p)+p)·ℓ(f(x),1) + (k(p−α)+(1−p))·ℓ(f(x),0), p ước lượng = trung bình α (như bài).
    GLWS     Chen et al. (ICML 2024): E-step = xác suất hậu nghiệm từng mẫu khi biết số đếm của túi (forward-backward
             trên NFA); loss = CE(nhãn giả, p) − log P(đếm = c | đếm ≤ c). Kho chính thức không có LICENSE nên KHÔNG sao
             chép; ở đây E-step tính bằng quy hoạch động theo số đếm (cùng đại lượng toán học), đã kiểm khớp số với hàm
             create_proportion_graph của kho (tests/test_baselines_ext.py, khi có kho). Không có tăng cường dữ liệu
             cho bảng, nên hai "view" yếu/mạnh là cùng một mẫu.
  PLL
    PRODEN   Lv et al. (ICML 2020): partial_loss của kho chính thức (nạp từ kho nếu có; nếu không, bản cài lại cùng
             công thức), SGD momentum 0,9 như main.py của kho; nhãn mềm cập nhật theo từng batch.
    GLWS/ILL Chen et al. (ICML 2024; NeurIPS 2024): với PLL hai phương pháp trùng nhau trong kho chính thức
             (imp_partial_label.py): nhãn giả = p chuẩn hoá trên tập ứng viên; loss = −Σ_{k∉S} log(1−p_k) + CE.
    RC, CC   Feng et al. (NeurIPS 2020): RC = CE có trọng số, trọng số cập nhật sau mỗi epoch bằng p chuẩn hoá trên
             tập ứng viên; CC = −log Σ_{k∈S} p_k. Không có code chính thức → cài theo bài.
Mạng: MLP (hidden = () là mô hình tuyến tính), đầu vào chuẩn hoá z theo train.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np

from .tuning import Choice, LogUniform, random_search, refit_best, weak_validation_split

NEG = -1e9   # thay −∞ trong quy hoạch động để gradient không thành NaN


def _torch():
    import torch
    return torch


def _device():
    torch = _torch()
    dev = os.environ.get("ICGIBBS_DEVICE")
    if dev:
        return torch.device(dev)
    return torch.device("cpu")      # mạng nhỏ: CPU thường nhanh hơn GPU vì chi phí mỗi bước


def _mlp(d_in, d_out, hidden):
    import torch.nn as nn
    layers, d = [], d_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


class _Std:
    def fit(self, X):
        X = np.asarray(X, np.float64)
        self.mu, self.sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-8
        self.mu = np.nan_to_num(self.mu)
        return self

    def __call__(self, X):
        Z = (np.asarray(X, np.float64) - self.mu) / self.sd
        return np.nan_to_num(Z).astype(np.float32)


def sigmoid_rampup(current, rampup_length):
    """exp(−5(1−t)²), Laine & Aila (ICLR 2017) — cùng công thức kho LLP-VAT dùng."""
    if rampup_length == 0:
        return 1.0
    t = float(np.clip(current, 0.0, rampup_length)) / rampup_length
    return float(np.exp(-5.0 * (1.0 - t) ** 2))


def load_official(repo_env: str, relpath: str, name: str):
    """Nạp một tệp .py của kho chính thức theo đường dẫn (không chạy __init__ của gói). None nếu không có kho."""
    root = os.environ.get(repo_env)
    if not root:
        return None
    path = os.path.join(root, relpath)
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# =====================================================================================================
# LLP
# =====================================================================================================
def _pad_bags(bags, counts):
    """bags: list mảng chỉ số; counts (G, 2) số mẫu mỗi lớp. Trả về idx (G, K) đệm −1, mask, c1 (số dương)."""
    K = max(len(b) for b in bags)
    idx = np.full((len(bags), K), -1, int)
    for g, b in enumerate(bags):
        idx[g, :len(b)] = b
    counts = np.asarray(counts)
    return idx, idx >= 0, counts[:, 1].astype(int)


def count_dp(log_q1, log_q0, mask):
    """log P(số dương = j), j = 0..K, cho từng túi; log_q* (B, K); vị trí bị mask không tính. Trả (B, K+1)."""
    torch = _torch()
    B, K = log_q1.shape
    logP = torch.full((B, K + 1), NEG, dtype=log_q1.dtype, device=log_q1.device)
    logP[:, 0] = 0.0
    for i in range(K):
        a, b, m = log_q1[:, i:i + 1], log_q0[:, i:i + 1], mask[:, i:i + 1]
        shifted = torch.cat([torch.full((B, 1), NEG, dtype=logP.dtype, device=logP.device), logP[:, :-1]], 1)
        new = torch.logaddexp(logP + b, shifted + a)
        logP = torch.where(m, new, logP)
    return logP


def glws_estep_and_loglik(probs, mask, c1):
    """Nhãn giả GLWS (hậu nghiệm từng mẫu khi biết đúng c1 dương) và log P(đếm = c | đếm ≤ c) — như
    create_proportion_graph của kho GLWS. probs (B, K, 2)."""
    torch = _torch()
    c1t = torch.as_tensor(c1, device=probs.device)
    # E-step trên bản tách đồ thị: ∂ log P(c) / ∂ log q1_i = P(y_i = 1 | đếm = c)
    a = torch.log(probs[..., 1].detach().clamp_min(1e-12)).requires_grad_(True)
    b = torch.log(probs[..., 0].detach().clamp_min(1e-12))
    logP = count_dp(a, b, mask)
    lc = logP.gather(1, c1t[:, None]).squeeze(1)
    (g,) = torch.autograd.grad(lc.sum(), a)
    pseudo1 = g.clamp(0, 1) * mask
    pseudo = torch.stack([1 - pseudo1, pseudo1], -1) * mask[..., None]
    # log-likelihood của túi (có gradient)
    logP2 = count_dp(torch.log(probs[..., 1].clamp_min(1e-12)), torch.log(probs[..., 0].clamp_min(1e-12)), mask)
    ar = torch.arange(logP2.shape[1], device=probs.device)[None, :]
    le = torch.where(ar <= c1t[:, None], logP2, torch.full_like(logP2, NEG))
    sup = logP2.gather(1, c1t[:, None]).squeeze(1) - torch.logsumexp(le, 1)
    return pseudo.detach(), sup


class LLPNet:
    """Cơ sở chung cho baseline LLP nhị phân. method ∈ {dllp, llp_vat, easyllp, glws}."""

    def __init__(self, method: str = "dllp", lr: float = 1e-3, weight_decay: float = 0.0, hidden=(64, 64),
                 epochs: int = 200, bags_per_batch: int = 8, cons: float = 0.05, xi: float = 1e-6, eps: float = 6.0,
                 seed: int = 0):
        self.method, self.lr, self.wd, self.hidden, self.epochs = method, lr, weight_decay, tuple(hidden), epochs
        self.bpb, self.cons, self.xi, self.eps, self.seed = bags_per_batch, cons, xi, eps, seed

    def fit(self, X, bags, counts):
        torch = _torch()
        import torch.nn.functional as F
        from .third_party.llp_vat_losses import VATLoss
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        dev = _device()
        self.std = _Std().fit(X)
        Xs = torch.tensor(self.std(X), device=dev)
        idx, mask, c1 = _pad_bags(bags, counts)
        sizes = mask.sum(1)
        alpha = c1 / sizes
        p_hat = float(alpha.mean())
        idx_t = torch.tensor(np.where(idx < 0, 0, idx), device=dev)
        mask_t = torch.tensor(mask, device=dev)
        props = torch.tensor(np.column_stack([1 - alpha, alpha]), dtype=torch.float32, device=dev)
        self.net = _mlp(X.shape[1], 2, self.hidden).to(dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        vat = VATLoss(xi=self.xi, eps=self.eps, ip=1)
        G = len(bags)
        n_steps = self.epochs * int(np.ceil(G / self.bpb))
        ramp = 0.4 * n_steps
        step = 0
        for _ in range(self.epochs):
            order = rng.permutation(G)
            for k in range(0, G, self.bpb):
                bb = order[k:k + self.bpb]
                ib, mb = idx_t[bb], mask_t[bb]
                xb = Xs[ib]                                        # (B, K, d)
                logits = self.net(xb)
                probs = torch.softmax(logits, -1)
                m = mb[..., None].float()
                if self.method in ("dllp", "llp_vat"):
                    avg = (probs * m).sum(1) / m.sum(1)
                    # đúng công thức ProportionLoss("ce") của kho LLP-VAT (clamp eps = 1e-8, tổng theo lớp, trung bình
                    # theo túi), viết dạng vector vì phép assert simplex.check của kho chỉ nhận vector 1 chiều ở
                    # torch mới; tests/test_baselines_ext.py kiểm khớp số với ProportionLoss gọi từng túi.
                    loss = (-props[bb] * torch.log(avg.clamp(1e-8, 1 - 1e-8))).sum(-1).mean()
                    if self.method == "llp_vat":
                        loss = loss + self.cons * sigmoid_rampup(step, ramp) * vat(self.net, xb[mb])
                elif self.method == "easyllp":
                    kk = torch.tensor(sizes[bb], dtype=torch.float32, device=dev)[:, None]
                    al = torch.tensor(alpha[bb], dtype=torch.float32, device=dev)[:, None]
                    w1 = kk * (al - p_hat) + p_hat
                    w0 = kk * (p_hat - al) + (1 - p_hat)
                    logp = torch.log_softmax(logits, -1)
                    li = -(w1 * logp[..., 1] + w0 * logp[..., 0])
                    loss = (li * mb).sum() / mb.sum()
                elif self.method == "glws":
                    pseudo, sup = glws_estep_and_loglik(probs, mb, c1[bb])
                    unsup = -(pseudo * torch.log(probs.clamp_min(1e-12))).sum(-1)
                    loss = (unsup * mb).sum() / mb.sum() - sup.mean()
                else:
                    raise ValueError(self.method)
                opt.zero_grad()
                loss.backward()
                opt.step()
                step += 1
        _ = F
        return self

    def predict_proba(self, X):
        torch = _torch()
        with torch.no_grad():
            dev = next(self.net.parameters()).device
            return torch.softmax(self.net(torch.tensor(self.std(X), device=dev)), -1).cpu().numpy()

    def weak_loss(self, X, bags, counts):
        """Loss trên nhãn yếu dùng để tune: CE giữa tỉ lệ thật của túi và trung bình xác suất dự đoán."""
        P = self.predict_proba(X)
        counts = np.asarray(counts, float)
        out = []
        for b, c in zip(bags, counts):
            prop = c / c.sum()
            avg = np.clip(P[b].mean(0), 1e-8, 1)
            out.append(-(prop * np.log(avg)).sum())
        return float(np.mean(out))


LLP_DEFAULTS = {
    "dllp": dict(lr=1e-3, weight_decay=0.0, hidden=(100,), epochs=200, bags_per_batch=8),
    "dllp_linear": dict(lr=1e-3, weight_decay=0.0, hidden=(), epochs=200, bags_per_batch=8),
    "llp_vat": dict(lr=3e-4, weight_decay=0.0, hidden=(100,), epochs=400, bags_per_batch=2, cons=0.05, xi=1e-6, eps=6.0),
    "easyllp": dict(lr=1e-3, weight_decay=0.0, hidden=(100,), epochs=200, bags_per_batch=8),
    "glws": dict(lr=1e-3, weight_decay=0.0, hidden=(100,), epochs=200, bags_per_batch=8),
}
_LLP_COMMON = {"lr": LogUniform(1e-4, 1e-2), "weight_decay": LogUniform(1e-6, 1e-2),
               "epochs": Choice([100, 200, 400]), "bags_per_batch": Choice([8, 32])}
LLP_SPACES = {
    "dllp": dict(_LLP_COMMON, hidden=Choice([(64,), (128, 128)])),
    "dllp_linear": dict(_LLP_COMMON, hidden=Choice([()])),
    "llp_vat": dict(_LLP_COMMON, hidden=Choice([(), (64,), (128, 128)]), xi=Choice([1e-6, 1e-3]),
                    eps=Choice([0.1, 0.5, 1.0, 2.0]), cons=LogUniform(1e-2, 1.0)),
    "easyllp": dict(_LLP_COMMON, hidden=Choice([(), (64,), (128, 128)])),
    "glws": dict(_LLP_COMMON, hidden=Choice([(64,), (128,), (128, 128)])),
}


def fit_llp(name: str, X, bags, counts, seed: int = 0, tune: bool = False, n_configs: int = 20):
    """name ∈ LLP_DEFAULTS. tune=False: cấu hình mặc định; tune=True: 20 cấu hình, validation 20% số TÚI."""
    method = "dllp" if name == "dllp_linear" else name
    if not tune:
        return LLPNet(method, seed=seed, **LLP_DEFAULTS[name]).fit(X, bags, counts)
    G = len(bags)
    btr, bva = weak_validation_split(G, seed, 0.2)
    counts = np.asarray(counts)

    def score(cfg):
        m = LLPNet(method, seed=seed, **cfg).fit(X, [bags[g] for g in btr], counts[btr])
        return m.weak_loss(X, [bags[g] for g in bva], counts[bva])

    res = random_search(score, LLP_SPACES[name], n=n_configs, seed=seed, default=LLP_DEFAULTS[name])
    m = refit_best(res, lambda cfg: LLPNet(method, seed=seed, **cfg).fit(X, bags, counts), X)
    m.tune_ = res
    return m


# =====================================================================================================
# PLL
# =====================================================================================================
def _proden_partial_loss():
    mod = load_official("PRODEN_REPO", os.path.join("utils", "utils_loss.py"), "proden_utils_loss")
    if mod is not None:
        return mod.partial_loss, "official"
    torch = _torch()
    import torch.nn.functional as F

    def partial_loss(output1, target, true):          # cùng công thức với utils/utils_loss.py của kho PRODEN
        output = F.softmax(output1, dim=1)
        loss = (-torch.sum(target * torch.log(output))) / output.size(0)
        revised = target.clone()
        revised[revised > 0] = 1
        revised = revised * output
        revised = revised / revised.sum(dim=1, keepdim=True)
        return loss, revised
    return partial_loss, "reimplemented"


class PLLNet:
    """method ∈ {proden, glws, rc, cc}."""

    def __init__(self, method: str = "proden", lr: float = 1e-3, weight_decay: float = 1e-5, hidden=(128, 128),
                 epochs: int = 200, batch_size: int = 128, seed: int = 0, optimizer: str | None = None,
                 average_entropy_loss: bool = True):
        self.method, self.lr, self.wd, self.hidden, self.epochs = method, lr, weight_decay, tuple(hidden), epochs
        self.bs, self.seed = batch_size, seed
        # GLWS/ILL cho PLL: số hạng cân bằng entropy trung bình 0,1 · CE(đều ‖ dự đoán trung bình của batch), như
        # `imp_partial_label.py`; MỌI config PLL chính thức đều bật (G6, 27/10/2026). Chỉ dùng khi method == "glws".
        self.average_entropy_loss = bool(average_entropy_loss)
        # PRODEN: SGD momentum 0,9 như kho chính thức (mặc định); không gian tune có thêm Adam (G6 lượt 2, mục 5)
        self.optimizer = optimizer or ("sgd" if method == "proden" else "adam")

    def fit(self, X, S, on_epoch=None):
        """on_epoch(model, epoch): gọi cuối mỗi epoch (chỉ để tái hiện số công bố, ví dụ trung bình 10 epoch cuối)."""
        torch = _torch()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        dev = _device()
        self.std = _Std().fit(X)
        Xs = torch.tensor(self.std(X), device=dev)
        S = torch.tensor(np.asarray(S, np.float32), device=dev)
        C = S.shape[1]
        W = S / S.sum(1, keepdim=True)                          # trọng số khởi đầu đều trên tập ứng viên
        self.net = _mlp(X.shape[1], C, self.hidden).to(dev)
        if self.method == "proden":
            partial_loss, self.source_ = _proden_partial_loss()
        if self.optimizer == "sgd":
            opt = torch.optim.SGD(self.net.parameters(), lr=self.lr, weight_decay=self.wd, momentum=0.9)
        elif self.optimizer == "adam":
            opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        else:
            raise ValueError(self.optimizer)
        n = len(S)
        for _ in range(self.epochs):
            for b in np.array_split(rng.permutation(n), max(1, n // self.bs)):
                b = torch.as_tensor(b, device=dev)
                logits = self.net(Xs[b])
                if self.method == "proden":
                    loss, new_w = partial_loss(logits, W[b], None)
                    W[b] = new_w.detach()
                elif self.method == "glws":
                    p = torch.softmax(logits, -1)
                    with torch.no_grad():
                        pseudo = S[b] * p
                        pseudo = pseudo / pseudo.sum(-1, keepdim=True)
                    sup = -torch.mean(torch.sum(torch.log(1.0000001 - p) * (1 - S[b]), 1))
                    unsup = -(pseudo * torch.log_softmax(logits, -1)).sum(1).mean()
                    loss = sup + unsup
                    if self.average_entropy_loss:
                        avg = torch.clamp(p.mean(0), min=1e-6, max=1.0)
                        loss = loss + 0.1 * (-(torch.full_like(avg, 1.0 / C) * torch.log(avg)).sum())
                elif self.method == "rc":
                    loss = -(W[b] * torch.log_softmax(logits, -1)).sum(1).mean()
                elif self.method == "cc":
                    p = torch.softmax(logits, -1)
                    loss = -torch.log((p * S[b]).sum(1).clamp_min(1e-12)).mean()
                else:
                    raise ValueError(self.method)
                opt.zero_grad()
                loss.backward()
                opt.step()
            if self.method == "rc":                              # RC: cập nhật trọng số sau mỗi epoch
                with torch.no_grad():
                    p = torch.softmax(self.net(Xs), -1) * S
                    W = p / p.sum(1, keepdim=True).clamp_min(1e-12)
            if on_epoch is not None:
                on_epoch(self, _)
        return self

    def predict_proba(self, X):
        torch = _torch()
        with torch.no_grad():
            dev = next(self.net.parameters()).device
            return torch.softmax(self.net(torch.tensor(self.std(X), device=dev)), -1).cpu().numpy()

    def weak_loss(self, X, S):
        P = self.predict_proba(X)
        return float(-np.mean(np.log(np.clip((P * np.asarray(S)).sum(1), 1e-12, None))))


PLL_DEFAULTS = {
    "proden": dict(lr=1e-3, weight_decay=1e-5, hidden=(128, 128), epochs=200, batch_size=128),
    "glws": dict(lr=1e-3, weight_decay=1e-5, hidden=(128, 128), epochs=200, batch_size=128, average_entropy_loss=True),
    "rc": dict(lr=1e-3, weight_decay=1e-5, hidden=(), epochs=200, batch_size=128),
    "cc": dict(lr=1e-3, weight_decay=1e-5, hidden=(), epochs=200, batch_size=128),
}
_PLL_SPACE = {"lr": LogUniform(1e-4, 1e-2), "weight_decay": LogUniform(1e-6, 1e-2),
              "hidden": Choice([(), (64,), (128, 128)]), "epochs": Choice([100, 200, 400]), "batch_size": Choice([128])}
PLL_SPACES = {k: dict(_PLL_SPACE) for k in PLL_DEFAULTS}
PLL_SPACES["glws"] = dict(_PLL_SPACE, hidden=Choice([(64,), (128,), (128, 128)]),
                          average_entropy_loss=Choice([True, False]))   # G6, 27/10/2026: như ILL nhãn nhiễu, SSL
# G6 lượt 2 (26/10/2026): SGD với lr ≤ 1e-2 làm PRODEN chưa hội tụ trên dữ liệu bảng nhỏ (bản Adam cũ của dự án mạnh hơn)
# → thêm bộ tối ưu vào không gian tune; cấu hình mặc định giữ SGD như kho chính thức. Ghi ở docs/tuning_spaces.md.
PLL_SPACES["proden"] = dict(_PLL_SPACE, optimizer=Choice(["sgd", "adam"]))


def fit_pll(name: str, X, S, seed: int = 0, tune: bool = False, n_configs: int = 20):
    if not tune:
        return PLLNet(name, seed=seed, **PLL_DEFAULTS[name]).fit(X, S)
    S = np.asarray(S)
    itr, iva = weak_validation_split(len(S), seed, 0.2)

    def score(cfg):
        return PLLNet(name, seed=seed, **cfg).fit(X[itr], S[itr]).weak_loss(X[iva], S[iva])

    res = random_search(score, PLL_SPACES[name], n=n_configs, seed=seed, default=PLL_DEFAULTS[name])
    m = refit_best(res, lambda cfg: PLLNet(name, seed=seed, **cfg).fit(X, S), X)
    m.tune_ = res
    return m


# =====================================================================================================
# PU (bối cảnh phụ): nnPU / uPU có tune; SAR-E (propensity đã biết) cho phân tích phụ của PU-SAR
# =====================================================================================================
NNPU_DEFAULT = dict(lr=1e-3, weight_decay=0.0, hidden=(100,), epochs=200)
NNPU_SPACE = {"lr": LogUniform(1e-4, 1e-2), "weight_decay": LogUniform(1e-6, 1e-2),
              "hidden": Choice([(64,), (128,), (128, 128)]), "epochs": Choice([100, 200, 400])}


def pu_risk(scores, s, prior):
    """Rủi ro PU không chệch với loss sigmoid (du Plessis et al.) — loss trên nhãn yếu để tune."""
    sig = lambda z: 1 / (1 + np.exp(np.clip(z, -50, 50)))     # ℓ(z, +1) = σ(−z)
    s = np.asarray(s)
    gp, gu = scores[s == 1], scores
    return float(prior * sig(gp).mean() + sig(-gu).mean() - prior * sig(-gp).mean())


def fit_nnpu(X, s, prior, nn_correction=True, seed=0, tune=False, n_configs=20):
    from .baselines import NNPU
    mk = lambda cfg: NNPU(prior=prior, nn_correction=nn_correction, seed=seed, lr=cfg["lr"],
                          weight_decay=cfg["weight_decay"], hidden=tuple(cfg["hidden"]), epochs=cfg["epochs"])
    if not tune:
        return mk(NNPU_DEFAULT).fit(X, s)
    s = np.asarray(s)
    itr, iva = weak_validation_split(len(s), seed, 0.2, stratify=s)

    def score(cfg):
        m = mk(cfg).fit(X[itr], s[itr])
        return pu_risk(m.decision_function(X[iva]), s[iva], prior)

    res = random_search(score, NNPU_SPACE, n=n_configs, seed=seed, default=NNPU_DEFAULT)
    m = refit_best(res, lambda cfg: mk(cfg).fit(X, s), X)
    m.tune_ = res
    return m


def fit_sar_e(X, s, e):
    """SAR-E (Bekker et al. 2019, code chính thức): bộ phân loại học với propensity ĐÃ BIẾT — đối chứng công bằng cho
    'Gibbs với propensity đã biết'."""
    from sarpu.pu_learning import pu_learn_sar_e
    from .baselines_sar import _lr_pu
    f, _ = pu_learn_sar_e(np.asarray(X, float), np.asarray(s, int), np.clip(np.asarray(e, float), 1e-3, 1),
                          classification_model=_lr_pu(1.0, 0))
    return f


# =====================================================================================================
# SSL và nhãn nhiễu phụ thuộc lớp (bối cảnh phụ): bản tune 20 cấu hình của label spreading, self-training,
# forward correction — không gian và loss validation đã ghi trước ở docs/tuning_spaces.md (G6, lượt 3)
# =====================================================================================================
class LabelSpreadingSSL:
    """Label spreading (Zhou et al., NIPS 2003) trên feature đã chuẩn hoá. Điểm test không nhận được nhãn lan
    truyền (hàng 0/0) được gán tiên nghiệm lớp của phần có nhãn, như bản chạy trong thí điểm (lượt v5)."""

    def __init__(self, kernel="knn", n_neighbors=7, gamma=1.0, alpha=0.2, max_iter=100):
        self.kernel, self.k, self.gamma, self.alpha, self.max_iter = kernel, n_neighbors, gamma, alpha, max_iter

    def fit(self, X, y_obs, n_classes=None):
        from sklearn.semi_supervised import LabelSpreading
        y_obs = np.asarray(y_obs)
        C = n_classes or int(y_obs.max()) + 1
        self.prior_ = np.bincount(y_obs[y_obs >= 0], minlength=C) / max(1, int((y_obs >= 0).sum()))
        self.std = _Std().fit(X)
        self.m = LabelSpreading(kernel=self.kernel, n_neighbors=self.k, gamma=self.gamma, alpha=self.alpha,
                                max_iter=self.max_iter).fit(self.std(X), y_obs)
        self.classes_ = self.m.classes_
        return self

    def predict_proba(self, X):
        C = len(self.prior_)
        with np.errstate(invalid="ignore", divide="ignore"):
            Pm = self.m.predict_proba(self.std(X))
        P = np.zeros((len(Pm), C))
        P[:, self.classes_] = Pm
        bad = ~np.isfinite(P).all(1) | (np.nan_to_num(P).sum(1) <= 0)
        self.fallback_rows_ = int(bad.sum())
        return np.where(bad[:, None], self.prior_[None, :], np.nan_to_num(P))


class SelfTrainingSSL:
    """Self-training (Yarowsky 1995) với hồi quy logistic, như bản của scikit-learn."""

    def __init__(self, C=1.0, threshold=0.75):
        self.C, self.threshold = C, threshold

    def fit(self, X, y_obs, n_classes=None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.semi_supervised import SelfTrainingClassifier
        y_obs = np.asarray(y_obs)
        self.n_classes = n_classes or int(y_obs.max()) + 1
        self.std = _Std().fit(X)
        self.m = SelfTrainingClassifier(LogisticRegression(C=self.C, max_iter=2000),
                                        threshold=self.threshold).fit(self.std(X), y_obs)
        return self

    def predict_proba(self, X):
        Pm = self.m.predict_proba(self.std(X))
        P = np.zeros((len(Pm), self.n_classes))
        P[:, self.m.classes_] = Pm
        return P


def _ssl_val_logloss(model, X, y_obs):
    lab = np.asarray(y_obs) >= 0
    if not lab.any():
        return np.inf
    P = np.clip(model.predict_proba(X[lab]), 1e-12, 1)
    return float(-np.mean(np.log(P[np.arange(lab.sum()), np.asarray(y_obs)[lab]])))


SSL_DEFAULTS = {"label_spreading": dict(kernel="knn", n_neighbors=7, gamma=1.0, alpha=0.2),
                "self_training": dict(C=1.0, threshold=0.75)}
SSL_SPACES = {
    "label_spreading": {"kernel": Choice(["knn", "rbf"]), "n_neighbors": Choice([3, 5, 7, 11, 15, 21]),
                        "gamma": LogUniform(1e-3, 10.0), "alpha": Choice([0.1, 0.2, 0.5, 0.8])},
    "self_training": {"C": LogUniform(1e-3, 1e3), "threshold": Choice([0.6, 0.75, 0.9, 0.95])},
}
_SSL_CLS = {"label_spreading": LabelSpreadingSSL, "self_training": SelfTrainingSSL}


def fit_ssl(name: str, X, y_obs, seed: int = 0, tune: bool = False, n_configs: int = 20, n_classes=None):
    """Chọn cấu hình theo log loss trên phần CÓ NHÃN của 20% validation (tách phân tầng theo nhãn quan sát, kể cả -1)."""
    y_obs = np.asarray(y_obs)
    C = n_classes or int(y_obs.max()) + 1
    if not tune:
        return _SSL_CLS[name](**SSL_DEFAULTS[name]).fit(X, y_obs, C)
    itr, iva = weak_validation_split(len(y_obs), seed, 0.2, stratify=y_obs)

    def score(cfg):
        if (y_obs[itr] >= 0).sum() < 2 or len(np.unique(y_obs[itr][y_obs[itr] >= 0])) < 2:
            raise ValueError("phần train của validation thiếu nhãn")
        return _ssl_val_logloss(_SSL_CLS[name](**cfg).fit(X[itr], y_obs[itr], C), X[iva], y_obs[iva])

    res = random_search(score, SSL_SPACES[name], n=n_configs, seed=seed, default=SSL_DEFAULTS[name])
    m = refit_best(res, lambda cfg: _SSL_CLS[name](**cfg).fit(X, y_obs, C), X)
    m.tune_ = res
    return m


FORWARD_DEFAULT = dict(lr=1e-3, weight_decay=0.0, hidden=(100,), epochs=200)      # "như nnPU" (tuning_spaces)
FORWARD_SPACE = {"lr": LogUniform(1e-4, 1e-2), "weight_decay": LogUniform(1e-6, 1e-2),
                 "hidden": Choice([(64,), (128,), (128, 128)]), "epochs": Choice([100, 200, 400])}


def forward_val_loss(model, X, w, T):
    """Log loss đã hiệu chỉnh tiến trên nhãn nhiễu: −log (p(x)ᵀ T)[w]."""
    Q = np.clip(model.predict_proba(X) @ np.asarray(T, float), 1e-12, 1)
    return float(-np.mean(np.log(Q[np.arange(len(w)), np.asarray(w)])))


def fit_forward(X, w, T, seed: int = 0, tune: bool = False, n_configs: int = 20):
    from .baselines import ForwardCorrection
    w = np.asarray(w)
    if not tune:
        return ForwardCorrection(T, seed=seed, **FORWARD_DEFAULT).fit(X, w)
    itr, iva = weak_validation_split(len(w), seed, 0.2, stratify=w)

    def score(cfg):
        return forward_val_loss(ForwardCorrection(T, seed=seed, **cfg).fit(X[itr], w[itr]), X[iva], w[iva], T)

    res = random_search(score, FORWARD_SPACE, n=n_configs, seed=seed, default=FORWARD_DEFAULT)
    m = refit_best(res, lambda cfg: ForwardCorrection(T, seed=seed, **cfg).fit(X, w), X)
    m.tune_ = res
    return m
