"""Phương pháp của G7 — đúng cấu hình đã chốt (giao thức Mục 4; METHOD_SPEC D1–D6) và đúng tên phương pháp như các
script thí điểm, để kết quả tập phát triển và tập đánh giá so được với nhau.

Phương pháp chỉ nhận `Cell` — không có nhãn sạch của train (R6). Oracle (TFM với nhãn sạch, trần tham chiếu) do bộ chạy
gọi riêng. Tỉ lệ lớp dương `prior` là giả định "tiên nghiệm lớp đã biết" chuẩn của PU (giao thức Mục 4).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .weakgen import WeakData

# cấu hình In-Context Gibbs đã chốt ở G4 (giao thức Mục 4)
GIBBS = dict(n_blocks=10, n_sweeps=30, burn_in=10, max_context=2000, n_use=10, em_iters=10, warm_folds=5)


@dataclass
class Cell:
    did: int
    name: str
    setting: str
    level: float
    seed: int
    n_classes: int
    Xtr: np.ndarray          # đầu vào TFM (mã thứ tự, NaN)
    Xte: np.ndarray
    Mtr: np.ndarray          # đầu vào MLP (Preprocessor fit trên train)
    Mte: np.ndarray
    weak: WeakData
    prior: float | None = None


@dataclass
class Out:
    P: np.ndarray            # (n_test, C)
    n_tfm_calls: int = 0
    extra: str = ""


def _P2(p):
    p = np.asarray(p, float)
    return np.column_stack([1 - p, p]) if p.ndim == 1 else p


def _cfg(cfg):
    return "|".join(f"{k}:{(round(v, 6) if isinstance(v, float) else v)}" for k, v in cfg.items()).replace(" ", "")


def tuned_extra(m):
    t = getattr(m, "tune_", None)
    return "" if t is None else f"best={_cfg(t.best)};val={t.best_score:.4f}"


# =====================================================================================================
# Mô hình quan sát cho sampler, nhãn ngây thơ và khởi động ấm (D6)
# =====================================================================================================
def weak_model(cell: Cell, known_eta: bool = False, prior_scale: float = 2.0):
    """prior_scale: độ lệch chuẩn của prior N(0, s² I) cho η của PU-SAR (mặc định 2; ablation giao thức Mục 7: 1 và 4)."""
    from ..weak import LLP, PUSAR, PUSCAR, SSL, NoisyLabel, PartialLabel
    w = cell.weak
    if w.setting == "pu_scar":
        return PUSCAR(w.s, c=w.level)
    if w.setting == "pu_sar":
        return PUSAR(w.s, Z=w.z[:, None], eta=list(w.eta)) if known_eta else \
            PUSAR(w.s, Z=w.z[:, None], prior_scale=prior_scale)
    if w.setting == "llp":
        return LLP(w.bags, w.counts, len(cell.Xtr))
    if w.setting == "ssl":
        return SSL(w.y_obs, cell.n_classes)
    if w.setting == "noisy":
        return NoisyLabel(w.w, 2, T=w.T)
    if w.setting in ("pll", "pll_real"):
        return PartialLabel(w.S)
    raise ValueError(w.setting)


def naive_labels(cell: Cell):
    """(nhãn ngây thơ, mặt nạ các điểm dùng làm ngữ cảnh) cho các bối cảnh cấp mẫu ngoài PU."""
    w = cell.weak
    if w.setting == "ssl":
        lab = w.y_obs >= 0
        return np.where(lab, w.y_obs, 0), lab
    if w.setting == "noisy":
        return w.w, None
    if w.setting in ("pll", "pll_real"):
        return w.naive_pll, None
    raise ValueError(w.setting)


def warm_start(cell: Cell, cond, known_eta: bool = False, extras: dict | None = None):
    """Điểm khởi động ấm P0 trên train (D6) và số lần gọi TFM. extras (dict, tuỳ chọn): với LLP ghi thêm
    extras["P0_test"] = dự đoán của cùng DLLP tuyến tính trên test (khởi động cho chế độ chuyển nạp)."""
    from ..baselines import DLLP, warm_start_cross, warm_start_pu
    w = cell.weak
    if w.setting in ("pu_scar", "pu_sar"):
        e = w.level if w.setting == "pu_scar" else (w.meta["e"] if known_eta else weak_model(cell).e())
        return warm_start_pu(cond, cell.Xtr, w.s, e, n_folds=GIBBS["warm_folds"], seed=cell.seed), GIBBS["warm_folds"]
    if w.setting == "llp":
        dl = DLLP(epochs=200, hidden=(), seed=cell.seed).fit(cell.Mtr, w.bags, w.counts)    # DLLP tuyến tính
        if extras is not None:
            extras["P0_test"] = dl.predict_proba(cell.Mte)
        return dl.predict_proba(cell.Mtr), 0
    pseudo, fit_mask = naive_labels(cell)
    return warm_start_cross(cond, cell.Xtr, pseudo, fit_mask=fit_mask, n_folds=GIBBS["warm_folds"],
                            seed=cell.seed), GIBBS["warm_folds"]


# =====================================================================================================
# Phần TFM (GPU): TFM ngây thơ, chuỗi Gibbs ấm (→ gibbs_warm và sem_warm), hard-EM ấm, propensity đã biết, CL + TFM
# =====================================================================================================
def run_tfm_methods(cell: Cell, cond, cond_final, todo, emit, monitor=None, guard=None, mh_check=False):
    """todo(method) → bool; emit(method, Out, seconds); guard(method, fn) bọc lỗi riêng cho từng phương pháp.
    Gọi theo đúng thứ tự của script thí điểm."""
    guard = guard or (lambda m, fn: fn())
    w = cell.weak
    if todo("naive"):
        guard("naive", lambda: _naive(cell, cond_final, emit))
    mh_on = mh_check and todo("gibbs_warm_mh")          # G4-5: chỉ chạy khi bật --mh-check
    need_warm = any(todo(m) for m in ("gibbs_warm", "sem_warm", "hard_em_warm")) or mh_on or \
        (w.setting == "noisy" and todo("cl_tfm"))
    state = {}
    if need_warm:
        def _warm():
            t0 = time.time()
            state["P0"], state["calls"] = warm_start(cell, cond)
            state["t"] = time.time() - t0
        guard("warm_start", _warm)
        if "P0" not in state:
            return
    if todo("gibbs_warm") or todo("sem_warm"):
        guard("gibbs_warm", lambda: _gibbs(cell, cond, cond_final, emit, monitor, state))
    if mh_on:
        guard("gibbs_warm_mh", lambda: _gibbs(cell, cond, cond_final, emit, monitor, state, mh=True))
    if todo("hard_em_warm"):
        guard("hard_em_warm", lambda: _hard_em(cell, cond, cond_final, emit, state))
    if w.setting == "pu_sar" and todo("gibbs_known_warm"):
        guard("gibbs_known_warm", lambda: _gibbs_known(cell, cond, cond_final, emit))
    if w.setting == "noisy" and todo("cl_tfm"):
        guard("cl_tfm", lambda: _cl_tfm(cell, cond_final, emit, state))


def run_rhat(cell, cond, cond_final=None, seeds=(0, 1000)):
    """Hai chuỗi Gibbs ấm cùng điểm khởi động, khác seed lấy mẫu; split-R̂ trên trạng thái nhãn sau burn-in.
    Có cond_final: trả thêm xác suất dự đoán trên test của từng chuỗi (cùng cách dự đoán như gibbs_warm)."""
    from ..sampler import InContextGibbs
    from .rhat import label_rhat
    P0, _ = warm_start(cell, cond)
    chains, preds = [], []
    for s in seeds:
        g = InContextGibbs(cond, weak_model(cell), n_blocks=GIBBS["n_blocks"], n_sweeps=GIBBS["n_sweeps"],
                           burn_in=GIBBS["burn_in"], max_context=GIBBS["max_context"], seed=cell.seed + s,
                           init=P0).run(cell.Xtr)
        chains.append(g.samples_)
        if cond_final is not None:
            preds.append(np.asarray(g.predict_proba(cell.Xtr, cell.Xte, cond=cond_final, n_use=GIBBS["n_use"]), float))
    st = label_rhat(*chains)
    return (st, preds) if cond_final is not None else st


# =====================================================================================================
# Ablation (giao thức Mục 7; các điểm định rõ ở G7_RUNNER_DESIGN §4 và giao thức §11, deviation 11)
# Mỗi biến thể là MỘT chuỗi Gibbs cùng seed với gibbs_warm của phần TFM, chỉ khác đúng một thành phần, nên so ghép cặp
# với gibbs_warm (K = 10, 30 vòng quét, khởi động ấm, quy nạp, không sửa MH, prior η N(0, 2²)).
# =====================================================================================================
ABLATION_VARIANTS = {
    "gibbs_cold": dict(init="cold"),                   # khởi tạo lạnh: y_i ~ ℓ_i chuẩn hoá (LLP: hoán vị khớp số đếm)
    "gibbs_warm_K5": dict(n_blocks=5),
    "gibbs_warm_K20": dict(n_blocks=20),
    "gibbs_warm_S15": dict(n_sweeps=15),               # burn-in = S/3; n_use = 10 mẫu cách đều sau burn-in
    "gibbs_warm_S60": dict(n_sweeps=60),
    "gibbs_warm_trans": dict(transductive=True),       # X_test vào chuỗi như điểm tự do (ℓ ≡ 1)
    "gibbs_warm_mh": dict(mh_correct=True),            # khối / bag-block có sửa MH (chính xác)
    "gibbs_warm_prior1": dict(prior_scale=1.0),        # PU-SAR: prior η ~ N(0, 1)
    "gibbs_warm_prior4": dict(prior_scale=4.0),        # PU-SAR: prior η ~ N(0, 4²)
}


def run_ablation(cell: Cell, cond, cond_final, variants, todo, emit, guard=None):
    guard = guard or (lambda m, fn: fn())
    need = [v for v in variants if todo(v)]
    if not need:
        return
    state = {}
    if any(ABLATION_VARIANTS[v].get("init") != "cold" for v in need):
        def _warm():
            t0 = time.time()
            ex = {}
            state["P0"], state["calls"] = warm_start(cell, cond, extras=ex)
            state["P0_test"] = ex.get("P0_test")
            state["t"] = time.time() - t0
        guard("warm_start", _warm)
    for v in need:
        if ABLATION_VARIANTS[v].get("init") != "cold" and "P0" not in state:
            continue
        guard(v, lambda v=v: _gibbs_variant(cell, cond, cond_final, emit, state, v, **ABLATION_VARIANTS[v]))


def _gibbs_variant(cell, cond, cond_final, emit, state, name, init="warm", n_blocks=None, n_sweeps=None,
                   transductive=False, mh_correct=False, prior_scale=None):
    from ..sampler import InContextGibbs
    t1 = time.time()
    cond.n_calls = 0
    S = GIBBS["n_sweeps"] if n_sweeps is None else int(n_sweeps)
    burn = GIBBS["burn_in"] if n_sweeps is None else S // 3
    K = GIBBS["n_blocks"] if n_blocks is None else int(n_blocks)
    wm = weak_model(cell) if prior_scale is None else weak_model(cell, prior_scale=prior_scale)
    cold = init == "cold"
    calls0, t0 = (0, 0.0) if cold else (state["calls"], state["t"])
    kw = dict(n_blocks=K, n_sweeps=S, burn_in=burn, max_context=GIBBS["max_context"], seed=cell.seed,
              mh_correct=mh_correct)
    info = f"K={K};sweeps={S};burn_in={burn};init={'cold' if cold else 'warm'}"
    if transductive:
        m = len(cell.Xte)
        wm.extend(m)
        P0t = state.get("P0_test")
        if P0t is None:                                    # ngoài LLP: điểm test khởi tạo theo ℓ ≡ 1 (đều)
            P0t = np.full((m, cell.n_classes), 1.0 / cell.n_classes)
        init_arg = "lik" if cold else np.vstack([np.asarray(state["P0"], float), np.asarray(P0t, float)])
        g = InContextGibbs(cond, wm, init=init_arg, track=np.arange(len(cell.Xtr), len(cell.Xtr) + m), **kw) \
            .run(np.vstack([cell.Xtr, cell.Xte]))
        P = g.transductive_proba()                         # trung bình Rao–Blackwell của điều kiện trong chuỗi
        n_calls = calls0 + cond.n_calls
        info += ";mode=transductive"
    else:
        g = InContextGibbs(cond, wm, init="lik" if cold else state["P0"], **kw).run(cell.Xtr)
        cond_final.n_calls = 0
        n_use = min(GIBBS["n_use"], max(len(g.samples_), 1))
        P = g.predict_proba(cell.Xtr, cell.Xte, cond=cond_final, n_use=n_use)
        n_calls = calls0 + cond.n_calls + cond_final.n_calls
    info += ";flips=" + "/".join(f"{f:.3f}" for f in g.trace_.flips)
    if mh_correct:
        info += ";accept=" + "/".join(f"{a:.3f}" for a in g.trace_.accept)
    emit(name, Out(P, n_calls, info), t0 + time.time() - t1)


def _naive(cell, cond_final, emit):
    from ..baselines import naive_tfm_llp, naive_tfm_pu
    w = cell.weak
    t0 = time.time()
    if w.setting in ("pu_scar", "pu_sar"):
        _, p = naive_tfm_pu(cond_final, cell.Xtr, w.s, cell.Xte, c=float(w.level))
        P = _P2(p)
    elif w.setting == "llp":
        rng = np.random.default_rng(1000 * cell.seed + cell.did + 99)           # dãy riêng cho nhãn ngây thơ của LLP
        P = _P2(naive_tfm_llp(cond_final, cell.Xtr, weak_model(cell), cell.Xte, rng))
    else:
        pseudo, fm = naive_labels(cell)
        fm = np.ones(len(pseudo), bool) if fm is None else fm
        P = cond_final.predict_proba(cell.Xtr[fm], pseudo[fm], cell.Xte)
    emit("naive", Out(P, 1), time.time() - t0)


def _gibbs(cell, cond, cond_final, emit, monitor, state, mh=False):
    from ..sampler import InContextGibbs
    w = cell.weak
    t1 = time.time()
    cond.n_calls = 0
    g = InContextGibbs(cond, weak_model(cell), n_blocks=GIBBS["n_blocks"], n_sweeps=GIBBS["n_sweeps"],
                       burn_in=GIBBS["burn_in"], max_context=GIBBS["max_context"], seed=cell.seed, init=state["P0"],
                       monitor=monitor, mh_correct=mh).run(cell.Xtr)
    if mh:                                                # G4-5: khối có sửa MH (chính xác), kèm tỉ lệ chấp nhận
        acc = "/".join(f"{a:.3f}" for a in g.trace_.accept)
        cond_final.n_calls = 0
        P = g.predict_proba(cell.Xtr, cell.Xte, cond=cond_final, n_use=GIBBS["n_use"])
        emit("gibbs_warm_mh", Out(P, state["calls"] + cond.n_calls + cond_final.n_calls,
                                  f"accept={acc};flips=" + "/".join(f"{f:.3f}" for f in g.trace_.flips)),
             state["t"] + time.time() - t1)
        return
    calls = cond.n_calls
    flips = "/".join(f"{f:.3f}" for f in g.trace_.flips)
    mon = "" if not g.trace_.monitor else ";monitor=" + "/".join(f"{m:.3f}" for m in g.trace_.monitor)
    info = f"flips={flips}{mon}"
    if w.setting == "pu_sar":
        info += f";eta_learned={np.round(g.weak.eta, 3).tolist()}"
    cond_final.n_calls = 0
    P = g.predict_proba(cell.Xtr, cell.Xte, cond=cond_final, n_use=GIBBS["n_use"])
    emit("gibbs_warm", Out(P, state["calls"] + calls + cond_final.n_calls, info), state["t"] + time.time() - t1)
    P = g.predict_proba(cell.Xtr, cell.Xte, cond=cond_final, last_only=True)
    emit("sem_warm", Out(P, state["calls"] + calls + 1, info), state["t"] + time.time() - t1)


def _hard_em(cell, cond, cond_final, emit, state):
    from ..baselines import hard_em_tfm
    t1 = time.time()
    cond.n_calls = 0
    wm = weak_model(cell, known_eta=(cell.weak.setting == "pu_sar"))      # hard-EM không học tham số phiền nhiễu
    init_y = wm.init_labels(np.random.default_rng(cell.seed), P0=state["P0"])
    P, _ = hard_em_tfm(cond, wm, cell.Xtr, cell.Xte, n_iter=GIBBS["em_iters"], n_blocks=GIBBS["n_blocks"],
                       seed=cell.seed, init=init_y, cond_final=cond_final)
    emit("hard_em_warm", Out(P, state["calls"] + cond.n_calls + 1), state["t"] + time.time() - t1)


def _gibbs_known(cell, cond, cond_final, emit):
    from ..sampler import InContextGibbs
    t1 = time.time()
    P0k, wc = warm_start(cell, cond, known_eta=True)
    cond.n_calls = 0
    g = InContextGibbs(cond, weak_model(cell, known_eta=True), n_blocks=GIBBS["n_blocks"],
                       n_sweeps=GIBBS["n_sweeps"], burn_in=GIBBS["burn_in"], max_context=GIBBS["max_context"],
                       seed=cell.seed, init=P0k).run(cell.Xtr)
    P = g.predict_proba(cell.Xtr, cell.Xte, cond=cond_final, n_use=GIBBS["n_use"])
    emit("gibbs_known_warm", Out(P, wc + cond.n_calls + GIBBS["n_use"]), time.time() - t1)


def _cl_tfm(cell, cond_final, emit, state):
    from cleanlab.filter import find_label_issues
    w = cell.weak
    t1 = time.time()
    issues = find_label_issues(labels=w.w, pred_probs=state["P0"])
    keep = ~issues
    emit("cl_tfm", Out(cond_final.predict_proba(cell.Xtr[keep], w.w[keep], cell.Xte), state["calls"] + 1,
                       f"pruned={int(issues.sum())}"), state["t"] + time.time() - t1)


# =====================================================================================================
# Phần baseline (CPU): mọi baseline peer-review, bản mặc định và bản tune — tên như các script thí điểm
# =====================================================================================================
def _pu(cell, kind, tune):
    from ..baselines import NNPU
    from ..baselines_ext import fit_nnpu
    from ..baselines_secondary import fit_pu2
    w = cell.weak
    if kind in ("nnpu", "upu") and tune is None:                          # bản mặc định như thí điểm (lớp NNPU cũ)
        m = NNPU(prior=cell.prior, nn_correction=(kind == "nnpu"), epochs=200, seed=cell.seed).fit(cell.Mtr, w.s)
        return Out(m.predict_proba(cell.Mte))
    if kind in ("nnpu", "upu"):
        m = fit_nnpu(cell.Mtr, w.s, prior=cell.prior, nn_correction=(kind == "nnpu"), seed=cell.seed, tune=True)
        return Out(m.predict_proba(cell.Mte), 0, tuned_extra(m))
    m = fit_pu2(kind, cell.Mtr, w.s, prior=cell.prior, seed=cell.seed, tune=bool(tune))
    return Out(m.predict_proba(cell.Mte), 0, tuned_extra(m))


def _z_test(cell):
    """z của tập test chuẩn hoá bằng trung bình và độ lệch chuẩn của train (như thí điểm)."""
    zr = np.nan_to_num(np.asarray(cell.Xtr[:, 0], float))
    return (np.nan_to_num(np.asarray(cell.Xte[:, 0], float)) - zr.mean()) / (zr.std() + 1e-8)


def _sar(cell, tuned):
    """SAR-EM (code chính thức) — như thí điểm: feature MLP chuẩn hoá z toàn bộ cột, nối thuộc tính propensity z."""
    from sklearn.preprocessing import StandardScaler
    from ..baselines_sar import SAREM, fit_sar_em_tuned
    w = cell.weak
    sc = StandardScaler().fit(cell.Mtr)
    Atr = np.column_stack([sc.transform(cell.Mtr), w.z])
    Ate = np.column_stack([sc.transform(cell.Mte), _z_test(cell)])
    pcol = [Atr.shape[1] - 1]
    if not tuned:
        m = SAREM(seed=cell.seed).fit(Atr, w.s, pcol)
        return Out(m.predict_proba(Ate), 0, f"its={m.info_['nb_iterations']}")
    m = fit_sar_em_tuned(Atr, w.s, pcol, n_configs=20, seed=cell.seed)
    return Out(m.predict_proba(Ate), 0, f"C_f={m.tune_.best['C_f']:.4g};C_e={m.tune_.best['C_e']:.4g};"
                                         f"val_weak_loss={m.tune_.best_score:.4f}")


def _sar_e_known(cell):
    from sklearn.preprocessing import StandardScaler
    from ..baselines_ext import fit_sar_e
    sc = StandardScaler().fit(cell.Mtr)
    f = fit_sar_e(sc.transform(cell.Mtr), cell.weak.s, cell.weak.meta["e"])
    return Out(_P2(np.clip(f.predict_proba(sc.transform(cell.Mte)), 1e-6, 1 - 1e-6)))


def _llp(cell, bname, tune):
    from ..baselines_ext import fit_llp
    m = fit_llp(bname, cell.Mtr, cell.weak.bags, cell.weak.counts, seed=cell.seed, tune=tune)
    return Out(m.predict_proba(cell.Mte), 0, tuned_extra(m))


def _pll(cell, bname, tune):
    from ..baselines_ext import fit_pll
    m = fit_pll(bname, cell.Mtr, cell.weak.S, seed=cell.seed, tune=tune)
    return Out(m.predict_proba(cell.Mte), 0, tuned_extra(m) if tune else f"source={getattr(m, 'source_', '-')}")


def _ssl(cell, kind, tune):
    from ..baselines_ext import fit_ssl
    from ..baselines_secondary import fit_ill, fit_vime
    y = cell.weak.y_obs
    if kind in ("label_spreading", "self_training"):
        m = fit_ssl(kind, cell.Mtr, y, seed=cell.seed, tune=tune, n_classes=cell.n_classes)
    elif kind == "ill":
        m = fit_ill("ssl", cell.Mtr, y, None, seed=cell.seed, tune=tune, n_classes=cell.n_classes)
    else:
        m = fit_vime(cell.Mtr, y, seed=cell.seed, tune=tune, n_classes=cell.n_classes)
    extra = tuned_extra(m)
    P = m.predict_proba(cell.Mte)
    if hasattr(m, "fallback_rows_"):
        extra = (extra + ";" if extra else "") + f"prior_fallback_rows={m.fallback_rows_}"
    return Out(P, 0, extra)


def _noisy(cell, kind, tune):
    from ..baselines import ForwardCorrection
    from ..baselines_ext import fit_forward
    from ..baselines_secondary import fit_ill
    w = cell.weak
    if kind == "forward_pilot":                                           # cấu hình của thí điểm (tên `forward`)
        return Out(ForwardCorrection(w.T, epochs=200, seed=cell.seed).fit(cell.Mtr, w.w).predict_proba(cell.Mte))
    if kind == "forward":
        m = fit_forward(cell.Mtr, w.w, w.T, seed=cell.seed, tune=tune)
    else:
        m = fit_ill("noisy", cell.Mtr, w.w, w.T, seed=cell.seed, tune=tune, n_classes=2)
    return Out(m.predict_proba(cell.Mte), 0, tuned_extra(m))


BASELINES = {
    "pu": {
        "nnpu": lambda c: _pu(c, "nnpu", None), "upu": lambda c: _pu(c, "upu", None),
        "nnpu_tun": lambda c: _pu(c, "nnpu", True), "upu_tun": lambda c: _pu(c, "upu", True),
        "glwspu_def": lambda c: _pu(c, "glws", False), "glwspu_tun": lambda c: _pu(c, "glws", True),
        "vpu_def": lambda c: _pu(c, "vpu", False), "vpu_tun": lambda c: _pu(c, "vpu", True),
    },
    "pu_sar_only": {
        "sar_em": lambda c: _sar(c, False), "sar_em_tuned": lambda c: _sar(c, True),
        "sar_e_known": _sar_e_known,
    },
    "llp": {f"{tag}_{suf}": (lambda c, b=b, t=t: _llp(c, b, t))
            for b, tag in (("dllp", "dllp"), ("dllp_linear", "dllplin"), ("llp_vat", "llpvat"),
                           ("easyllp", "easyllp"), ("glws", "glws"))
            for t, suf in ((False, "def"), (True, "tun"))},
    "pll": {f"{tag}_{suf}": (lambda c, b=b, t=t: _pll(c, b, t))
            for b, tag in (("proden", "proden"), ("glws", "glwspll"), ("rc", "rc"), ("cc", "cc"))
            for t, suf in ((False, "def"), (True, "tun"))},
    "ssl": {"label_spreading": lambda c: _ssl(c, "label_spreading", False),
            "ls_tun": lambda c: _ssl(c, "label_spreading", True),
            "self_training": lambda c: _ssl(c, "self_training", False),
            "st_tun": lambda c: _ssl(c, "self_training", True),
            "illssl_def": lambda c: _ssl(c, "ill", False), "illssl_tun": lambda c: _ssl(c, "ill", True),
            "vime_def": lambda c: _ssl(c, "vime", False), "vime_tun": lambda c: _ssl(c, "vime", True)},
    "noisy": {"forward": lambda c: _noisy(c, "forward_pilot", False),
              "forward_def": lambda c: _noisy(c, "forward", False), "forward_tun": lambda c: _noisy(c, "forward", True),
              "illnz_def": lambda c: _noisy(c, "ill", False), "illnz_tun": lambda c: _noisy(c, "ill", True)},
}


def baselines_for(setting: str) -> dict:
    if setting == "pu_scar":
        return dict(BASELINES["pu"])
    if setting == "pu_sar":
        return {**BASELINES["pu"], **BASELINES["pu_sar_only"]}
    if setting == "pll_real":
        return dict(BASELINES["pll"])
    return dict(BASELINES[setting])


TFM_METHODS = {
    "pu_scar": ["naive", "gibbs_warm", "sem_warm", "hard_em_warm"],
    "pu_sar": ["naive", "gibbs_warm", "sem_warm", "hard_em_warm", "gibbs_known_warm"],
    "llp": ["naive", "gibbs_warm", "sem_warm", "hard_em_warm"],
    "ssl": ["naive", "gibbs_warm", "sem_warm", "hard_em_warm"],
    "noisy": ["naive", "gibbs_warm", "sem_warm", "hard_em_warm", "cl_tfm"],
    "pll": ["naive", "gibbs_warm", "sem_warm", "hard_em_warm"],
    "pll_real": ["naive", "gibbs_warm", "sem_warm", "hard_em_warm"],
}
