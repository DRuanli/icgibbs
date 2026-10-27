"""
tuning.py — chọn siêu tham số cho baseline khi CHỈ có nhãn yếu (giao thức G5, Mục 3 và Mục 4 quy tắc 1).

Mọi baseline cần tune nhận cùng ngân sách: 20 cấu hình random search; validation = 20% train, tách theo seed;
điểm = loss của chính baseline trên NHÃN YẾU của phần validation (không bao giờ dùng nhãn sạch).
Không gian tìm kiếm của từng baseline ghi trước trong de-tai-moi/docs/tuning_spaces.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np


@dataclass
class LogUniform:
    low: float
    high: float

    def sample(self, rng):
        return float(np.exp(rng.uniform(np.log(self.low), np.log(self.high))))


@dataclass
class Choice:
    values: list

    def sample(self, rng):
        return self.values[int(rng.integers(len(self.values)))]


@dataclass
class TuneResult:
    best: dict
    best_score: float
    trials: List[dict] = field(default_factory=list)


def sample_configs(space: Dict[str, object], n: int, seed: int, include_default: dict | None = None) -> List[dict]:
    """n cấu hình ngẫu nhiên (cấu hình đầu là cấu hình mặc định nếu được cho — vẫn tính trong ngân sách n)."""
    rng = np.random.default_rng(seed)
    out = [dict(include_default)] if include_default is not None else []
    while len(out) < n:
        out.append({k: (v.sample(rng) if hasattr(v, "sample") else v) for k, v in space.items()})
    return out[:n]


def random_search(fit_and_score: Callable[[dict], float], space: Dict[str, object], n: int = 20, seed: int = 0,
                  default: dict | None = None) -> TuneResult:
    """fit_and_score(cfg) -> loss trên validation nhãn yếu (càng nhỏ càng tốt; lỗi hoặc NaN = +inf)."""
    trials, best, best_score = [], None, np.inf
    for cfg in sample_configs(space, n, seed, default):
        try:
            score = float(fit_and_score(cfg))
        except Exception as e:  # cấu hình hỏng không làm hỏng cả lần tune
            score, cfg = np.inf, dict(cfg, error=repr(e)[:120])
        if not np.isfinite(score):
            score = np.inf
        trials.append(dict(cfg, score=score))
        if score < best_score:
            best, best_score = {k: v for k, v in cfg.items() if k != "error"}, score
    if best is None:
        best = dict(default or {})
    return TuneResult(best=best, best_score=best_score, trials=trials)


def refit_best(res: TuneResult, fit: Callable[[dict], object], X, max_tries: int = 5):
    """Fit lại trên toàn bộ train với cấu hình tốt nhất. Quy tắc dự phòng (ghi trước G7, G6 lượt 5): nếu mô hình fit
    lại cho xác suất không hữu hạn trên chính X train (huấn luyện phân kỳ), dùng cấu hình xếp hạng kế tiếp theo loss
    validation, tối đa max_tries cấu hình. Chỉ nhìn X train — không nhìn test, không nhìn nhãn. Khi không có lỗi, kết
    quả trùng hoàn toàn với `fit(res.best)`. Ghi thứ hạng đã dùng vào res.refit_rank (0 = tốt nhất; -1 = mọi lần thử
    đều hỏng, trả mô hình cuối)."""
    ranked = sorted((t for t in res.trials if "error" not in t and np.isfinite(t["score"])), key=lambda t: t["score"])
    cfgs = [{k: v for k, v in t.items() if k != "score"} for t in ranked] or [dict(res.best)]
    m = None
    for r, cfg in enumerate(cfgs[:max_tries]):
        m = fit(cfg)
        try:
            ok = bool(np.all(np.isfinite(np.asarray(m.predict_proba(X), float))))
        except Exception:
            ok = False
        if ok:
            res.refit_rank, res.refit_config = r, cfg
            return m
    res.refit_rank, res.refit_config = -1, cfgs[min(len(cfgs), max_tries) - 1]
    return m


def weak_validation_split(n: int, seed: int, frac: float = 0.2, stratify=None):
    """Tách chỉ số train/validation từ phần train (stratify theo NHÃN YẾU nếu có, ví dụ s của PU)."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(n)
    try:
        return train_test_split(idx, test_size=frac, random_state=seed, stratify=stratify)
    except ValueError:
        return train_test_split(idx, test_size=frac, random_state=seed)
