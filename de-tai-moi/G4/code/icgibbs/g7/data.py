"""Tải và chuẩn hoá dữ liệu cho G7 (giao thức Mục 2; DATASET_CARDS A2).

- `canonical_table`: bỏ hàng thiếu target; loại hàng trùng hoàn toàn (X, y), giữ bản đầu (R1, L1.4); mã hoá nhãn
  (nhị phân: 1 = lớp thiểu số, hoà thì theo thứ tự chữ cái; đa lớp: mã theo thứ tự chữ cái của nhãn gốc).
- `encode_tfm`: đầu vào của TFM — cột số giữ nguyên (thiếu = NaN), cột phân loại → mã thứ tự theo thứ tự chữ cái của
  các mức (thiếu = NaN). Không dùng nhãn nên không cần fit.
- `Preprocessor`: đầu vào của baseline MLP — điền trung vị, chuẩn hoá z, one-hot; **fit trên train của từng lần chia**,
  mức phân loại chưa gặp trong train → toàn 0 (R2, L1.2).
- `table_sha256`: mã băm của bảng đã chuẩn hoá (R7), ghi vào docs/dataset_hashes.csv và mỗi dòng kết quả.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass
class Table:
    did: int
    name: str
    version: int | None
    md5_openml: str | None
    X: "object"            # pandas DataFrame, cột số và cột phân loại (dtype category/object)
    y: np.ndarray          # mã lớp 0..C-1
    classes: list          # nhãn gốc theo thứ tự mã
    n_raw: int
    n_missing_target: int
    n_duplicates_removed: int

    @property
    def n_classes(self):
        return len(self.classes)


def canonical_table(df, target, did=0, name="", version=None, md5=None, binary=None) -> Table:
    import pandas as pd
    n_raw = len(df)
    keep = ~pd.isna(target).to_numpy()
    df = df.loc[keep].reset_index(drop=True)
    t = pd.Series(np.asarray(target)[keep]).astype(str).reset_index(drop=True)
    full = df.copy()
    full["__y__"] = t.values
    dup = full.duplicated(keep="first").to_numpy()
    df, t = df.loc[~dup].reset_index(drop=True), t.loc[~dup].reset_index(drop=True)
    vals, cnt = np.unique(t.to_numpy(), return_counts=True)          # np.unique: thứ tự chữ cái
    if binary is None:
        binary = len(vals) == 2
    if binary:
        if len(vals) != 2:
            raise ValueError(f"dataset {did}: cần 2 lớp, có {len(vals)}")
        minority = vals[np.lexsort((vals, cnt))[0]]                 # ít mẫu nhất; hoà thì theo chữ cái
        y = (t.to_numpy() == minority).astype(int)
        classes = [vals[vals != minority][0], minority]
    else:
        y = np.searchsorted(vals, t.to_numpy()).astype(int)
        classes = list(vals)
    return Table(did, name, version, md5, df, y, classes, n_raw, int((~keep).sum()), int(dup.sum()))


def _is_numeric(s):
    import pandas as pd
    return pd.api.types.is_numeric_dtype(s) and not isinstance(s.dtype, pd.CategoricalDtype)


def encode_tfm(X) -> np.ndarray:
    import pandas as pd
    cols = []
    for c in X.columns:
        s = X[c]
        if _is_numeric(s):
            cols.append(pd.to_numeric(s, errors="coerce").to_numpy(np.float64))
        else:
            s = s.astype("object")
            levels = sorted({str(v) for v in s if not pd.isna(v)})
            m = {v: i for i, v in enumerate(levels)}
            cols.append(np.array([np.nan if pd.isna(v) else m[str(v)] for v in s], np.float64))
    return np.column_stack(cols) if cols else np.zeros((len(X), 0))


class Preprocessor:
    """Fit trên train: trung vị và (mean, std) của cột số, các mức của cột phân loại. Transform: điền, chuẩn hoá,
    one-hot theo các mức đã gặp (mức mới → toàn 0)."""

    def fit(self, X):
        import pandas as pd
        self.num = [c for c in X.columns if _is_numeric(X[c])]
        self.cat = [c for c in X.columns if c not in self.num]
        N = X[self.num].apply(pd.to_numeric, errors="coerce") if self.num else None
        self.med = N.median().fillna(0.0) if self.num else None
        if self.num:
            F = N.fillna(self.med)
            self.mu, self.sd = F.mean(), F.std(ddof=0).replace(0, 1.0).fillna(1.0)
        self.levels = {c: sorted({str(v) for v in X[c] if not pd.isna(v)}) for c in self.cat}
        return self

    def transform(self, X) -> np.ndarray:
        import pandas as pd
        parts = []
        if self.num:
            F = X[self.num].apply(pd.to_numeric, errors="coerce").fillna(self.med)
            parts.append(((F - self.mu) / self.sd).to_numpy(np.float32))
        for c in self.cat:
            lv = self.levels[c]
            idx = {v: i for i, v in enumerate(lv)}
            M = np.zeros((len(X), len(lv)), np.float32)
            for r, v in enumerate(X[c]):
                if not pd.isna(v) and str(v) in idx:
                    M[r, idx[str(v)]] = 1.0
            parts.append(M)
        return np.hstack(parts) if parts else np.zeros((len(X), 0), np.float32)


def table_sha256(X_tfm: np.ndarray, y: np.ndarray) -> str:
    """SHA-256 của (shape, X dạng float64 với NaN chuẩn hoá, y dạng int64)."""
    X = np.ascontiguousarray(np.asarray(X_tfm, np.float64)).copy()
    X[np.isnan(X)] = np.nan                                           # một biểu diễn NaN duy nhất
    h = hashlib.sha256()
    h.update(np.asarray(X.shape, np.int64).tobytes())
    h.update(np.where(np.isnan(X), np.float64(np.nan), X).tobytes())
    h.update(np.ascontiguousarray(np.asarray(y, np.int64)).tobytes())
    return h.hexdigest()


def load_real_pll(path: str, name: str, expected_sha256: str | None = None):
    """Bản .npz do g5_verify_pll_real.py lưu (việc F7): X (M×d), S (M×Q, tập ứng viên), y (nhãn đúng), sha256 của
    tệp gốc. Số lớp Q lấy từ S (Lost có 2 lớp không bao giờ là nhãn đúng). Trả (Table, S)."""
    import pandas as pd
    z = np.load(path, allow_pickle=False)
    sha = str(z["sha256"])
    if expected_sha256 and sha != expected_sha256:
        raise ValueError(f"{name}: SHA-256 của tệp gốc {sha[:12]} khác DATASET_CARDS {expected_sha256[:12]}")
    X = pd.DataFrame(np.asarray(z["X"], np.float64), columns=[f"f{i}" for i in range(z["X"].shape[1])])
    S = np.asarray(z["S"]).astype(bool)
    y = np.asarray(z["y"]).astype(int)
    t = Table(0, name, None, sha, X, y, list(range(S.shape[1])), len(y), 0, 0)
    return t, S


def load_openml(did: int, binary: bool | None = None) -> Table:
    from sklearn.datasets import fetch_openml
    b = fetch_openml(data_id=did, as_frame=True, parser="auto")
    det = b.details or {}
    ver = det.get("version")
    return canonical_table(b.data.copy(), b.target, did=did, name=det.get("name", str(did)),
                           version=int(ver) if ver is not None else None, md5=det.get("md5_checksum"), binary=binary)
