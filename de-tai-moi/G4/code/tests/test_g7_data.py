"""Kiểm thử phần dữ liệu của bộ chạy G7 (G6/G7_RUNNER_DESIGN.md: R1, R2, R7) — không cần mạng."""
import numpy as np
import pandas as pd
import pytest

from icgibbs.g7.data import Preprocessor, canonical_table, encode_tfm, table_sha256
from icgibbs.g7.registry import DEV_BINARY, DEV_MULTICLASS, EVAL_BINARY, EVAL_MULTICLASS
from icgibbs.g7.splits import random_split, split_sha256, stratified_split


def _frame():
    df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 2.0, 5.0, 1.0, 7.0],
                       "b": pd.Categorical(["x", "y", "y", "y", None, "x", "z"]),
                       "c": [0, 1, 1, 1, 0, 0, 1]})
    t = pd.Series(["no", "yes", "yes", "yes", "no", "no", None])
    return df, t


def test_registry_sizes_and_disjoint():
    assert len(EVAL_BINARY) == 10 and len(EVAL_MULTICLASS) == 14
    assert not (set(EVAL_BINARY) | set(EVAL_MULTICLASS)) & (set(DEV_BINARY) | set(DEV_MULTICLASS))


def test_canonical_drops_missing_target_and_exact_duplicates():
    df, t = _frame()
    tab = canonical_table(df, t)
    assert tab.n_raw == 7 and tab.n_missing_target == 1
    # hàng 1 và 3 trùng hoàn toàn (2.0, y, 1, yes); hàng 0 và 5 trùng hoàn toàn (1.0, x, 0, no)
    assert tab.n_duplicates_removed == 2 and len(tab.X) == 4
    assert tab.classes == ["yes", "no"] or tab.classes == ["no", "yes"]
    assert tab.y.sum() == min((tab.y == 1).sum(), (tab.y == 0).sum())      # 1 = lớp thiểu số


def test_conflicting_duplicates_are_kept():
    df = pd.DataFrame({"a": [1.0, 1.0, 2.0]})
    tab = canonical_table(df, pd.Series(["p", "q", "p"]))
    assert tab.n_duplicates_removed == 0 and len(tab.X) == 3


def test_binary_minority_tie_alphabetical_and_multiclass_codes():
    tab = canonical_table(pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}), pd.Series(["b", "a", "b", "a"]))
    assert tab.classes == ["b", "a"] and list(tab.y) == [0, 1, 0, 1]
    tab = canonical_table(pd.DataFrame({"a": [1.0, 2.0, 3.0]}), pd.Series(["c", "a", "b"]), binary=False)
    assert tab.classes == ["a", "b", "c"] and list(tab.y) == [2, 0, 1]


def test_encode_tfm_keeps_nan_and_orders_levels():
    df, t = _frame()
    X = encode_tfm(df)
    assert np.isnan(X[2, 0]) and np.isnan(X[4, 1])
    assert list(X[[0, 1, 6], 1]) == [0.0, 1.0, 2.0]                  # x < y < z
    assert np.array_equal(encode_tfm(df), X, equal_nan=True)


def test_preprocessor_fits_on_train_only_and_ignores_unseen_levels():
    df, _ = _frame()
    tr, te = df.iloc[:4], df.iloc[4:]
    p = Preprocessor().fit(tr)
    assert p.med["a"] == 2.0                                          # trung vị của train (1, 2, 2), không phải của toàn bộ
    Z = p.transform(te)
    # cột phân loại b chỉ có mức {x, y} trong train → 'z' và thiếu thành toàn 0
    j0 = len(p.num)
    assert np.all(Z[[0, 2], j0:j0 + 2] == 0)
    Ztr = p.transform(tr)
    assert np.allclose(Ztr[:, :len(p.num)].mean(0), 0, atol=1e-6)


def test_hashes_are_stable_and_sensitive():
    df, t = _frame()
    tab = canonical_table(df, t)
    X = encode_tfm(tab.X)
    h = table_sha256(X, tab.y)
    assert h == table_sha256(X.copy(), tab.y.copy())
    X2 = X.copy(); X2[0, 0] += 1e-9
    assert table_sha256(X2, tab.y) != h
    y = np.array([0, 1] * 50)
    a = stratified_split(y, 0); b = stratified_split(y, 0); c = stratified_split(y, 1)
    assert split_sha256(*a) == split_sha256(*b) != split_sha256(*c)


def test_stratified_split_matches_pilot_scripts():
    from sklearn.model_selection import train_test_split
    y = np.random.default_rng(0).integers(0, 3, 500)
    for seed in (0, 1, 2):
        ours = stratified_split(y, seed)
        ref = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=seed)
        assert all(np.array_equal(a, b) for a, b in zip(ours, ref))
    tr, te = random_split(1000, 3)
    assert len(te) == 100 and len(set(tr) & set(te)) == 0
