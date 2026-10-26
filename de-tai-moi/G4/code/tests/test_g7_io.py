import csv

import numpy as np

from icgibbs.g7.io import ResultWriter, config_hash, weak_hash
from icgibbs.g7.weakgen import make_weak


def test_writer_resumes_and_skips_done(tmp_path):
    p = tmp_path / "r.csv"
    fields = ["backend", "dataset_id", "setting", "level", "seed", "method", "auc", "status"]
    w = ResultWriter(str(p), fields, str(tmp_path / "p.jsonl"))
    row = dict(backend="b", dataset_id=3, setting="llp", level=8, seed=0, method="gibbs_warm", auc=0.9, status="ok")
    w.write(row, P=np.array([[0.2, 0.8]]), y=[1])
    w.write(dict(row, method="naive", status="error: x"))
    w.close()
    w2 = ResultWriter(str(p), fields)
    assert w2.is_done(backend="b", dataset_id=3, setting="llp", level=8, seed=0, method="gibbs_warm")
    assert not w2.is_done(backend="b", dataset_id=3, setting="llp", level=8, seed=0, method="naive")
    w2.close()
    rows = list(csv.DictReader(open(p)))
    assert len(rows) == 2 and rows[0]["status"] == "ok"


def test_hashes_deterministic():
    y = np.array([0, 1] * 40); X = np.zeros((80, 1))
    a = make_weak("llp", 8, y, X, np.random.default_rng(0)); b = make_weak("llp", 8, y, X, np.random.default_rng(0))
    c = make_weak("llp", 8, y, X, np.random.default_rng(1))
    assert weak_hash(a) == weak_hash(b) != weak_hash(c)
    assert config_hash({"a": 1, "b": (2,)}) == config_hash({"b": (2,), "a": 1})
