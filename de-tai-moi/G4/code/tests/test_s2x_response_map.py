"""S2_PLAN §5.1: ánh xạ phản ứng chạy được với TFM giả; nhãn gieo có đúng tỉ lệ; chạy tiếp bỏ qua dòng đã xong."""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import s2x_response_map as rm  # noqa: E402


def test_planted_labels_rate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 4000)
    S = rng.uniform(size=(4000, 5)) < 0.6
    S[np.arange(4000), y] = True
    z = rm.planted_labels(y, S, 0.5, rng)
    assert S[np.arange(4000), z].all()                       # luôn nằm trong tập ứng viên
    single = S.sum(1) == 1
    r = (z == y)[~single].mean()
    assert abs(r - 0.5) < 0.03
    zu = rm.planted_labels(y, S, None, rng)
    assert abs((zu == y).mean() - np.mean(1 / S.sum(1))) < 0.03


def test_response_map_dry(tmp_path):
    out = str(tmp_path / "m.csv")
    args = ["--backends", "dry", "--source", "synthetic", "--datasets", "12", "--q", "0.6", "--seeds", "0",
            "--rho", "u,0.8", "--out", out]
    assert rm.main(args) == 0
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 2 and all(r["status"] == "ok" for r in rows)
    lo, hi = sorted(rows, key=lambda r: float(r["r_actual"]))
    assert float(hi["phi"]) > float(lo["phi"])                 # gán đúng nhiều hơn → TFM giả đẩy về nhãn đúng mạnh hơn
    assert rm.main(args) == 0 and len(list(csv.DictReader(open(out)))) == 2
