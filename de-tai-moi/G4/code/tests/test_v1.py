"""V2 giai đoạn 1 (V2/PHASE1_PLAN.md): lưu Q̄ và dự đoán mỗi chuỗi (npz), dự đoán ngoài fold của learner khởi động,
chạy tiếp không ghi trùng, tạo lại tệp lưu khi thiếu, --explore cho v0 trên S2."""
import csv
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import run_g7  # noqa: E402

DRY = ["--backend", "dry", "--source", "synthetic", "--seeds", "0"]


def _rows(p):
    return list(csv.DictReader(open(p)))


@pytest.mark.parametrize("setting,level,did", [("pll", "0.6", "54"), ("llp", "8", "1510")])
def test_v0_save_and_oof_dry(tmp_path, setting, level, did):
    out, sd = str(tmp_path / "v1.csv"), str(tmp_path / "npz")
    args = ["--part", "v0", "--settings", setting, "--levels", f"{setting}={level}", "--datasets", did, *DRY,
            "--save_dir", sd, "--oof", "--out", out]
    assert run_g7.main(args) == 0
    rows = _rows(out)
    assert all(r["status"] == "ok" for r in rows) and len(rows) == 5
    files = sorted(os.listdir(sd))
    assert len(files) == 1 and files[0].startswith("v1_dry_")
    z = np.load(os.path.join(sd, files[0]))
    n, m = len(z["ytr"]), len(z["yte"])
    for st in ("warm", "base", "cold"):
        assert z[f"Q_{st}"].shape[0] == n and z[f"P_{st}"].shape[0] == m
        assert np.isfinite(z[f"wpll_{st}"]) and len(z[f"monitor_{st}"]) == 30
        r = next(x for x in rows if x["method"] == f"gibbs_init_{st}")
        key = "auc" if setting == "llp" else "acc"
        P = z[f"P_{st}"]
        if setting == "llp":
            from sklearn.metrics import roc_auc_score
            got = roc_auc_score(z["yte"], P[:, 1])
        else:
            got = float(np.mean(P.argmax(1) == z["yte"]))
        assert abs(got - float(r[key])) < 1e-4                 # npz khớp dòng CSV
    Po = z["Poof_L"]
    assert Po.shape == (n, z["P_L"].shape[1]) and np.all(np.isfinite(Po)) and np.allclose(Po.sum(1), 1, atol=1e-5)
    if setting == "llp":
        assert (z["bag_of"] >= 0).all() and z["counts"].sum() == n
    else:
        S = z["S"]
        assert S.shape == (n, Po.shape[1])
        free = S.sum(1) >= 2
        for st in ("warm", "base", "cold"):
            assert np.all(z[f"Q_{st}"][free].sum(1) > 0.99)     # điểm tự do luôn có Q̄
    # chạy lại: không thêm dòng, không chạy lại (tệp đã có)
    t0 = os.path.getmtime(os.path.join(sd, files[0]))
    assert run_g7.main(args) == 0 and len(_rows(out)) == 5
    assert os.path.getmtime(os.path.join(sd, files[0])) == t0
    # thiếu tệp lưu: chạy lại chuỗi, tạo lại tệp giống hệt, không ghi trùng dòng
    zz = {k: z[k] for k in z.files}
    os.remove(os.path.join(sd, files[0]))
    assert run_g7.main(args) == 0 and len(_rows(out)) == 5
    z2 = np.load(os.path.join(sd, files[0]))
    for k in ("Q_warm", "P_base", "Poof_L", "P_L"):
        assert np.array_equal(zz[k], z2[k]), k


def test_oof_requires_base_and_save_requires_v0(tmp_path):
    out = str(tmp_path / "x.csv")
    with pytest.raises(SystemExit):
        run_g7.main(["--part", "v0", "--starts", "warm,cold", "--oof", *DRY, "--out", out])
    with pytest.raises(SystemExit):
        run_g7.main(["--part", "tfm", "--save_dir", str(tmp_path), *DRY, "--out", out])


def test_v0_on_s2_needs_explore(tmp_path):
    out = str(tmp_path / "s2.csv")
    with pytest.raises(SystemExit):
        run_g7.main(["--part", "v0", "--split", "s2", "--eval-ok", *DRY, "--out", out])
    assert run_g7.main(["--part", "v0", "--split", "s2", "--eval-ok", "--explore", "--settings", "pll",
                        "--levels", "pll=0.6", "--datasets", "181", "--starts", "warm,cold", *DRY, "--out", out]) == 0
    rows = _rows(out)
    assert {r["method"] for r in rows} == {"gibbs_init_warm", "gibbs_init_cold", "gibbs_ms"}
    assert all(r["status"] == "ok" for r in rows)
