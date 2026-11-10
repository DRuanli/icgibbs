"""V2 V1D (V1D_PLAN §3.1): bộ nhớ đệm của learner L không đổi kết quả; --l_only tính trước rồi lượt thường trúng."""
import csv
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import run_g7  # noqa: E402


def _run(tmp, tag, setting, level, did, extra=()):
    out = str(tmp / f"{tag}.csv")
    sd = str(tmp / f"{tag}_npz")
    assert run_g7.main(["--part", "v0", "--settings", setting, "--levels", f"{setting}={level}", "--datasets", did,
                        "--backend", "dry", "--source", "synthetic", "--seeds", "0", "--save_dir", sd,
                        "--out", out, *extra]) == 0
    rows = {r["method"]: r for r in csv.DictReader(open(out))}
    z = np.load(os.path.join(sd, os.listdir(sd)[0]))
    return rows, z


@pytest.mark.parametrize("setting,level,did", [("pll", "0.6", "54"), ("llp", "8", "1510")])
def test_l_cache_same_results(tmp_path, setting, level, did):
    cache = str(tmp_path / "lc")
    r0, z0 = _run(tmp_path, "nocache", setting, level, did)
    r1, z1 = _run(tmp_path, "miss", setting, level, did, ("--l_cache", cache))
    r2, z2 = _run(tmp_path, "hit", setting, level, did, ("--l_cache", cache))
    assert len(os.listdir(cache)) == 1
    assert "l_cache=hit" not in r1["base_start"]["extra"] and "l_cache=hit" in r2["base_start"]["extra"]
    key = "auc" if setting == "llp" else "acc"
    for m in r0:
        assert r0[m][key] == r1[m][key] == r2[m][key], m
    for k in ("P_L", "P_base", "Q_base", "P_warm"):
        assert np.array_equal(z0[k], z1[k]) and np.array_equal(z0[k], z2[k]), k


def test_l_only_prefetch_then_hit(tmp_path):
    cache = str(tmp_path / "lc")
    out = str(tmp_path / "lo.csv")
    assert run_g7.main(["--part", "v0", "--settings", "pll", "--levels", "pll=0.6", "--datasets", "54",
                        "--backend", "dry", "--source", "synthetic", "--seeds", "0,1", "--l_cache", cache,
                        "--l_only", "--out", out]) == 0
    rows = list(csv.DictReader(open(out)))
    assert [r["method"] for r in rows] == ["l_cache", "l_cache"] and all(r["extra"] == "computed" for r in rows)
    r, _ = _run(tmp_path, "after", "pll", "0.6", "54", ("--l_cache", cache))
    assert "l_cache=hit" in r["base_start"]["extra"]
    with pytest.raises(SystemExit):
        run_g7.main(["--part", "v0", "--backend", "dry", "--source", "synthetic", "--l_only", "--out", out])


def test_choose_q_rule(tmp_path, capsys):
    import v1d_choose_q as V
    cal, ref = tmp_path / "cal", tmp_path / "ref"
    cal.mkdir(); ref.mkdir()
    rng = np.random.default_rng(0)
    yte = rng.integers(0, 3, 100)

    def put(path, acc):
        P = np.full((100, 3), 0.1)
        k = int(acc * 100)
        P[np.arange(k), yte[:k]] = 0.8
        P[np.arange(k, 100), (yte[k:] + 1) % 3] = 0.8
        np.savez(path, yte=yte, P_warm=P, P_base=P, P_cold=P)
    for d in V.DATASETS:
        for s in V.SEEDS:
            put(ref / f"v1_tabicl_v2_{d}_pll_0.6_s{s}.npz", 0.90)
            for q, acc in ((0.2, 0.85), (0.3, 0.81), (0.4, 0.70)):
                put(cal / f"v1_tabicl_v2_{d}_pll_inst_{q}_s{s}.npz", acc)
    assert V.choose(str(cal), str(ref))[0] == 0.3            # 0,4 tụt 0,20 > 0,10; 0,3 tụt 0,09
    os.remove(cal / "v1_tabicl_v2_54_pll_inst_0.2_s0.npz")
    V.main(["--calib_dir", str(cal), "--ref_dir", str(ref)])
    assert capsys.readouterr().out.strip() == ""                # chưa đủ ô → không chọn


def test_init_labels_zero_mass_rows_fall_back_to_candidates():
    from icgibbs.weak import PartialLabel
    S = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], bool)
    wm = PartialLabel(S)
    P0 = np.array([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0], [0.2, 0.3, 0.5]])     # hàng 2: P0 = 0 trên {1, 2}
    rng = np.random.default_rng(0)
    y = wm.init_labels(rng, P0)
    assert all(S[i, y[i]] for i in range(3))
    # hàng có khối lượng dương không đổi so với cách cũ (cùng rng)
    W = (S * P0)[[0, 2]]
    W = W / W.sum(1, keepdims=True)
    rng2 = np.random.default_rng(0)
    old0 = rng2.choice(3, p=W[0])
    assert y[0] == old0
