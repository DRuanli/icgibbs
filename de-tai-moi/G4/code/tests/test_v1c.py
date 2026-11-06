"""V2 lượt V1C (V2/PHASE1B_PLAN.md): Φ_chuỗi dựng lại đúng ô V1 từ npz, và thử tốc độ W4 — chạy khô (backend dry)."""
import csv
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import run_g7  # noqa: E402
import v1c_chain_map  # noqa: E402
import w4_pilot  # noqa: E402


def _rows(p):
    return list(csv.DictReader(open(p)))


@pytest.fixture(scope="module")
def npz_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("v1npz")
    for setting, level, did in [("pll", "0.6", "54"), ("llp", "8", "1510")]:
        assert run_g7.main(["--part", "v0", "--settings", setting, "--levels", f"{setting}={level}", "--datasets", did,
                            "--backend", "dry", "--source", "synthetic", "--seeds", "0", "--save_dir", str(d),
                            "--out", str(d / f"{setting}.csv")]) == 0
    return d


def test_chain_map_rebuilds_cells_and_measures(npz_dir, tmp_path):
    out = str(tmp_path / "cm.csv")
    assert v1c_chain_map.main(["--npz_glob", str(npz_dir / "v1_*.npz"), "--source", "synthetic", "--seeds", "0",
                               "--rho", "0.5,0.9", "--out", out]) == 0
    rows = _rows(out)
    assert rows and all(r["status"] == "ok" for r in rows)
    plants = {(r["setting"], r["plant"]) for r in rows}
    assert plants == {("pll", "uniform"), ("pll", "struct"), ("pll", "chain"), ("llp", "uniform"), ("llp", "chain")}
    for r in rows:
        assert 0.0 <= float(r["phi"]) <= 1.0 and 0.0 <= float(r["r_final_warm"]) <= 1.0
        if r["setting"] == "pll":                       # tỉ lệ đúng của z bám theo ρ (điểm một ứng viên luôn đúng)
            assert float(r["r_actual"]) >= float(r["rho"]) - 0.02
    # chạy lại: không ghi thêm dòng nào (bỏ qua ô đã xong)
    n = len(rows)
    v1c_chain_map.main(["--npz_glob", str(npz_dir / "v1_*.npz"), "--source", "synthetic", "--seeds", "0",
                        "--rho", "0.5,0.9", "--out", out])
    assert len(_rows(out)) == n


def test_chain_map_flags_mismatch(npz_dir, tmp_path):
    src = sorted(p for p in os.listdir(npz_dir) if p.endswith(".npz") and "_pll_" in p)[0]
    bad = tmp_path / src
    z = dict(np.load(npz_dir / src))
    z["weak_hash"] = np.array("0" * 16)
    np.savez(bad, **z)
    out = str(tmp_path / "mm.csv")
    v1c_chain_map.main(["--npz_glob", str(bad), "--source", "synthetic", "--rho", "0.5", "--out", out])
    rows = _rows(out)
    assert rows and all(r["status"] == "mismatch" for r in rows)


def test_planted_llp_chain_keeps_counts():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 80)
    bags = [np.arange(i, i + 8) for i in range(0, 80, 8)]
    Q = np.column_stack([1 - rng.uniform(size=80), np.zeros(80)])
    Q[:, 1] = 1 - Q[:, 0]
    z = v1c_chain_map.planted_llp_chain(y, bags, 0.6, Q, rng)
    assert all(z[b].sum() == y[b].sum() for b in bags) and (z != y).any()


def test_w4_pilot_dry_and_unsupported(tmp_path):
    out = str(tmp_path / "t.csv")
    assert w4_pilot.main(["--backends", "dry", "--grid", "300x2,300x3", "--reps", "2", "--n_query", "50",
                          "--out", out]) == 0
    rows = _rows(out)
    ch = [r for r in rows if r["kind"] == "chain"]
    assert len(ch) == 4 and all(r["status"] == "ok" for r in rows)
    assert all(float(r["max_abs_diff"]) < 1e-12 for r in ch)              # LogisticRegression tất định
    out2 = str(tmp_path / "u.csv")
    w4_pilot.main(["--backends", "tabpfn_v2", "--grid", "300x26", "--out", out2])     # không cần cài tabpfn
    assert [r["status"] for r in _rows(out2)] == ["unsupported"]


def test_sweeps_option_short_chains_and_reset(tmp_path):
    from icgibbs.g7.methods import GIBBS
    sd = tmp_path / "npz"
    assert run_g7.main(["--part", "v0", "--settings", "pll", "--levels", "pll=0.6", "--datasets", "54",
                        "--backend", "dry", "--source", "synthetic", "--seeds", "0", "--sweeps", "3",
                        "--save_dir", str(sd), "--out", str(tmp_path / "p.csv")]) == 0
    z = np.load(sd / os.listdir(sd)[0])
    assert len(z["monitor_warm"]) == 3 and len(z["monitor_cold"]) == 3
    assert run_g7.main(["--part", "v0", "--settings", "pll", "--levels", "pll=0.6", "--datasets", "54",
                        "--backend", "dry", "--source", "synthetic", "--seeds", "1", "--out", str(tmp_path / "q.csv")]) == 0
    assert GIBBS["n_sweeps"] == 30 and GIBBS["burn_in"] == 10        # lần gọi sau trở lại mặc định
    with pytest.raises(SystemExit):
        run_g7.main(["--part", "tfm", "--backend", "dry", "--source", "synthetic", "--sweeps", "3",
                     "--out", str(tmp_path / "r.csv")])


def test_choose_rule(tmp_path, capsys):
    rows = []
    for be, sec in [("tabpfn_v3", 1.0), ("tabpfn_v3_5", 1.4)]:
        for n, C in [(500, 2), (2000, 2), (500, 10), (2000, 10)]:
            for rep in (1, 2):
                rows.append(dict(backend=be, n_ctx=n, n_classes=C, kind="chain", rep=rep, seconds=sec * n / 2000,
                                 max_abs_diff=0.0, status="ok"))
    import pandas as pd
    p = tmp_path / "t.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    assert w4_pilot.choose_pfn(p)["chosen"] == "tabpfn_v3_5"               # tỉ số 1,4 ≤ 1,5
    df = pd.DataFrame(rows)
    df.loc[df.backend == "tabpfn_v3_5", "seconds"] *= 1.2                    # tỉ số 1,68 > 1,5
    df.to_csv(p, index=False)
    assert w4_pilot.choose_pfn(p)["chosen"] == "tabpfn_v3"
    df.loc[df.backend == "tabpfn_v3", "max_abs_diff"] = 1e-3                 # v3 không tất định → chỉ 3.5 hợp lệ
    df.to_csv(p, index=False)
    w4_pilot.main(["--choose", str(p)])
    assert capsys.readouterr().out.strip() == "tabpfn_v3_5"
    w4_pilot.main(["--choose", str(tmp_path / "khong_co.csv")])
    assert capsys.readouterr().out.strip() == ""
