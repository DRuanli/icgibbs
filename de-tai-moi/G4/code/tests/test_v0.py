"""V2 giai đoạn 0 (V2/PHASE0_PREDICTIONS.md): thu dự đoán ngoài khối không đổi chuỗi; WPLL đúng định nghĩa; phần v0
của bộ chạy (nhiều điểm xuất phát, chọn theo WPLL, chạy tiếp); R̂ thăm dò trên S2 cần --explore; vết R̂; ánh xạ phản
ứng mở rộng (lỗi có cấu trúc, LLP giữ số đếm, xác suất biên có điều kiện số đếm)."""
import csv
import itertools
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import run_g7  # noqa: E402
import v0_response_map as vm  # noqa: E402
from icgibbs.conditional import FunctionConditional  # noqa: E402
from icgibbs.llp import count_constrained_log_normalizer  # noqa: E402
from icgibbs.sampler import InContextGibbs  # noqa: E402
from icgibbs.weak import LLP, PartialLabel, make_bags  # noqa: E402

DRY = ["--backend", "dry", "--source", "synthetic", "--seeds", "0"]


def _cell_cond(C):
    """TFM giả: tần suất nhãn của ngữ cảnh trong cùng ô (cột 0), làm trơn — phụ thuộc thật vào nhãn ngữ cảnh."""
    def fn(Xc, yc, Xq):
        out = np.empty((len(Xq), C))
        for r, c in enumerate(Xq[:, 0]):
            cnt = np.bincount(yc[Xc[:, 0] == c], minlength=C) + 0.5
            out[r] = cnt / cnt.sum()
        return out
    return FunctionConditional(fn, C)


def _pll_problem(n=80, C=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 5, size=(n, 1)).astype(float)
    y = (X[:, 0].astype(int) + rng.integers(0, 2, n)) % C
    S = rng.uniform(size=(n, C)) < 0.5
    S[np.arange(n), y] = True
    return X, y, S


def _llp_problem(n=64, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 4, size=(n, 1)).astype(float)
    y = (rng.uniform(size=n) < np.where(X[:, 0] >= 2, 0.8, 0.2)).astype(int)
    bags, counts = make_bags(n, 8, rng, y)
    return X, y, bags, counts


@pytest.mark.parametrize("scheme", ["block", "single"])
def test_collect_q_does_not_change_chain_pll(scheme):
    X, y, S = _pll_problem()
    runs = []
    for cq in (False, True):
        g = InContextGibbs(_cell_cond(3), PartialLabel(S), scheme=scheme, n_blocks=5, n_sweeps=6, burn_in=2, seed=3,
                           collect_q=cq).run(X)
        runs.append(g)
    assert all(np.array_equal(a, b) for a, b in zip(runs[0].samples_, runs[1].samples_))
    assert np.array_equal(runs[0].y0_, runs[1].y0_)
    assert np.isfinite(runs[1].weak_fit())
    with pytest.raises(ValueError):
        runs[0].weak_fit()


def test_collect_q_does_not_change_chain_llp():
    X, y, bags, counts = _llp_problem()
    runs = [InContextGibbs(_cell_cond(2), LLP(bags, counts, len(y)), n_blocks=4, n_sweeps=6, burn_in=2, seed=5,
                           collect_q=cq).run(X) for cq in (False, True)]
    assert all(np.array_equal(a, b) for a, b in zip(runs[0].samples_, runs[1].samples_))
    assert np.isfinite(runs[1].weak_fit())


def test_weak_fit_uniform_conditional_closed_form():
    uni = lambda C: FunctionConditional(lambda Xc, yc, Xq: np.full((len(Xq), C), 1.0 / C), C)
    X, y, S = _pll_problem()
    g = InContextGibbs(uni(3), PartialLabel(S), n_blocks=5, n_sweeps=4, burn_in=1, seed=0, collect_q=True).run(X)
    free = S.sum(1) >= 2
    assert np.isclose(g.weak_fit(), np.mean(np.log(S[free].sum(1) / 3)))
    X, y, bags, counts = _llp_problem()
    g = InContextGibbs(uni(2), LLP(bags, counts, len(y)), n_blocks=4, n_sweeps=4, burn_in=1, seed=0,
                       collect_q=True).run(X)
    want = np.mean([count_constrained_log_normalizer(np.full((len(b), 2), 0.5), c) / len(b) for b, c in zip(bags, counts)])
    assert np.isclose(g.weak_fit(), want)


def test_weak_fit_prefers_true_labels_in_context():
    """Cùng TFM giả: chuỗi bắt đầu từ nhãn thật (chạy 0 vòng sau burn-in ít) có WPLL cao hơn chuỗi bắt đầu lệch."""
    X, y, S = _pll_problem(n=200, seed=1)
    wrong = np.array([np.flatnonzero(s & (np.arange(3) != t))[0] if (s & (np.arange(3) != t)).any() else t
                      for s, t in zip(S, y)])
    fits = [InContextGibbs(_cell_cond(3), PartialLabel(S), n_blocks=5, n_sweeps=2, burn_in=0, seed=0, init=init,
                           collect_q=True, mode="argmax").run(X).weak_fit() for init in (y, wrong)]
    assert fits[0] > fits[1]


def test_v0_part_multistart_dry(tmp_path):
    out = str(tmp_path / "v0.csv")
    args = ["--part", "v0", "--settings", "llp,pll", "--levels", "llp=8;pll=0.6", "--datasets", "1510,54", *DRY,
            "--out", out]
    assert run_g7.main(args) == 0
    rows = list(csv.DictReader(open(out)))
    assert all(r["status"] == "ok" for r in rows), [r["status"] for r in rows if r["status"] != "ok"]
    want = {"base_start", "gibbs_init_warm", "gibbs_init_base", "gibbs_init_cold", "gibbs_ms"}
    for st in ("llp", "pll"):
        got = [r for r in rows if r["setting"] == st]
        assert {r["method"] for r in got} == want, st
        inits = {r["method"]: r for r in got if r["method"].startswith("gibbs_init_")}
        wp = {m.split("_")[-1]: float(dict(kv.split("=", 1) for kv in r["extra"].split(";") if "=" in kv)["wpll"])
              for m, r in inits.items()}
        ms = next(r for r in got if r["method"] == "gibbs_ms")
        assert f"chosen={max(wp, key=wp.get)}" in ms["extra"]
        assert all("r_start=" in r["extra"] and "r_final=" in r["extra"] for r in inits.values())
        best = inits[f"gibbs_init_{max(wp, key=wp.get)}"]
        key = "auc" if st == "llp" else "acc"
        assert ms[key] == best[key]
    n = len(rows)
    assert run_g7.main(args) == 0 and len(list(csv.DictReader(open(out)))) == n
    # chạy tiếp khi thiếu gibbs_ms: chạy lại các điểm xuất phát nhưng không ghi trùng dòng
    keep = [r for r in rows if not (r["method"] == "gibbs_ms" and r["setting"] == "pll")]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(keep)
    assert run_g7.main(args) == 0
    rows2 = list(csv.DictReader(open(out)))
    assert len(rows2) == n and sum(r["method"] == "gibbs_ms" for r in rows2) == 2


def test_v0_starts_validation(tmp_path):
    with pytest.raises(ValueError):
        from icgibbs.g7.methods import run_multistart
        run_multistart(None, None, None, ["nope"], lambda m: True, None)


def test_rhat_on_s2_requires_explore(tmp_path):
    out = str(tmp_path / "r.csv")
    with pytest.raises(SystemExit):
        run_g7.main(["--part", "rhat", "--split", "s2", "--eval-ok", "--settings", "llp", *DRY, "--out", out])
    with pytest.raises(SystemExit):
        run_g7.main(["--part", "tfm", "--split", "s2", "--eval-ok", "--explore", *DRY, "--out", out])
    with pytest.raises(SystemExit):
        run_g7.main(["--part", "v0", "--split", "s2", "--eval-ok", *DRY, "--out", out])


def test_rhat_trace_dry(tmp_path):
    out, tdir = str(tmp_path / "r.csv"), str(tmp_path / "tr")
    assert run_g7.main(["--part", "rhat", "--settings", "llp", "--levels", "llp=8", "--datasets", "1510", *DRY,
                        "--trace_dir", tdir, "--out", out]) == 0
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 1 and rows[0]["status"] == "ok"
    files = os.listdir(tdir)
    assert len(files) == 1
    z = np.load(os.path.join(tdir, files[0]))
    for k in ("samples_a", "samples_b", "monitor_a", "monitor_b", "flips_a", "pred_a", "ytr", "yte", "y0_a"):
        assert k in z.files
    assert z["samples_a"].shape[0] == 20 and len(z["monitor_a"]) == 30


def test_count_marginals_brute_force():
    rng = np.random.default_rng(0)
    for m, k in [(5, 2), (6, 3), (4, 0), (4, 4), (7, 1)]:
        p = rng.uniform(0.05, 0.95, m)
        P = np.column_stack([1 - p, p])
        num, Z = np.zeros(m), 0.0
        for yy in itertools.product([0, 1], repeat=m):
            yy = np.array(yy)
            if yy.sum() != k:
                continue
            w = np.prod(np.where(yy == 1, p, 1 - p))
            Z += w
            num += w * yy
        assert np.allclose(vm.count_marginals(P, k), num / Z)


def test_planted_llp_keeps_counts_and_rate():
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=4000) < 0.5).astype(int)
    bags, _ = make_bags(4000, 8, rng, y)
    for rho in (None, 0.2, 0.8, 1.0):
        z = vm.planted_llp(y, bags, rho, rng)
        assert all(z[b].sum() == y[b].sum() for b in bags)
    r_hi = (vm.planted_llp(y, bags, 0.9, rng) == y).mean()
    r_lo = (vm.planted_llp(y, bags, 0.3, rng) == y).mean()
    assert r_hi > r_lo and (vm.planted_llp(y, bags, 1.0, rng) == y).all()


def test_planted_struct_rate_and_structure():
    rng = np.random.default_rng(0)
    n, C = 3000, 5
    y = rng.integers(0, C, n)
    S = rng.uniform(size=(n, C)) < 0.6
    S[np.arange(n), y] = True
    conf = rng.uniform(size=n)                                 # P0 gán conf cho nhãn thật, phần còn lại chia đều
    P0 = np.full((n, C), 0.0) + ((1 - conf) / (C - 1))[:, None]
    P0[np.arange(n), y] = conf
    z = vm.planted_struct(y, S, 0.6, P0, rng)
    assert S[np.arange(n), z].all()
    free = S.sum(1) >= 2
    assert abs((z == y)[free].mean() - 0.6) < 0.01
    assert conf[free & (z != y)].mean() < conf[free & (z == y)].mean()     # điểm khó bị sai trước


def test_v0_response_map_dry(tmp_path):
    out = str(tmp_path / "llp.csv")
    args = ["--setting", "llp", "--backends", "dry", "--source", "synthetic", "--datasets", "1510", "--seeds", "0",
            "--rho", "u,0.3,0.9", "--out", out]
    assert vm.main(args) == 0
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 3 and all(r["status"] == "ok" for r in rows)
    lo, hi = sorted(rows, key=lambda r: float(r["r_actual"]))[0], sorted(rows, key=lambda r: float(r["r_actual"]))[-1]
    assert float(hi["phi"]) > float(lo["phi"])
    assert vm.main(args) == 0 and len(list(csv.DictReader(open(out)))) == 3
    out2 = str(tmp_path / "pll.csv")
    assert vm.main(["--setting", "pll", "--plant", "struct", "--backends", "dry", "--source", "synthetic",
                    "--datasets", "12", "--seeds", "0", "--rho", "0.5,0.9", "--out", out2]) == 0
    rows = list(csv.DictReader(open(out2)))
    assert len(rows) == 2 and all(r["status"] == "ok" and r["plant"] == "struct" for r in rows)
    with pytest.raises(SystemExit):
        vm.main(["--setting", "llp", "--plant", "struct", "--out", str(tmp_path / "x.csv")])
