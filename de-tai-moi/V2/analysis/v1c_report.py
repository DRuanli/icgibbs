"""v1c_report.py — chấm lượt V1C theo V2/PHASE1B_PLAN.md (viết và commit TRƯỚC khi có số).

  §4  chọn TabPFN-3 hay 3.5 (hợp lệ: chạy không lỗi với C ≤ 10, chênh hai lượt gọi giống hệt ≤ 1e−4; tỉ số tốc độ)
  P11 TFM mới không có điểm hút thấp trên ô F1 (g > 0 tại ρ ∈ {0,2; 0,35; 0,5}, gieo uniform) ở ≥ 3/4 dataset
  P8  sai số tuyệt đối trung vị của dự đoán r_cuối từ Φ_chuỗi (gieo chain) ≤ 0,10
  P9  sai số trung vị của chain < của uniform (cùng ô)
Chạy: python -I V2/analysis/v1c_report.py [--raw V2/raw/V1C] [--out_dir V2/results]   (từ thư mục de-tai-moi)
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

F1 = [12, 1501, 40966, 28]
LOW_RHO = [0.2, 0.35, 0.5]
NEW_PFN = ["tabpfn_v3", "tabpfn_v3_5"]


def _read(raw, pattern):
    fs = sorted(glob.glob(os.path.join(raw, pattern)))
    if not fs:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f, low_memory=False) for f in fs], ignore_index=True)


def choose_pfn(tm):
    """PHASE1B_PLAN §4."""
    out = {}
    for be in NEW_PFN:
        d = tm[tm.backend == be]
        ch = d[(d.kind == "chain") & (d.n_classes <= 10)]
        cells = d[d.n_classes <= 10][["n_ctx", "n_classes"]].drop_duplicates()
        ok_cells = ch[ch.status == "ok"][["n_ctx", "n_classes"]].drop_duplicates()
        no_error = len(cells) > 0 and len(ok_cells) == len(cells) and not d.status.astype(str).str.startswith("error").any()
        diff = pd.to_numeric(ch[ch.status == "ok"].max_abs_diff, errors="coerce")
        det = bool(len(diff) and diff.max() <= 1e-4)
        med = ch[(ch.status == "ok") & (ch.n_ctx == 2000)].groupby("n_classes").seconds.median()
        out[be] = dict(valid=bool(no_error and det), no_error=bool(no_error), max_abs_diff=float(diff.max()) if len(diff) else None,
                       med_sec_n2000={int(k): float(v) for k, v in med.items()})
    v3, v35 = out["tabpfn_v3"], out["tabpfn_v3_5"]
    ratio = None
    if v3["valid"] and v35["valid"]:
        rs = [v35["med_sec_n2000"][c] / v3["med_sec_n2000"][c] for c in (2, 10)
              if c in v35["med_sec_n2000"] and c in v3["med_sec_n2000"]]
        ratio = float(np.median(rs)) if rs else None
        chosen = "tabpfn_v3_5" if ratio is not None and ratio <= 1.5 else "tabpfn_v3"
    elif v3["valid"]:
        chosen = "tabpfn_v3"
    elif v35["valid"]:
        chosen = "tabpfn_v3_5"
    else:
        chosen = None
    return dict(per_backend=out, ratio_v35_over_v3=ratio, chosen=chosen)


def low_attractor_table(mp):
    """g = Φ − r tại ρ thấp, gieo uniform, seed 0, các dataset F1; đạt nếu g > 0 tại cả ba mức."""
    d = mp[(mp.status == "ok") & mp.dataset_id.isin(F1)].copy()
    d["rho"] = pd.to_numeric(d.rho, errors="coerce")
    d = d[d.rho.round(3).isin(LOW_RHO)]
    d["g"] = d.phi - d.r_actual
    tab = d.pivot_table(index=["backend", "dataset_id"], columns="rho", values="g")
    tab["pass"] = (tab[LOW_RHO] > 0).all(axis=1) & tab[LOW_RHO].notna().all(axis=1)
    return tab


def predict_rfinal(r, g, phi, r0):
    """PHASE1B_PLAN §2."""
    df = pd.DataFrame(dict(r=r, g=g, phi=phi)).groupby("r", as_index=False).mean().sort_values("r")
    r, g, phi = df.r.to_numpy(), df.g.to_numpy(), df.phi.to_numpy()
    if len(r) == 0:
        return np.nan
    if len(r) == 1:
        return float(min(1.0, phi[0]))
    r0 = float(np.clip(r0, r[0], r[-1]))
    g0 = float(np.interp(r0, r, g))

    def root(a, ga, b, gb):
        return a + (0.0 - ga) * (b - a) / (gb - ga)

    if g0 >= 0:
        pts = [(r0, g0)] + [(rk, gk) for rk, gk in zip(r, g) if rk > r0]
        for (a, ga), (b, gb) in zip(pts[:-1], pts[1:]):
            if ga >= 0 and gb < 0:
                return float(root(a, ga, b, gb))
        return float(min(1.0, phi[-1]))
    pts = [(r0, g0)] + [(rk, gk) for rk, gk in sorted(zip(r, g), reverse=True) if rk < r0]
    for (a, ga), (b, gb) in zip(pts[:-1], pts[1:]):
        if ga < 0 and gb >= 0:
            return float(root(a, ga, b, gb))
    return float(phi[0])


def chain_predictions(cm):
    d = cm[cm.status == "ok"].copy()
    keys = ["backend", "dataset_id", "setting", "level", "seed", "plant"]
    rows = []
    for k, h in d.groupby(keys):
        rhat = predict_rfinal(h.r_actual.to_numpy(float), (h.phi - h.r_actual).to_numpy(float), h.phi.to_numpy(float),
                              float(h.r_start_warm.iloc[0]))
        rf = float(h.r_final_warm.iloc[0])
        rows.append(dict(zip(keys, k), r_start=float(h.r_start_warm.iloc[0]), r_final=rf, r_hat=rhat,
                         abs_err=abs(rhat - rf)))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--raw", default=os.path.join(here, "raw", "V1C"))
    ap.add_argument("--out_dir", default=os.path.join(here, "results"))
    a = ap.parse_args(argv)
    pd.set_option("display.width", 220)
    summ = {}
    print("# Lượt V1C — chấm theo PHASE1B_PLAN (thăm dò)\n")

    tm = _read(a.raw, "v1c_w4_timing.csv")
    if len(tm):
        ok = tm[(tm.status == "ok") & (tm.kind == "chain")]
        print("## A. Thời gian trung vị mỗi lượt gọi (giây, n_estimators = 1)")
        print(ok.pivot_table(index="backend", columns=["n_ctx", "n_classes"], values="seconds", aggfunc="median").round(3).to_string())
        fin = tm[(tm.status == "ok") & (tm.kind == "final")]
        if len(fin):
            print("\nDự đoán cuối, (2.000, 10):", fin.groupby("backend").seconds.median().round(3).to_dict())
        print("Bộ nhớ GPU đỉnh (MB):", pd.to_numeric(ok.peak_gpu_mb, errors="coerce").groupby(ok.backend).max().to_dict())
        print("Chênh lớn nhất giữa hai lượt gọi giống hệt:", pd.to_numeric(ok.max_abs_diff, errors="coerce").groupby(ok.backend).max().to_dict())
        print("Trạng thái khác ok/unsupported:", tm[~tm.status.isin(["ok", "unsupported"])][["backend", "n_ctx", "n_classes", "status"]].to_dict("records"))
        summ["choice"] = choose_pfn(tm)
        print("\n## §4 Chọn TabPFN-3.x:", json.dumps(summ["choice"], ensure_ascii=False, indent=1))

    mp = _read(a.raw, "v1c_map_pll_uniform.csv")
    if len(mp):
        tab = low_attractor_table(mp)
        print("\n## P11. g = Φ − r tại ρ thấp, gieo uniform, PLL q 0,6, các dataset F1")
        print(tab.round(3).to_string())
        n_pass = tab.groupby(level=0)["pass"].sum().astype(int).to_dict()
        chosen = summ.get("choice", {}).get("chosen")
        p11 = None if chosen is None else bool(n_pass.get(chosen, 0) >= 3)
        cons = dict(tabpfn_v2_fails_ge3=bool(4 - n_pass.get("tabpfn_v2", 0) >= 3) if "tabpfn_v2" in n_pass else None,
                    tabicl_v2_passes_ge3=bool(n_pass.get("tabicl_v2", 0) >= 3) if "tabicl_v2" in n_pass else None)
        summ["P11"] = dict(n_pass_of_4=n_pass, chosen=chosen, verdict=None if p11 is None else ("ĐÚNG" if p11 else "SAI"),
                           consistency_check=cons)
        print("P11:", json.dumps(summ["P11"], ensure_ascii=False))
    for nm in ("v1c_map_pll_struct.csv", "v1c_map_llp_uniform.csv"):
        m = _read(a.raw, nm)
        if len(m):
            m = m[m.status == "ok"].assign(g=lambda x: x.phi - x.r_actual)
            print(f"\n## Báo cáo kèm: {nm} (g theo ρ)")
            print(m.pivot_table(index=["backend", "dataset_id"], columns="rho", values="g").round(3).to_string())

    cm = _read(a.raw, "v1c_chainmap_*.csv")
    if len(cm):
        print(f"\n## C. Φ_chuỗi: {len(cm)} dòng; trạng thái {cm.status.astype(str).str[:8].value_counts().to_dict()}")
        pr = chain_predictions(cm)
        os.makedirs(a.out_dir, exist_ok=True)
        pr.to_csv(os.path.join(a.out_dir, "v1c_chainmap_pred.csv"), index=False)
        print(pr.groupby(["backend", "setting", "plant"]).abs_err.agg(["size", "median", "mean"]).round(4).to_string())
        ch = pr[pr.plant == "chain"]
        med_chain = float(ch.abs_err.median()) if len(ch) else None
        cell = ["backend", "dataset_id", "setting", "level", "seed"]
        both = pr[pr.plant.isin(["chain", "uniform"])].pivot_table(index=cell, columns="plant", values="abs_err").dropna()
        summ["P8"] = dict(median_abs_err_chain=med_chain, n_cells=int(len(ch)),
                          verdict=None if med_chain is None else ("ĐÚNG" if med_chain <= 0.10 else "SAI"))
        summ["P9"] = dict(n_cells=int(len(both)),
                          median_chain=float(both["chain"].median()) if len(both) else None,
                          median_uniform=float(both["uniform"].median()) if len(both) else None)
        summ["P9"]["verdict"] = None if not len(both) else \
            ("ĐÚNG" if summ["P9"]["median_chain"] < summ["P9"]["median_uniform"] else "SAI")
        print("P8:", json.dumps(summ["P8"], ensure_ascii=False))
        print("P9:", json.dumps(summ["P9"], ensure_ascii=False))
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "v1c_summary.json"), "w") as f:
        json.dump(summ, f, ensure_ascii=False, indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
