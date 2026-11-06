"""v1c_report.py — chấm lượt V1C theo V2/PHASE1B_PLAN.md (viết và commit TRƯỚC khi có số).

  §4  chọn TabPFN-3 hay 3.5 (hợp lệ: chạy không lỗi với C ≤ 10, chênh hai lượt gọi giống hệt ≤ 1e−4; tỉ số tốc độ)
  P11 TFM mới không có điểm hút thấp trên ô F1 (g > 0 tại ρ ∈ {0,2; 0,35; 0,5}, gieo uniform) ở ≥ 3/4 dataset
  P8  sai số tuyệt đối trung vị của dự đoán r_cuối từ Φ_chuỗi (gieo chain) ≤ 0,10
  P9  sai số trung vị của chain < của uniform (cùng ô)
  Bổ sung 07/11 (PHASE1B_PLAN §7, viết trước khi có số):
  P15 (D) AUC của s1 (TV trung bình giữa Q̄ của 3 chuỗi dò 3 vòng quét) cho T1 (spread r_cuối V1 ≥ 0,03) ≥ 0,75
  P14 (E) A1 TabPFN-3.x ≥ A1 TabPFN v2 + 0,05 trên ≥ 3/4 dataset F1 PLL
  P13 (E) gộp đều A1 của 4 TFM: trung bình (X − TFM tốt nhất) ≥ −0,01 và thua > 0,02 ở ≤ 2 dataset
  P12 (F) A1 cùng chi phí (3 × 10 vòng quét): G > 0 ở cả 4 nhóm và tổng thụt lùi ≤ 3
  G       cơ chế nhãn yếu mới: chỉ báo cáo
Chạy: python -I V2/analysis/v1c_report.py [--raw V2/raw/V1C] [--out_dir V2/results]   (từ thư mục de-tai-moi)
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "G4", "code"))
from w4_pilot import choose_pfn  # noqa: E402  (một nguồn duy nhất cho quy tắc §4)

F1 = [12, 1501, 40966, 28]
F1_LLP = [50, 6332]
DEV_PLL, DEV_LLP = [54, 36, 23], [31, 37, 1464, 1462, 1480, 1510]
STARTS = ("warm", "base", "cold")
PAT = re.compile(r"v1_(?P<be>.+?)_(?P<did>\d+)_(?P<st>llp|pll)(?:_(?P<var>feat|inst))?_(?P<lev>[\d.]+)_s(?P<seed>\d+)\.npz$")


def load_npz_dir(d):
    """{(backend, did, setting_label, level, seed): npz} cho mọi tệp v1_*.npz trong thư mục d."""
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "v1_*.npz"))):
        m = PAT.search(os.path.basename(f))
        if m:
            st = m["st"] + (f"_{m['var']}" if m["var"] else "")
            out[(m["be"], int(m["did"]), st, m["lev"], int(m["seed"]))] = np.load(f)
    return out


def metric(P, y, st):
    from sklearn.metrics import roc_auc_score
    P = np.asarray(P, float)
    return float(roc_auc_score(y, P[:, 1])) if st.startswith("llp") else float(np.mean(P.argmax(1) == y))


def a1(z):
    return np.mean([np.asarray(z[f"P_{s}"], float) for s in STARTS], axis=0)


def r_final(z, s):
    return float(np.asarray(z[f"monitor_{s}"], float)[-5:].mean())


def mean_tv(z):
    Q = [np.asarray(z[f"Q_{s}"], float) for s in STARTS]
    return float(np.mean([0.5 * np.abs(Q[i] - Q[j]).sum(1).mean() for i in range(3) for j in range(i + 1, 3)]))


def auc_or_none(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, int)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None


def part_d(v1, probe):
    rows = []
    for k, z in probe.items():
        o = v1.get(k)
        if o is None or str(o["sha256_split"]) != str(z["sha256_split"]):
            continue
        rf = [r_final(o, s) for s in STARTS]
        st = k[2]
        y = o["yte"]
        m_v1, m_A1, m_L = metric(o["P_warm"], y, st), metric(a1(o), y, st), metric(o["P_L"], y, st)
        w = [float(z[f"wpll_{s}"]) for s in STARTS]
        rows.append(dict(backend=k[0], dataset_id=k[1], setting=st, seed=k[4], s1=mean_tv(z), s2=max(w) - min(w),
                         T1=int(max(rf) - min(rf) >= 0.03), T2=int(m_v1 < max(m_A1, m_L) - 0.02)))
    return pd.DataFrame(rows)


def part_f(v1, f):
    rows = []
    for k, z in f.items():
        o = v1.get(k)
        if o is None or str(o["sha256_split"]) != str(z["sha256_split"]):
            continue
        y, st = o["yte"], k[2]
        rows.append(dict(backend=k[0], dataset_id=k[1], setting=st, seed=k[4], v1=metric(o["P_warm"], y, st),
                         A1=metric(a1(o), y, st), A1e=metric(a1(z), y, st), L=metric(o["P_L"], y, st)))
    d = pd.DataFrame(rows)
    if not len(d):
        return d, {}
    g = d.groupby(["backend", "setting", "dataset_id"]).mean(numeric_only=True).reset_index()
    g["reg_A1e"] = g.A1e < np.maximum(g.v1, g.L) - 0.02
    grp = g.groupby(["backend", "setting"]).apply(lambda h: pd.Series(dict(
        n=int(len(h)), G_A1e=(h.A1e - h.v1).mean(), G_A1=(h.A1 - h.v1).mean(), R_A1e=int(h.reg_A1e.sum()))),
        include_groups=False)
    ok = bool(len(grp) == 4 and (grp.G_A1e > 0).all() and grp.R_A1e.sum() <= 3)
    return grp, dict(groups=grp.round(4).reset_index().to_dict("records"), total_R=int(grp.R_A1e.sum()),
                     verdict="ĐÚNG" if ok else "SAI", n_groups=int(len(grp)))


def part_e(v1, e, chosen):
    """A1 theo (TFM, dataset), seed 0; mốc TabPFN v2/TabICL v2 của tập phát triển lấy từ V1."""
    tfms = ["tabpfn_v2", "tabicl_v2"] + [b for b in (chosen, "tabdpt") if b]
    pool = {}
    for src in (e, v1):
        for k, z in src.items():
            be, did, st, lev, seed = k
            if seed != 0 or be not in tfms:
                continue
            if (st == "pll" and did in F1 + DEV_PLL and lev == "0.6") or (st == "llp" and did in F1_LLP + DEV_LLP and lev == "8"):
                pool.setdefault((did, st), {}).setdefault(be, z)        # ưu tiên E (đọc trước) nếu trùng
    rows = []
    for (did, st), byb in pool.items():
        hs = {str(z["sha256_split"]) for z in byb.values()}
        y = next(iter(byb.values()))["yte"]
        r = dict(dataset_id=did, setting=st, split_ok=len(hs) == 1)
        for be, z in byb.items():
            r[be] = metric(a1(z), y, st)
        if len(hs) == 1 and all(b in byb for b in tfms):
            r["X"] = metric(np.mean([a1(byb[b]) for b in tfms], axis=0), y, st)
        rows.append(r)
    d = pd.DataFrame(rows).sort_values(["setting", "dataset_id"])
    summ = {}
    if chosen and len(d):
        f1 = d[(d.setting == "pll") & d.dataset_id.isin(F1)].dropna(subset=[chosen, "tabpfn_v2"])
        n_ok = int((f1[chosen] >= f1["tabpfn_v2"] + 0.05).sum())
        summ["P14"] = dict(n_datasets=int(len(f1)), n_pass=n_ok, verdict="ĐÚNG" if n_ok >= 3 else "SAI")
    if "X" in d and d.X.notna().any():
        full = d.dropna(subset=["X"])
        best = full[tfms].max(axis=1)
        mdiff, nlose = float((full.X - best).mean()), int((best - full.X > 0.02).sum())
        summ["P13"] = dict(n_datasets=int(len(full)), mean_X_minus_best=mdiff, n_lose_gt_002=nlose,
                           verdict="ĐÚNG" if (mdiff >= -0.01 and nlose <= 2) else "SAI")
    return d, summ


def part_g(v1, g):
    rows = []
    for k, z in g.items():
        be, did, st, lev, seed = k
        y = z["yte"]
        r = dict(backend=be, dataset_id=did, setting=st, level=lev, seed=seed, v1=metric(z["P_warm"], y, st),
                 A1=metric(a1(z), y, st), L=metric(z["P_L"], y, st))
        ref = v1.get((be, did, st.split("_")[0], "8" if st.startswith("llp") else "0.6", seed))
        if ref is not None and str(ref["sha256_split"]) == str(z["sha256_split"]):
            r["A1_iid_ref"] = metric(a1(ref), ref["yte"], st)
        rows.append(r)
    return pd.DataFrame(rows)
LOW_RHO = [0.2, 0.35, 0.5]


def _read(raw, pattern):
    fs = sorted(glob.glob(os.path.join(raw, pattern)))
    if not fs:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f, low_memory=False) for f in fs], ignore_index=True)


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
    ap.add_argument("--v1_npz", default=",".join(os.path.join(here, "raw", x, "v1_npz") for x in ("V1A", "V1B")))
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
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
        summ["choice"] = choose_pfn(os.path.join(a.raw, "v1c_w4_timing.csv"))
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
    v1 = {}
    for d in a.v1_npz.split(","):
        v1.update(load_npz_dir(d))
    probe = load_npz_dir(os.path.join(a.raw, "v1c_probe_npz"))
    if probe:
        dd = part_d(v1, probe)
        print(f"\n## D. Chuỗi dò: {len(dd)} ô khớp V1; T1 = {int(dd.T1.sum())}, T2 = {int(dd.T2.sum())}")
        res = {}
        for sig in ("s1", "s2"):
            for tg in ("T1", "T2"):
                res[f"AUC_{sig}_{tg}"] = auc_or_none(dd[tg], dd[sig])
        for be, h in dd.groupby("backend"):
            res[f"AUC_s1_T1_{be}"] = auc_or_none(h.T1, h.s1)
        a15 = res["AUC_s1_T1"]
        summ["P15"] = dict(res, n_cells=int(len(dd)), verdict=None if a15 is None else ("ĐÚNG" if a15 >= 0.75 else "SAI"))
        print("P15:", json.dumps(summ["P15"], ensure_ascii=False))
        dd.to_csv(os.path.join(a.out_dir, "v1c_probe_cells.csv"), index=False)
    fn = load_npz_dir(os.path.join(a.raw, "v1c_f_npz"))
    if fn:
        grp, s12 = part_f(v1, fn)
        print("\n## F. A1 cùng chi phí (3 × 10 vòng quét) so với v1 (1 × 30) và A1 (3 × 30)")
        print(grp.round(4).to_string())
        summ["P12"] = s12
        print("P12:", json.dumps(s12, ensure_ascii=False, default=str))
    en = load_npz_dir(os.path.join(a.raw, "v1c_e_npz"))
    if en:
        de, s13 = part_e(v1, en, summ.get("choice", {}).get("chosen"))
        print("\n## E. A1 theo TFM, seed 0 (X = trung bình đều 4 TFM)")
        print(de.round(4).to_string(index=False))
        summ.update(s13)
        for kk in ("P14", "P13"):
            if kk in s13:
                print(f"{kk}:", json.dumps(s13[kk], ensure_ascii=False))
        de.to_csv(os.path.join(a.out_dir, "v1c_e_table.csv"), index=False)
    gn = load_npz_dir(os.path.join(a.raw, "v1c_g_npz"))
    if gn:
        dg = part_g(v1, gn)
        print("\n## G. Cơ chế nhãn yếu mới (TabICL v2), trung bình 3 seed — chỉ báo cáo")
        print(dg.groupby(["setting", "level", "dataset_id"]).mean(numeric_only=True).drop(columns="seed").round(4).to_string())
        dg.to_csv(os.path.join(a.out_dir, "v1c_g_cells.csv"), index=False)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "v1c_summary.json"), "w") as f:
        json.dump(summ, f, ensure_ascii=False, indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
