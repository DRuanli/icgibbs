"""v1d_report.py — chấm lượt V1D theo V2/V1D_PLAN.md §2–§3 (viết và commit TRƯỚC khi có số).

  q*   quy tắc §2 (dùng chung G4/code/v1d_choose_q.py)
  P16  v2-đa-TFM: X4 (trung bình A1 của 4 TFM) so với tiên tri và với từng TFM, 23 dataset chung
  P17  A1 với TabPFN-3 và TabDPT: G(A1 − v1) > 0 và thụt lùi ≤ 2, mỗi TFM
  P18  H8′: AUC(c3 → T2) ≥ 0,80 trên các ô mới (TabPFN-3, TabDPT)
  V1D-4 bối cảnh khó (túi 32, PLL phụ thuộc mẫu ở q*): chỉ báo cáo
  kèm  thời gian mỗi ô theo TFM; so khớp từng số với V1C phần E
Chạy: python -I V2/analysis/v1d_report.py [--raw V2/raw/V1D]   (từ thư mục de-tai-moi)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(V2), "G4", "code"))
from v1c_report import a1, auc_or_none, load_npz_dir, mean_tv, metric  # noqa: E402
import v1d_choose_q  # noqa: E402

COMMON = {"llp": [3, 29, 44, 1049, 1489, 1494, 1464, 1462, 1480, 1510],
          "pll": [11, 46, 182, 188, 458, 469, 1475, 1497, 40975, 40982, 54, 36, 23]}
LEV = {"llp": "8", "pll": "0.6"}
TFMS = ["tabpfn_v2", "tabicl_v2", "tabpfn_v3", "tabdpt"]
NEW = ["tabpfn_v3", "tabdpt"]
SEEDS = (0, 1, 2)


def per_cell(pool):
    """Bảng ô: (tfm, setting, did, seed) → v1, A1, L, c3, mã băm lần chia."""
    rows = []
    for (be, did, st, lev, seed), z in pool.items():
        y = z["yte"]
        rows.append(dict(tfm=be, setting=st, level=lev, dataset_id=did, seed=seed, split=str(z["sha256_split"]),
                         v1=metric(z["P_warm"], y, st), A1=metric(a1(z), y, st), L=metric(z["P_L"], y, st),
                         c3=0.5 * float(np.abs(np.asarray(z["Q_warm"], float) - np.asarray(z["Q_base"], float)).sum(1).mean())))
    return pd.DataFrame(rows)


def x4_table(pool, st, datasets, lev):
    out = []
    for did in datasets:
        for seed in SEEDS:
            zs = {t: pool.get((t, did, st, lev, seed)) for t in TFMS}
            if any(v is None for v in zs.values()) or len({str(v["sha256_split"]) for v in zs.values()}) != 1:
                continue
            y = zs["tabpfn_v2"]["yte"]
            r = dict(setting=st, dataset_id=did, seed=seed, X4=metric(np.mean([a1(zs[t]) for t in TFMS], 0), y, st))
            r.update({t: metric(a1(zs[t]), y, st) for t in TFMS})
            out.append(r)
    return pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(V2, "raw", "V1D"))
    ap.add_argument("--out_dir", default=os.path.join(V2, "results"))
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    pd.set_option("display.width", 220)
    summ = {}
    v1 = {}
    for d in ("V1A", "V1B"):
        v1.update(load_npz_dir(os.path.join(V2, "raw", d, "v1_npz")))
    new = load_npz_dir(os.path.join(a.raw, "v1d_npz"))
    pool = dict(v1)
    pool.update(new)
    print(f"# Lượt V1D — chấm theo V1D_PLAN (thăm dò). npz V1: {len(v1)}; npz V1D: {len(new)}\n")

    q, gap = v1d_choose_q.choose(os.path.join(a.raw, "v1d_npz"), os.path.join(V2, "raw", "V1B", "v1_npz"))
    summ["q_star"] = dict(q=q, gap_vs_iid06=gap)
    print("## q* (§2):", json.dumps(summ["q_star"], ensure_ascii=False))

    # P16
    xt = pd.concat([x4_table(pool, st, COMMON[st], LEV[st]) for st in COMMON], ignore_index=True)
    if len(xt):
        full = xt.groupby(["setting", "dataset_id"]).filter(lambda h: len(h) == len(SEEDS))
        g = full.groupby(["setting", "dataset_id"]).mean(numeric_only=True).drop(columns="seed")
        g["oracle"] = g[TFMS].max(axis=1)
        print("\n## P16. A1 theo TFM và X4, trung bình 3 seed")
        print(g.round(4).to_string())
        d = g.X4 - g.oracle
        each = {t: float((g.X4 - g[t]).mean()) for t in TFMS}
        ok = bool(len(g) and d.mean() >= -0.01 and int((d < -0.02).sum()) <= 3 and all(v >= 0 for v in each.values()))
        summ["P16"] = dict(n_datasets=int(len(g)), mean_X4_minus_oracle=float(d.mean()), n_lose_gt_002=int((d < -0.02).sum()),
                           mean_X4_minus_each=each, verdict=("ĐÚNG" if ok else "SAI") if len(g) == 23 else
                           f"{'ĐÚNG' if ok else 'SAI'} (tạm thời, {len(g)}/23 dataset)")
        print("P16:", json.dumps(summ["P16"], ensure_ascii=False))
        g.to_csv(os.path.join(a.out_dir, "v1d_p16_table.csv"))

    # P17, P18
    pc = per_cell({k: z for k, z in pool.items() if k[0] in NEW and k[3] == LEV.get(k[2]) and k[1] in COMMON.get(k[2], [])
                   and k[4] in SEEDS})
    if len(pc):
        summ["P17"] = {}
        for t, h in pc.groupby("tfm"):
            gd = h.groupby(["setting", "dataset_id"]).mean(numeric_only=True)
            R = int((gd.A1 < np.maximum(gd.v1, gd.L) - 0.02).sum())
            G = float((gd.A1 - gd.v1).mean())
            summ["P17"][t] = dict(n_datasets=int(len(gd)), G=G, R=R, ok=bool(G > 0 and R <= 2))
        summ["P17"]["verdict"] = "ĐÚNG" if all(summ["P17"].get(t, {}).get("ok") for t in NEW) else "SAI"
        print("\nP17:", json.dumps(summ["P17"], ensure_ascii=False))
        pc["T2"] = (pc.v1 < np.maximum(pc.A1, pc.L) - 0.02).astype(int)
        npos = int(pc.T2.sum())
        auc = auc_or_none(pc.T2, pc.c3)
        summ["P18"] = dict(n_cells=int(len(pc)), n_pos=npos, AUC=auc,
                           AUC_by_tfm={t: auc_or_none(h.T2, h.c3) for t, h in pc.groupby("tfm")},
                           verdict="không đủ ô" if npos < 5 else ("ĐÚNG" if auc is not None and auc >= 0.80 else "SAI"))
        print("P18:", json.dumps(summ["P18"], ensure_ascii=False))
        pc.to_csv(os.path.join(a.out_dir, "v1d_cells_new_tfm.csv"), index=False)

    # V1D-4
    hard = {k: z for k, z in pool.items() if (k[2] == "llp" and k[3] == "32") or k[2] == "pll_inst"}
    for k, z in load_npz_dir(os.path.join(V2, "raw", "V1C", "v1c_g_npz")).items():
        if k[2] == "llp" and k[3] == "32":
            hard.setdefault(k, z)
    if hard:
        hc = per_cell(hard)
        if q is not None:
            hc = hc[(hc.setting != "pll_inst") | (hc.level.astype(float).round(3) == round(q, 3))]
        print("\n## V1D-4. Bối cảnh khó, trung bình 3 seed (chỉ báo cáo)")
        tab = hc.groupby(["setting", "level", "dataset_id", "tfm"])[["v1", "A1", "L"]].mean().unstack("tfm")
        print(tab.round(4).to_string())
        flip = hc[hc.setting.str.startswith("llp") & (hc.v1 < 0.5)]
        print("Ô LLP có v1 AUC < 0,5 (nhãn đảo):", flip[["tfm", "dataset_id", "seed", "v1", "A1"]].round(3).to_dict("records"))
        hc.to_csv(os.path.join(a.out_dir, "v1d_hard_cells.csv"), index=False)

    # thời gian mỗi ô và so khớp với V1C E
    fs = sorted(glob.glob(os.path.join(a.raw, "v1d_c_*.csv")))
    if fs:
        t = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
        t = t[t.status == "ok"]
        per = t.groupby(["backend", "dataset_id", "setting", "seed"]).seconds.sum()
        hit = t[t.method == "base_start"].assign(hit=lambda x: x.extra.astype(str).str.contains("l_cache=hit"))
        print("\n## Thời gian (giây) mỗi ô (3 chuỗi + L):", per.groupby(level=[0, 2]).median().round(0).to_dict(),
              "| tỉ lệ ô trúng bộ nhớ đệm L:", hit.groupby("backend").hit.mean().round(3).to_dict())
    e = load_npz_dir(os.path.join(V2, "raw", "V1C", "v1c_e_npz"))
    diffs = []
    for k, z in e.items():
        if k[0] in NEW and k in new:
            diffs.append(max(float(np.abs(np.asarray(z[f"P_{s}"], float) - np.asarray(new[k][f"P_{s}"], float)).max())
                             for s in ("warm", "base", "cold")))
    if diffs:
        summ["match_V1C_E"] = dict(n_cells=len(diffs), max_abs_diff=max(diffs))
        print("So khớp với V1C phần E:", summ["match_V1C_E"])
    with open(os.path.join(a.out_dir, "v1d_summary.json"), "w") as f:
        json.dump(summ, f, ensure_ascii=False, indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
