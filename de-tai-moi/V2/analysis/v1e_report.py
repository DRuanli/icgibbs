"""v1e_report.py — chấm lượt V1E theo V2/V1E_PLAN.md (viết và commit TRƯỚC khi có số).

  E1   q* (V1D_PLAN §2) và PLL phụ thuộc mẫu ở q* với 4 TFM — chỉ báo cáo
  P19  X3 (TabICL, TabPFN-3, TabDPT) trên 32 dataset V1 chỉ có TabICL
  P20  A1 với TabPFN-3 và TabDPT trên 32 dataset đó: G > 0 và thụt lùi ≤ 3, mỗi TFM
  P21  H8′: AUC(c3 → T2) ≥ 0,80 trên 192 ô mới
  E2   tóm tắt đếm dataset mở rộng
Chạy: python -I V2/analysis/v1e_report.py [--raw V2/raw/V1E]   (từ thư mục de-tai-moi)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(V2), "G4", "code"))
from v1c_report import a1, auc_or_none, load_npz_dir, metric  # noqa: E402
from v1d_report import per_cell  # noqa: E402
import v1d_choose_q  # noqa: E402

T3 = ["tabicl_v2", "tabpfn_v3", "tabdpt"]
NEW = ["tabpfn_v3", "tabdpt"]
SEEDS = (0, 1, 2)
LEV = {"llp": "8", "pll": "0.6"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(V2, "raw", "V1E"))
    ap.add_argument("--out_dir", default=os.path.join(V2, "results"))
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    pd.set_option("display.width", 220)
    summ = {}
    v1a = load_npz_dir(os.path.join(V2, "raw", "V1A", "v1_npz"))
    v1b = load_npz_dir(os.path.join(V2, "raw", "V1B", "v1_npz"))
    new = load_npz_dir(os.path.join(a.raw, "v1e_npz"))
    calib_dir = os.path.join(a.raw, "v1d_npz")
    if not os.path.isdir(calib_dir):
        calib_dir = os.path.join(V2, "raw", "V1D", "v1d_npz")
    print(f"# Lượt V1E — chấm theo V1E_PLAN (thăm dò). npz V1E: {len(new)}\n")

    q, gap = v1d_choose_q.choose(calib_dir, os.path.join(V2, "raw", "V1B", "v1_npz"))
    summ["q_star"] = dict(q=q, gap_vs_iid06=gap)
    print("## E1. q*:", json.dumps(summ["q_star"], ensure_ascii=False))
    inst = {k: z for k, z in {**load_npz_dir(calib_dir), **new}.items() if k[2] == "pll_inst"}
    if inst and q is not None:
        hc = per_cell(inst)
        hc = hc[hc.level.astype(float).round(3) == round(q, 3)]
        print(hc.groupby(["dataset_id", "tfm"])[["v1", "A1", "L"]].mean().unstack("tfm").round(4).to_string())
        hc.to_csv(os.path.join(a.out_dir, "v1e_inst_cells.csv"), index=False)

    # 32 dataset V1 chỉ có TabICL
    pfn_ds = {(k[1], k[2]) for k in v1a}
    targets = sorted({(k[1], k[2]) for k in v1b if (k[1], k[2]) not in pfn_ds and k[3] == LEV.get(k[2])})
    pool = {**v1b, **new}
    rows = []
    for did, st in targets:
        for seed in SEEDS:
            zs = {t: pool.get((t, did, st, LEV[st], seed)) for t in T3}
            if any(v is None for v in zs.values()) or len({str(v["sha256_split"]) for v in zs.values()}) != 1:
                continue
            y = zs["tabicl_v2"]["yte"]
            r = dict(setting=st, dataset_id=did, seed=seed, X3=metric(np.mean([a1(zs[t]) for t in T3], 0), y, st))
            r.update({t: metric(a1(zs[t]), y, st) for t in T3})
            rows.append(r)
    xt = pd.DataFrame(rows)
    print(f"\n## P19. {len(targets)} dataset đích; ô đủ 3 TFM: {len(xt)}")
    if len(xt):
        full = xt.groupby(["setting", "dataset_id"]).filter(lambda h: len(h) == len(SEEDS))
        g = full.groupby(["setting", "dataset_id"]).mean(numeric_only=True).drop(columns="seed")
        g["oracle"] = g[T3].max(axis=1)
        print(g.round(4).to_string())
        d = g.X3 - g.oracle
        each = {t: float((g.X3 - g[t]).mean()) for t in T3}
        ok = bool(len(g) and d.mean() >= -0.01 and int((d < -0.02).sum()) <= 4 and all(v >= 0 for v in each.values()))
        summ["P19"] = dict(n_datasets=int(len(g)), mean_X3_minus_oracle=float(d.mean()), n_lose_gt_002=int((d < -0.02).sum()),
                           mean_X3_minus_each=each, verdict=("ĐÚNG" if ok else "SAI") if len(g) == len(targets) else
                           f"{'ĐÚNG' if ok else 'SAI'} (tạm thời, {len(g)}/{len(targets)} dataset)")
        print("P19:", json.dumps(summ["P19"], ensure_ascii=False))
        g.to_csv(os.path.join(a.out_dir, "v1e_p19_table.csv"))

    pc = per_cell({k: z for k, z in new.items() if k[0] in NEW and (k[1], k[2]) in set(targets) and k[3] == LEV.get(k[2])})
    if len(pc):
        summ["P20"] = {}
        for t, h in pc.groupby("tfm"):
            gd = h.groupby(["setting", "dataset_id"]).mean(numeric_only=True)
            R = int((gd.A1 < np.maximum(gd.v1, gd.L) - 0.02).sum())
            G = float((gd.A1 - gd.v1).mean())
            summ["P20"][t] = dict(n_datasets=int(len(gd)), G=G, R=R, ok=bool(G > 0 and R <= 3))
        summ["P20"]["verdict"] = "ĐÚNG" if all(summ["P20"].get(t, {}).get("ok") for t in NEW) else "SAI"
        print("\nP20:", json.dumps(summ["P20"], ensure_ascii=False))
        pc["T2"] = (pc.v1 < np.maximum(pc.A1, pc.L) - 0.02).astype(int)
        npos = int(pc.T2.sum())
        auc = auc_or_none(pc.T2, pc.c3)
        summ["P21"] = dict(n_cells=int(len(pc)), n_pos=npos, AUC=auc,
                           AUC_by_tfm={t: auc_or_none(h.T2, h.c3) for t, h in pc.groupby("tfm")},
                           verdict="không đủ ô" if npos < 5 else ("ĐÚNG" if auc is not None and auc >= 0.80 else "SAI"))
        print("P21:", json.dumps(summ["P21"], ensure_ascii=False))
        pc.to_csv(os.path.join(a.out_dir, "v1e_cells_new_tfm.csv"), index=False)

    cs = os.path.join(a.raw, "s3_census_summary.txt")
    if os.path.exists(cs):
        print("\n## E2. Đếm dataset mở rộng\n" + "".join(l for l in open(cs) if "bậc" in l or "mọi dataset" in l or "pool" in l))
    with open(os.path.join(a.out_dir, "v1e_summary.json"), "w") as f:
        json.dump(summ, f, ensure_ascii=False, indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
