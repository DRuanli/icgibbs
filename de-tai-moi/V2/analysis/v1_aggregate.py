"""v1_aggregate.py — chấm giai đoạn 1 của V2, VIẾT TRƯỚC khi có số (05/11/2026), theo V2/PHASE1_PLAN.md.

Đầu vào: mọi tệp v1_<backend>_<did>_<setting>_<level>_s<seed>.npz dưới V2/raw/V1*/ (run_g7.py --part v0 --save_dir
--oof). Mỗi tệp có Q_<start>, P_<start>, wpll_<start> cho start ∈ {warm, base, cold}; P_L, Poof_L (learner khởi động);
S (PLL) hoặc bag_of, counts (LLP); ytr, yte.
Bộ gộp (PHASE1_PLAN §3): A0 argmax WPLL (đối chiếu), A1 trung bình đều 3 chuỗi, A2 stacking 3 chuỗi, A3 stacking 3 chuỗi
+ learner. Stacking: w = argmax_{w ∈ Δ} Σ_u log Σ_k w_k ℓ_uk, EM từ w đều, dừng khi max |Δw| < 1e-10 hoặc 5.000 vòng.
Quy tắc chọn và P5–P7: PHASE1_PLAN §4–§5. In kết quả, ghi V2/results/v1_summary.json và v1_cells.csv.
Chạy: python -I V2/analysis/v1_aggregate.py   (từ thư mục de-tai-moi)
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(HERE)
ROOT = os.path.dirname(V2)
sys.path.insert(0, os.path.join(ROOT, "G4", "code"))
from icgibbs.llp import count_constrained_log_normalizer  # noqa: E402

RAW, OUT = os.path.join(V2, "raw"), os.path.join(V2, "results")
pd.set_option("display.width", 230)
pd.set_option("display.max_columns", 30)
STARTS = ("warm", "base", "cold")
PHASE0 = {12, 28, 40966, 1501, 50, 6332}                     # loại khỏi giai đoạn 1 (PHASE1_PLAN §1)
AGG = ("A0", "A1", "A2", "A3")
CHOICE = ("A1", "A2", "A3")                                  # thứ tự đơn giản dần → phức tạp (quy tắc hoà)
REG_MARGIN, TIE_G, P6_TOL, P7_TOL = 0.02, 0.002, 0.005, -0.002
PAT = re.compile(r"v1_(?P<backend>.+?)_(?P<did>\d+)_(?P<setting>llp|pll)_(?P<level>[\d.]+)_s(?P<seed>\d+)\.npz$")


def metric(P, y, setting):
    P = np.asarray(P, float)
    return float(roc_auc_score(y, P[:, 1])) if setting == "llp" else float(np.mean(P.argmax(1) == y))


def unit_loglik(Q, z, setting):
    """log ℓ_u cho mỗi đơn vị quan sát yếu: PLL — điểm tự do, log Σ_{c∈S_u} Q_u(c); LLP — túi, log P(số đếm | Q độc lập)."""
    Q = np.clip(np.asarray(Q, float), 1e-12, None)
    Q = Q / Q.sum(1, keepdims=True)
    if setting == "pll":
        S = z["S"].astype(bool)
        free = S.sum(1) >= 2
        return np.log(np.maximum((Q[free] * S[free]).sum(1), 1e-300))
    bag_of, counts = z["bag_of"], z["counts"]
    out = np.empty(len(counts))
    for g in range(len(counts)):
        idx = np.flatnonzero(bag_of == g)
        out[g] = count_constrained_log_normalizer(Q[idx], counts[g])
    return out


def stack_weights(L, max_iter=5000, tol=1e-10):
    """L: (U, K) log ℓ_uk. EM cho trọng số hỗn hợp trên simplex (bài toán lõm ⇒ hội tụ về cực đại toàn cục)."""
    U, K = L.shape
    w = np.full(K, 1.0 / K)
    for _ in range(max_iter):
        lw = np.log(np.maximum(w, 1e-300)) + L
        R = np.exp(lw - logsumexp(lw, axis=1, keepdims=True))
        w_new = R.mean(0)
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    return w / w.sum()


def load_cells():
    rows, weights = [], []
    for f in sorted(glob.glob(os.path.join(RAW, "**", "v1_*.npz"), recursive=True)):
        mt = PAT.search(os.path.basename(f))
        if not mt:
            continue
        backend, did, setting = mt["backend"], int(mt["did"]), mt["setting"]
        level, seed = float(mt["level"]), int(mt["seed"])
        assert did not in PHASE0, f"dataset {did} đã dùng ở giai đoạn 0"
        z = np.load(f)
        y = z["yte"].astype(int)
        Pk = {s: z[f"P_{s}"] for s in STARTS}
        LL = np.column_stack([unit_loglik(z[f"Q_{s}"], z, setting) for s in STARTS] +
                             [unit_loglik(z["Poof_L"], z, setting)])
        wp = {s: float(z[f"wpll_{s}"]) for s in STARTS}
        best = max(wp, key=wp.get)
        w2 = stack_weights(LL[:, :3])
        w3 = stack_weights(LL)
        P = {"A0": Pk[best], "A1": np.mean([Pk[s] for s in STARTS], 0),
             "A2": sum(w2[k] * Pk[s] for k, s in enumerate(STARTS)),
             "A3": sum(w3[k] * Pk[s] for k, s in enumerate(STARTS)) + w3[3] * z["P_L"]}
        r = dict(backend=backend, dataset_id=did, setting=setting, level=level, seed=seed,
                 v1=metric(Pk["warm"], y, setting), L=metric(z["P_L"], y, setting),
                 **{a: metric(P[a], y, setting) for a in AGG}, chosen_A0=best,
                 oof_nonfinite=int(z["oof_nonfinite"]) if "oof_nonfinite" in z.files else 0)
        for s in STARTS:
            r[f"m_{s}"] = metric(Pk[s], y, setting)
        rows.append(r)
        weights.append(dict(backend=backend, setting=setting, dataset_id=did, seed=seed,
                            **{f"A2_w_{s}": w2[k] for k, s in enumerate(STARTS)},
                            **{f"A3_w_{s}": w3[k] for k, s in enumerate(STARTS)}, A3_w_L=w3[3]))
    return pd.DataFrame(rows), pd.DataFrame(weights)


def group_tables(cells):
    """Trung bình 3 seed theo dataset; thụt lùi; R, G theo nhóm (backend, setting)."""
    ds = cells.groupby(["backend", "setting", "dataset_id"])[["v1", "L", *AGG]].mean().reset_index()
    ref = np.maximum(ds.v1, ds.L) - REG_MARGIN
    for a in AGG:
        ds[f"reg_{a}"] = ds[a] < ref
        ds[f"reg_v1_{a}"] = ds[a] < ds.v1 - REG_MARGIN
        ds[f"reg_L_{a}"] = ds[a] < ds.L - REG_MARGIN
        ds[f"d_{a}"] = ds[a] - ds.v1
    grp = ds.groupby(["backend", "setting"]).agg(
        n=("dataset_id", "size"), **{f"R_{a}": (f"reg_{a}", "sum") for a in AGG},
        **{f"G_{a}": (f"d_{a}", "mean") for a in AGG},
        **{f"Rv1_{a}": (f"reg_v1_{a}", "sum") for a in AGG}, **{f"RL_{a}": (f"reg_L_{a}", "sum") for a in AGG})
    return ds, grp


def choose(grp):
    R = {a: int(grp[f"R_{a}"].sum()) for a in CHOICE}
    G = {a: float(grp[f"G_{a}"].mean()) for a in CHOICE}
    rmin = min(R.values())
    cand = [a for a in CHOICE if R[a] == rmin]
    if len(cand) > 1:
        gmax = max(G[a] for a in cand)
        top = [a for a in cand if G[a] == gmax]
        close = [a for a in cand if gmax - G[a] < TIE_G]
        chosen = min(close, key=CHOICE.index) if len(close) > 1 else top[0]
    else:
        chosen = cand[0]
    return chosen, R, G


def main():
    os.makedirs(OUT, exist_ok=True)
    cells, wts = load_cells()
    if cells.empty:
        print("chưa có tệp v1_*.npz dưới V2/raw/")
        return
    cells.to_csv(os.path.join(OUT, "v1_cells.csv"), index=False)
    ds, grp = group_tables(cells)
    print("# Giai đoạn 1 của V2 — chọn bộ gộp chuỗi (PHASE1_PLAN; thăm dò)\n")
    print(f"Số ô: {len(cells)}; số dataset theo nhóm:", grp["n"].to_dict(), "\n")
    print("## Trung bình theo nhóm (AUC cho LLP, accuracy cho PLL)")
    print(cells.groupby(["backend", "setting"])[["v1", "L", *AGG]].mean().round(4).to_string(), "\n")
    print("## Thụt lùi R (so với max(v1, L) − 0,02) và G (trung bình m_A − m_v1) theo nhóm")
    print(grp.round(4).to_string(), "\n")
    chosen, R, G = choose(grp)
    print("## Quy tắc chọn: R tổng", R, "; G trung bình nhóm", {k: round(v, 4) for k, v in G.items()})
    print("   Bộ gộp được chọn:", chosen, "\n")
    pf = grp.loc[grp.index.get_level_values(0) == "tabpfn_v2"]
    res = {"chosen": chosen, "R": R, "G": G}
    if len(pf):
        r = {a: int(pf[f"R_{a}"].sum()) for a in CHOICE}
        res["P5"] = dict(R_tabpfn=r, verdict="ĐÚNG" if r["A3"] < r["A1"] and r["A3"] < r["A2"] else "SAI")
    ic = grp.loc[grp.index.get_level_values(0) == "tabicl_v2"]
    if len(ic):
        gaps = {}
        for idx, row in ic.iterrows():
            gs = [row[f"G_{a}"] for a in CHOICE]
            gaps[idx[1]] = float(max(gs) - min(gs))
        res["P6"] = dict(max_gap_by_setting=gaps, verdict="ĐÚNG" if all(v < P6_TOL for v in gaps.values()) else "SAI")
    gch = {f"{i[0]}|{i[1]}": float(row[f"G_{chosen}"]) for i, row in grp.iterrows()}
    res["P7"] = dict(G_chosen_by_group=gch, verdict="ĐÚNG" if all(v >= P7_TOL for v in gch.values()) else "SAI")
    many = {f"{i[0]}|{i[1]}": int(row[f"R_{chosen}"]) for i, row in grp.iterrows() if row[f"R_{chosen}"] >= 3}
    res["groups_with_3plus_regressions"] = many
    for k in ("P5", "P6", "P7"):
        if k in res:
            print(f"## {k}", json.dumps(res[k], ensure_ascii=False, indent=1))
    print("\n## Nhóm có ≥ 3 thụt lùi với bộ được chọn (giới hạn của v2, PHASE1_PLAN §7):", many or "không có")
    print("\n## Báo cáo kèm: trọng số trung bình")
    print(wts.groupby(["backend", "setting"]).mean(numeric_only=True).drop(columns=["dataset_id", "seed"]).round(3).to_string())
    frac = wts.groupby(["backend", "setting"]).A3_w_L.apply(lambda s: float((s > 0.5).mean()))
    print("Tỉ lệ ô có w_L > 0,5:", frac.round(3).to_dict())
    res["frac_wL_gt_half"] = {f"{k[0]}|{k[1]}": v for k, v in frac.items()}
    print("\n## Kiểm: A0 có trùng chuỗi có WPLL lớn nhất (gibbs_ms) — số ô A0 = m của start được chọn:",
          int(np.sum([np.isclose(r.A0, r[f"m_{r.chosen_A0}"]) for _, r in cells.iterrows()])), "/", len(cells))
    print("Số hàng ngoài fold không hữu hạn (đã thay bằng đều):", int(cells.oof_nonfinite.sum()))
    print("\n## Theo dataset (trung bình 3 seed)")
    print(ds[["backend", "setting", "dataset_id", "v1", "L", *AGG] + [f"reg_{a}" for a in CHOICE]].round(4).to_string(index=False))
    with open(os.path.join(OUT, "v1_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
