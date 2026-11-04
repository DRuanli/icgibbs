"""v0_report.py — phân tích giai đoạn 0 của V2, VIẾT TRƯỚC khi có số (04/11/2026), theo V2/PHASE0_PREDICTIONS.md.

Đầu vào (tìm đệ quy dưới V2/raw/):
  v0_*.csv     run_g7.py --part v0       (gibbs_init_<start>, base_start, gibbs_ms; extra có wpll, r_start, r_final)
  v0x_*.csv    v0_response_map.py        (ánh xạ phản ứng: PLL lưới ρ cao, PLL lỗi có cấu trúc, LLP)
  s2r4_*.csv   run_g7.py --part rhat --split s2 --explore   (R4: chuỗi thứ hai trên Study 2)
In kết luận P1–P4, cổng V0, các bảng mô tả, và R4 (REVIEW_RESPONSE_PLAN §3.R4 (a)–(d)). Ghi JSON tóm tắt.
Chạy: python -I V2/analysis/v0_report.py  (từ thư mục de-tai-moi)
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(HERE)
ROOT = os.path.dirname(V2)
RAW = os.path.join(V2, "raw")
OUT = os.path.join(V2, "results")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

P1_DS, P2_DS, P3_DS = [12, 28, 40966], 1501, [50, 6332]
NAMES = {12: "mfeat-factors", 28: "optdigits", 40966: "MiceProtein", 1501: "semeion", 50: "tic-tac-toe",
         6332: "cylinder-bands"}


def kv(extra):
    out = {}
    for part in str(extra).split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def load(pattern):
    fs = sorted(glob.glob(os.path.join(RAW, "**", pattern), recursive=True))
    if not fs:
        return pd.DataFrame()
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    return d[d.status == "ok"].copy() if "status" in d else d


def metric_col(setting):
    return "auc" if setting == "llp" else "acc"


def multistart():
    d = load("v0_*.csv")
    if d.empty:
        return d
    d = d.drop_duplicates(["backend", "dataset_id", "setting", "level", "seed", "method"], keep="first").copy()
    d["level"] = d.level.astype(float)
    d["m"] = [r["auc"] if r["setting"] == "llp" else r["acc"] for _, r in d.iterrows()]
    ex = d.extra.map(kv)
    for k in ("wpll", "r_start", "r_final"):
        d[k] = ex.map(lambda e, k=k: fnum(e.get(k)))          # gibbs_ms ghi wpll của mọi điểm, không phải một số
    d["start"] = ex.map(lambda e: e.get("start", ""))
    return d


def resp():
    d = load("v0x_*.csv")
    if d.empty:
        return d
    d = d.drop_duplicates(["backend", "dataset_id", "setting", "level", "plant", "seed", "rho"], keep="first").copy()
    d = d[d.rho.astype(str) != "u"].copy()
    d["rho"] = d.rho.astype(float)
    d["gain"] = d.phi - d.r_actual
    return d.groupby(["backend", "dataset_id", "setting", "level", "plant", "rho"], as_index=False)[
        ["r_actual", "phi", "gain"]].mean()


def p1(ms):
    x = ms[(ms.backend == "tabpfn_v2") & (ms.setting == "pll") & (ms.level == 0.6) & ms.dataset_id.isin(P1_DS)]
    gi, bs = x[x.method == "gibbs_init_base"], x[x.method == "base_start"]
    res = dict(n_runs=int(len(gi)), complete=bool(len(gi) == 9 and len(bs) == 9))
    if not len(gi):
        return res | dict(verdict="thiếu số")
    res["r_final"] = {NAMES[k]: g.r_final.round(3).tolist() for k, g in gi.groupby("dataset_id")}
    res["a_all_ge_0.80"] = bool((gi.r_final >= 0.80).all())
    per = {}
    for k in P1_DS:
        a, b = gi[gi.dataset_id == k].m.mean(), bs[bs.dataset_id == k].m.mean()
        per[NAMES[k]] = dict(gibbs_init_base=round(float(a), 4), base_start=round(float(b), 4),
                             ok=bool(a >= b - 0.02))
    res["b_per_dataset"] = per
    res["b_all"] = bool(all(v["ok"] for v in per.values()))
    v = "ĐÚNG" if res["a_all_ge_0.80"] and res["b_all"] else "SAI"
    res["verdict"] = v if res["complete"] else f"chưa đủ ô (tạm: {v})"     # chỉ chấm khi đủ 9 lần chạy
    return res


def p2(ms, rp):
    g = rp[(rp.backend == "tabpfn_v2") & (rp.setting == "pll") & (rp.plant == "uniform") & (rp.dataset_id == P2_DS)
           & (rp.level == 0.6) & rp.rho.isin([0.85, 0.9, 0.95])].sort_values("rho")
    x = ms[(ms.backend == "tabpfn_v2") & (ms.setting == "pll") & (ms.dataset_id == P2_DS) & (ms.method == "gibbs_init_base")]
    res = dict(gains={f"{r.rho}": dict(r=round(r.r_actual, 4), gain=round(r.gain, 4)) for r in g.itertuples()})
    if len(g) < 3 or not len(x):
        return res | dict(verdict="thiếu số")
    r_start = float(x.r_start.mean())
    res["r_start_mean"] = round(r_start, 4)
    if (g.gain < 0).all():
        pred = "sụp"
        hit = int((x.r_final < 0.5).sum()) >= 2
    elif ((g.gain > 0) & (g.r_actual <= r_start)).any():
        pred = "không sụp"
        hit = int((x.r_final >= 0.8).sum()) >= 2
    else:
        return res | dict(prediction="không có", verdict="không chấm")
    res |= dict(prediction=pred, r_final=x.r_final.round(3).tolist(), verdict="ĐÚNG" if hit else "SAI")
    return res


def p3(ms, rp):
    out = {}
    for k in P3_DS:
        g = rp[(rp.backend == "tabpfn_v2") & (rp.setting == "llp") & (rp.dataset_id == k) & (rp.level == 8)]
        x = ms[(ms.backend == "tabpfn_v2") & (ms.setting == "llp") & (ms.level == 8) & (ms.dataset_id == k)]
        gi, bs = x[x.method == "gibbs_init_base"], x[x.method == "base_start"]
        if g.empty or gi.empty or bs.empty:
            out[NAMES[k]] = dict(verdict="thiếu số")
            continue
        r_start = float(gi.r_start.mean())
        j = (g.r_actual - r_start).abs().idxmin()
        near = g.loc[j]
        a, b = float(gi.m.mean()), float(bs.m.mean())
        res = dict(r_start=round(r_start, 4), nearest_r=round(float(near.r_actual), 4), gain=round(float(near.gain), 4),
                   auc_gibbs_init_base=round(a, 4), auc_base_start=round(b, 4))
        if abs(near.r_actual - r_start) > 0.10 or near.gain == 0:
            out[NAMES[k]] = res | dict(prediction="không có", verdict="không chấm")
            continue
        pred_up = near.gain > 0
        hit = (a >= b - 0.01) if pred_up else (a < b - 0.01)
        out[NAMES[k]] = res | dict(prediction="Gibbs ≥ start − 0,01" if pred_up else "Gibbs < start − 0,01",
                                   verdict="ĐÚNG" if hit else "SAI")
    return out


def p4(ms):
    x = ms[ms.method.str.startswith("gibbs_init_")]
    rows = []
    for key, g in x.groupby(["backend", "dataset_id", "setting", "level", "seed"]):
        if len(g) < 2 or g.wpll.isna().any():
            continue
        spread = g.m.max() - g.m.min()
        ch = g.loc[g.wpll.idxmax()]
        rows.append(dict(backend=key[0], dataset=NAMES.get(key[1], key[1]), setting=key[2], level=key[3], seed=key[4],
                         n_starts=len(g), spread=spread, chosen=ch.start, best=g.loc[g.m.idxmax()].start,
                         correct=bool(ch.m >= g.m.max() - 0.005)))
    t = pd.DataFrame(rows)
    if t.empty:
        return dict(verdict="thiếu số"), t
    el = t[t.spread > 0.01]
    rate = float(el.correct.mean()) if len(el) else float("nan")
    res = dict(n_cells=int(len(t)), n_eligible=int(len(el)), n_correct=int(el.correct.sum()), rate=rate,
               verdict="không chấm (không có ô đủ điều kiện)" if not len(el) else ("ĐÚNG" if rate >= 0.8 else "SAI"),
               rate_all_cells=float(t.correct.mean()))
    return res, t


def descriptive(ms):
    x = ms[ms.method.isin(["gibbs_init_warm", "gibbs_init_base", "gibbs_init_cold", "gibbs_ms", "base_start"])]
    tab = x.pivot_table(index=["backend", "setting", "level", "dataset_id"], columns="method", values="m", aggfunc="mean")
    tab.index = tab.index.set_levels([tab.index.levels[3].map(lambda k: NAMES.get(k, k))], level=[3])
    return tab


def fixed_points(rp):
    rows = []
    for key, g in rp.groupby(["backend", "dataset_id", "setting", "level", "plant"]):
        g = g.sort_values("r_actual")
        r, gain = g.r_actual.values, g.gain.values
        fps = []
        for i in range(len(r) - 1):
            if np.sign(gain[i]) != np.sign(gain[i + 1]):
                rs = r[i] - gain[i] * (r[i + 1] - r[i]) / (gain[i + 1] - gain[i])
                fps.append(f"{rs:.2f}{'s' if gain[i] > 0 else 'u'}")
        rows.append(dict(backend=key[0], dataset=NAMES.get(key[1], key[1]), setting=key[2], level=key[3], plant=key[4],
                         r_min=round(r.min(), 2), r_max=round(r.max(), 2), min_gain=round(gain.min(), 3),
                         fixed_points=" ".join(fps) or "-"))
    return pd.DataFrame(rows)


def r4():
    rh = load("s2r4_*.csv")
    if rh.empty:
        return dict(verdict="thiếu số"), None
    sys.path.insert(0, os.path.join(ROOT, "S2", "analysis"))
    import s2_confirmatory as sc
    d, _ = sc.load()
    rh = rh.drop_duplicates(["dataset_id", "setting", "level", "seed"], keep="first").copy()
    rh["level"] = rh.level.astype(float)
    ex = rh.extra.map(kv)
    for k in ("metric_chain_a", "metric_chain_b", "metric_absdiff", "frac_rhat_gt_1_1", "rhat_median", "rhat_p90"):
        rh[k] = ex.map(lambda e, k=k: fnum(e.get(k)))
    key = ["dataset_id", "setting", "level", "seed"]
    gw = d[d.method == "gibbs_warm"].copy()
    gw["m"] = np.where(gw.setting == "llp", gw.auc, gw.acc)
    j = gw[key + ["m"]].merge(rh[key + ["metric_chain_a", "metric_chain_b"]], on=key, how="inner")
    res = dict(n_cells=int(len(rh)), expected_cells=114, n_matched=int(len(j)),     # 18 LLP × 3 + 10 PLL × 2 × 3
               chain_a_vs_registered_maxabs=float((j.metric_chain_a - j.m).abs().max()) if len(j) else None)
    # (a) H2′, H4′ với chuỗi b thay cho gibbs_warm
    db = d.copy()
    mb = rh.set_index(key).metric_chain_b
    idx = db.method == "gibbs_warm"
    k_ = pd.MultiIndex.from_frame(db.loc[idx, key])
    vals = mb.reindex(k_).values
    col = np.where(db.loc[idx, "setting"] == "llp", "auc", "acc")
    for c in ("auc", "acc"):
        sel = idx.copy()
        sel[idx] = col == c
        db.loc[sel, c] = vals[col == c]
    db = db[~(idx & db[["auc", "acc"]].isna().all(axis=1))]
    h2b, _ = sc.h2(db)
    h4b, _ = sc.h4(db)
    res["a_H2prime_chain_b"] = dict(pass_=h2b["pass"], comparator=h2b["comparator"], naive=h2b["naive"],
                                    frac_wins_strongest=h2b["frac_wins_strongest"])
    res["a_H4prime_chain_b"] = dict(pass_=h4b["pass"], **{q: dict(comparator=h4b[q]["comparator"], naive=h4b[q]["naive"])
                                                          for q in ("0.3", "0.6")})
    # (b) |a − b| theo ô; MCSE của trung bình theo dataset (3 seed, sd(a − b)/√2 mỗi ô)
    rh["chain_diff"] = rh.metric_chain_a - rh.metric_chain_b
    res["b_absdiff"] = {f"{st[0]}_{st[1]}": dict(median=float(g.metric_absdiff.median()), p90=float(g.metric_absdiff.quantile(0.9)),
                                 max=float(g.metric_absdiff.max()))
                        for st, g in rh.groupby(["setting", "level"])}
    mc = rh.groupby(["setting", "level", "dataset_id"])["chain_diff"].apply(lambda s: float(np.sqrt(np.mean(s ** 2 / 2)) / np.sqrt(len(s))))
    res["b_mcse_dataset_mean"] = {f"{k[0]}_{k[1]}": dict(median=float(g.median()), max=float(g.max()))
                                  for k, g in mc.groupby(level=[0, 1])}
    # (c) số dataset đổi dấu chênh so với đối thủ chính giữa chuỗi a và chuỗi b
    flips = {}
    for (st, lv), comp in ((("llp", 8.0), "dllplin_def"), (("pll", 0.3), "cc_tun"), (("pll", 0.6), "cc_tun")):
        t = d[(d.setting == st) & (d.level == lv) & (d.method == comp)].copy()
        t["m"] = t.auc if st == "llp" else t.acc
        cm = t.groupby("dataset_id").m.mean()
        r = rh[(rh.setting == st) & (rh.level == lv)].groupby("dataset_id")[["metric_chain_a", "metric_chain_b"]].mean()
        r = r.join(cm.rename("comp"), how="inner")
        sa, sb = np.sign((r.metric_chain_a - r.comp).round(10)), np.sign((r.metric_chain_b - r.comp).round(10))
        flips[f"{st}_{lv}"] = dict(n_datasets=int(len(r)), n_sign_flips=int((sa != sb).sum()))
    res["c_sign_flips"] = flips
    # (d) R̂ trên Study 2
    res["d_rhat"] = {f"{k[0]}_{k[1]}": dict(frac_gt_1_1_median=float(g.frac_rhat_gt_1_1.median()),
                                           frac_gt_1_1_p90=float(g.frac_rhat_gt_1_1.quantile(0.9)),
                                           rhat_p90_median=float(g.rhat_p90.median()))
                     for k, g in rh.groupby(["setting", "level"])}
    res["verdict"] = "đã tính (thăm dò)"
    return res, rh


def main():
    os.makedirs(OUT, exist_ok=True)
    ms, rp = multistart(), resp()
    summary = {}
    print("# Giai đoạn 0 của V2 — kết quả theo PHASE0_PREDICTIONS (thăm dò)\n")
    if not ms.empty:
        print("## Trung bình chỉ số test theo điểm xuất phát (AUC cho LLP, accuracy cho PLL)")
        print(descriptive(ms).round(4).to_string(), "\n")
        summary["P1"] = p1(ms)
        print("## P1", json.dumps(summary["P1"], ensure_ascii=False, indent=1), "\n")
    if not rp.empty:
        print("## Ánh xạ phản ứng: điểm bất động (s = hút, u = không ổn định)")
        print(fixed_points(rp).to_string(index=False), "\n")
    if not ms.empty and not rp.empty:
        summary["P2"] = p2(ms, rp)
        print("## P2", json.dumps(summary["P2"], ensure_ascii=False, indent=1), "\n")
        summary["P3"] = p3(ms, rp)
        print("## P3", json.dumps(summary["P3"], ensure_ascii=False, indent=1), "\n")
    if not ms.empty:
        summary["P4"], t4 = p4(ms)
        print("## P4", json.dumps(summary["P4"], ensure_ascii=False, indent=1))
        if len(t4):
            print(t4.round(4).to_string(index=False), "\n")
        p1v, p4v = summary.get("P1", {}).get("verdict", ""), summary["P4"].get("verdict", "")
        if p1v.startswith("ĐÚNG") and p4v == "ĐÚNG":
            gate = "V0 QUA: nhiều điểm xuất phát + chọn bằng WPLL thành mặc định v2"
        elif p1v.startswith("SAI"):
            gate = "V0 KHÔNG QUA: P1 sai — xem lại cơ chế (W1) trước Study 3"
        elif p1v.startswith("ĐÚNG"):
            gate = "V0 QUA MỘT PHẦN: giữ nhiều điểm xuất phát, tìm tiêu chí chọn khác trên tập phát triển"
        else:
            gate = "chưa đủ số"
        summary["gate_V0"] = gate
        print("## Cổng V0:", gate, "\n")
    summary["R4"], _ = r4()
    print("## R4 (thăm dò)", json.dumps(summary["R4"], ensure_ascii=False, indent=1, default=float))
    with open(os.path.join(OUT, "v0_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
