"""s3_round3_explore.py — phân tích THĂM DÒ cho vòng phản biện 3 (viết ngày 13/11/2026, SAU khi Study 3 và S3B đã chấm).

Không phần nào ở đây nằm trong STUDY3_PLAN hay S3B_PLAN. Mọi số từ script này phải được ghi là "exploratory" trong bài.
Script không chạy lại gì: chỉ đọc kết quả và dự đoán đã lưu.

  R1  tách H1'' (d1) và H2'' (d2) theo bối cảnh (phản biện TMLR F12)
  R2  so sánh B1 theo bối cảnh: X - GBDT, L - GBDT, L - PL-kNN, v1 - PL-kNN, v1 - GBDT; L so với baseline CPU chọn theo
      bối cảnh như kế hoạch viết (sai lệch D2) (F7, kiểm toán L4, N4, N7)
  R3  tỉ lệ lớp đông nhất trên tập kiểm tra của từng dataset PLL, đặt cạnh GBDT-PLL và PL-kNN (F7)
  R4  độ lệch chuẩn theo seed của từng phương pháp (F12)
  R5  bộ so sánh cùng thông tin cho multi-TFM: chuỗi đơn của mọi TFM trung bình với dự đoán của L; chuỗi khởi từ L
      của mọi TFM; chuỗi đơn TabICL v2 + L; multi-start của TabICL v2 (F8)
  R6  độ đúng nhãn huấn luyện mà chuỗi đơn khôi phục từ số đếm của túi (LLP), cạnh điểm xuất phát lạnh (F20)
  R7  chuỗi 100 vòng: số ô còn tăng theo TFM, giá trị tại vòng 30 và 100, độ lặp lại của 30 vòng đầu (kiểm toán N1-N3)
  R8  số lượt gọi TFM của từng cấu hình, đọc từ CSV thô (kiểm toán L1, N14)
Chạy: python -I S3/s3_round3_explore.py   (từ thư mục de-tai-moi)
Ra:   S3/results/s3_round3_explore.json, s3_round3_datasets.csv
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "V2", "analysis"))
sys.path.insert(0, os.path.join(ROOT, "G4", "code"))
from v1c_report import load_npz_dir, metric, r_final  # noqa: E402

RES = os.path.join(HERE, "results")
RAW3 = os.path.join(HERE, "raw", "S3")
RAWB = os.path.join(HERE, "raw", "S3B")
TFMS = ["tabicl_v2", "tabpfn_v3", "tabdpt", "tabpfn_v2"]
STARTS = ("warm", "base", "cold")
TIE = 0.001


def summ(d, alt="two-sided"):
    d = np.asarray(d, float)
    out = dict(n=int(len(d)), mean=float(d.mean()), median=float(np.median(d)), win=int((d >= TIE).sum()),
               tie=int((np.abs(d) < TIE).sum()), loss=int((d <= -TIE).sum()), min=float(d.min()), max=float(d.max()))
    if np.any(d != 0):
        out["p"] = float(wilcoxon(d, alternative=alt).pvalue)
        out["alternative"] = alt
    return out


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj.tolist()


def main():
    res = {"note": "THĂM DÒ (exploratory) — viết 13/11/2026 sau khi Study 3 và S3B đã chấm; không thuộc kế hoạch nào"}
    ds = pd.read_csv(os.path.join(RES, "s3_datasets.csv"))
    cpu = pd.read_csv(os.path.join(RES, "s3b_cpu_datasets.csv"))
    cells = pd.read_csv(os.path.join(RES, "s3_cells.csv"))

    # R1 ------------------------------------------------------------------------------------------------------------
    res["R1"] = {}
    for st in ("llp", "pll"):
        h = ds[ds.setting == st]
        res["R1"][st] = {"H1": summ(h.d1, "greater"), "H2": summ(h.d2, "greater")}

    # R2 ------------------------------------------------------------------------------------------------------------
    r2 = {}
    for st in ("llp", "pll"):
        h = cpu[cpu.setting == st]
        r2[st] = {"X_gbdt": summ(h.X - h.gbdtself_def), "L_gbdt": summ(h.meanL - h.gbdtself_def),
                  "v1_gbdt": summ(h.meanv1 - h.gbdtself_def)}
        if st == "pll":
            r2[st].update(X_plknn=summ(h.X - h.plknn_def), L_plknn=summ(h.meanL - h.plknn_def),
                          v1_plknn=summ(h.meanv1 - h.plknn_def), L_best=summ(h.meanL - h.best_cpu))
    # L against the baseline of each setting with the better mean (as S3B_PLAN §3 says; deviation D2)
    means = {st: {m: cpu[cpu.setting == st][m].mean() for m in ("gbdtself_def", "plknn_def")} for st in ("llp", "pll")}
    pick = {st: max((m for m in means[st] if not np.isnan(means[st][m])), key=lambda m: means[st][m]) for st in means}
    ref = np.where(cpu.setting == "llp", cpu[pick["llp"]], cpu[pick["pll"]])
    r2["L_vs_setting_baseline"] = summ(cpu.meanL - ref)
    r2["L_vs_setting_baseline"]["baseline"] = pick
    ps =[summ(cpu.X - cpu.gbdtself_def)["p"], summ((cpu.X - cpu.plknn_def).dropna())["p"],
          r2["L_vs_setting_baseline"]["p"]]
    r2["L_vs_setting_baseline"]["p_holm_with_other_two"] = holm(ps)[2]
    res["R2"] = r2

    # R3, R5, R6, seeds of X ---------------------------------------------------------------------------------------
    npz = load_npz_dir(os.path.join(RAW3, "s3_npz"))
    counts = cells.groupby(["dataset_id", "tfm"]).size().unstack()
    by = {}
    for (be, did, st, lev, seed), z in npz.items():
        by.setdefault((did, st, seed), {})[be] = z
    rows = []
    for (did, st, seed), zs in sorted(by.items()):
        ok = [t_ for t_ in TFMS if t_ in zs and counts.loc[did].get(t_, 0) == 3]
        z0 = zs[ok[0]]
        y = np.asarray(z0["yte"])
        assert len({str(zs[t_]["sha256_split"]) for t_ in ok}) == 1
        P = lambda z, s_: np.asarray(z[f"P_{s_}"], float)
        PL = P(z0, "L")
        Pwarm = np.mean([P(zs[t_], "warm") for t_ in ok], 0)
        Pbase = np.mean([P(zs[t_], "base") for t_ in ok], 0)
        PX = np.mean([np.mean([P(zs[t_], s_) for s_ in STARTS], 0) for t_ in ok], 0)
        r = dict(dataset_id=did, setting=st, seed=seed, n_tfm=len(ok), X=metric(PX, y, st), L=metric(PL, y, st),
                 Xwarm=metric(Pwarm, y, st), XwarmL=metric(0.5 * Pwarm + 0.5 * PL, y, st), Xbase=metric(Pbase, y, st))
        if "tabicl_v2" in ok:
            zi = zs["tabicl_v2"]
            r["icl_v1"] = metric(P(zi, "warm"), y, st)
            r["icl_v1L"] = metric(0.5 * P(zi, "warm") + 0.5 * PL, y, st)
            r["icl_base"] = metric(P(zi, "base"), y, st)
            r["icl_A1"] = metric(np.mean([P(zi, s_) for s_ in STARTS], 0), y, st)
        if st == "pll":
            r["majority"] = float(np.mean(y == np.bincount(y).argmax()))
            r["n_classes"] = int(P(z0, "warm").shape[1])
        else:
            # training-label accuracy recovered from bag counts (single chain, mean over TFMs), and the cold start
            r["rfin_warm"] = float(np.mean([r_final(zs[t_], "warm") for t_ in ok]))
            r["rstart_warm"] = float(np.mean([float(zs[t_]["rstart_warm"]) for t_ in ok]))
            r["rstart_cold"] = float(np.mean([float(zs[t_]["rstart_cold"]) for t_ in ok]))
            ytr = np.asarray(z0["ytr"])
            r["majority_train"] = float(max(np.mean(ytr == 0), np.mean(ytr == 1)))
        rows.append(r)
    xc = pd.DataFrame(rows)
    xd = xc.groupby(["setting", "dataset_id"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    xd = xd.merge(cpu[["setting", "dataset_id", "gbdtself_def", "plknn_def", "meanv1", "meanA1"]], on=["setting", "dataset_id"])
    # R3
    pl = xd[xd.setting == "pll"].sort_values("dataset_id")
    res["R3"] = {"by_dataset": pl[["dataset_id", "n_classes", "majority", "gbdtself_def", "plknn_def"]].round(4).to_dict("records"),
                 "n_gbdt_within_002_of_majority": int((pl.gbdtself_def <= pl.majority + 0.02).sum()),
                 "n_gbdt_below_majority": int((pl.gbdtself_def < pl.majority).sum()),
                 "n_gbdt_near_majority_X_too": int(((pl.gbdtself_def <= pl.majority + 0.02) & (pl.X <= pl.majority + 0.02)).sum()),
                 "mean_majority": float(pl.majority.mean()), "n": int(len(pl))}
    # R5
    r5 = {}
    for c in ("Xwarm", "XwarmL", "Xbase", "icl_v1", "icl_v1L", "icl_base", "icl_A1", "meanv1", "meanA1"):
        h = xd.dropna(subset=[c])
        r5[c] = {"vs_X": summ(h[c] - h.X), "vs_L": summ(h[c] - h.L),
                 "mean_llp": float(h[h.setting == "llp"][c].mean()), "mean_pll": float(h[h.setting == "pll"][c].mean()),
                 "n": int(len(h))}
    r5["X"] = {"mean_llp": float(xd[xd.setting == "llp"].X.mean()), "mean_pll": float(xd[xd.setting == "pll"].X.mean())}
    r5["L"] = {"mean_llp": float(xd[xd.setting == "llp"].L.mean()), "mean_pll": float(xd[xd.setting == "pll"].L.mean())}
    res["R5"] = r5
    # R6
    ll = xd[xd.setting == "llp"]
    res["R6"] = {"rfin_warm_median": float(ll.rfin_warm.median()), "rfin_warm_min": float(ll.rfin_warm.min()),
                 "rfin_warm_max": float(ll.rfin_warm.max()), "rstart_cold_median": float(ll.rstart_cold.median()),
                 "rstart_warm_median": float(ll.rstart_warm.median()), "majority_train_median": float(ll.majority_train.median()),
                 "n": int(len(ll))}

    # R4: seed s.d., median over datasets (or pairs) ----------------------------------------------------------------
    sd = {}
    for st in ("llp", "pll"):
        c = cells[cells.setting == st]
        g = c.groupby(["tfm", "dataset_id"])
        sd[st] = {"v1": float(g.v1.std(ddof=1).median()), "A1": float(g.A1.std(ddof=1).median()),
                  "L": float(c.groupby("dataset_id").L.std(ddof=1).median()),
                  "X": float(xc[xc.setting == st].groupby("dataset_id").X.std(ddof=1).median())}
        rawc = pd.read_csv(os.path.join(RAWB, f"s3b_cpu_{st}.csv"))
        m = "auc" if st == "llp" else "acc"
        for meth in rawc.method.unique():
            sd[st][meth] = float(rawc[rawc.method == meth].groupby("dataset_id")[m].std(ddof=1).median())
    res["R4"] = sd

    # R7: 100-sweep chains -----------------------------------------------------------------------------------------
    lc = pd.read_csv(os.path.join(RES, "s3b_long_cells.csv"))
    longz = load_npz_dir(os.path.join(RAWB, "s3b_npz"))
    r7 = {}
    for be in ("tabicl_v2", "tabpfn_v3", "tabdpt"):
        h = lc[lc.backend == be]
        diffs, at30, at100 = [], [], []
        for (b_, did, st, lev, seed), z in longz.items():
            if b_ != be:
                continue
            m100 = np.asarray(z["monitor_warm"], float)
            m30 = np.asarray(npz[(be, did, st, lev, seed)]["monitor_warm"], float)
            diffs.append(float(np.abs(m100[:30] - m30).max()))
            at30.append(m100[29])
            at100.append(m100[99])
        diffs = np.array(diffs)
        r7[be] = {"n": int(len(h)), "still_rising": int(h.still_rising.sum()),
                  "n_differ_first30": int((diffs > 1e-6).sum()), "max_diff_first30": float(diffs.max()),
                  "r_at30": float(np.mean(at30)), "r_at100": float(np.mean(at100)), "n_npz": int(len(diffs))}
    res["R7"] = r7

    # R8: forward passes per configuration -------------------------------------------------------------------------
    calls = {}
    for f in sorted(glob.glob(os.path.join(RAW3, "s3_*_*.csv"))) + sorted(glob.glob(os.path.join(RAWB, "s3b_tfm_*.csv"))):
        if "lprefetch" in f or "hashes" in f:
            continue
        d = pd.read_csv(f)
        if "n_tfm_calls" not in d:
            continue
        d = d[d.status == "ok"]
        for (meth, st), g in d.groupby(["method", "setting"]):
            vals = sorted(set(int(x) for x in g.n_tfm_calls.dropna()))
            key = f"{meth}|{st}"
            calls[key] = sorted(set(calls.get(key, [])) | set(vals))
    res["R8"] = calls

    # R9: the interim look at B3 (cells present after GPU session 1, read from the commit that stored session 1)
    import io
    import subprocess
    have = set()
    for be in ("tabicl_v2", "tabpfn_v3", "tabdpt"):
        path = f"de-tai-moi/S3/raw/S3B/s3b_long_{be}_pll.csv"
        txt = subprocess.run(["git", "-C", ROOT, "show", f"392b8de:{path}"], capture_output=True, text=True, check=True).stdout
        d = pd.read_csv(io.StringIO(txt))
        d = d[(d.status == "ok") & (d.method == "gibbs_init_warm")]
        have |= {(be, int(r.dataset_id), int(r.seed)) for r in d.itertuples()}
    key = list(zip(lc.backend, lc.dataset_id.astype(int), lc.seed.astype(int)))
    m = np.array([k in have for k in key])
    g67 = lc.gain.to_numpy()[m]
    nmiss = int((~m).sum())
    need = -(0.03 * len(lc) - g67.sum()) / nmiss          # mean gain the missing cells needed for P22's 2nd part
    res["R9"] = {"n_session1": int(m.sum()), "n_missing": nmiss, "mean_gain_session1": float(g67.mean()),
                 "loss_needed_per_missing_cell": float(need),
                 "n_looked_other_tfms": int(sum(1 for k, mm in zip(key, m) if mm and k[0] != "tabdpt")),
                 "missing_r30_acc_mean": float(lc[~m].v1.mean())}
    xd.to_csv(os.path.join(RES, "s3_round3_datasets.csv"), index=False)
    json.dump(res, open(os.path.join(RES, "s3_round3_explore.json"), "w"), indent=1)
    print(json.dumps(res, indent=1)[:6000])


if __name__ == "__main__":
    main()
