"""s3b_report.py — chấm lượt bổ sung S3B theo S3/S3B_PLAN.md (viết và commit TRƯỚC khi có số).

  B1  bản gộp nhiều TFM (X, số của lượt S3) so với baseline CPU: PL-kNN, GBDT tự huấn luyện
  B2  chuỗi đơn so với naive, hard-EM (kiểm định về log loss) và oracle
  B3  chuỗi warm 100 vòng quét trên PLL; dự đoán P22
Chạy: python -I S3/s3b_report.py [--raw S3/raw/S3B]      (từ thư mục de-tai-moi)
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
TFMS = ["tabicl_v2", "tabpfn_v3", "tabdpt", "tabpfn_v2"]
TIE = 0.001
KEY = ["backend", "dataset_id", "setting", "seed"]


def summ(d, test=True):
    d = np.asarray(d, float)
    out = dict(n=int(len(d)), mean=float(d.mean()), median=float(np.median(d)), win=int((d >= TIE).sum()),
               tie=int((np.abs(d) < TIE).sum()), loss=int((d <= -TIE).sum()), n_worse_002=int((d < -0.02).sum()))
    bs = np.random.default_rng(0).choice(d, size=(10000, len(d)), replace=True).mean(1)
    out["ci95"] = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
    if test:
        out["p_two_sided"] = float(wilcoxon(d, alternative="two-sided").pvalue) if np.any(d != 0) else 1.0
    return out


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, run)
    return [float(a) for a in adj]


def load(raw, pattern):
    fs = sorted(glob.glob(os.path.join(raw, pattern)))
    if not fs:
        return pd.DataFrame()
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d = d[d.status == "ok"].drop_duplicates(KEY + ["method"], keep="last")
    d["metric"] = np.where(d.setting == "llp", d.auc, d.acc)
    return d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(HERE, "raw", "S3B"))
    ap.add_argument("--s3raw", default=os.path.join(HERE, "raw", "S3"))
    ap.add_argument("--out_dir", default=os.path.join(HERE, "results"))
    a = ap.parse_args(argv)
    pd.set_option("display.width", 220)
    res = {"note": "S3B: bổ sung, ghi trước sau khi Study 3 đã chấm; không thay đổi kết luận của Study 3"}
    ds = pd.read_csv(os.path.join(a.out_dir, "s3_datasets.csv"))          # X, meanv1, meanA1, meanL theo dataset
    cells = pd.read_csv(os.path.join(a.out_dir, "s3_cells.csv"))          # v1, A1, L, T2 theo ô
    # mã băm lần chia và nhãn yếu của lượt S3, để kiểm S3B dùng đúng cùng ô
    ref = load(a.s3raw, "s3_*_[lp]l[pl].csv")
    ref = ref[ref.method == "gibbs_init_warm"].drop_duplicates(["dataset_id", "setting", "seed"])
    ref = ref.set_index(["dataset_id", "setting", "seed"])[["sha256_split", "weak_hash"]]

    def check(d, name):
        j = d.join(ref, on=["dataset_id", "setting", "seed"], rsuffix="_s3")
        ok = (j.sha256_split == j.sha256_split_s3) & (j.weak_hash == j.weak_hash_s3)
        res[f"hash_mismatch_{name}"] = int((~ok).sum())
        if (~ok).any():
            print(f"!! {name}: {int((~ok).sum())} dòng không khớp mã băm với lượt S3 — bị loại")
        return d[ok.to_numpy()]

    # ------------------------------------------------------------------ B1
    cpu = load(a.raw, "s3b_cpu_*.csv")
    if len(cpu):
        cpu = check(cpu, "cpu")
        full = cpu.groupby(["dataset_id", "setting", "method"]).filter(lambda h: len(h) == 3)
        b = full.groupby(["setting", "dataset_id", "method"]).metric.mean().unstack("method").reset_index()
        b = b.merge(ds[["dataset_id", "X", "meanv1", "meanA1", "meanL"]], on="dataset_id")
        b["best_cpu"] = b[[c for c in ("gbdtself_def", "plknn_def") if c in b]].max(axis=1)
        b.to_csv(os.path.join(a.out_dir, "s3b_cpu_datasets.csv"), index=False)
        print("## B1. Baseline CPU theo dataset\n", b.round(4).to_string(index=False))
        tests = {}
        g = b.dropna(subset=["gbdtself_def"])
        tests["X_vs_gbdt"] = summ(g.X - g.gbdtself_def)
        if "plknn_def" in b:
            k = b.dropna(subset=["plknn_def"])
            tests["X_vs_plknn"] = summ(k.X - k.plknn_def)
        tests["L_vs_bestcpu"] = summ(b.meanL - b.best_cpu)
        names = list(tests)
        for nm, pa in zip(names, holm([tests[n]["p_two_sided"] for n in names])):
            tests[nm]["p_holm"] = pa
        desc = {"v1_vs_gbdt": summ(g.meanv1 - g.gbdtself_def, test=False), "A1_vs_gbdt": summ(g.meanA1 - g.gbdtself_def, test=False),
                "X_vs_bestcpu": summ(b.X - b.best_cpu, test=False)}
        for st in ("llp", "pll"):
            h = b[b.setting == st]
            desc[f"means_{st}"] = {c: float(h[c].mean()) for c in ("X", "meanv1", "meanA1", "meanL", "gbdtself_def", "plknn_def", "best_cpu")
                                   if c in h and h[c].notna().any()}
        res["B1"] = dict(n_datasets=int(len(b)), tests=tests, descriptive=desc,
                         complete=bool(len(g) == 20 and ("plknn_def" not in b or b.plknn_def.notna().sum() == 8)))
        print(json.dumps(res["B1"], ensure_ascii=False, indent=1))

    # ------------------------------------------------------------------ B2
    tfm = load(a.raw, "s3b_tfm_*.csv")
    if len(tfm):
        tfm = check(tfm, "tfm")
        s3 = load(a.s3raw, "s3_*_[lp]l[pl].csv")
        v1 = s3[s3.method == "gibbs_init_warm"][KEY + ["metric", "logloss"]].rename(columns={"metric": "v1", "logloss": "ll_v1"})
        w = tfm.pivot_table(index=KEY, columns="method", values=["metric", "logloss"]).reset_index()
        w.columns = [c[0] if not c[1] else f"{'ll' if c[0] == 'logloss' else 'm'}_{c[1]}" for c in w.columns]
        w = w.merge(v1, on=KEY).merge(cells.rename(columns={"tfm": "backend"})[KEY + ["A1", "L"]], on=KEY)
        need = [c for c in ("m_naive", "m_hard_em_warm", "m_oracle") if c in w]
        pr = w.dropna(subset=need).groupby(["backend", "setting", "dataset_id"]).filter(lambda h: len(h) == 3)
        pr = pr.groupby(["backend", "setting", "dataset_id"]).mean(numeric_only=True).drop(columns="seed").reset_index()
        pr.to_csv(os.path.join(a.out_dir, "s3b_tfm_pairs.csv"), index=False)
        by = pr.groupby(["backend", "setting"]).agg(n=("v1", "size"), naive=("m_naive", "mean"), hard_em=("m_hard_em_warm", "mean"),
                                                    v1=("v1", "mean"), A1=("A1", "mean"), oracle=("m_oracle", "mean"),
                                                    ll_v1=("ll_v1", "mean"), ll_hard_em=("ll_hard_em_warm", "mean"))
        print("\n## B2. Trung bình theo (TFM, setting)\n", by.round(4).to_string())
        dd = pr.groupby(["setting", "dataset_id"]).mean(numeric_only=True).reset_index().merge(ds[["dataset_id", "X"]], on="dataset_id")
        best_or = pr.groupby("dataset_id").m_oracle.max().rename("best_oracle")
        dd = dd.join(best_or, on="dataset_id")
        res["B2"] = dict(
            n_pairs=int(len(pr)), by_tfm_setting=by.round(5).reset_index().to_dict("records"),
            test_logloss_v1_vs_hardem=summ(dd.ll_hard_em_warm - dd.ll_v1),            # dương = chuỗi đơn có log loss thấp hơn
            v1_minus_naive=summ(dd.v1 - dd.m_naive, test=False), v1_minus_hardem=summ(dd.v1 - dd.m_hard_em_warm, test=False),
            oracle_gap_v1=summ(dd.m_oracle - dd.v1, test=False), oracle_gap_A1=summ(dd.m_oracle - dd.A1, test=False),
            X_minus_best_oracle=summ(dd.X - dd.best_oracle, test=False),
            by_setting={st: dict(v1_minus_naive=float((h.v1 - h.m_naive).mean()), v1_minus_hardem=float((h.v1 - h.m_hard_em_warm).mean()),
                                 oracle_gap_v1=float((h.m_oracle - h.v1).mean()), oracle_gap_A1=float((h.m_oracle - h.A1).mean()),
                                 X_minus_best_oracle=float((h.X - h.best_oracle).mean()))
                        for st, h in dd.groupby("setting")},
            complete=bool(len(pr) == 71))
        print(json.dumps({k: v for k, v in res["B2"].items() if k != "by_tfm_setting"}, ensure_ascii=False, indent=1))

    # ------------------------------------------------------------------ B3
    lg = load(a.raw, "s3b_long_*.csv")
    if len(lg):
        lg = check(lg[lg.method == "gibbs_init_warm"], "long")
        c3 = cells.rename(columns={"tfm": "backend"})
        j = lg[KEY + ["metric"]].rename(columns={"metric": "v100"}).merge(c3[KEY + ["v1", "A1", "L", "T2"]], on=KEY)
        rows = []
        for r in j.itertuples():
            f = os.path.join(a.raw, "s3b_npz", f"v1_{r.backend}_{r.dataset_id}_pll_0.6_s{r.seed}.npz")
            mon = np.load(f, allow_pickle=False)["monitor_warm"] if os.path.exists(f) else None
            rows.append(dict(r30=float(np.mean(mon[27:32])) if mon is not None and len(mon) >= 100 else np.nan,
                             r100=float(np.mean(mon[-5:])) if mon is not None and len(mon) >= 100 else np.nan))
        j = pd.concat([j.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
        j["gain"] = j.v100 - j.v1
        j["repaired"] = j.v100 >= np.maximum(j.A1, j.L) - 0.02
        j["still_rising"] = (j.r100 - j.r30) > 0.03
        j.to_csv(os.path.join(a.out_dir, "s3b_long_cells.csv"), index=False)
        done = [t for t, h in j.groupby("backend") if len(h) == 24]
        jj = j[j.backend.isin(done)]
        stuck = jj[jj.T2 == 1]
        frac = float(stuck.repaired.mean()) if len(stuck) else float("nan")
        res["B3"] = dict(tfms_complete=done, n_cells=int(len(jj)), n_stuck=int(len(stuck)),
                         n_repaired=int(stuck.repaired.sum()), frac_repaired=frac, mean_gain=float(jj.gain.mean()) if len(jj) else float("nan"),
                         mean_gain_stuck=float(stuck.gain.mean()) if len(stuck) else float("nan"),
                         n_still_rising=int(jj.still_rising.sum()),
                         by_tfm={t: dict(n=int(len(h)), n_stuck=int((h.T2 == 1).sum()), n_repaired=int(h[h.T2 == 1].repaired.sum()),
                                         mean_gain=float(h.gain.mean()), mean_gain_stuck=float(h[h.T2 == 1].gain.mean()) if (h.T2 == 1).any() else None,
                                         r30=float(h.r30.mean()), r100=float(h.r100.mean())) for t, h in jj.groupby("backend")})
        if len(stuck) and len(done):
            ok = frac < 0.25 and res["B3"]["mean_gain"] < 0.03
            res["B3"]["P22"] = ("ĐÚNG" if ok else "SAI") + ("" if len(done) == 3 else f" (tạm thời, {len(done)}/3 TFM)")
        else:
            res["B3"]["P22"] = "chưa chấm được"
        print("\n## B3.", json.dumps(res["B3"], ensure_ascii=False, indent=1))

    with open(os.path.join(a.out_dir, "s3b_summary.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
