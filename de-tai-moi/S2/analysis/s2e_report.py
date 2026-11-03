"""
s2e_report.py — lượt thăm dò S2E (G9_PLAN O3, O4): TabICL v2 trên dataset cũ của NC1 (tập phát triển của S2).
Viết 03/11/2026, trước khi có kết quả. Chỉ mô tả, không kiểm định (thăm dò).

  O3 (R̂):  hai chuỗi Gibbs ấm (seed 0 và 1000), split-R̂ trên nhãn sau burn-in; đặt cạnh R̂ của TabPFN v2 ở NC1
            (G8/results/rhat_summary.csv, cùng 3 dataset nhỏ mỗi nhóm). Thêm 4 dataset ≥ 8 lớp ở PLL q 0,6.
  O4 (MH):  chuỗi khối có sửa MH (chính xác theo Định lý 1 / Mệnh đề 4(a)) so với chuỗi khối xấp xỉ mặc định, cùng seed,
            ghép cặp với gibbs_warm TabICL v2 của lượt L6 (G7/raw/L6). Metric chính: AUC (LLP), accuracy (PLL); kèm
            log loss và tỉ lệ chấp nhận MH.
Nhóm dataset: "small3" = 3 dataset nhỏ nhất mỗi nhóm (quy tắc R̂ của NC1); "many" = mfeat-factors, optdigits, semeion,
MiceProtein (≥ 8 lớp).
Chạy:  python s2e_report.py   → S2/results/s2e_*.csv, s2e_summary.json
"""
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
S2 = os.path.dirname(HERE)
ROOT = os.path.dirname(S2)
RAW = os.path.join(S2, "raw", "S2E")
MANY = {12, 28, 1501, 40966}
KEY = ["dataset_id", "setting", "level", "seed"]
PRIMARY = {"llp": "auc", "pll": "acc"}


def parse(extra):
    return dict(kv.split("=", 1) for kv in str(extra).split(";") if "=" in kv)


def read(pattern):
    fs = sorted(glob.glob(os.path.join(RAW, pattern)))
    if not fs:
        return pd.DataFrame()
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d = d[d.status == "ok"].copy()
    d["level"] = d.level.astype(float)
    d["group"] = np.where(d.dataset_id.isin(MANY), "many", "small3")
    return d.drop_duplicates(KEY + ["method"])


def rhat():
    d = read("s2e_rhat*.csv")
    if d.empty:
        return {}
    p = d.extra.map(parse)
    for k in ("frac_rhat_gt_1_1", "rhat_global", "metric_chain_a", "metric_chain_b", "metric_absdiff"):
        d[k] = p.map(lambda o: float(o.get(k, "nan")))
    s = d.groupby(["group", "setting", "level"]).agg(
        n=("seed", "size"), frac_gt_1_1_median=("frac_rhat_gt_1_1", "median"), frac_gt_1_1_max=("frac_rhat_gt_1_1", "max"),
        rhat_global_median=("rhat_global", "median"), metric_absdiff_median=("metric_absdiff", "median"),
        metric_absdiff_max=("metric_absdiff", "max")).reset_index()
    nc1 = pd.read_csv(os.path.join(ROOT, "G8", "results", "rhat_summary.csv"))
    nc1 = nc1[nc1.setting.isin(["llp", "pll"]) & nc1.level.isin([8.0, 0.3, 0.6])]
    s = s.merge(nc1[["setting", "level", "frac_gt_1_1_median", "metric_absdiff_median"]].rename(
        columns={"frac_gt_1_1_median": "tabpfn_frac_gt_1_1_median", "metric_absdiff_median": "tabpfn_metric_absdiff_median"}),
        on=["setting", "level"], how="left")
    s.loc[s.group == "many", ["tabpfn_frac_gt_1_1_median", "tabpfn_metric_absdiff_median"]] = np.nan  # NC1 không đo
    s.round(4).to_csv(os.path.join(S2, "results", "s2e_rhat_summary.csv"), index=False)
    d[["group", "dataset", "setting", "level", "seed", "frac_rhat_gt_1_1", "rhat_global", "metric_chain_a",
       "metric_chain_b", "metric_absdiff"]].round(4).to_csv(os.path.join(S2, "results", "s2e_rhat_cells.csv"), index=False)
    return s.round(4).to_dict("records")


def mh():
    a = read("s2e_mh*.csv")
    if a.empty:
        return {}
    ref = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(ROOT, "G7", "raw", "L6", "g7_tabicl_*.csv")))])
    ref = ref[(ref.status == "ok") & (ref.method == "gibbs_warm")].copy()
    ref["level"] = ref.level.astype(float)
    ref = ref.drop_duplicates(KEY).set_index(KEY)
    acc = a.extra.map(parse).map(lambda o: o.get("accept", ""))
    a["accept_mean"] = acc.map(lambda v: float(np.mean([float(x) for x in v.split("/")])) if v else np.nan)
    rows, cells = [], []
    for (grp, st, lv), g in a.groupby(["group", "setting", "level"]):
        met = PRIMARY[st]
        j = g.set_index(KEY).join(ref[[met, "logloss"]], rsuffix="_approx", how="inner")
        dm = j[met] - j[met + "_approx"]                   # dương = chuỗi có sửa MH tốt hơn
        dl = j["logloss_approx"] - j["logloss"]
        rows.append(dict(group=grp, setting=st, level=lv, n=len(j), n_datasets=int(j.index.get_level_values(0).nunique()),
                         metric=met, mh_minus_approx=float(dm.mean()), min=float(dm.min()), max=float(dm.max()),
                         mh_better=int((dm > 0).sum()), ties=int((dm == 0).sum()), mh_worse=int((dm < 0).sum()),
                         logloss_mh_better=float(dl.mean()), accept_mean=float(j.accept_mean.mean())))
        cells.append(j.assign(group=grp, diff=dm, diff_logloss=dl)[
            ["group", "dataset", met, met + "_approx", "diff", "diff_logloss", "accept_mean"]].reset_index())
    s = pd.DataFrame(rows)
    s.round(4).to_csv(os.path.join(S2, "results", "s2e_mh_summary.csv"), index=False)
    pd.concat(cells).round(4).to_csv(os.path.join(S2, "results", "s2e_mh_cells.csv"), index=False)
    return s.round(4).to_dict("records")


def main():
    os.makedirs(os.path.join(S2, "results"), exist_ok=True)
    out = dict(rhat=rhat(), mh=mh())
    json.dump(out, open(os.path.join(S2, "results", "s2e_summary.json"), "w"), indent=1, ensure_ascii=False, default=float)
    print(json.dumps(out, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
