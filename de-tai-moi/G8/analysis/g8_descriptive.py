"""
g8_descriptive.py — G8: các phần MÔ TẢ của giao thức (Mục 5 chẩn đoán chuỗi, Mục 7 ablation và độ nhạy) và H5 (RQ5).
Không có kiểm định xác nhận nào ở đây trừ H5 (thứ hạng, không có α). Chỉ đọc G7/raw/.

  R̂        : G7/raw/L4/g7_rhat.csv — 2 chuỗi (seed 0 và 1000) trên 3 dataset nhỏ mỗi nhóm, mọi bối cảnh × mức.
  H5 (RQ5)  : L4/g7_rq5.csv (TFM: TabICL v2) + C3/g7_rq5_base.csv (8 baseline PLL). Xếp hạng accuracy trong mỗi lần chia
              giữa Gibbs ấm, TFM ngây thơ và 8 baseline peer-review (oracle, hard-EM, SEM không xếp hạng: oracle không
              phải phương pháp; hard-EM, SEM là đối chứng RQ3). Tiêu chí ghi trước: Gibbs trong top 2. Báo cáo cả thứ
              hạng gộp (nguồn = PLL thật) lẫn từng dataset; kết luận theo cách đọc chặt hơn.
  Ablation  : mỗi biến thể so với `gibbs_warm` của phần chính trên cùng ô (dataset, bối cảnh, mức, seed): chênh metric
              chính (AUC: LLP, PU-SAR; accuracy: PLL) và chênh log loss (dương = biến thể tốt hơn), thắng/thua.
  Cỡ dữ liệu: L4/g7_size_tfm.csv + C3/g7_size_base.csv + phần chính (cỡ đầy đủ).
  TabICL    : L6/g7_tabicl_*.csv, so với TabPFN (phần chính) và với đối thủ chính (mô tả, Wilcoxon không hiệu chỉnh).
Chạy:  python g8_descriptive.py --raw ../../G7/raw --out ../results
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from g8_confirmatory import COMPARATOR, S1_EXCLUDE, h2_like, h4_block, holm, load, load_files

PRIMARY = {"llp": "auc", "pu_sar": "auc", "pll": "acc"}
PLL_PEERS = ["proden_def", "proden_tun", "glwspll_def", "glwspll_tun", "rc_def", "rc_tun", "cc_def", "cc_tun"]
KEY = ["dataset_id", "setting", "level", "seed"]


def parse(extra):
    return dict(kv.split("=", 1) for kv in str(extra).split(";") if "=" in kv)


def rhat(raw, out):
    d = pd.read_csv(os.path.join(raw, "L4", "g7_rhat.csv"))
    d = d[d.status == "ok"].copy()
    p = d.extra.map(parse)
    for k in ("rhat_median", "rhat_p90", "frac_rhat_gt_1_1", "rhat_global", "metric_absdiff", "metric_chain_a",
              "metric_chain_b"):
        d[k] = p.map(lambda o: float(o.get(k, "nan")))
    s = d.groupby(["setting", "level"]).agg(n=("seed", "size"), frac_gt_1_1_median=("frac_rhat_gt_1_1", "median"),
                                            frac_gt_1_1_max=("frac_rhat_gt_1_1", "max"),
                                            rhat_global_median=("rhat_global", "median"),
                                            metric_absdiff_median=("metric_absdiff", "median"),
                                            metric_absdiff_max=("metric_absdiff", "max")).reset_index()
    s.round(4).to_csv(os.path.join(out, "rhat_summary.csv"), index=False)
    d[["dataset", "setting", "level", "seed", "frac_rhat_gt_1_1", "rhat_global", "metric_chain_a", "metric_chain_b",
       "metric_absdiff"]].round(4).to_csv(os.path.join(out, "rhat_cells.csv"), index=False)
    return dict(n=int(len(d)), frac_gt_1_1_median_overall=float(d.frac_rhat_gt_1_1.median()),
                metric_absdiff_median_overall=float(d.metric_absdiff.median()),
                metric_absdiff_max=float(d.metric_absdiff.max()))


def h5(raw, out):
    t = load_files(raw, ["L4"], "g7_rq5.csv")
    b = load_files(raw, ["C3"], "g7_rq5_base.csv")
    d = pd.concat([t, b])
    d = d[d.setting == "pll_real"]
    ranked = ["gibbs_warm", "naive"] + PLL_PEERS
    acc = d.pivot_table(index=["dataset", "seed"], columns="method", values="acc")
    r = acc[ranked].rank(axis=1, ascending=False, method="average")
    per_ds = r.groupby(level="dataset").mean()
    pooled = r.mean()
    per_ds.loc["(gộp: nguồn PLL thật)"] = pooled
    per_ds.T.round(2).to_csv(os.path.join(out, "h5_ranks.csv"))
    acc.groupby(level="dataset").mean().T.round(4).to_csv(os.path.join(out, "rq5_acc.csv"))
    pos = {ds: int(per_ds.loc[ds].rank(method="min")["gibbs_warm"])     # vị trí của Gibbs (1 = tốt nhất) theo hạng TB
           for ds in per_ds.index}
    best_base = acc[PLL_PEERS].groupby(level="dataset").mean().idxmax(axis=1)
    vs = {}
    for ds in acc.index.get_level_values(0).unique():
        a = acc.loc[ds]
        bb = best_base[ds]
        diff = a["gibbs_warm"] - a[bb]
        vs[ds] = dict(best_baseline=bb, mean_diff=float(diff.mean()), wins=int((diff > 0).sum()),
                      losses=int((diff < 0).sum()), p_uncorrected=float(wilcoxon(diff).pvalue) if (diff != 0).any() else 1.0)
    return dict(rank_position=pos, pass_pooled=pos["(gộp: nguồn PLL thật)"] <= 2,
                pass_every_dataset=all(v <= 2 for k, v in pos.items() if not k.startswith("(")),
                gibbs_vs_best_baseline=vs)


def ablations(raw, out, main):
    files = {"blocks": "L4/g7_abl_blocks.csv", "sweeps": "L4/g7_abl_sweeps.csv", "mh": "L4/g7_abl_mh.csv",
             "trans": "L4/g7_abl_trans.csv", "init": "L5*/g7_abl_init_*.csv", "prior": "L5*/g7_abl_prior_*.csv"}
    ref = main[main.method == "gibbs_warm"].set_index(KEY)
    rows, cells = [], []
    for name, pat in files.items():
        fs = sorted(glob.glob(os.path.join(raw, pat)))
        if not fs:
            continue
        a = pd.concat([pd.read_csv(f) for f in fs])
        a = a[a.status == "ok"].copy()
        a["level"] = a.level.astype(float)
        a = a.drop_duplicates(KEY + ["method"])
        for (var, st, lv), g in a.groupby(["method", "setting", "level"]):
            met = PRIMARY[st]
            j = g.set_index(KEY).join(ref[[met, "logloss"]], rsuffix="_ref", how="inner")
            dm = j[met] - j[met + "_ref"]
            dl = j["logloss_ref"] - j["logloss"]
            rows.append(dict(ablation=name, variant=var, setting=st, level=lv, n=len(j), metric=met,
                             mean_diff=float(dm.mean()), wins=int((dm > 0).sum()), ties=int((dm == 0).sum()),
                             losses=int((dm < 0).sum()), mean_diff_logloss=float(dl.mean()),
                             n_datasets=int(j.index.get_level_values(0).nunique())))
            cells.append(j.assign(ablation=name, variant=var, diff=dm, diff_logloss=dl)[
                ["ablation", "variant", met, met + "_ref", "diff", "diff_logloss"]].reset_index())
    s = pd.DataFrame(rows)
    s.round(4).to_csv(os.path.join(out, "ablation_summary.csv"), index=False)
    pd.concat(cells).round(4).to_csv(os.path.join(out, "ablation_cells.csv"), index=False)
    # khởi tạo lạnh ở các dataset sụp (PLL q = 0,6, ≥ 8 lớp): có thoát được không?
    return s


def size_curve(raw, out, main):
    t = load_files(raw, ["L4"], "g7_size_tfm.csv", extra_key=("n_train",))
    b = load_files(raw, ["C3"], "g7_size_base.csv", extra_key=("n_train",))
    s = pd.concat([t, b])
    ids = s.dataset_id.unique()
    full = main[main.dataset_id.isin(ids) & (((main.setting == "llp") & (main.level == 8)) |
                                             ((main.setting == "pll") & (main.level == 0.3)))]
    full = full[full.method.isin(["gibbs_warm", "naive", "hard_em_warm", "oracle", "dllplin_def", "cc_tun"])]
    full = full[~((full.setting == "llp") & (full.method == "cc_tun")) & ~((full.setting == "pll") & (full.method == "dllplin_def"))]
    full = full.assign(size="đầy đủ")
    s = s.assign(size=s.n_train.astype(int).astype(str))
    a = pd.concat([s, full])
    a["metric"] = np.where(a.setting == "pll", a.acc, a.auc)
    a["method"] = a.method.replace({"dllplin_def": "doi_thu_chinh", "cc_tun": "doi_thu_chinh"})
    tab = a.pivot_table(index=["setting", "dataset", "size"], columns="method", values="metric", aggfunc="mean")
    ntr = a.groupby(["setting", "dataset", "size"]).n_train.max()
    tab["n_train"] = ntr
    tab["gibbs_minus_comp"] = tab.gibbs_warm - tab.doi_thu_chinh
    tab["gibbs_minus_naive"] = tab.gibbs_warm - tab.naive
    tab.round(4).to_csv(os.path.join(out, "size_curve.csv"))
    return tab


def tabicl(raw, out, main):
    t = load_files(raw, ["L6"], "g7_tabicl_*.csv")
    rows = []
    for (st, lv), g in t.groupby(["setting", "level"]):
        met = PRIMARY[st]
        ti = g.pivot_table(index="dataset", columns="method", values=met)
        m = main[(main.setting == st) & (main.level == lv)].pivot_table(index="dataset", columns="method", values=met)
        comp = COMPARATOR[st]
        r = dict(setting=st, level=lv, n=int(ti.shape[0]))
        for k in ("gibbs_warm", "naive", "hard_em_warm", "oracle"):
            r[f"tabicl_{k}"] = float(ti[k].mean())
            r[f"tabpfn_{k}"] = float(m[k].mean())
        r["comparator"] = comp
        r["comparator_mean"] = float(m[comp].mean())
        j = pd.concat([ti["gibbs_warm"], m[comp]], axis=1).dropna()
        dd = j.iloc[:, 0] - j.iloc[:, 1]
        r.update(tabicl_gibbs_minus_comp=float(dd.mean()), wins=int((dd > 0).sum()), losses=int((dd < 0).sum()),
                 p_uncorrected=float(wilcoxon(dd).pvalue))
        dn = ti["gibbs_warm"] - ti["naive"]
        r.update(tabicl_gibbs_minus_naive=float(dn.mean()), wins_naive=int((dn > 0).sum()))
        de = ti["gibbs_warm"] - m["gibbs_warm"]
        r.update(tabicl_minus_tabpfn_gibbs=float(de.mean()))
        rows.append(r)
        ti.round(4).to_csv(os.path.join(out, f"tabicl_{st}_{lv}.csv"))
    s = pd.DataFrame(rows)
    s.round(4).to_csv(os.path.join(out, "tabicl_summary.csv"), index=False)
    return s


def tabicl_exploratory(raw, main):
    """THĂM DÒ, chọn SAU khi thấy kết quả: chạy đúng kiểm định của H2-LLP và H4-PLL nhưng với TabICL v2 thay TabPFN v2
    cho mọi phương pháp dựa trên TFM (Gibbs, TFM ngây thơ). Giao thức chỉ định TabPFN v2 là TFM chính và TabICL v2 là
    ablation (Mục 7), nên kết quả ở đây KHÔNG phải kiểm định xác nhận: chọn TFM sau khi xem số là một lối rẽ phân tích."""
    t = load_files(raw, ["L6"], "g7_tabicl_*.csv")
    base = main[~main.method.isin(t.method.unique())]
    d = pd.concat([base, t], ignore_index=True)
    r, _ = h2_like(d, "llp", 8.0, "dllplin_def")
    adj = holm([r["comparator"]["p"], r["naive"]["p"]])
    r["comparator"]["p_holm"], r["naive"]["p_holm"] = float(adj[0]), float(adj[1])
    sig = all(r[k]["p_holm"] < 0.05 and r[k]["mean_diff"] > 0 for k in ("comparator", "naive"))
    r["pass_if_confirmatory"] = bool(sig and r["strongest"]["wins"] >= 7)
    return dict(H2_llp8=r, H4_pll=h4_block(d, None), H4_pll_S1=h4_block(d, None, exclude=S1_EXCLUDE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="../../G7/raw")
    ap.add_argument("--out", default="../results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    main_ = load(a.raw)
    res = dict(rhat=rhat(a.raw, a.out), H5=h5(a.raw, a.out))
    ab = ablations(a.raw, a.out, main_)
    sz = size_curve(a.raw, a.out, main_)
    tb = tabicl(a.raw, a.out, main_)
    res["tabicl_exploratory"] = tabicl_exploratory(a.raw, main_)
    json.dump(res, open(os.path.join(a.out, "g8_descriptive.json"), "w"), indent=1, default=float)
    pd.set_option("display.width", 220)
    print(json.dumps(res, indent=1, default=float, ensure_ascii=False))
    print(pd.read_csv(os.path.join(a.out, "rhat_summary.csv")).to_string())
    print(pd.read_csv(os.path.join(a.out, "h5_ranks.csv"), index_col=0).to_string())
    print(ab.round(4).to_string())
    print(sz.round(3).to_string())
    print(tb.round(4).T.to_string())


if __name__ == "__main__":
    main()
