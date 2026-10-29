"""
g8_confirmatory.py — G8: kiểm định xác nhận đúng kế hoạch thống kê đã đóng băng (G5/EXPERIMENT_PROTOCOL Mục 6, cùng
các deviation đến 13 và tiêu chí thành công ở G3/RESEARCH_PLAN §5). Chỉ đọc kết quả thô G7/raw/ (không sửa tay).

  H2-LLP (chính): LLP túi 8, AUC. Đơn vị = trung bình 3 seed mỗi dataset. Wilcoxon signed-rank hai phía, Gibbs ấm so
      với (i) DLLP tuyến tính mặc định, (ii) TFM ngây thơ; Holm trên 2 phép so sánh (deviation 7). Đạt nếu cả hai có ý
      nghĩa sau Holm theo hướng Gibbs tốt hơn VÀ Gibbs thắng ≥ 7/10 dataset trước baseline peer-review mạnh nhất trên
      tập đánh giá (ứng viên có AUC trung bình cao nhất trên tập đánh giá).
  H4-PLL (chính): accuracy, q = 0,3 và 0,6, 14 dataset. Wilcoxon hai phía với CC tune (đối thủ chính, deviation 13) và
      TFM ngây thơ; Holm trên 4 phép so sánh. Tiêu chí ghi trước: đạt hoặc vượt baseline mạnh nhất trên ≥ 60% dataset
      (mọi cách đọc được báo cáo, xem h4_block). Chạy hai lần: 14 dataset và S1 (bỏ 4 dataset, giao thức §1.5).
  H3: Gibbs ấm so với hard-EM (cùng TFM, cùng khởi động), log loss (PU-SCAR: bản khớp tiên nghiệm), đơn vị = ô
      (dataset × mức), các bối cảnh có RQ3 ở giao thức Mục 0 (LLP 8, 16; PU-SAR; PU-SCAR; PLL; SSL; nhiễu). Bayesian
      signed-rank (Benavoli et al. 2017), ROPE ±0,005. Đạt nếu Gibbs tốt hơn ở ≥ 70% ô; bác bỏ nếu P(ROPE) > 0,9 hoặc
      Spearman(mức yếu, khoảng cách) ≤ 0. PU-SAR: hard-EM được biết propensity nên so với `gibbs_known_warm` (cùng
      thông tin); bản so với `gibbs_warm` báo cáo kèm.
  Bối cảnh phụ: cùng kiểm định như H2 với đối thủ chính của bối cảnh (docs/primary_comparators.md), báo cáo đầy đủ.
  Phân tích phụ (luôn báo cáo): Gibbs so với từng baseline peer-review, Holm trên mọi cặp baseline × mức.
  Thăm dò (deviation 6): D1–D3.
Chạy:  python g8_confirmatory.py --raw ../../G7/raw --out ../results
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

METRIC = {"llp": ("auc", True), "pu_sar": ("auc", True), "pu_scar": ("logloss_pm", False), "ssl": ("auc", True),
          "noisy": ("auc", True), "pll": ("acc", True)}
COMPARATOR = {"llp": "dllplin_def", "pll": "cc_tun", "pu_sar": "nnpu", "pu_scar": "vpu_tun", "noisy": "cl_tfm",
              "ssl": "st_tun"}
CANDIDATES = {
    "llp": ["dllp_def", "dllp_tun", "dllplin_def", "dllplin_tun", "llpvat_def", "llpvat_tun", "easyllp_def", "easyllp_tun",
            "glws_def", "glws_tun"],
    "pll": ["proden_def", "proden_tun", "glwspll_def", "glwspll_tun", "rc_def", "rc_tun", "cc_def", "cc_tun"],
    "pu_sar": ["nnpu", "nnpu_tun", "upu", "upu_tun", "sar_em", "sar_em_tuned", "glwspu_def", "glwspu_tun", "vpu_def",
               "vpu_tun"],
    "pu_scar": ["nnpu", "nnpu_tun", "upu", "upu_tun", "glwspu_def", "glwspu_tun", "vpu_def", "vpu_tun"],
    "noisy": ["forward", "forward_def", "forward_tun", "cl_tfm", "illnz_def", "illnz_tun"],
    "ssl": ["label_spreading", "ls_tun", "self_training", "st_tun", "illssl_def", "illssl_tun", "vime_def", "vime_tun"],
}
WEAKER = {"llp": 1, "pu_sar": -1, "pu_scar": -1, "ssl": -1, "noisy": 1, "pll": 1}   # dấu: mức lớn hơn là yếu hơn?
RQ3_LEVELS = {"llp": [8, 16], "pu_sar": [0.1, 0.3], "pu_scar": [0.1, 0.3], "pll": [0.3, 0.6], "ssl": [0.05, 0.2],
              "noisy": [0.2, 0.4]}


def load(raw):
    fs = glob.glob(os.path.join(raw, "L[123]", "g7_tfm_*.csv")) + glob.glob(os.path.join(raw, "C[123]", "g7_base_*.csv"))
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d = d[d.status == "ok"].copy()
    d["level"] = d.level.astype(float)
    return d.drop_duplicates(["dataset_id", "setting", "level", "seed", "method"])


def ds_table(d, setting, level, metric):
    """bảng dataset × phương pháp: trung bình qua seed; kèm số seed có mặt."""
    x = d[(d.setting == setting) & (d.level == level)]
    t = x.pivot_table(index="dataset", columns="method", values=metric, aggfunc="mean")
    n = x.pivot_table(index="dataset", columns="method", values=metric, aggfunc="count")
    return t, n


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    m = len(ps)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def boot_ci(x, B=10000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def compare(a, b, higher):
    """a = Gibbs, b = đối thủ (Series theo dataset). Trả chênh có hướng (dương = Gibbs tốt hơn), p Wilcoxon hai phía."""
    j = pd.concat([a, b], axis=1).dropna()
    diff = (j.iloc[:, 0] - j.iloc[:, 1]) * (1 if higher else -1)
    p = float(wilcoxon(diff).pvalue) if (diff != 0).any() else 1.0
    lo, hi = boot_ci(diff)
    return dict(n=len(diff), mean_diff=float(diff.mean()), ci_lo=lo, ci_hi=hi, wins=int((diff > 0).sum()),
                ties=int((diff == 0).sum()), losses=int((diff < 0).sum()), p=p)


def bayes_signed_rank(z, rope=0.005, prior=1.0, samples=20000, seed=0):
    """Benavoli et al. (2017): z = chênh (dương = Gibbs tốt hơn). Trả P(trái), P(ROPE), P(phải) — xác suất hậu nghiệm
    mỗi vùng chiếm khối lượng lớn nhất."""
    z = np.concatenate([[0.0], np.asarray(z, float)])
    rng = np.random.default_rng(seed)
    alpha = np.concatenate([[prior], np.ones(len(z) - 1)])
    W = rng.dirichlet(alpha, samples)
    S = (z[:, None] + z[None, :]) / 2
    regions = [(S < -rope), (np.abs(S) <= rope), (S > rope)]
    mass = np.stack([((W @ R.astype(float)) * W).sum(1) for R in regions], 1)
    win = np.bincount(mass.argmax(1), minlength=3) / samples
    return dict(p_left=float(win[0]), p_rope=float(win[1]), p_right=float(win[2]))


def h2_like(d, setting, level, comp, n_holm_extra=None):
    met, higher = METRIC[setting]
    t, n = ds_table(d, setting, level, met)
    g = t["gibbs_warm"]
    res = {"comparator": compare(g, t[comp], higher), "naive": compare(g, t["naive"], higher)}
    cands = [c for c in CANDIDATES[setting] if c in t]
    means = t[cands].mean()
    strongest = means.idxmax() if higher else means.idxmin()
    res["strongest"] = dict(name=strongest, **compare(g, t[strongest], higher))
    res["n_datasets"] = int(t.shape[0])
    res["min_seeds"] = int(n["gibbs_warm"].min())
    res["means"] = {k: float(v) for k, v in t[["gibbs_warm", "naive", "hard_em_warm", comp, "oracle"]
                                               if "oracle" in t else ["gibbs_warm", "naive", comp]].mean().items()}
    return res, t


S1_EXCLUDE = ("MiceProtein", "wall-robot-navigation", "balance-scale", "car")   # giao thức §1.5, đăng ký trước
PLL_TUNED = ["proden_tun", "glwspll_tun", "rc_tun", "cc_tun"]


def h4_block(d, outdir, exclude=()):
    """H4-PLL. Kiểm định ghi trước: Wilcoxon hai phía với cc_tun và naive ở q = 0,3 và 0,6, Holm trên 4 phép so sánh.
    Tiêu chí ghi trước (giao thức Mục 6): 'đạt hoặc vượt baseline mạnh nhất trên ≥ 60% dataset mỗi loại'. Câu này không
    nói rõ (a) tính theo từng mức hay gộp hai mức, (b) 'baseline mạnh nhất' là baseline có trung bình cao nhất hay
    baseline tốt nhất của từng dataset. Script báo cáo mọi cách đọc; báo cáo G8 dùng kết luận yếu nhất (nguyên tắc S1)."""
    d = d[~d.dataset.isin(exclude)]
    h4, ps, tabs = {}, [], {}
    for q in (0.3, 0.6):
        r, t = h2_like(d, "pll", q, "cc_tun")
        h4[q], tabs[q] = r, t
        ps += [r["comparator"]["p"], r["naive"]["p"]]
        if outdir:
            t.to_csv(os.path.join(outdir, f"table_pll_{q}.csv"))
    adj = holm(ps)
    k = 0
    for q in (0.3, 0.6):
        for key in ("comparator", "naive"):
            h4[q][key]["p_holm"] = float(adj[k]); k += 1
    res = {str(q): h4[q] for q in h4}
    # các cách đọc quy tắc 60%: số ô Gibbs >= mốc so sánh
    reads = {}
    for name, fn in {
        "vs_mean_strongest": lambda q, t: t[h4[q]["strongest"]["name"]],
        "vs_cc_tun": lambda q, t: t["cc_tun"],
        "vs_max_tuned_per_dataset": lambda q, t: t[PLL_TUNED].max(axis=1),
        "vs_max_all_and_naive_per_dataset (quy tắc thí điểm RQ4)": lambda q, t: pd.concat(
            [t[CANDIDATES["pll"]].max(axis=1), t["naive"]], axis=1).max(axis=1),
    }.items():
        per = {str(q): int((tabs[q]["gibbs_warm"] >= fn(q, tabs[q])).sum()) for q in (0.3, 0.6)}
        n = {str(q): int(tabs[q].shape[0]) for q in (0.3, 0.6)}
        reads[name] = dict(count=per, n=n, frac_per_level={q: per[q] / n[q] for q in per},
                           frac_pooled=sum(per.values()) / sum(n.values()),
                           pass_per_level=all(per[q] / n[q] >= 0.6 for q in per),
                           pass_pooled=sum(per.values()) / sum(n.values()) >= 0.6)
    res["rule60"] = reads
    res["all_tests_significant"] = bool(all(h4[q][key]["p_holm"] < 0.05 and h4[q][key]["mean_diff"] > 0
                                            for q in (0.3, 0.6) for key in ("comparator", "naive")))
    res["any_reading_passes"] = bool(any(v["pass_per_level"] or v["pass_pooled"] for v in reads.values()))
    res["weakest_reading_passes"] = bool(res["all_tests_significant"] and all(v["pass_per_level"] and v["pass_pooled"]
                                                                            for v in reads.values()))
    res["n_datasets"] = int(tabs[0.3].shape[0])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="../../G7/raw")
    ap.add_argument("--out", default="../results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = load(a.raw)
    out, lines = {}, []
    P = lines.append

    # ---------------- H2-LLP (túi 8) ----------------
    r, t = h2_like(d, "llp", 8.0, "dllplin_def")
    adj = holm([r["comparator"]["p"], r["naive"]["p"]])
    r["comparator"]["p_holm"], r["naive"]["p_holm"] = float(adj[0]), float(adj[1])
    sig = all(r[k]["p_holm"] < 0.05 and r[k]["mean_diff"] > 0 for k in ("comparator", "naive"))
    r["pass"] = bool(sig and r["strongest"]["wins"] >= 7)
    out["H2_llp8"] = r
    t.to_csv(os.path.join(a.out, "table_llp8.csv"))

    # ---------------- H4-PLL (cả 14 dataset; và S1: bỏ 4 dataset, giao thức §1.5) ----------------
    out["H4_pll"] = h4_block(d, a.out, exclude=())
    out["H4_pll_S1"] = h4_block(d, None, exclude=S1_EXCLUDE)

    # ---------------- H3 ----------------
    rows = []
    for st, lvls in RQ3_LEVELS.items():
        met = "logloss_pm" if st == "pu_scar" else "logloss"
        for lv in lvls:
            t, _ = ds_table(d, st, float(lv), met)
            g = "gibbs_known_warm" if st == "pu_sar" else "gibbs_warm"
            for ds in t.index:
                rows.append(dict(setting=st, level=lv, dataset=ds, gap=t.loc[ds, "hard_em_warm"] - t.loc[ds, g],
                                 gap_unknown=(t.loc[ds, "hard_em_warm"] - t.loc[ds, "gibbs_warm"]) if st == "pu_sar" else np.nan))
    h3 = pd.DataFrame(rows).dropna(subset=["gap"])
    lv_rank = {st: {lv: (i / (len(l) - 1) if WEAKER[st] > 0 else 1 - i / (len(l) - 1)) for i, lv in enumerate(sorted(l))}
               for st, l in RQ3_LEVELS.items()}
    h3["weakness"] = [lv_rank[s][l] for s, l in zip(h3.setting, h3.level)]
    h3.to_csv(os.path.join(a.out, "h3_cells.csv"), index=False)
    bsr = bayes_signed_rank(h3.gap.values)
    rho = spearmanr(h3.weakness, h3.gap)
    frac = float((h3.gap > 0).mean())
    out["H3"] = dict(n_cells=len(h3), frac_gibbs_better=frac, bayes=bsr, spearman_rho=float(rho.statistic),
                     spearman_p=float(rho.pvalue), mean_gap=float(h3.gap.mean()),
                     by_setting={s: dict(n=int(len(g)), frac=float((g.gap > 0).mean()), mean_gap=float(g.gap.mean()),
                                         rho=float(spearmanr(g.weakness, g.gap).statistic) if g.weakness.nunique() > 1 else None)
                                 for s, g in h3.groupby("setting")},
                     pass_=bool(frac >= 0.7 and bsr["p_rope"] <= 0.9 and rho.statistic > 0))
    g = h3[h3.setting == "pu_sar"]
    out["H3"]["pu_sar_vs_gibbs_warm"] = dict(frac=float((g.gap_unknown > 0).mean()), mean_gap=float(g.gap_unknown.mean()))

    # ---------------- Bối cảnh phụ ----------------
    sec = {}
    for st, lvls in {"llp": [16, 32], "pu_sar": [0.1, 0.3], "pu_scar": [0.1, 0.3], "ssl": [0.05, 0.2], "noisy": [0.2, 0.4]}.items():
        ps, rs = [], {}
        for lv in lvls:
            r, t = h2_like(d, st, float(lv), COMPARATOR[st])
            rs[lv] = r
            ps += [r["comparator"]["p"], r["naive"]["p"]]
            t.to_csv(os.path.join(a.out, f"table_{st}_{lv}.csv"))
        adj = holm(ps) if st != "llp" else np.concatenate([holm(ps[:2]), holm(ps[2:])])
        k = 0
        for lv in lvls:
            for key in ("comparator", "naive"):
                rs[lv][key]["p_holm"] = float(adj[k]); k += 1
        sec[st] = {str(lv): rs[lv] for lv in lvls}
    out["secondary"] = sec

    # ---------------- Phân tích phụ: so với từng baseline ----------------
    per = []
    for st, lvls in {"llp": [8.0], "pll": [0.3, 0.6]}.items():
        met, higher = METRIC[st]
        for lv in lvls:
            t, _ = ds_table(d, st, lv, met)
            for c in CANDIDATES[st]:
                r = compare(t["gibbs_warm"], t[c], higher)
                per.append(dict(setting=st, level=lv, baseline=c, **r))
    per = pd.DataFrame(per)
    for st in per.setting.unique():
        m = per.setting == st
        per.loc[m, "p_holm"] = holm(per.loc[m, "p"].values)
    per.to_csv(os.path.join(a.out, "per_baseline.csv"), index=False)

    # ---------------- Thăm dò D1–D3 ----------------
    def gap(st, lv):
        met, higher = METRIC[st]
        t, _ = ds_table(d, st, float(lv), met)
        return (t.gibbs_warm - t.naive) * (1 if higher else -1)
    g03, g06 = gap("pll", 0.3), gap("pll", 0.6)
    j = pd.concat([g06, g03], axis=1).dropna()
    D1 = dict(mean_q03=float(g03.mean()), mean_q06=float(g06.mean()),
              p_one_sided=float(wilcoxon(j.iloc[:, 0] - j.iloc[:, 1], alternative="greater").pvalue))
    g8, g16 = gap("llp", 8), gap("llp", 16)
    j = pd.concat([g16, g8], axis=1).dropna()
    D2 = dict(mean_bag8=float(g8.mean()), mean_bag16=float(g16.mean()),
              p_one_sided=float(wilcoxon(j.iloc[:, 0] - j.iloc[:, 1], alternative="greater").pvalue))
    D3 = {}
    for st, lvls in {"pu_scar": [0.1, 0.3], "noisy": [0.2, 0.4], "ssl": [0.05, 0.2]}.items():
        for lv in lvls:
            t, _ = ds_table(d, st, float(lv), "auc")
            D3[f"{st}_{lv}"] = float((t.gibbs_warm - t.naive).mean())
    out["exploratory"] = dict(D1=D1, D2=D2, D3=D3, D3_holds=bool(all(v <= 0.02 for v in D3.values())))

    # tính đầy đủ
    out["coverage"] = d.groupby(["setting"]).size().to_dict()
    json.dump(out, open(os.path.join(a.out, "g8_confirmatory.json"), "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float)[:20000])


if __name__ == "__main__":
    main()
