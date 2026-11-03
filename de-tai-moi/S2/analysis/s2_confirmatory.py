"""
s2_confirmatory.py — Nghiên cứu 2: kiểm định xác nhận đúng S2_PLAN §1 và §4 (đóng băng ở commit aaee9fd; danh sách cuối
ở S2/s2_final.csv theo §3.4). Chỉ đọc S2/raw/. Dùng lại các hàm thống kê của G8/analysis/g8_confirmatory.py.

  H2-LLP′: LLP túi 8, AUC trung bình 3 seed mỗi dataset. Wilcoxon hai phía, Gibbs ấm (TabICL v2) so với DLLP tuyến tính
           mặc định và TFM ngây thơ (TabICL v2); Holm trên 2. Đạt nếu cả hai có ý nghĩa sau Holm theo hướng Gibbs tốt hơn
           VÀ Gibbs thắng ≥ 70% dataset trước baseline LLP có AUC trung bình cao nhất trên tập đánh giá S2.
  H4-PLL′: PLL q 0,3 và 0,6, accuracy; Wilcoxon hai phía với CC tune và TFM ngây thơ ở hai mức; Holm trên 4. Đạt nếu cả 4
           có ý nghĩa sau Holm VÀ ở MỖI mức, Gibbs ≥ baseline PLL có accuracy trung bình cao nhất ở mức đó trên ≥ 60% dataset.
  H3′:     log loss, đơn vị là ô (dataset × mức) của LLP túi 8 và PLL; Bayesian signed-rank, ROPE ±0,005; đạt nếu Gibbs tốt
           hơn hard-EM ở ≥ 70% ô, P(ROPE) ≤ 0,9 và Spearman(mức yếu, chênh) > 0. Làm rõ lúc viết script (trước khi chạy):
           LLP chỉ có một mức nên không mang thông tin về Spearman; ρ tính trên các ô PLL (q 0,3 → 0, q 0,6 → 1).
  S1:      nhị phân: chạy lại mọi kiểm định của H2′ khi bỏ dataset gắn cờ; kết luận khác → dùng kết luận yếu hơn.
           Đa lớp (còn < 10 dataset): "khác kết luận" nếu chênh trung bình so với CC tune đổi dấu, hoặc Gibbs thắng < 60%
           dataset còn lại, ở một mức bất kỳ (S2_PLAN §1, thầy duyệt 01/11). p-value vẫn báo cáo.
Dataset trượt tiêu chí sau loại trùng (irish, maternal_health_risk) bị bỏ theo §3.4 dù có kết quả thô (các lượt chạy
bằng bản code đóng băng, danh sách cũ).
Chạy:  python s2_confirmatory.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
S2 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(S2, "..", "G8", "analysis"))
from g8_confirmatory import CANDIDATES, bayes_signed_rank, compare, holm  # noqa: E402
from g8_confirmatory import ds_table as _ds_table  # noqa: E402

S1_FLAG = {40536, 44130, 375, 23380, 1476}


def ds_table(d, setting, level, metric):
    """Như G8, nhưng làm tròn trung bình theo seed tới 10 chữ số thập phân. Sửa 03/11/2026, sau khi đọc bảng kết quả:
    trung bình 3 accuracy bằng nhau, cộng theo thứ tự khác nhau, có thể lệch 2e-16 (cjs, PLL q 0,3: Gibbs và CC tune
    cùng 0,99693667) và bị đếm là thua. Làm tròn chỉ đổi những chênh nhỏ hơn 1e-10; S2_RESULTS ghi số trước và sau."""
    t, n = _ds_table(d, setting, level, metric)
    return t.round(10), n


def load():
    fs = sorted(glob.glob(os.path.join(S2, "raw", "S2T*", "s2_tfm_*.csv"))) + \
        sorted(glob.glob(os.path.join(S2, "raw", "S2C*", "s2_base_*.csv"))) + \
        sorted(glob.glob(os.path.join(S2, "raw", "S2G*", "s2_*.csv"))) + \
        sorted(glob.glob(os.path.join(S2, "raw", "S2R*", "s2_*.csv")))
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d = d[d.status == "ok"].copy()
    d["level"] = d.level.astype(float)
    d = d.drop_duplicates(["dataset_id", "setting", "level", "seed", "method"], keep="first")
    fin = pd.read_csv(os.path.join(S2, "s2_final.csv"))
    keep = {g: set(fin[(fin.group == g) & fin.kept].did) for g in ("binary", "multiclass")}
    d = d[((d.setting == "llp") & d.dataset_id.isin(keep["binary"])) | ((d.setting == "pll") & d.dataset_id.isin(keep["multiclass"]))]
    return d, keep


def h2(d, exclude=()):
    x = d[~d.dataset_id.isin(exclude)]
    t, n = ds_table(x, "llp", 8.0, "auc")
    g = t["gibbs_warm"]
    r = {"comparator": compare(g, t["dllplin_def"], True), "naive": compare(g, t["naive"], True)}
    adj = holm([r["comparator"]["p"], r["naive"]["p"]])
    r["comparator"]["p_holm"], r["naive"]["p_holm"] = float(adj[0]), float(adj[1])
    cands = [c for c in CANDIDATES["llp"] if c in t]
    best = t[cands].mean().idxmax()
    r["strongest"] = dict(name=best, **compare(g, t[best], True))
    r["frac_wins_strongest"] = r["strongest"]["wins"] / r["strongest"]["n"]
    r["n_datasets"], r["min_seeds"] = int(t.shape[0]), int(n["gibbs_warm"].min())
    r["means"] = {k: float(v) for k, v in t[["gibbs_warm", "naive", "hard_em_warm", "dllplin_def", best, "oracle"]].mean().items()}
    sig = all(r[k]["p_holm"] < 0.05 and r[k]["mean_diff"] > 0 for k in ("comparator", "naive"))
    r["pass"] = bool(sig and r["frac_wins_strongest"] >= 0.7)
    return r, t


def h4(d, exclude=()):
    x = d[~d.dataset_id.isin(exclude)]
    res, ps, tabs = {}, [], {}
    for q in (0.3, 0.6):
        t, n = ds_table(x, "pll", q, "acc")
        g = t["gibbs_warm"]
        r = {"comparator": compare(g, t["cc_tun"], True), "naive": compare(g, t["naive"], True)}
        cands = [c for c in CANDIDATES["pll"] if c in t]
        best = t[cands].mean().idxmax()
        ge = int((g >= t[best]).sum())
        r["strongest"] = dict(name=best, n_ge=ge, n_datasets_level=int(len(g)), frac_ge=ge / len(g), **compare(g, t[best], True))
        r["n_datasets"], r["min_seeds"] = int(t.shape[0]), int(n["gibbs_warm"].min())
        r["means"] = {k: float(v) for k, v in t[["gibbs_warm", "naive", "hard_em_warm", "cc_tun", best, "oracle"]].mean().items()}
        res[str(q)], tabs[q] = r, t
        ps += [r["comparator"]["p"], r["naive"]["p"]]
    adj = holm(ps)
    k = 0
    for q in ("0.3", "0.6"):
        for key in ("comparator", "naive"):
            res[q][key]["p_holm"] = float(adj[k]); k += 1
    sig = all(res[q][key]["p_holm"] < 0.05 and res[q][key]["mean_diff"] > 0 for q in ("0.3", "0.6") for key in ("comparator", "naive"))
    rule60 = all(res[q]["strongest"]["frac_ge"] >= 0.6 for q in ("0.3", "0.6"))
    res["all_tests_significant"], res["rule60_each_level"] = bool(sig), bool(rule60)
    res["pass"] = bool(sig and rule60)
    return res, tabs


def h3(d):
    rows = []
    for st, lvls in (("llp", [8.0]), ("pll", [0.3, 0.6])):
        for lv in lvls:
            t, _ = ds_table(d, st, lv, "logloss")
            for ds in t.index:
                rows.append(dict(setting=st, level=lv, dataset=ds, gap=t.loc[ds, "hard_em_warm"] - t.loc[ds, "gibbs_warm"]))
    c = pd.DataFrame(rows).dropna()
    bsr = bayes_signed_rank(c.gap.values)
    p = c[c.setting == "pll"]
    rho = spearmanr((p.level == 0.6).astype(float), p.gap)
    frac = float((c.gap > 0).mean())
    return dict(n_cells=len(c), frac_gibbs_better=frac, bayes=bsr, spearman_rho_pll=float(rho.statistic),
                spearman_p=float(rho.pvalue), by_setting={s: float((g.gap > 0).mean()) for s, g in c.groupby("setting")},
                pass_=bool(frac >= 0.7 and bsr["p_rope"] <= 0.9 and rho.statistic > 0)), c


def per_baseline(d):
    """Phân tích phụ đã đăng ký (S2_PLAN §4): Gibbs so với TỪNG baseline, Holm trên mọi cặp baseline × mức của bối cảnh."""
    rows = []
    for st, lvls, met in (("llp", [8.0], "auc"), ("pll", [0.3, 0.6], "acc")):
        for lv in lvls:
            t, _ = ds_table(d, st, lv, met)
            for c in CANDIDATES[st]:
                if c in t:
                    rows.append(dict(setting=st, level=lv, baseline=c, n_missing=int(t[c].isna().sum()),
                                     **compare(t["gibbs_warm"], t[c], True)))
    per = pd.DataFrame(rows)
    for st in per.setting.unique():
        m = per.setting == st
        per.loc[m, "p_holm"] = holm(per.loc[m, "p"].values)
    return per


def main():
    d, keep = load()
    out = {}
    os.makedirs(os.path.join(S2, "results"), exist_ok=True)
    out["H2_llp8"], t = h2(d)
    t.to_csv(os.path.join(S2, "results", "s2_table_llp8.csv"))
    out["H2_llp8_S1"], _ = h2(d, exclude=S1_FLAG)
    have_multi = set(d[d.setting == "pll"].dataset_id.unique())
    if have_multi != keep["multiclass"]:
        out["H4_pll"] = f"CHƯA ĐỦ DỮ LIỆU: thiếu {sorted(keep['multiclass'] - have_multi)} — không tính H4′, H3′"
    else:
        out["H4_pll"], tabs = h4(d)
        for q, t in tabs.items():
            t.to_csv(os.path.join(S2, "results", f"s2_table_pll_{q}.csv"))
        s1, _ = h4(d, exclude=S1_FLAG)
        out["H4_pll_S1"] = s1
        out["H4_pll_S1_differs"] = bool(any(np.sign(s1[q]["comparator"]["mean_diff"]) != np.sign(out["H4_pll"][q]["comparator"]["mean_diff"])
                                            or s1[q]["comparator"]["wins"] / s1[q]["comparator"]["n"] < 0.6 for q in ("0.3", "0.6")))
        out["H3"], cells = h3(d)
        cells.to_csv(os.path.join(S2, "results", "s2_h3_cells.csv"), index=False)
        per = per_baseline(d)
        per.to_csv(os.path.join(S2, "results", "s2_per_baseline.csv"), index=False)
        out["per_baseline_all_holm_sig"] = {st: int(((g.p_holm < 0.05) & (g.mean_diff > 0)).sum()) for st, g in per.groupby("setting")}
        out["per_baseline_n"] = {st: int(len(g)) for st, g in per.groupby("setting")}
    json.dump(out, open(os.path.join(S2, "results", "s2_confirmatory.json"), "w"), indent=1, default=float, ensure_ascii=False)
    print(json.dumps(out, indent=1, default=float, ensure_ascii=False))


if __name__ == "__main__":
    main()
