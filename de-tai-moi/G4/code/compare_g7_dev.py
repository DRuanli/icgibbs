"""
compare_g7_dev.py — so kết quả của bộ chạy G7 trên tập phát triển (G6 lượt 5) với kết quả thí điểm cùng ô, để thấy
sai khác do R1–R3 (G7_RUNNER_DESIGN), và tóm tắt PROOFS G4-5 (sửa MH) và R̂.
Chạy:  python compare_g7_dev.py --pilot_dir ../pilot --g7 "../pilot/g6r5_*.csv"
"""
import argparse
import glob
import os

import pandas as pd


def load_pilot(pilot_dir):
    fr = []
    for f in ("g4_pilot_v2.csv", "g4_rq4_pilot.csv", "g6r2_llp.csv", "g6r2_pll.csv", "g6r3_pll.csv"):
        p = os.path.join(pilot_dir, f)
        if os.path.exists(p):
            d = pd.read_csv(p)
            d = d[d.status == "ok"]
            d["src"] = f
            fr.append(d)
    d = pd.concat(fr, ignore_index=True)
    d["level"] = d.level.astype(float)
    # g6r3 (PRODEN có Adam) thay proden_tun của g6r2
    d = d.sort_values("src", key=lambda s: s.map({"g6r3_pll.csv": 0}).fillna(1)).drop_duplicates(
        ["dataset_id", "setting", "level", "seed", "method"], keep="first")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot_dir", default="../pilot")
    ap.add_argument("--g7", default="../pilot/g6r5_*.csv")
    a = ap.parse_args()
    g = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(a.g7)) if "rhat" not in f], ignore_index=True)
    g = g[g.status == "ok"].copy()
    g["level"] = g.level.astype(float)
    p = load_pilot(a.pilot_dir)
    key = ["dataset_id", "setting", "level", "seed", "method"]
    for st, met in (("llp", "auc"), ("pll", "acc")):
        m = g[g.setting == st].merge(p[p.setting == st], on=key, suffixes=("_g7", "_pilot"))
        if m.empty:
            continue
        m["diff"] = m[f"{met}_g7"] - m[f"{met}_pilot"]
        print(f"\n=== {st} ({met}): bộ chạy G7 − thí điểm, cùng ô")
        print(m.groupby("method")["diff"].agg(["mean", "std", "min", "max", "count"]).round(4).to_string())
        nt = m.groupby("dataset_g7")[["n_train_g7", "n_train_pilot"]].first() if "dataset_g7" in m else None
        if nt is not None:
            print("n_train (G7 sau loại trùng, thí điểm):"); print(nt.to_string())
    # G4-5
    mh = g[g.method.isin(["gibbs_warm", "gibbs_warm_mh"]) & (g.setting == "llp")]
    if (mh.method == "gibbs_warm_mh").any():
        t = mh.pivot_table(index=["dataset", "seed"], columns="method", values="auc")
        t["diff_mh_minus_approx"] = t.gibbs_warm_mh - t.gibbs_warm
        acc = mh[mh.method == "gibbs_warm_mh"].extra.str.extract(r"accept=([^;]*)")[0].map(
            lambda s: pd.to_numeric(pd.Series(s.split("/")), errors="coerce").mean())
        print("\n=== PROOFS G4-5: LLP túi 8, AUC của bag-block có sửa MH − bản xấp xỉ")
        print(t.round(4).to_string())
        print(f"trung bình chênh {t.diff_mh_minus_approx.mean():.4f}; |chênh| lớn nhất {t.diff_mh_minus_approx.abs().max():.4f}; "
              f"tỉ lệ chấp nhận MH trung bình {acc.mean():.3f}")
    rh = [f for f in glob.glob(a.g7) if "rhat" in f]
    if rh:
        r = pd.concat([pd.read_csv(f) for f in rh])
        r = r[r.status == "ok"]
        ex = r.extra.str.extract(r"rhat_median=([\d.]+);rhat_p90=([\d.]+);frac_rhat_gt_1_1=([\d.]+)").astype(float)
        ex.columns = ["rhat_median", "rhat_p90", "frac_gt_1.1"]
        print("\n=== R̂ (2 chuỗi)"); print(pd.concat([r[["dataset", "setting", "level", "seed"]], ex], axis=1).to_string())


if __name__ == "__main__":
    main()
