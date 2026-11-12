"""v3_headroom.py — hai phép tính THĂM DÒ trên dữ liệu đã có, làm ngày 12/11/2026 để định hướng bài v3 (không chạy GPU).

  H-A  Trần của việc CHỌN chuỗi trên Study 3: nếu có một bộ chọn hoàn hảo (biết nhãn test) thì hơn bản gộp X bao nhiêu?
  H-B  Tỉ lệ khoảng cách "TFM ngây thơ → oracle có nhãn thật" mà chuỗi đơn đã lấy lại ở Study 2 (TabICL v2).
Chạy: python -I V3/v3_headroom.py   (từ thư mục de-tai-moi)
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "V2", "analysis"))
sys.path.insert(0, os.path.join(ROOT, "G4", "code"))
from v1c_report import load_npz_dir, metric  # noqa: E402

S = ("warm", "base", "cold")


def main():
    npz = load_npz_dir(os.path.join(ROOT, "S3", "raw", "S3", "s3_npz"))
    by = {}
    for (be, did, st, lev, seed), z in npz.items():
        by.setdefault((did, st, seed), {})[be] = z
    rows = []
    for (did, st, seed), zs in by.items():
        y = next(iter(zs.values()))["yte"]
        ch = {(t, s): (np.asarray(z[f"P_{s}"], float), float(z[f"wpll_{s}"])) for t, z in zs.items() for s in S}
        m = {k: metric(P, y, st) for k, (P, w) in ch.items()}
        X = metric(np.mean([np.mean([ch[(t, s)][0] for s in S], 0) for t in zs], 0), y, st)
        sel = np.mean([ch[max([(t, s) for s in S], key=lambda k: ch[k][1])][0] for t in zs], 0)
        orc = np.mean([ch[max([(t, s) for s in S], key=lambda k: m[k])][0] for t in zs], 0)
        rows.append(dict(setting=st, dataset_id=did, seed=seed, X=X, L=metric(np.asarray(next(iter(zs.values()))["P_L"], float), y, st),
                         best_single_chain=max(m.values()), wpll_pick_global=m[max(ch, key=lambda k: ch[k][1])],
                         X_wpll_pick_per_tfm=metric(sel, y, st), X_oracle_pick_per_tfm=metric(orc, y, st)))
    g = pd.DataFrame(rows).groupby(["setting", "dataset_id"]).mean(numeric_only=True).drop(columns="seed")
    print("## H-A. Study 3: trần của việc chọn chuỗi (trung bình theo dataset)")
    print(g.groupby("setting").mean().round(4).to_string())
    for c in ("best_single_chain", "X_oracle_pick_per_tfm", "X_wpll_pick_per_tfm", "wpll_pick_global"):
        d = g[c] - g.X
        print(f"  {c:24s} − X: {d.mean():+.4f}   thắng/thua {int((d > 0.001).sum())}/{int((d < -0.001).sum())}")
    print("\n## H-B. Study 2 (TabICL v2, một chuỗi): phần khoảng cách naive → oracle đã lấy lại")
    for f, lab in (("s2_table_llp8.csv", "LLP túi 8 (AUC)"), ("s2_table_pll_0.3.csv", "PLL q 0,3 (acc)"), ("s2_table_pll_0.6.csv", "PLL q 0,6 (acc)")):
        t = pd.read_csv(os.path.join(ROOT, "S2", "results", f), index_col=0)
        nv, gb, orc = t.naive.mean(), t.gibbs_warm.mean(), t.oracle.mean()
        best = t[[c for c in t.columns if c.endswith("_tun")]].mean().max()
        print(f"  {lab:18s} naive {nv:.3f} | baseline tune tốt nhất {best:.3f} | chuỗi đơn {gb:.3f} | oracle {orc:.3f} | "
              f"lấy lại {100 * (gb - nv) / (orc - nv):.0f}% (baseline: {100 * (best - nv) / (orc - nv):.0f}%)")


if __name__ == "__main__":
    main()
