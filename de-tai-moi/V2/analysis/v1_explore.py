"""v1_explore.py — V2 giai đoạn 1, phân tích THĂM DÒ viết SAU khi xem số của v1_aggregate.py (05/11/2026).

  E1  Các điểm xuất phát có về cùng một lưu vực không: r_cuối (trung bình 5 vòng quét cuối của monitor, độ đúng của nhãn
      chuỗi trên train) cho warm/base/cold; "cùng lưu vực" nếu max − min < 0,03. Liên hệ Mệnh đề 4(c)/(d) (W1_R1A_REPORT).
  E2  Lợi ích của A1 (trung bình đều) so với v1, tách theo cùng/khác lưu vực, và so với trung bình của các chuỗi đơn
      (phần lợi do giảm biến thiên Monte Carlo).
      Chuỗi "r_cuối cao nhất" dùng nhãn thật trên train, tức là một bộ chọn tiên tri, không dùng được khi triển khai;
      A0_picks_bestr: tỉ lệ ô mà A0 (WPLL lớn nhất, dùng được) chọn trúng chuỗi đó.
  E3  Dataset có thụt lùi với A1.
  E4  A1 so với v1 theo dataset (trung bình 3 seed): thắng/hoà/thua (hoà: |chênh| < 0,001) và kiểm định dấu hạng
      Wilcoxon hai phía trong từng nhóm. Không hiệu chỉnh đa kiểm định; chỉ để mô tả.
Chạy: python -I V2/analysis/v1_explore.py   (từ thư mục de-tai-moi)
"""
from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(HERE)
PAT = re.compile(r"v1_(?P<backend>.+?)_(?P<did>\d+)_(?P<setting>llp|pll)_(?P<level>[\d.]+)_s(?P<seed>\d+)\.npz$")
pd.set_option("display.width", 220)


def metric(P, y, setting):
    P = np.asarray(P, float)
    return float(roc_auc_score(y, P[:, 1])) if setting == "llp" else float(np.mean(P.argmax(1) == y))


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(V2, "raw", "V1*", "v1_npz", "v1_*.npz"))):
        mt = PAT.search(os.path.basename(f))
        z = np.load(f)
        st = mt["setting"]
        r = dict(backend=mt["backend"], setting=st, dataset_id=int(mt["did"]), seed=int(mt["seed"]))
        for s in ("warm", "base", "cold"):
            mon = np.asarray(z[f"monitor_{s}"], float)
            r[f"rfin_{s}"] = float(mon[-5:].mean())
            r[f"m_{s}"] = metric(z[f"P_{s}"], z["yte"], st)
        w = {s: float(np.asarray(z[f"wpll_{s}"], float).ravel()[-1]) for s in ("warm", "base", "cold")}
        r["A0_pick"] = max(w, key=w.get)
        r["m_A1"] = metric(np.mean([z[f"P_{s}"] for s in ("warm", "base", "cold")], axis=0), z["yte"], st)
        r["m_L"] = metric(z["P_L"], z["yte"], st)
        rows.append(r)
    d = pd.DataFrame(rows)
    rf = d[["rfin_warm", "rfin_base", "rfin_cold"]]
    d["spread_r"] = rf.max(axis=1) - rf.min(axis=1)
    d["same_basin"] = d.spread_r < 0.03
    ms = d[["m_warm", "m_base", "m_cold"]]
    d["spread_m"] = ms.max(axis=1) - ms.min(axis=1)
    d["A1_minus_v1"] = d.m_A1 - d.m_warm
    d["A1_minus_meanchain"] = d.m_A1 - ms.mean(axis=1)
    d.to_csv(os.path.join(V2, "results", "v1_explore_cells.csv"), index=False)

    print("# E1. Cùng lưu vực (spread r_cuối < 0,03) theo nhóm")
    print(d.groupby(["backend", "setting"]).agg(cells=("seed", "size"), frac_same_basin=("same_basin", "mean"),
                                                 median_spread_r=("spread_r", "median"), p90_spread_r=("spread_r", lambda v: v.quantile(.9)))
          .round(3).to_string(), "\n")
    print("# E2. Lợi ích của A1, tách theo cùng/khác lưu vực (trung bình theo ô)")
    print(d.groupby(["backend", "setting", "same_basin"]).agg(cells=("seed", "size"), A1_minus_v1=("A1_minus_v1", "mean"),
                                                               A1_minus_meanchain=("A1_minus_meanchain", "mean"),
                                                               spread_metric=("spread_m", "median")).round(4).to_string(), "\n")
    print("# E2b. Ô khác lưu vực: chuỗi nào có r_cuối cao nhất (đếm), và A1 so với chuỗi đó")
    nb = d[~d.same_basin].copy()
    nb["best_r"] = nb[["rfin_warm", "rfin_base", "rfin_cold"]].idxmax(1).str.replace("rfin_", "")
    nb["m_bestr"] = [r[f"m_{r.best_r}"] for _, r in nb.iterrows()]
    nb["A0_is_bestr"] = nb.A0_pick == nb.best_r
    print(nb.groupby(["backend", "setting"]).best_r.value_counts().unstack(fill_value=0).to_string())
    print(nb.groupby(["backend", "setting"]).apply(lambda g: pd.Series(dict(cells=len(g), A1_minus_bestr=(g.m_A1 - g.m_bestr).mean(),
                                                                                  A0_picks_bestr=g.A0_is_bestr.mean())),
                                                   include_groups=False).round(4).to_string(), "\n")
    print("# E3. blood-transfusion (1464), LLP: L so với các chuỗi")
    print(d[d.dataset_id == 1464][["backend", "seed", "rfin_warm", "rfin_base", "rfin_cold", "m_warm", "m_base", "m_cold",
                                   "m_A1", "m_L"]].round(4).to_string(index=False))

    print("\n# E4. A1 so với v1 theo dataset (trung bình 3 seed)")
    g = d.groupby(["backend", "setting", "dataset_id"])[["m_A1", "m_warm"]].mean()
    g["diff"] = g.m_A1 - g.m_warm
    out = []
    for key, h in g.groupby(level=[0, 1]):
        x = h["diff"].to_numpy()
        nz = x[np.abs(x) > 1e-12]
        out.append(dict(backend=key[0], setting=key[1], datasets=len(x), win=int((x >= 1e-3).sum()),
                        tie=int((np.abs(x) < 1e-3).sum()), loss=int((x <= -1e-3).sum()), mean=x.mean(), median=np.median(x),
                        min=x.min(), max=x.max(), wilcoxon_p=wilcoxon(nz).pvalue if len(nz) >= 6 else np.nan))
    print(pd.DataFrame(out).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
