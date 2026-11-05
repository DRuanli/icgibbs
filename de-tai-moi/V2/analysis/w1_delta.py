"""w1_delta.py — V2 W1 (05/11/2026, thăm dò, chỉ đọc kết quả đã có): đo trên TFM thật đại lượng δ của giả định đồng
nhất (H) trong Mệnh đề drift (V2/W1_R1A_REPORT.md §2), và so với độ lớn gain |Φ(r) − r|.

  δ̂_plant  = |Φ_đều(r) − Φ_cấu trúc(r)|: hai cách gieo lỗi khác nhau cho cùng r; nếu (H) đúng với tolerance δ thì
             δ̂_plant ≤ 2δ (cả hai ánh xạ cùng cách Φ không quá δ). Đo tại các r của ánh xạ lỗi có cấu trúc (0,2–0,9).
  δ̂_chain  = |(r_{t+1} − r_t) − g_đều(r_t)| dọc vết THẬT của chuỗi (V0: monitor mỗi vòng quét, mọi điểm xuất phát),
             chỉ các bước mà r_t nằm trong lưới đo của ánh xạ. Vòng quét hệ thống đi nhanh hơn một bước Jacobi khi gain
             lớn (W1_R1A_REPORT §2.3), nên δ̂_chain lớn ở vòng đầu là bình thường; mục quan trọng là ở các vòng sau
             (t ≥ 5), khi chuỗi đã gần điểm dừng: ở đó (H) đúng thì r_{t+1} − r_t ≈ g(r_t).
  γ̂         = trung vị |g_đều(r)| trên đoạn r mà vết chuỗi đi qua.
  resid_late (CÓ DẤU) = (r_{t+1} − r_t) − g_đều(r_t) ở t ≥ 5: dương nghĩa là chuỗi tiến tốt hơn ánh xạ gieo đều báo.
             Bổ đề 1 của Mệnh đề 4 tách hai phía, nên chỉ phía dưới (H⁻_δ: Φ_chuỗi ≥ Φ − δ) là cần cho các kết luận
             "đi lên / ở lại trên"; p10 của resid_late đo phía đó.
  endpoint_signed = r_final thật − r dự đoán bằng lặp ánh xạ gieo đều (như E2), trung vị theo chuỗi.
Chạy: python -I V2/analysis/w1_delta.py   (từ thư mục de-tai-moi)
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import v0_explore as ve  # noqa: E402
import v0_report as vr  # noqa: E402

pd.set_option("display.width", 220)


def gain_fn(g):
    g = g.sort_values("r_actual")
    r, gain = g.r_actual.values, (g.phi - g.r_actual).values
    return lambda x: np.interp(x, r, gain), r.min(), r.max()


def main():
    mp = ve._maps()
    mp = mp[(mp.setting == "pll") & (mp.level == 0.6)]
    ms = vr.multistart()
    ms = ms[(ms.setting == "pll") & ms.method.str.startswith("gibbs_init_")]
    rows = []
    for (be, did), g in mp.groupby(["backend", "dataset_id"]):
        gu, gs = g[g.plant == "uniform"], g[g.plant == "struct"]
        if len(gu) < 4:
            continue
        fu, lo, hi = gain_fn(gu)
        r = dict(backend=be, dataset=vr.NAMES.get(did, did))
        if len(gs) >= 4:
            gs = gs.sort_values("r_actual")
            dp = np.abs((gs.phi - gs.r_actual).values - fu(gs.r_actual.values))
            r.update(delta_plant_median=float(np.median(dp)), delta_plant_max=float(dp.max()))
        x = ms[(ms.backend == be) & (ms.dataset_id == did)]
        steps, late, late_signed = [], [], []
        rs_seen = []
        for _, c in x.iterrows():
            mon = [float(v) for v in vr.kv(c.extra).get("monitor", "").split("/") if v]
            tr = np.r_[c.r_start, mon]
            for t in range(len(tr) - 1):
                if lo <= tr[t] <= hi:
                    d = abs((tr[t + 1] - tr[t]) - fu(tr[t]))
                    steps.append(d)
                    rs_seen.append(tr[t])
                    if t >= 5:
                        late.append(d)
                        late_signed.append((tr[t + 1] - tr[t]) - fu(tr[t]))
        errs = [c.r_final - ve.iterate(gu.r_actual.values, gu.phi.values, c.r_start) for _, c in x.iterrows()]
        if errs:
            r["endpoint_err_median"] = float(np.median(np.abs(errs)))      # như E2: lặp ánh xạ gieo đều 30 lần
            r["endpoint_signed_median"] = float(np.median(errs))
        if steps:
            r.update(delta_chain_median=float(np.median(steps)), delta_chain_late_median=float(np.median(late)) if late else np.nan,
                     delta_chain_late_p90=float(np.quantile(late, 0.9)) if late else np.nan,
                     gamma_median=float(np.median(np.abs(fu(np.array(rs_seen))))), n_steps=len(steps))
            if late_signed:
                r.update(resid_late_signed_median=float(np.median(late_signed)),
                         resid_late_signed_p10=float(np.quantile(late_signed, 0.1)))
        rows.append(r)
    t = pd.DataFrame(rows)
    t = t[t.n_steps.notna()] if "n_steps" in t else t                # bỏ dataset không có chuỗi PLL q 0,6 ở V0
    print("# W1: δ của giả định đồng nhất (H), đo trên TFM thật (PLL q 0,6, dataset của NC1; thăm dò)\n")
    print(t.round(4).to_string(index=False))
    t.to_csv(os.path.join(V2, "results", "w1_delta.csv"), index=False)


if __name__ == "__main__":
    main()
