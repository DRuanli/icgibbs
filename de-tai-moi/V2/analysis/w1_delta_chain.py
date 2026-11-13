"""w1_delta_chain.py — THĂM DÒ (viết 13/11/2026, vòng phản biện 3, F14): một cận dưới ước lượng của δ trong Giả thiết 1
(đồng nhất), từ các ánh xạ đáp ứng đã đo ở lượt V1C. Không chạy lại gì.

Lập luận. Nếu (H_δ) đúng thì với mọi trạng thái y, trung bình trên một vòng quét (mọi khối, phủ đủ F, nên các h_i triệt
tiêu) của xác suất vẽ lại đúng nằm trong khoảng δ quanh Φ(r) (bỏ qua chênh nhỏ giữa r_{-B} và r). Hai cách gieo lỗi
cho cùng một r (lỗi theo biên của chính chuỗi, và lỗi đều) vì thế cho hai giá trị cách nhau tối đa 2δ. Vậy
    δ ≥ max_r |Φ_chain(r) − Φ_uniform(r)| / 2,
với Φ_uniform nội suy tuyến tính tại các r của ánh xạ "chain" (chỉ trong khoảng r đã đo của cả hai). Đây là ước lượng:
mỗi Φ là trung bình của các lần vẽ lại, có nhiễu đo (cột phi_sd_blocks), nên cận dưới này mang nhiễu đó.

Phạm vi như Định lý 3: PLL (nhãn yếu cấp mẫu), context không bị cắt (0,9·n_train ≤ 2.000 với K = 10).
Ra: V2/results/w1_delta_chain.csv (mỗi ô một dòng), in tóm tắt.
Chạy: python -I V2/analysis/w1_delta_chain.py   (từ thư mục de-tai-moi)
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(HERE)
RAW = os.path.join(V2, "raw", "V1C")


def main():
    d = pd.concat([pd.read_csv(os.path.join(RAW, f)) for f in ("v1c_chainmap_tabicl.csv", "v1c_chainmap_tabpfn.csv")])
    d = d[(d.status == "ok") & (d.setting == "pll")]
    rows = []
    for (be, ds, seed), g in d.groupby(["backend", "dataset", "seed"]):
        n_train = int(g.n_train.iloc[0])
        u = g[g.plant == "uniform"].sort_values("r_actual")
        c = g[g.plant == "chain"].sort_values("r_actual")
        s = g[g.plant == "struct"].sort_values("r_actual")
        if len(u) < 3 or len(c) < 3:
            continue
        inside = c[(c.r_actual >= u.r_actual.min()) & (c.r_actual <= u.r_actual.max())]
        if inside.empty:
            continue
        phi_u = np.interp(inside.r_actual, u.r_actual, u.phi)
        dc = np.abs(inside.phi.to_numpy() - phi_u)
        r = dict(backend=be, dataset=ds, seed=seed, n_train=n_train, in_scope=bool(0.9 * n_train <= 2000),
                 delta_lb_chain=float(dc.max() / 2), n_points=int(len(inside)),
                 max_abs_gain_uniform=float((u.phi - u.r_actual).abs().max()),
                 max_abs_gain_chain=float((c.phi - c.r_actual).abs().max()))
        if len(s) >= 3:
            ins = s[(s.r_actual >= u.r_actual.min()) & (s.r_actual <= u.r_actual.max())]
            if not ins.empty:
                r["delta_lb_struct"] = float(np.abs(ins.phi.to_numpy() - np.interp(ins.r_actual, u.r_actual, u.phi)).max() / 2)
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(V2, "results", "w1_delta_chain.csv"), index=False)
    sc = t[t.in_scope]
    print(f"cells {len(t)}, in scope {len(sc)} ({sc.dataset.nunique()} datasets)")
    for name, h in (("all", t), ("in scope", sc)):
        print(f"{name}: delta_lb_chain median {h.delta_lb_chain.median():.3f}, p90 {h.delta_lb_chain.quantile(0.9):.3f}, "
              f"max {h.delta_lb_chain.max():.3f}; struct median {h.delta_lb_struct.median():.3f}; "
              f"share with lb > max|gain_uniform| {(h.delta_lb_chain > h.max_abs_gain_uniform).mean():.2f}")
    print(sc.groupby("backend").delta_lb_chain.describe().round(3))


if __name__ == "__main__":
    main()
