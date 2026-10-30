"""
g8_pll_collapse.py — G8, phân tích lỗi (THĂM DÒ, hậu kiểm, không phải kiểm định xác nhận): chuỗi Gibbs ở PLL.
Đọc cột `extra` của bộ chạy G7 (avg_candidates; flips = tỉ lệ nhãn đổi mỗi vòng quét; monitor = tỉ lệ nhãn gán trong
chuỗi trùng nhãn thật của train, chỉ dùng để chẩn đoán) và ghi một dòng cho mỗi (dataset, mức, seed).
Chạy:  python g8_pll_collapse.py --raw ../../G7/raw --out ../results
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd


def parse(extra):
    out = {}
    for kv in str(extra).split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="../../G7/raw")
    ap.add_argument("--out", default="../results")
    a = ap.parse_args()
    from g8_confirmatory import load
    d = load(a.raw)
    d = d[d.setting == "pll"]
    g = d[d.method == "gibbs_warm"].copy()
    p = g.extra.map(parse)
    g["avg_candidates"] = p.map(lambda o: float(o.get("avg_candidates", "nan")))
    mon = p.map(lambda o: [float(v) for v in o["monitor"].split("/")] if "monitor" in o else [np.nan])
    fl = p.map(lambda o: [float(v) for v in o["flips"].split("/")] if "flips" in o else [np.nan])
    g["monitor_sweep1"] = mon.map(lambda m: m[0])
    g["monitor_last"] = mon.map(lambda m: m[-1])
    g["flips_last5"] = fl.map(lambda f: float(np.mean(f[-5:])))
    g["frac_classes_in_set"] = g.avg_candidates / g.n_classes
    acc = d.pivot_table(index=["dataset", "level", "seed"], columns="method", values="acc")
    keep = ["naive", "hard_em_warm", "cc_tun", "glwspll_tun", "oracle"]
    g = g.set_index(["dataset", "level", "seed"])[["n_classes", "n_train", "d", "avg_candidates", "frac_classes_in_set",
                                                    "monitor_sweep1", "monitor_last", "flips_last5", "acc"]]
    g = g.rename(columns={"acc": "gibbs_warm"}).join(acc[[k for k in keep if k in acc]]).sort_index()
    # "sụp": Gibbs kém CC tune (đối thủ chính của H4) cùng ô từ 0,2 accuracy trở lên — định nghĩa mô tả, chọn SAU khi
    # xem số liệu, để khoanh các ô quyết định kết quả H4; không dùng cho kiểm định nào
    g["collapsed"] = g.gibbs_warm < g.cc_tun - 0.2
    g.round(4).to_csv(os.path.join(a.out, "pll_chain_diag.csv"))
    s = g.reset_index()
    print(s.groupby(["level", "collapsed"]).agg(n=("seed", "size"), C_min=("n_classes", "min"),
                                               n_train_max=("n_train", "max")).to_string())
    print(s[s.collapsed][["dataset", "level", "seed", "n_classes", "n_train", "monitor_sweep1", "monitor_last",
                          "flips_last5", "gibbs_warm", "naive", "cc_tun"]].round(3).to_string())
    healthy = s[~s.collapsed & (s.level == 0.6)]
    print("q=0,6, chuỗi không sụp: flips_last5 trung vị %.3f; sụp: %.3f" % (
        healthy.flips_last5.median(), s[s.collapsed & (s.level == 0.6)].flips_last5.median()))


if __name__ == "__main__":
    main()
