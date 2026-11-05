"""v0_explore.py — phân tích THĂM DÒ của giai đoạn 0, viết SAU khi đã xem kết quả của v0_report.py (05/11/2026).
Không thay kết luận đã ghi trước (P1–P4, cổng V0). Mỗi mục trả lời một câu hỏi nảy ra khi đọc kết quả:

  E1  P4 sai ở đâu: tách theo bối cảnh; ở các ô chọn sai, các điểm xuất phát có kết thúc ở cùng một chỗ không
  E2  Ánh xạ phản ứng dự đoán r cuối của chuỗi tốt đến đâu: lặp r ← Φ(r) 30 lần từ r_start, so với r_final thật;
      so ánh xạ gieo đều (S2X + ρ cao) và ánh xạ lỗi có cấu trúc (W1, ngưỡng đề xuất: sai số tuyệt đối trung vị ≤ 0,1)
  E3  TabPFN, PLL q 0,6, bốn dataset từng sụp: Study 1 (gibbs_warm) so với v2 (gibbs_ms) và CC tune
  E4  R4: chênh hai chuỗi lớn ở đâu; bộ ước lượng trung bình HAI chuỗi (nhiều mẫu Monte Carlo hơn) cho H2′, H4′
  E5  Chuỗi còn trôi ở vòng quét 30 không (từ vết R4): |r_30 − r_20|, tỉ lệ đổi nhãn 5 vòng cuối
Chạy: python -I V2/analysis/v0_explore.py   (từ thư mục de-tai-moi)
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(HERE)
ROOT = os.path.dirname(V2)
sys.path.insert(0, HERE)
import v0_report as vr  # noqa: E402

pd.set_option("display.width", 230)
pd.set_option("display.max_columns", 30)


def e1(ms):
    print("# E1. P4 theo bối cảnh (ô đủ điều kiện: chênh chỉ số test giữa các điểm xuất phát > 0,01)")
    _, t = vr.p4(ms)
    el = t[t.spread > 0.01]
    print(el.groupby(["setting"]).correct.agg(["sum", "count"]).rename(columns={"sum": "đúng", "count": "ô"}).to_string())
    x = ms[ms.method.str.startswith("gibbs_init_")]
    rf = x.groupby(["backend", "dataset", "setting", "seed"]).r_final.agg(lambda s: s.max() - s.min()).rename("r_final_range")
    bad = el[~el.correct].merge(rf.reset_index(), on=["backend", "dataset", "setting", "seed"], how="left")
    print("\nÔ chọn sai: khoảng r_final giữa các điểm xuất phát so với khoảng chỉ số test")
    print(bad[["backend", "dataset", "setting", "seed", "spread", "r_final_range", "chosen", "best"]].round(3).to_string(index=False))
    ok = el[el.correct].merge(rf.reset_index(), on=["backend", "dataset", "setting", "seed"], how="left")
    print(f"\nTrung vị khoảng r_final: ô chọn đúng {ok.r_final_range.median():.3f}; ô chọn sai {bad.r_final_range.median():.3f}\n")


def _maps():
    """Ánh xạ trung bình 3 seed: gieo đều (S2X: ρ ≤ 0,8; V0: ρ 0,85–0,95) và lỗi có cấu trúc (V0)."""
    s2x = pd.read_csv(os.path.join(ROOT, "S2", "raw", "S2X", "s2x_tabpfn.csv"))
    s2x = pd.concat([s2x, pd.read_csv(os.path.join(ROOT, "S2", "raw", "S2X", "s2x_tabicl.csv"))])
    s2x = s2x[(s2x.status == "ok") & (s2x.rho.astype(str) != "u")].copy()
    s2x["setting"], s2x["level"], s2x["plant"] = "pll", s2x.q, "uniform"
    v0x = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(V2, "raw", "**", "v0x_*.csv"), recursive=True)])
    v0x = v0x[(v0x.status == "ok") & (v0x.rho.astype(str) != "u")].copy()
    cols = ["backend", "dataset_id", "setting", "level", "plant", "rho", "r_actual", "phi"]
    m = pd.concat([s2x[cols], v0x[cols]], ignore_index=True)
    m["rho"], m["level"] = m.rho.astype(float), m.level.astype(float)
    return m.groupby(["backend", "dataset_id", "setting", "level", "plant", "rho"], as_index=False)[["r_actual", "phi"]].mean()


def iterate(r_grid, phi_grid, r0, T=30):
    o = np.argsort(r_grid)
    r_grid, g = np.asarray(r_grid)[o], (np.asarray(phi_grid) - np.asarray(r_grid))[o]
    r = float(r0)
    for _ in range(T):
        gain = np.interp(r, r_grid, g)          # ngoài lưới: giữ gain ở đầu mút
        r = float(np.clip(r + gain, 0.0, 1.0))
    return r


def e2(ms):
    print("# E2. Dự đoán r_final bằng cách lặp ánh xạ phản ứng 30 lần từ r_start")
    mp = _maps()
    x = ms[ms.method.str.startswith("gibbs_init_")].copy()
    rows = []
    for r in x.itertuples():
        for plant in ("uniform", "struct"):
            g = mp[(mp.backend == r.backend) & (mp.dataset_id == r.dataset_id) & (mp.setting == r.setting)
                   & (mp.level == r.level) & (mp.plant == plant)]
            if len(g) < 4:
                continue
            pred = iterate(g.r_actual.values, g.phi.values, r.r_start)
            rows.append(dict(backend=r.backend, dataset=vr.NAMES.get(r.dataset_id, r.dataset_id), setting=r.setting,
                             start=r.start, seed=r.seed, plant=plant, r_start=r.r_start, r_final=r.r_final, pred=pred,
                             abserr=abs(pred - r.r_final), same_side=(pred >= 0.5) == (r.r_final >= 0.5)))
    t = pd.DataFrame(rows)
    s = t.groupby(["plant", "backend", "setting"]).agg(n=("abserr", "size"), median_abserr=("abserr", "median"),
                                                       p90_abserr=("abserr", lambda v: v.quantile(0.9)),
                                                       frac_same_side_of_0_5=("same_side", "mean"))
    print(s.round(3).to_string())
    print("\nTheo dataset (trung vị sai số tuyệt đối):")
    print(t.pivot_table(index=["backend", "setting", "dataset"], columns="plant", values="abserr", aggfunc="median").round(3).to_string())
    print("\nVí dụ dự đoán so với thật (ánh xạ lỗi có cấu trúc, PLL):")
    y = t[(t.plant == "struct")].groupby(["backend", "dataset", "start"])[["r_start", "pred", "r_final"]].mean()
    print(y.round(3).to_string(), "\n")
    return t


def e3(ms):
    print("# E3. TabPFN v2, PLL q 0,6, bốn dataset ≥ 8 lớp: Study 1 so với v2")
    x = ms[(ms.backend == "tabpfn_v2") & (ms.setting == "pll") & (ms.level == 0.6)]
    t = x.pivot_table(index="dataset_id", columns="method", values="m", aggfunc="mean")
    t = t.rename(index=vr.NAMES)[["gibbs_init_warm", "gibbs_ms", "base_start"]]
    t.columns = ["v1 (Study 1, gibbs_warm)", "v2 (gibbs_ms)", "CC tune (base_start)"]
    t.loc["trung bình"] = t.mean()
    t["v2 − CC tune"] = t.iloc[:, 1] - t.iloc[:, 2]
    print(t.round(4).to_string(), "\n")


def _metric(P, y, binary):
    P = np.asarray(P, float)
    return roc_auc_score(y, P[:, 1]) if binary else float(np.mean(P.argmax(1) == y))


def e4():
    print("# E4. R4: chênh hai chuỗi và bộ ước lượng trung bình hai chuỗi")
    sys.path.insert(0, os.path.join(ROOT, "S2", "analysis"))
    import s2_confirmatory as sc
    d, _ = sc.load()
    rows = []
    for f in sorted(glob.glob(os.path.join(V2, "raw", "V0A", "r4_traces", "*.npz"))):
        _, did, setting, level, seed = os.path.basename(f)[:-4].split("_")
        z = np.load(f)
        binary = setting == "llp"
        y = z["yte"].astype(int)
        ma, mb = _metric(z["pred_a"], y, binary), _metric(z["pred_b"], y, binary)
        mab = _metric((z["pred_a"].astype(float) + z["pred_b"].astype(float)) / 2, y, binary)
        rows.append(dict(dataset_id=int(did), setting=setting, level=float(level), seed=int(seed[1:]), a=ma, b=mb, ab=mab))
    t = pd.DataFrame(rows)
    gw = d[d.method == "gibbs_warm"].copy()
    gw["m"] = np.where(gw.setting == "llp", gw.auc, gw.acc)
    t = t.merge(gw[["dataset_id", "setting", "level", "seed", "m", "dataset"]], on=["dataset_id", "setting", "level", "seed"])
    print(f"Kiểm: chuỗi a từ vết (float16) so với gibbs_warm đã đăng ký: |chênh| tối đa {np.abs(t.a - t.m).max():.4f}")
    t["gap"] = (t.a - t.b).abs()
    big = t[t.gap > 0.03].sort_values("gap", ascending=False)
    print(f"Ô có |a − b| > 0,03: {len(big)} / {len(t)}")
    print(big[["dataset", "setting", "level", "seed", "a", "b", "ab", "gap"]].round(4).to_string(index=False))
    out, js = {}, {}
    for est in ("a", "b", "ab"):
        dd = d.copy()
        key = ["dataset_id", "setting", "level", "seed"]
        mp = t.set_index(key)[est]
        idx = dd.method == "gibbs_warm"
        vals = mp.reindex(pd.MultiIndex.from_frame(dd.loc[idx, key])).values
        is_llp = (dd.loc[idx, "setting"] == "llp").values
        dd.loc[idx & (dd.setting == "llp"), "auc"] = vals[is_llp]
        dd.loc[idx & (dd.setting == "pll"), "acc"] = vals[~is_llp]
        h2, _ = sc.h2(dd)
        h4, _ = sc.h4(dd)
        js[est] = {"h2_llp": dict(h2["comparator"], pass_=h2["pass"]),
                   "h4_pll3": dict(h4["0.3"]["comparator"]), "h4_pll6": dict(h4["0.6"]["comparator"]),
                   "h4_pass": h4["pass"]}
        out[est] = {"H2′ LLP vs DLLP-lin": (h2["comparator"]["mean_diff"], h2["comparator"]["p_holm"], h2["pass"]),
                    "H4′ q0,3 vs CC tune": (h4["0.3"]["comparator"]["mean_diff"], h4["0.3"]["comparator"]["p_holm"], None),
                    "H4′ q0,6 vs CC tune": (h4["0.6"]["comparator"]["mean_diff"], h4["0.6"]["comparator"]["p_holm"], h4["pass"])}
    print("\nPhép thử của Study 2 với từng bộ ước lượng (Δ trung bình, p_Holm, đạt?) — a: chuỗi đã đăng ký; "
          "b: chuỗi thứ hai; ab: trung bình hai chuỗi")
    for k in out["a"]:
        print(f"  {k:22s} " + "  ".join(f"{e}: {out[e][k][0]:+.4f} p={out[e][k][1]:.4f}" + ("" if out[e][k][2] is None else
                                       (" ĐẠT" if out[e][k][2] else " KHÔNG")) for e in ("a", "b", "ab")))
    print()
    import json                                     # thêm 05/11: để bài báo lấy số bằng script (\\val{r4-*})
    with open(os.path.join(V2, "results", "v0_r4.json"), "w", encoding="utf-8") as fh:
        json.dump(js, fh, ensure_ascii=False, indent=1, default=float)
    return t


def e5():
    print("# E5. Chuỗi còn trôi ở vòng quét 30? (vết R4, TabICL v2, Study 2; nhãn thật chỉ dùng để chẩn đoán)")
    rows = []
    for f in sorted(glob.glob(os.path.join(V2, "raw", "V0A", "r4_traces", "*.npz"))):
        _, did, setting, level, seed = os.path.basename(f)[:-4].split("_")
        z = np.load(f)
        for c in "ab":
            mon, fl = z[f"monitor_{c}"], z[f"flips_{c}"]
            rows.append(dict(setting=f"{setting}_{level}", drift_20_30=abs(float(mon[-1] - mon[19])),
                             flips_last5=float(fl[-5:].mean()), r_final=float(mon[-1])))
    t = pd.DataFrame(rows)
    s = t.groupby("setting").agg(chains=("drift_20_30", "size"), drift_median=("drift_20_30", "median"),
                                 drift_p90=("drift_20_30", lambda v: v.quantile(0.9)), drift_max=("drift_20_30", "max"),
                                 flips_last5_median=("flips_last5", "median"))
    print(s.round(4).to_string(), "\n")


def main():
    ms = vr.multistart()
    e1(ms)
    e2(ms)
    e3(ms)
    e4()
    e5()


if __name__ == "__main__":
    main()


def e6():
    """E6 (thêm 05/11, sau khi xem E1): chọn một chuỗi theo WPLL so với trung bình mọi chuỗi (không tham số), trên
    mọi ô nhiều điểm xuất phát. Chỉ để gợi ý quy tắc cho giai đoạn 1; không chọn ngưỡng nào trên dữ liệu này."""
    import json
    print("# E6. Chọn theo WPLL so với trung bình dự đoán của mọi chuỗi (gợi ý cho giai đoạn 1)")
    P = {}
    for f in glob.glob(os.path.join(V2, "raw", "**", "v0_*_preds.jsonl"), recursive=True):
        for line in open(f):
            r = json.loads(line)
            be, did, st, lv, sd, m = r["key"].split("|")[:6]
            P[(be, int(did), st, float(lv), int(sd), m)] = (np.asarray(r["P"], float), np.asarray(r["y"], int))
    rows = []
    for k in sorted({k[:5] for k in P}):
        inits = {m: P[k + (m,)] for m in ("gibbs_init_warm", "gibbs_init_base", "gibbs_init_cold") if k + (m,) in P}
        if len(inits) < 2 or k + ("gibbs_ms",) not in P:
            continue
        binary = k[2] == "llp"
        y = next(iter(inits.values()))[1]
        sc = {m: _metric(v[0], y, binary) for m, v in inits.items()}
        avg = _metric(np.mean([v[0] for v in inits.values()], 0), y, binary)
        rows.append(dict(backend=k[0], setting=k[2], dataset=vr.NAMES.get(k[1], k[1]), seed=k[4],
                         select=_metric(P[k + ("gibbs_ms",)][0], y, binary), average=avg, best=max(sc.values()),
                         warm=sc["gibbs_init_warm"]))
    t = pd.DataFrame(rows)
    s = t.groupby(["backend", "setting"])[["warm", "select", "average", "best"]].mean()
    print(s.round(4).to_string())
    print("\nTheo dataset:")
    print(t.groupby(["backend", "setting", "dataset"])[["warm", "select", "average", "best"]].mean().round(4).to_string(), "\n")


if __name__ == "__main__" and os.environ.get("V0_E6", "1") == "1":
    e6()
