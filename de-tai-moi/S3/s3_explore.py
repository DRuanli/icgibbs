"""s3_explore.py — phân tích THĂM DÒ trên dữ liệu Study 3 (viết ngày 12/11/2026, SAU khi đã chấm xác nhận).

Không phần nào ở đây nằm trong STUDY3_PLAN. Mọi số từ script này phải được ghi là "thăm dò" (exploratory) trong bài.
Script không chạy lại chuỗi nào: chỉ đọc các dự đoán đã lưu trong S3/raw/S3/s3_npz.

  E1  từng chuỗi (warm = v1, base, cold), learner L, và các tổ hợp trung bình (bỏ một điểm xuất phát, v1 + L, ...)
  E2  gộp nhiều TFM: X (đã ghi trước), bỏ một TFM, X của riêng chuỗi warm, X không dùng chuỗi base; so với L
  E3  ba chuỗi có về cùng lưu vực không (độ lệch r_cuối giữa các điểm xuất phát), và lợi A1 − v1 theo hai nhóm
  E4  "tự đặt trọng số": độ tự tin (xác suất lớn nhất trung bình) của chuỗi kém nhất và tốt nhất trong ô PLL
  E5  cấu hình của Study 2 (TabICL v2, một chuỗi) so với L trên dataset của Study 3
  E6  X so với A1 của từng TFM
  E7  chi phí (giây, số lượt gọi TFM) từ các tệp csv thô
Chạy: python -I S3/s3_explore.py   (từ thư mục de-tai-moi)
"""
from __future__ import annotations

import glob
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "V2", "analysis"))
sys.path.insert(0, os.path.join(ROOT, "G4", "code"))
from v1c_report import load_npz_dir, metric, r_final  # noqa: E402

RAW = os.path.join(HERE, "raw", "S3")
OUT = os.path.join(HERE, "results")
STARTS = ("warm", "base", "cold")
TFMS = ["tabicl_v2", "tabpfn_v3", "tabdpt", "tabpfn_v2"]
TIE = 0.001


def wtl(d):
    d = np.asarray(d, float)
    return int((d >= TIE).sum()), int((np.abs(d) < TIE).sum()), int((d <= -TIE).sum())


def summ(d, test=True):
    d = np.asarray(d, float)
    w, t, l = wtl(d)
    out = dict(n=int(len(d)), mean=float(d.mean()), median=float(np.median(d)), win=w, tie=t, loss=l,
               n_worse_002=int((d < -0.02).sum()), min=float(d.min()))
    if test and np.any(d != 0):
        out["p_two_sided"] = float(wilcoxon(d, alternative="two-sided").pvalue)
    rng = np.random.default_rng(0)
    bs = rng.choice(d, size=(10000, len(d)), replace=True).mean(1)
    out["ci95"] = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
    return out


def P(z, s):
    return np.asarray(z[f"P_{s}"], float)


def main():
    npz = load_npz_dir(os.path.join(RAW, "s3_npz"))
    res = {"note": "THĂM DÒ — viết sau khi chấm xác nhận Study 3; không nằm trong STUDY3_PLAN"}
    pd.set_option("display.width", 250)

    # ---------- E1: từng chuỗi và các tổ hợp trong một TFM ----------
    combos = {
        "warm": ("warm",), "base": ("base",), "cold": ("cold",), "L": ("L",),
        "A1": STARTS, "warm+cold": ("warm", "cold"), "warm+base": ("warm", "base"), "base+cold": ("base", "cold"),
        "warm+L": ("warm", "L"), "A1+L": STARTS + ("L",),
    }
    rows = []
    for (be, did, st, lev, seed), z in npz.items():
        y = z["yte"]
        r = dict(tfm=be, dataset_id=did, setting=st, seed=seed)
        for name, parts in combos.items():
            r[name] = metric(np.mean([P(z, p) for p in parts], 0), y, st)
        rf = {s: r_final(z, s) for s in STARTS}
        r.update({f"rfin_{s}": rf[s] for s in STARTS})
        r["spread"] = max(rf.values()) - min(rf.values())
        if st == "pll":
            accs = {s: metric(P(z, s), y, st) for s in STARTS}
            conf = {s: float(P(z, s).max(1).mean()) for s in STARTS}
            lo, hi = min(accs, key=accs.get), max(accs, key=accs.get)
            r.update(acc_gap=accs[hi] - accs[lo], conf_lo=conf[lo], conf_hi=conf[hi])
        rows.append(r)
    cells = pd.DataFrame(rows)
    cells.to_csv(os.path.join(OUT, "s3_explore_cells.csv"), index=False)
    pairs = cells.groupby(["tfm", "setting", "dataset_id"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    names = list(combos)
    e1 = pairs.groupby(["tfm", "setting"])[names].mean()
    e1["n"] = pairs.groupby(["tfm", "setting"]).size()
    print("## E1. Trung bình theo (TFM, setting) của từng chuỗi và tổ hợp\n", e1.round(4).to_string())
    res["E1_by_tfm_setting"] = e1.round(5).reset_index().to_dict("records")
    byds = pairs.groupby(["setting", "dataset_id"])[names].mean().reset_index()      # trung bình trên TFM chạy được
    res["E1_vs_v1"] = {c: summ(byds[c] - byds["warm"]) for c in names if c != "warm"}
    res["E1_vs_A1"] = {c: summ(byds[c] - byds["A1"]) for c in names if c != "A1"}
    res["E1_vs_L"] = {c: summ(byds[c] - byds["L"]) for c in names if c != "L"}
    print("\nSo với v1 (theo dataset, trung bình trên TFM):")
    for c, s in res["E1_vs_v1"].items():
        print(f"  {c:10s} {s['mean']:+.4f}  W/T/L {s['win']}/{s['tie']}/{s['loss']}  p={s.get('p_two_sided', float('nan')):.4f}")
    print("So với A1:")
    for c, s in res["E1_vs_A1"].items():
        print(f"  {c:10s} {s['mean']:+.4f}  W/T/L {s['win']}/{s['tie']}/{s['loss']}  p={s.get('p_two_sided', float('nan')):.4f}")
    for st in ("llp", "pll"):
        h = byds[byds.setting == st]
        res[f"E1_means_{st}"] = {c: float(h[c].mean()) for c in names}
        print(f"Trung bình theo dataset, {st}:", {c: round(float(h[c].mean()), 4) for c in names})

    # ---------- E2: gộp nhiều TFM ----------
    by = {}
    for (be, did, st, lev, seed), z in npz.items():
        by.setdefault((did, st, seed), {})[be] = z
    full = pairs.groupby(["dataset_id", "tfm"]).size().unstack().notna()              # cặp có số
    counts = cells.groupby(["dataset_id", "tfm"]).size().unstack()
    rows = []
    for (did, st, seed), zs in by.items():
        ok = [t for t in TFMS if t in zs and counts.loc[did].get(t, 0) == 3]
        y = zs[ok[0]]["yte"]
        assert len({str(zs[t]["sha256_split"]) for t in ok}) == 1

        def X(parts, tf=ok):
            return metric(np.mean([np.mean([P(zs[t], p) for p in parts], 0) for t in tf], 0), y, st)

        r = dict(dataset_id=did, setting=st, seed=seed, n_tfm=len(ok), X=X(STARTS), X_warm=X(("warm",)),
                 X_warmcold=X(("warm", "cold")), X_base=X(("base",)), L=metric(P(zs[ok[0]], "L"), y, st))
        PX = np.mean([np.mean([P(zs[t], p) for p in STARTS], 0) for t in ok], 0)
        r["X+L"] = metric((PX * len(ok) + P(zs[ok[0]], "L")) / (len(ok) + 1), y, st)
        r["XL_half"] = metric(0.5 * PX + 0.5 * P(zs[ok[0]], "L"), y, st)
        for t in ok:
            rest = [u for u in ok if u != t]
            if len(rest) >= 2:
                r[f"X_wo_{t}"] = X(STARTS, rest)
            r[f"A1_{t}"] = metric(np.mean([P(zs[t], p) for p in STARTS], 0), y, st)
            r[f"v1_{t}"] = metric(P(zs[t], "warm"), y, st)
        core = [t for t in ("tabicl_v2", "tabpfn_v3", "tabdpt") if t in ok]
        r["X3"] = X(STARTS, core)
        for a, b in itertools.combinations(core, 2):
            r[f"X2_{a}+{b}"] = X(STARTS, [a, b])
        rows.append(r)
    xc = pd.DataFrame(rows)
    xd = xc.groupby(["setting", "dataset_id"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    xd.to_csv(os.path.join(OUT, "s3_explore_datasets.csv"), index=False)
    res["E2_vs_L"] = {c: summ(xd[c] - xd["L"]) for c in ("X", "X_warm", "X_warmcold", "X_base", "X3", "X+L", "XL_half")}
    res["E2_vs_X"] = {c: summ(xd[c] - xd["X"]) for c in ("X_warm", "X_warmcold", "X_base", "X3", "X+L", "XL_half")}
    for st in ("llp", "pll"):
        h = xd[xd.setting == st]
        res[f"E2_X_vs_L_{st}"] = summ(h["X"] - h["L"], test=False)
        res[f"E2_means_{st}"] = {c: float(h[c].mean()) for c in ("X", "X_warm", "X_warmcold", "X_base", "X3", "L")}
    print("\n## E2. Bản gộp nhiều TFM so với L (theo dataset)")
    for c, s in res["E2_vs_L"].items():
        print(f"  {c:11s} − L: {s['mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] trung vị {s['median']:+.4f}  "
              f"W/T/L {s['win']}/{s['tie']}/{s['loss']}  kém >0,02: {s['n_worse_002']}  p={s.get('p_two_sided', float('nan')):.2g}")
    print("So với X:")
    for c, s in res["E2_vs_X"].items():
        print(f"  {c:11s} − X: {s['mean']:+.4f}  W/T/L {s['win']}/{s['tie']}/{s['loss']}  p={s.get('p_two_sided', float('nan')):.2g}")
    lo = {}
    for t in TFMS:
        c = f"X_wo_{t}"
        h = xd.dropna(subset=[c])
        lo[t] = dict(summ(h[c] - h["X"]), n=int(len(h)))
    res["E2_leave_one_out_vs_X"] = lo
    print("Bỏ một TFM (so với X):", {t: (round(v["mean"], 4), v["n"], f"{v['win']}/{v['tie']}/{v['loss']}") for t, v in lo.items()})
    pr = {c: summ(xd.dropna(subset=[c])[c] - xd.dropna(subset=[c])["X3"]) for c in xd.columns if c.startswith("X2_")}
    res["E2_pairs_vs_X3"] = pr
    print("Cặp TFM so với X3:", {c: round(v["mean"], 4) for c, v in pr.items()})

    # ---------- E6: X so với A1 và v1 của từng TFM ----------
    res["E6_X_vs_A1_tfm"], res["E6_X_vs_v1_tfm"], res["E6_A1_vs_L_tfm"], res["E6_v1_vs_L_tfm"] = {}, {}, {}, {}
    print("\n## E6. Theo từng TFM")
    for t in TFMS:
        h = xd.dropna(subset=[f"A1_{t}"])
        res["E6_X_vs_A1_tfm"][t] = summ(h["X"] - h[f"A1_{t}"])
        res["E6_X_vs_v1_tfm"][t] = summ(h["X"] - h[f"v1_{t}"])
        res["E6_A1_vs_L_tfm"][t] = summ(h[f"A1_{t}"] - h["L"])
        res["E6_v1_vs_L_tfm"][t] = summ(h[f"v1_{t}"] - h["L"])
        for k in ("E6_X_vs_A1_tfm", "E6_A1_vs_L_tfm", "E6_v1_vs_L_tfm"):
            s = res[k][t]
            print(f"  {k[3:]:14s} {t:10s} n={s['n']:2d} {s['mean']:+.4f} W/T/L {s['win']}/{s['tie']}/{s['loss']} "
                  f"kém>0,02: {s['n_worse_002']} p={s.get('p_two_sided', float('nan')):.3g}")
    # E5: cấu hình Study 2 (TabICL v2, một chuỗi) so với L, theo setting
    res["E5_tabicl_v1_vs_L"] = {}
    for st in ("llp", "pll"):
        h = xd[xd.setting == st]
        res["E5_tabicl_v1_vs_L"][st] = summ(h["v1_tabicl_v2"] - h["L"], test=False)
        res["E5_tabicl_v1_vs_L"][st + "_A1"] = summ(h["A1_tabicl_v2"] - h["L"], test=False)
    print("\n## E5. TabICL v2 so với L theo setting:", json.dumps(
        {k: (round(v["mean"], 4), f"{v['win']}/{v['tie']}/{v['loss']}") for k, v in res["E5_tabicl_v1_vs_L"].items()}))

    # ---------- E3: cùng lưu vực hay không ----------
    cells["same_basin"] = cells.spread < 0.03
    cells["G"] = cells["A1"] - cells["warm"]
    e3 = cells.groupby(["setting", "same_basin"]).agg(n=("G", "size"), G_mean=("G", "mean"), G_median=("G", "median"))
    res["E3_basin"] = dict(frac_same=float(cells.same_basin.mean()),
                           by_setting={st: float(h.same_basin.mean()) for st, h in cells.groupby("setting")},
                           by_tfm={t: float(h.same_basin.mean()) for t, h in cells.groupby("tfm")},
                           table=e3.round(5).reset_index().to_dict("records"),
                           G_same=float(cells[cells.same_basin].G.mean()), G_diff=float(cells[~cells.same_basin].G.mean()),
                           n_same=int(cells.same_basin.sum()), n_diff=int((~cells.same_basin).sum()))
    print("\n## E3. Ba chuỗi về cùng lưu vực (lệch r_cuối < 0,03):", round(res["E3_basin"]["frac_same"], 3),
          res["E3_basin"]["by_setting"], res["E3_basin"]["by_tfm"])
    print(e3.round(4).to_string())

    # ---------- E4: tự đặt trọng số (PLL) ----------
    pl = cells[(cells.setting == "pll") & (cells.acc_gap >= 0.10)]
    res["E4_self_weighting"] = dict(n_cells=int(len(pl)), n_pll_cells=int((cells.setting == "pll").sum()),
                                    conf_worst_mean=float(pl.conf_lo.mean()), conf_best_mean=float(pl.conf_hi.mean()),
                                    frac_worst_less_confident=float((pl.conf_lo < pl.conf_hi).mean()))
    print("\n## E4. Ô PLL có chênh accuracy giữa các chuỗi ≥ 0,10:", json.dumps(res["E4_self_weighting"]))

    # ---------- E7: chi phí ----------
    fr = []
    for f in sorted(glob.glob(os.path.join(RAW, "s3_*_[lp]l[pl].csv"))):
        if "lprefetch" in f:
            continue
        fr.append(pd.read_csv(f))
    raw = pd.concat(fr, ignore_index=True)
    raw = raw[(raw.status == "ok") & raw.method.str.startswith("gibbs_init_")]
    key = ["backend", "dataset_id", "setting", "seed", "method"]
    raw = raw.sort_values("seconds").drop_duplicates(key, keep="last")
    ch = raw.groupby("backend").agg(n=("seconds", "size"), sec_median=("seconds", "median"), sec_max=("seconds", "max"),
                                    calls_median=("n_tfm_calls", "median"), gpu_mb_max=("peak_gpu_mb", "max"))
    res["E7_cost_per_chain"] = ch.round(2).reset_index().to_dict("records")
    res["E7_n_train_max"] = int(raw.n_train.max())
    tot = raw.groupby(["dataset_id", "setting", "seed"]).seconds.sum()
    res["E7_X_seconds_per_cell"] = dict(median=float(tot.median()), max=float(tot.max()))
    one = raw[raw.method == "gibbs_init_warm"].groupby("backend").seconds.median()
    res["E7_v1_seconds_median"] = {k: float(v) for k, v in one.items()}
    print("\n## E7. Chi phí mỗi chuỗi\n", ch.round(1).to_string(), "\nTổng giây của X mỗi ô:", res["E7_X_seconds_per_cell"])

    with open(os.path.join(OUT, "s3_explore.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
