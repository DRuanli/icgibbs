"""s3_report.py — chấm Study 3 bản gọn theo S3/STUDY3_PLAN.md §3–§4 (viết và commit TRƯỚC khi có số).

  H1  v2 (A1) hơn v1: δ₁ = trung bình trên TFM của (A1 − v1), Wilcoxon một phía trên dataset
  H2  đa-TFM hơn một TFM: δ₂ = X − trung bình trên TFM của A1, Wilcoxon một phía; Holm trên {H1, H2}, α = 0,05
  H3  H8′: AUC(c3 → T2) ≥ 0,80 gộp mọi ô; < 5 ô dương tính → "không đủ ô"
  C4  tỉ lệ cặp (TFM, dataset) có A1 < max(v1, L) − 0,02 ≤ 10%
  C5  X so với TFM tốt nhất: trung bình ≥ −0,01 và thua > 0,02 ở ≤ 3 dataset
  kèm theo TFM, theo nhóm, A1 so với L, độ nhạy S1, ô nhãn đảo
Chạy: python -I S3/s3_report.py [--raw S3/raw/S3]   (từ thư mục de-tai-moi)
"""
from __future__ import annotations

import argparse
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
from v1c_report import a1, auc_or_none, load_npz_dir, metric  # noqa: E402
from icgibbs.g7.registry import S3_BINARY, S3_MULTICLASS, S3_S1_FLAG, S3_TABPFN_V2  # noqa: E402

TFMS = ["tabicl_v2", "tabpfn_v3", "tabdpt", "tabpfn_v2"]
SEEDS = (0, 1, 2)
LEV = {"llp": "8", "pll": "0.6"}
TIE = 1e-3


def setting_of(did):
    return "llp" if did in S3_BINARY else "pll"


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def one_sided(x):
    x = np.asarray(x, float)
    nz = x[np.abs(x) > 1e-12]
    p = float(wilcoxon(nz, alternative="greater").pvalue) if len(nz) >= 1 else 1.0
    return dict(n=int(len(x)), mean=float(x.mean()), median=float(np.median(x)), ci95=boot_ci(x) if len(x) > 1 else None,
                win=int((x >= TIE).sum()), tie=int((np.abs(x) < TIE).sum()), loss=int((x <= -TIE).sum()), p_one_sided=p)


def holm(ps):
    order = np.argsort(ps)
    adj, run = [0.0] * len(ps), 0.0
    for k, i in enumerate(order):
        run = max(run, (len(ps) - k) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def build(pool, valid_ds):
    """Bảng theo (TFM, dataset) và theo dataset; chỉ cặp đủ 3 seed, TabPFN v2 chỉ trong giới hạn công bố."""
    pairs, cells = [], []
    per_ds = []
    for did in sorted(valid_ds):
        st = setting_of(did)
        ok = []
        for t in TFMS:
            if t == "tabpfn_v2" and did not in S3_TABPFN_V2:
                continue
            zs = [pool.get((t, did, st, LEV[st], s)) for s in SEEDS]
            if any(z is None for z in zs):
                continue
            ok.append(t)
            v1 = [metric(z["P_warm"], z["yte"], st) for z in zs]
            A = [metric(a1(z), z["yte"], st) for z in zs]
            L = [metric(z["P_L"], z["yte"], st) for z in zs]
            pairs.append(dict(tfm=t, dataset_id=did, setting=st, v1=np.mean(v1), A1=np.mean(A), L=np.mean(L)))
            for s, z, a_, b_, c_ in zip(SEEDS, zs, v1, A, L):
                c3 = 0.5 * float(np.abs(np.asarray(z["Q_warm"], float) - np.asarray(z["Q_base"], float)).sum(1).mean())
                cells.append(dict(tfm=t, dataset_id=did, setting=st, seed=s, v1=a_, A1=b_, L=c_, c3=c3,
                                  T2=int(a_ < max(b_, c_) - 0.02)))
        row = dict(dataset_id=did, setting=st, n_tfm=len(ok), tfms=",".join(ok))
        if len(ok) >= 2:
            xs, same = [], True
            for s in SEEDS:
                zs = [pool[(t, did, st, LEV[st], s)] for t in ok]
                same &= len({str(z["sha256_split"]) for z in zs}) == 1
                xs.append(metric(np.mean([a1(z) for z in zs], axis=0), zs[0]["yte"], st))
            row.update(X=float(np.mean(xs)), split_same=bool(same))
        per_ds.append(row)
    P, C, D = pd.DataFrame(pairs), pd.DataFrame(cells), pd.DataFrame(per_ds)
    if len(P):
        g = P.groupby("dataset_id")
        D = D.merge(g.apply(lambda h: pd.Series(dict(d1=(h.A1 - h.v1).mean(), meanA1=h.A1.mean(), bestA1=h.A1.max(),
                                                      meanL=h.L.mean(), meanv1=h.v1.mean())), include_groups=False),
                    left_on="dataset_id", right_index=True, how="left")
        D["d2"] = D.X - D.meanA1 if "X" in D else np.nan
    return P, C, D


def tests(D):
    d1 = D.dropna(subset=["d1"]).d1.to_numpy()
    d2 = D[(D.n_tfm >= 2)].dropna(subset=["d2"]).d2.to_numpy()
    h1, h2 = one_sided(d1), one_sided(d2)
    adj = holm([h1["p_one_sided"], h2["p_one_sided"]])
    for h, p in ((h1, adj[0]), (h2, adj[1])):
        h["p_holm"] = float(p)
        h["verdict"] = "ĐÚNG" if p < 0.05 else "KHÔNG BÁC BỎ ĐƯỢC H0"
    return h1, h2


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(HERE, "raw", "S3"))
    ap.add_argument("--out_dir", default=os.path.join(HERE, "results"))
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    pd.set_option("display.width", 220)
    summ = {}
    pool = load_npz_dir(os.path.join(a.raw, "s3_npz"))
    hp = os.path.join(a.raw, "s3_dataset_hashes.csv")
    all_ds = set(S3_BINARY) | set(S3_MULTICLASS)
    dropped = {}
    if os.path.exists(hp):
        h = pd.read_csv(hp)
        unchecked = []
        for r in h.itertuples():
            if str(r.status).startswith("ok"):
                if not bool(r.pass_criteria):          # trượt tiêu chí sau loại trùng → loại (STUDY3_PLAN §2)
                    dropped[int(r.did)] = f"trượt tiêu chí: n={getattr(r, 'n', '')}, d={getattr(r, 'd', '')}"
            else:                                       # lỗi tải ở bước kiểm (không phải trượt tiêu chí): chưa kiểm
                unchecked.append(int(r.did))
        missing = all_ds - set(h.did.astype(int))
        unchecked += sorted(missing)
        if unchecked:
            print(f"[cảnh báo] chưa kiểm được tiêu chí sau tải cho {sorted(unchecked)} (lỗi tải ở bước mã băm); giữ lại")
        summ["unchecked_datasets"] = sorted(unchecked)
    else:
        print("[cảnh báo] không có s3_dataset_hashes.csv: không áp được bước kiểm sau khi tải")
    valid = all_ds - set(dropped)
    print(f"# Study 3 bản gọn — chấm theo STUDY3_PLAN. npz: {len(pool)}; dataset hợp lệ {len(valid)}/20; loại: {dropped}\n")
    P, C, D = build(pool, valid)
    if not len(P):
        print("chưa có ô nào đủ 3 seed")
        return 0
    n_expected = 3 * len(valid) + len(set(S3_TABPFN_V2) & valid)
    summ["coverage"] = dict(pairs=int(len(P)), pairs_expected=int(n_expected), datasets_with_data=int(D.d1.notna().sum()),
                            dropped_datasets=dropped, complete=bool(len(P) == n_expected))
    print("## Độ phủ:", json.dumps(summ["coverage"], ensure_ascii=False))
    print("\n## Theo dataset (trung bình trên TFM chạy được)")
    print(D.drop(columns=["tfms"]).round(4).to_string(index=False))
    print("\n## Theo TFM")
    print(P.groupby(["tfm", "setting"]).agg(n=("A1", "size"), v1=("v1", "mean"), A1=("A1", "mean"), L=("L", "mean"))
          .assign(G=lambda x: x.A1 - x.v1).round(4).to_string())

    h1, h2 = tests(D)
    summ["H1"], summ["H2"] = h1, h2
    print("\nH1 (v2 hơn v1):", json.dumps(h1, ensure_ascii=False))
    print("H2 (đa-TFM hơn một TFM):", json.dumps(h2, ensure_ascii=False))

    npos = int(C.T2.sum())
    auc = auc_or_none(C.T2, C.c3)
    ci = None
    if auc is not None and npos >= 5:
        rng = np.random.default_rng(0)
        ids = C.dataset_id.unique()
        bs = []
        for _ in range(2000):
            samp = pd.concat([C[C.dataset_id == d] for d in rng.choice(ids, len(ids), replace=True)])
            v = auc_or_none(samp.T2, samp.c3)
            if v is not None:
                bs.append(v)
        ci = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))] if bs else None
    summ["H3"] = dict(n_cells=int(len(C)), n_pos=npos, AUC=auc, ci95_cluster_boot=ci,
                      AUC_by_tfm={t: auc_or_none(g.T2, g.c3) for t, g in C.groupby("tfm")},
                      verdict="không đủ ô" if npos < 5 else ("ĐÚNG" if auc is not None and auc >= 0.80 else "SAI"))
    print("H3 (H8′):", json.dumps(summ["H3"], ensure_ascii=False))

    reg = P[P.A1 < np.maximum(P.v1, P.L) - 0.02]
    rate = len(reg) / len(P)
    summ["C4"] = dict(n_pairs=int(len(P)), n_regress=int(len(reg)), rate=float(rate), verdict="ĐẠT" if rate <= 0.10 else "KHÔNG ĐẠT",
                      regressions=reg[["tfm", "dataset_id", "v1", "A1", "L"]].round(4).to_dict("records"))
    print("C4 (thụt lùi):", json.dumps(summ["C4"], ensure_ascii=False))
    Dx = D[(D.n_tfm >= 2)].dropna(subset=["X"])
    gap = Dx.X - Dx.bestA1
    summ["C5"] = dict(n=int(len(Dx)), mean_gap=float(gap.mean()), n_lose_gt_002=int((gap < -0.02).sum()),
                      verdict="ĐẠT" if (gap.mean() >= -0.01 and int((gap < -0.02).sum()) <= 3) else "KHÔNG ĐẠT")
    print("C5 (so với tiên tri):", json.dumps(summ["C5"], ensure_ascii=False))

    s1 = D[~D.dataset_id.isin(S3_S1_FLAG)]
    a_, b_ = tests(s1)
    summ["S1_sensitivity"] = dict(H1=a_, H2=b_)
    print("\nĐộ nhạy S1 (bỏ", sorted(S3_S1_FLAG), "): H1", a_["verdict"], f"p_holm={a_['p_holm']:.4f}; H2", b_["verdict"],
          f"p_holm={b_['p_holm']:.4f}")
    by = {}
    for st, g in D.groupby("setting"):
        by[st] = dict(n=int(len(g)), mean_d1=float(g.d1.mean()), mean_d2=float(g.d2.mean()) if g.d2.notna().any() else None)
    summ["by_setting"] = by
    dl = (D.meanA1 - D.meanL).dropna().to_numpy()
    nz = dl[np.abs(dl) > 1e-12]
    summ["A1_vs_L"] = dict(mean=float(dl.mean()), win=int((dl >= TIE).sum()), loss=int((dl <= -TIE).sum()),
                           p_two_sided=float(wilcoxon(nz).pvalue) if len(nz) else None)
    flips = C[(C.setting == "llp") & (C.v1 < 0.5)]
    summ["label_flip_cells_v1"] = int(len(flips))
    print("Theo nhóm:", json.dumps(by, ensure_ascii=False), "| A1 so với L:", json.dumps(summ["A1_vs_L"], ensure_ascii=False),
          "| ô v1 AUC < 0,5:", len(flips))
    P.to_csv(os.path.join(a.out_dir, "s3_pairs.csv"), index=False)
    C.to_csv(os.path.join(a.out_dir, "s3_cells.csv"), index=False)
    D.to_csv(os.path.join(a.out_dir, "s3_datasets.csv"), index=False)
    with open(os.path.join(a.out_dir, "s3_summary.json"), "w") as f:
        json.dump(summ, f, ensure_ascii=False, indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
