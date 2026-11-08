"""h8_explore.py — THĂM DÒ (08/11/2026, sau khi P15 sai): tìm tín hiệu KHÔNG cần nhãn cho H8 trên dữ liệu đã có.
Đích T2: v1 (chuỗi warm) thua max(A1, L) quá 0,02 trên ô. Tín hiệu (không dùng nhãn thật, đo sau chuỗi đầy đủ):
  c1 = độ tự tin của chuỗi warm (xác suất lớn nhất trung bình trên test) trừ max của hai chuỗi còn lại
  c2 = TV trung bình giữa dự đoán test của warm và của A1
  c3 = TV trung bình giữa Q̄_warm và Q̄_base (trên train)
Dữ liệu: V1 (234 ô) + V1C phần E (seed 0, mọi TFM). Chạy: python -I V2/analysis/h8_explore.py
"""
import glob, os, re
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAT = re.compile(r"v1_(?P<be>.+?)_(?P<did>\d+)_(?P<st>llp|pll)_(?P<lev>[\d.]+)_s(?P<seed>\d+)\.npz$")
S = ("warm", "base", "cold")
def met(P, y, st): return roc_auc_score(y, P[:, 1]) if st == "llp" else float(np.mean(P.argmax(1) == y))
rows = []
for src, pat in [("V1", "raw/V1*/v1_npz/*.npz"), ("V1C_E", "raw/V1C/v1c_e_npz/*.npz")]:
    for f in sorted(glob.glob(os.path.join(V2, pat))):
        m = PAT.search(os.path.basename(f)); z = np.load(f); st = m["st"]; y = z["yte"]
        P = {s: np.asarray(z[f"P_{s}"], float) for s in S}; A = np.mean(list(P.values()), 0)
        conf = {s: P[s].max(1).mean() for s in S}
        tv = lambda a, b: 0.5 * np.abs(a - b).sum(1).mean()
        m1, mA, mL = met(P["warm"], y, st), met(A, y, st), met(z["P_L"], y, st)
        rows.append(dict(src=src, be=m["be"], did=int(m["did"]), st=st, seed=int(m["seed"]),
                         c1=max(conf["base"], conf["cold"]) - conf["warm"], c2=tv(P["warm"], A),
                         c3=tv(np.asarray(z["Q_warm"]), np.asarray(z["Q_base"])),
                         T2=int(m1 < max(mA, mL) - 0.02), gainA1=mA - m1))
d = pd.DataFrame(rows).drop_duplicates(["be", "did", "st", "seed"])
d.to_csv(os.path.join(V2, "results", "h8_explore_cells.csv"), index=False)
print("ô:", len(d), "T2:", int(d.T2.sum()))
for c in ("c1", "c2", "c3"):
    print(c, "AUC T2 gộp %.3f" % roc_auc_score(d.T2, d[c]),
          {k: round(roc_auc_score(h.T2, h[c]), 3) for k, h in d.groupby("be") if 0 < h.T2.sum() < len(h)},
          "| Spearman với lợi của A1: %.3f" % d[c].corr(d.gainA1, method="spearman"))
