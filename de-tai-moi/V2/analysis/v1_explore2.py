"""v1_explore2.py — THĂM DÒ (viết 07/11/2026, sau khi đã xem số V1) để thiết kế lượt V1C mở rộng.
  X1  Gộp chéo TFM: trên các ô có cả TabPFN v2 và TabICL v2 (cùng lần chia, đã kiểm mã băm), trung bình dự đoán của
      6 chuỗi (3 mỗi TFM) so với A1 của từng TFM và với TFM tốt hơn trên ô đó (tiên tri).
  X2  Hội tụ theo vòng quét: |r_t − r_cuối| theo t (monitor, nhãn thật) — có cần 30 vòng quét không?
Chạy: python -I V2/analysis/v1_explore2.py
"""
import glob, os, re
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAT = re.compile(r"v1_(?P<be>.+?)_(?P<did>\d+)_(?P<st>llp|pll)_(?P<lev>[\d.]+)_s(?P<seed>\d+)\.npz$")
S = ("warm", "base", "cold")
def met(P, y, st):
    return roc_auc_score(y, P[:, 1]) if st == "llp" else float(np.mean(P.argmax(1) == y))
cells, conv = {}, []
for f in sorted(glob.glob(os.path.join(V2, "raw", "V1*", "v1_npz", "*.npz"))):
    m = PAT.search(os.path.basename(f)); z = np.load(f)
    cells[(m["be"], int(m["did"]), m["st"], int(m["seed"]))] = dict(
        P=np.mean([z[f"P_{s}"] for s in S], 0), y=z["yte"], h=str(z["sha256_split"]), L=z["P_L"])
    for s in S:
        mon = np.asarray(z[f"monitor_{s}"], float); rf = mon[-5:].mean()
        conv.append(dict(be=m["be"], st=m["st"], start=s, **{f"t{t}": abs(mon[t - 1] - rf) for t in (3, 5, 8, 10, 12, 15, 20)}))
rows = []
for (be, did, st, seed), c in cells.items():
    if be != "tabpfn_v2": continue
    o = cells.get(("tabicl_v2", did, st, seed))
    if o is None or o["h"] != c["h"]: continue
    a, b = met(c["P"], c["y"], st), met(o["P"], c["y"], st)
    x = met((c["P"] + o["P"]) / 2, c["y"], st)
    rows.append(dict(did=did, st=st, seed=seed, A1_pfn=a, A1_icl=b, X=x, L=met(c["L"], c["y"], st)))
d = pd.DataFrame(rows)
g = d.groupby(["st", "did"]).mean(numeric_only=True)
g["best1"] = g[["A1_pfn", "A1_icl"]].max(1)
print("# X1. Gộp chéo TFM (trung bình 3 seed, theo dataset)")
for st, h in g.groupby(level=0):
    print(st, "n =", len(h), "| X − A1_pfn", round((h.X - h.A1_pfn).mean(), 4), "| X − A1_icl", round((h.X - h.A1_icl).mean(), 4),
          "| X − tốt hơn (tiên tri)", round((h.X - h.best1).mean(), 4), "| X thua tốt hơn > 0,02:", int(((h.best1 - h.X) > 0.02).sum()),
          "| X thua max(A1_icl, L) > 0,02:", int((np.maximum(h.A1_icl, h.L) - h.X > 0.02).sum()))
print(g.round(4).to_string())
c = pd.DataFrame(conv)
print("\n# X2. Trung vị |r_t − r_cuối| theo vòng quét t")
print(c.groupby(["be", "st"]).median(numeric_only=True).round(4).to_string())
print("tỉ lệ chuỗi có |r_10 − r_cuối| < 0,01:", c.groupby(["be", "st"]).t10.apply(lambda v: (v < 0.01).mean()).round(3).to_dict())
