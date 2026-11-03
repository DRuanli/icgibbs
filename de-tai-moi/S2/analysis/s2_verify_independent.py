"""s2_verify_independent.py — kiểm lại độc lập phép thử chính của S2 (H2′, H4′) thẳng từ raw/, không dùng hàm của dự án.
Chạy:  python s2_verify_independent.py <thư mục S2>"""
import glob, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
R = sys.argv[1]
fs = glob.glob(f"{R}/raw/S2T/s2_tfm_*.csv") + glob.glob(f"{R}/raw/S2C*/s2_base_*.csv") + glob.glob(f"{R}/raw/S2G/s2_g_*.csv")
d = pd.concat([pd.read_csv(f) for f in fs])
d = d[d.status == "ok"]
fin = pd.read_csv(f"{R}/s2_final.csv"); fin = fin[fin.kept]
B = set(fin[fin.group == "binary"].did); M = set(fin[fin.group == "multiclass"].did)
print("binary", len(B), "multi", len(M))
def table(setting, level, metric, ids):
    x = d[(d.setting == setting) & (np.isclose(d.level.astype(float), level)) & d.dataset_id.isin(ids)]
    c = x.groupby(["dataset_id", "method"]).seed.nunique().unstack()
    t = x.groupby(["dataset_id", "method"])[metric].mean().unstack().round(10)
    return t, c
def holm(ps):
    ps = np.array(ps); o = np.argsort(ps); m = len(ps); adj = np.zeros(m); run = 0
    for k, i in enumerate(o):
        run = max(run, (m - k) * ps[i]); adj[i] = min(1, run)
    return adj
t, c = table("llp", 8, "auc", B)
print("LLP n", len(t), "min seeds gibbs", c.gibbs_warm.min(), "dllplin", c.dllplin_def.min())
ps = []
for b in ("dllplin_def", "naive"):
    diff = t.gibbs_warm - t[b]; ps.append(wilcoxon(diff).pvalue)
    print(" ", b, round(diff.mean(), 4), (diff > 0).sum(), (diff == 0).sum(), (diff < 0).sum())
print("  holm", holm(ps))
base = ["dllp_def","dllp_tun","dllplin_def","dllplin_tun","llpvat_def","llpvat_tun","easyllp_def","easyllp_tun","glws_def","glws_tun"]
best = t[base].mean().idxmax(); print("  strongest", best, ((t.gibbs_warm - t[best]) > 0).mean())
ps = []; res = {}
for q in (0.3, 0.6):
    t, c = table("pll", q, "acc", M)
    print("PLL", q, "n", len(t), "min seeds", c.gibbs_warm.min(), c.cc_tun.min(), c.naive.min())
    for b in ("cc_tun", "naive"):
        diff = t.gibbs_warm - t[b]; ps.append(wilcoxon(diff).pvalue)
        print(" ", b, round(diff.mean(), 4), (diff > 0).sum(), (diff == 0).sum(), (diff < 0).sum())
    pb = ["proden_def","proden_tun","glwspll_def","glwspll_tun","rc_def","rc_tun","cc_def","cc_tun"]
    best = t[pb].mean().idxmax(); print("  strongest", best, (t.gibbs_warm >= t[best]).mean())
print("  holm4", holm(ps).round(4))
