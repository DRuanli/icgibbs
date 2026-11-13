"""
make_figures.py — paper figures, regenerated from raw results (never edited by hand). Vector PDF, sized for a 5.5 in
text width; colours are a CVD-checked pair and every series also differs by marker / line style (readable in greyscale).

  figures/s2_gaps.pdf    Study 2, per dataset: Gibbs − primary comparator (LLP bag 8: AUC vs DLLP-linear;
                         PLL: accuracy vs CC-tuned, q = 0.3 and 0.6). Points are means over 3 seeds.
  figures/mechanism_map.pdf  top row of mechanism.pdf only (main text)
  figures/mechanism.pdf  Why the TFM matters (PLL, q = 0.6, the four datasets with >= 8 classes, Study 1 datasets):
                         (top) one-sweep response gain Phi(r) − r of a Gibbs sweep started from an assignment with
                         fraction r of correct labels (S2X, exploratory); (bottom) fraction of correct training labels
                         along the actual chains, one line per seed (Study 1 main run and TFM ablation).
Run:  python paper/scripts/make_figures.py
"""
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(PAPER)
sys.path.insert(0, os.path.join(ROOT, "G8", "analysis"))
from g8_confirmatory import load as load_s1, load_files  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1a1a1a", "#5a5a5a", "#e2e2e2"
COLLAPSE = ["mfeat-factors", "semeion", "MiceProtein", "optdigits"]
SHORT = {"hazelnut-spread-contaminant-detection": "hazelnut-contaminant",
         "students_dropout_and_academic_success": "students-dropout", "Is-this-a-good-customer": "good-customer",
         "Bank_Customer_Churn": "bank-churn", "Marketing_Campaign": "marketing-campaign",
         "wine-quality-white": "wine-quality-white", "compas-two-years": "compas-two-years"}
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "font.family": "serif", "font.serif": ["Times", "Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                     "pdf.fonttype": 42, "mathtext.fontset": "stix", "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def s2_gaps(out):
    R = os.path.join(ROOT, "S2", "results")
    l = pd.read_csv(os.path.join(R, "s2_table_llp8.csv"), index_col=0)
    gl = (l.gibbs_warm - l.dllplin_def).round(10).sort_values()
    p = {q: pd.read_csv(os.path.join(R, f"s2_table_pll_{q}.csv"), index_col=0) for q in (0.3, 0.6)}
    gp = pd.DataFrame({q: (p[q].gibbs_warm - p[q].cc_tun).round(10) for q in p}).sort_values(0.6)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.45), gridspec_kw=dict(width_ratios=[1.1, 1], wspace=0.75))
    ax = axes[0]
    y = np.arange(len(gl))
    ax.axvline(0, color=MUTED, lw=0.7)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.scatter(gl.values, y, s=13, color=BLUE, marker="o", zorder=3, lw=0)
    ax.set_yticks(y, [SHORT.get(n, n) for n in gl.index])
    ax.set_xlabel("AUC difference")
    ax.set_title(f"(a) LLP, bag 8: Gibbs $-$ DLLP (linear)", loc="left")
    ax = axes[1]
    y = np.arange(len(gp))
    ax.axvline(0, color=MUTED, lw=0.7)
    ax.grid(axis="x", color=GRID, lw=0.5)
    for i in y:
        ax.plot(gp.iloc[i].values, [i, i], color=GRID, lw=1.5, zorder=1)
    ax.scatter(gp[0.3], y, s=13, color=BLUE, marker="o", zorder=3, lw=0, label="$q=0.3$")
    ax.scatter(gp[0.6], y, s=13, color=ORANGE, marker="D", zorder=3, lw=0, label="$q=0.6$")
    ax.set_yticks(y, [SHORT.get(n, n) for n in gp.index])
    ax.set_xlabel("accuracy difference")
    ax.set_title("(b) PLL: Gibbs $-$ CC (tuned)", loc="left")
    ax.legend(frameon=False, loc="lower right", handletextpad=0.2, borderaxespad=0.1)
    fig.savefig(os.path.join(out, "s2_gaps.pdf"))
    plt.close(fig)


def monitors(d):
    rows = []
    for _, r in d.iterrows():
        o = dict(kv.split("=", 1) for kv in str(r.extra).split(";") if "=" in kv)
        if "monitor" in o:
            rows.append((r.dataset, r.seed, [float(v) for v in o["monitor"].split("/")]))
    return rows


def mechanism(out, both=True):
    x = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(ROOT, "S2", "raw", "S2X", "s2x_*.csv")))])
    x = x[x.status == "ok"].copy()
    x["rho"] = x.rho.astype(str)
    x["gain_s"] = x.phi - x.r_actual
    g = x.groupby(["q", "dataset", "n_classes", "backend", "rho"]).agg(r=("r_actual", "mean"), phi=("phi", "mean"),
                                                                     lo=("gain_s", "min"), hi=("gain_s", "max"),
                                                                     nseed=("seed", "nunique")).reset_index()
    g["gain"] = g.phi - g.r
    assert (g[g.q == 0.6].nseed == 3).all()             # caption: mean over 3 seeds, range shaded
    raw = os.path.join(ROOT, "G7", "raw")
    sel = lambda d: d[(d.setting == "pll") & (d.level == 0.6) & (d.method == "gibbs_warm") & d.dataset.isin(COLLAPSE)]
    mon = {"tabpfn_v2": monitors(sel(load_s1(raw))), "tabicl_v2": monitors(sel(load_files(raw, ["L6"], "g7_tabicl_*.csv")))}
    style = {"tabpfn_v2": dict(color=BLUE, marker="o", ls="-", label="TabPFN v2"),
             "tabicl_v2": dict(color=ORANGE, marker="D", ls="--", label="TabICL v2")}
    fig, axes = plt.subplots(2 if both else 1, 4, figsize=(5.5, 2.65 if both else 1.3), sharex="row", sharey="row",
                             squeeze=False)
    for col, ds in enumerate(COLLAPSE):
        C = int(x[x.dataset == ds].n_classes.iloc[0])
        ax = axes[0, col]
        ax.axhline(0, color=MUTED, lw=0.7)
        ax.grid(axis="y", color=GRID, lw=0.5)
        for be, s in style.items():
            h = g[(g.q == 0.6) & (g.dataset == ds) & (g.backend == be)].sort_values("r")
            ax.fill_between(h.r, h.lo, h.hi, color=s["color"], alpha=0.18, lw=0)
            ax.plot(h.r, h.gain, color=s["color"], ls=s["ls"], lw=1.1, marker=s["marker"], ms=2.6, label=s["label"])
        ax.set_title(f"{ds} ($C={C}$)", loc="left", fontsize=6.8)
        ax.set_xlim(0, 0.9)
        ax.set_xticks([0, 0.4, 0.8])
        ax.set_xlabel("$r$ (fraction correct)")
        if not both:
            continue
        ax = axes[1, col]
        ax.grid(axis="y", color=GRID, lw=0.5)
        for be, s in style.items():
            for k, (_, _, m) in enumerate([r for r in mon[be] if r[0] == ds]):
                ax.plot(np.arange(1, len(m) + 1), m, color=s["color"], ls=s["ls"], lw=0.9)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("sweep")
    axes[0, 0].set_ylabel(r"gain $\Phi(r)-r$")
    if both:
        axes[1, 0].set_ylabel("fraction correct")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, loc="lower center", ncol=2, handlelength=2.6, bbox_to_anchor=(0.5, -0.02 if both else -0.06))
    fig.tight_layout(h_pad=0.6, w_pad=0.4, rect=(0, 0.05 if both else 0.08, 1, 1))
    fig.savefig(os.path.join(out, "mechanism.pdf" if both else "mechanism_map.pdf"))
    plt.close(fig)


def main():
    out = os.path.join(PAPER, "figures")
    os.makedirs(out, exist_ok=True)
    s2_gaps(out)
    mechanism(out)
    mechanism(out, both=False)
    print("figures written to", out)


if __name__ == "__main__":
    main()
