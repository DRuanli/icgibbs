"""
s2x_response_map_report.py — S2_PLAN §5.1 (THĂM DÒ, dataset cũ của NC1): tóm tắt và vẽ ánh xạ phản ứng Φ(r) từ lượt S2X.
Đọc S2/raw/S2X/s2x_*.csv; ghi S2/results/s2x_summary.csv và S2/figures/s2x_response_map.png.
  gain(r) = Φ(r) − r: dương = một vòng quét đẩy cách gán về phía nhãn đúng; âm = kéo về phía ngẫu nhiên.
Chạy:  python s2x_response_map_report.py
"""
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
S2 = os.path.dirname(HERE)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ORDER = ["mfeat-factors", "semeion", "MiceProtein", "optdigits", "eucalyptus", "steel-plates-fault", "satimage"]
NAME = {"tabpfn_v2": "TabPFN v2", "tabicl_v2": "TabICL v2"}


def main():
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(S2, "raw", "S2X", "s2x_*.csv")))])
    d = d[d.status == "ok"].copy()
    d["rho"] = d.rho.astype(str)
    g = d.groupby(["q", "dataset", "n_classes", "backend", "rho"]).agg(
        r=("r_actual", "mean"), phi=("phi", "mean"), phi_sd_seeds=("phi", "std"), n=("phi", "size")).reset_index()
    g["gain"] = g.phi - g.r
    os.makedirs(os.path.join(S2, "results"), exist_ok=True)
    g.round(4).to_csv(os.path.join(S2, "results", "s2x_summary.csv"), index=False)
    u = g[g.rho == "u"].pivot_table(index=["q", "dataset", "n_classes"], columns="backend", values="gain").round(3)
    print("gain tại trạng thái gán đều trong tập ứng viên (r ≈ 1/|S|):\n", u.to_string())
    lo = g[(g.q == 0.6) & g.rho.isin(["0.2", "0.3", "0.45"])].pivot_table(index="dataset", columns="backend",
                                                                          values="gain", aggfunc="min").round(3)
    print("\nq = 0,6: gain nhỏ nhất trong r ≈ 0,2–0,45:\n", lo.to_string())

    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
                         "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(2, len(ORDER), figsize=(15, 4.9), sharex=True)
    for row, q in enumerate((0.6, 0.3)):
        for col, ds in enumerate(ORDER):
            ax = axes[row, col]
            ax.plot([0, 1], [0, 0], color=MUTED, lw=1)
            for be, colr, mk in (("tabpfn_v2", BLUE, "o"), ("tabicl_v2", ORANGE, "D")):
                x = g[(g.q == q) & (g.dataset == ds) & (g.backend == be)].sort_values("r")
                ax.plot(x.r, x.gain, color=colr, lw=2, marker=mk, ms=4, label=NAME[be])
                xu = x[x.rho == "u"]
                ax.scatter(xu.r, xu.gain, s=70, facecolor="none", edgecolor=colr, lw=1.5, zorder=4)
            ax.grid(axis="y", color=GRID, lw=0.8)
            C = int(d[d.dataset == ds].n_classes.iloc[0])
            if row == 0:
                ax.set_title(f"{ds}\n({C} lớp)", fontsize=8.5, color=INK, loc="left")
            if col == 0:
                ax.set_ylabel(f"q = {str(q).replace('.', ',')}\nΦ(r) − r", color=INK)
            if row == 1:
                ax.set_xlabel("r: tỉ lệ nhãn gán đúng")
            ax.set_xlim(0, 0.9)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.995, 0.995))
    fig.suptitle("Một vòng quét Gibbs (PLL) đẩy cách gán về phía nhãn đúng (> 0) hay về phía ngẫu nhiên (< 0); "
                 "vòng tròn rỗng = trạng thái gán đều trong tập ứng viên", x=0.01, ha="left", fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.join(S2, "figures"), exist_ok=True)
    fig.savefig(os.path.join(S2, "figures", "s2x_response_map.png"), dpi=150)


if __name__ == "__main__":
    main()
