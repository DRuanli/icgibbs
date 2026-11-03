"""s2_figures.py — hình kết quả xác nhận S2: chênh Gibbs (TabICL v2) − đối thủ chính theo dataset.
Đọc S2/results/s2_table_*.csv (do s2_confirmatory.py tạo); ghi S2/figures/s2_gap_by_dataset.png."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

S2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"


def main():
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
                         "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    R = os.path.join(S2, "results")
    l = pd.read_csv(os.path.join(R, "s2_table_llp8.csv"), index_col=0)
    gl = (l.gibbs_warm - l.dllplin_def).sort_values()
    p = {q: pd.read_csv(os.path.join(R, f"s2_table_pll_{q}.csv"), index_col=0) for q in (0.3, 0.6)}
    gp = pd.DataFrame({q: p[q].gibbs_warm - p[q].cc_tun for q in p}).sort_values(0.6)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw=dict(width_ratios=[1, 1]))
    ax = axes[0]
    y = np.arange(len(gl))
    ax.axvline(0, color=MUTED, lw=1)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.scatter(gl.values, y, s=36, color=BLUE, zorder=3, edgecolor="#fcfcfb", lw=1.5)
    ax.set_yticks(y, gl.index, fontsize=8)
    ax.set_title("LLP túi 8 — AUC: Gibbs − DLLP tuyến tính\n18 dataset; thắng 16/18, Holm p = 0,0001", loc="left",
                 fontsize=9, color=INK)
    ax = axes[1]
    y = np.arange(len(gp))
    ax.axvline(0, color=MUTED, lw=1)
    ax.grid(axis="x", color=GRID, lw=0.8)
    for i in y:
        ax.plot(gp.iloc[i].values, [i, i], color=GRID, lw=2, zorder=1)
    ax.scatter(gp[0.3], y, s=36, color=BLUE, zorder=3, label="q = 0,3", edgecolor="#fcfcfb", lw=1.5)
    ax.scatter(gp[0.6], y, s=36, color=ORANGE, marker="D", zorder=3, label="q = 0,6", edgecolor="#fcfcfb", lw=1.5)
    ax.set_yticks(y, gp.index, fontsize=8)
    ax.set_title("PLL — accuracy: Gibbs − CC tune\n10 dataset; q 0,3: 8 thắng/2 hoà; q 0,6: 8/1/1; Holm p = 0,016",
                 loc="right", fontsize=9, color=INK)
    ax.legend(frameon=False, loc="lower right")
    fig.suptitle("Nghiên cứu 2 (dataset mới, đăng ký trước, TabICL v2): chênh theo dataset, trung bình 3 seed; "
                 "dương = Gibbs tốt hơn", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.join(S2, "figures"), exist_ok=True)
    fig.savefig(os.path.join(S2, "figures", "s2_gap_by_dataset.png"), dpi=160)


if __name__ == "__main__":
    main()
