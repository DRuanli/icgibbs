"""
g8_figures.py — hình của báo cáo G8, tạo lại từ G7/raw/ (không sửa tay).
  fig1_gap_by_dataset.png : chênh Gibbs − đối thủ chính theo dataset, TabPFN v2 (TFM chính) và TabICL v2 (ablation TFM),
                            LLP túi 8 (AUC) và PLL q = 0,3 / 0,6 (accuracy). Trung bình 3 seed.
  fig2_chain_monitor.png  : tỉ lệ nhãn gán đúng theo vòng quét (chỉ để chẩn đoán) ở PLL q = 0,6, bốn dataset ≥ 8 lớp.
  fig3_size.png           : chênh Gibbs − đối thủ chính theo cỡ train (ablation cỡ dữ liệu, TabPFN v2).
Chạy:  python g8_figures.py --raw ../../G7/raw --out ../figures
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from g8_confirmatory import load, load_files  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # thứ tự cố định của bảng màu tham chiếu (đã kiểm CVD)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
COLLAPSE = ["mfeat-factors", "semeion", "MiceProtein", "optdigits"]


def fig1(main, tabicl, out):
    panels = [("llp", 8.0, "auc", "dllplin_def", "LLP túi 8 — AUC\nGibbs − DLLP tuyến tính"),
              ("pll", 0.3, "acc", "cc_tun", "PLL q = 0,3 — accuracy\nGibbs − CC tune"),
              ("pll", 0.6, "acc", "cc_tun", "PLL q = 0,6 — accuracy\nGibbs − CC tune")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 5.2), gridspec_kw=dict(width_ratios=[10, 14, 14]))
    for ax, (st, lv, met, comp, title) in zip(axes, panels):
        m = main[(main.setting == st) & (main.level == lv)].pivot_table(index="dataset", columns="method", values=met)
        t = tabicl[(tabicl.setting == st) & (tabicl.level == lv)].pivot_table(index="dataset", columns="method",
                                                                              values=met)
        g1 = (m.gibbs_warm - m[comp]).rename("TabPFN v2")
        g2 = (t.gibbs_warm - m[comp]).rename("TabICL v2")
        g = pd.concat([g1, g2], axis=1).sort_values("TabPFN v2")
        y = np.arange(len(g))
        ax.axvline(0, color=MUTED, lw=1)
        ax.grid(axis="x", color=GRID, lw=0.8)
        for i in y:
            ax.plot([g.iloc[i, 0], g.iloc[i, 1]], [i, i], color=GRID, lw=2, zorder=1)
        ax.scatter(g["TabPFN v2"], y, s=36, color=BLUE, zorder=3, label="TabPFN v2 (TFM chính)",
                   edgecolor="#fcfcfb", lw=1.5)
        ax.scatter(g["TabICL v2"], y, s=36, color=ORANGE, zorder=3, marker="D", label="TabICL v2 (ablation)",
                   edgecolor="#fcfcfb", lw=1.5)
        ax.set_yticks(y, g.index, fontsize=8)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        lo = min(-0.12, g.min().min() - 0.03)
        ax.set_xlim(lo, max(0.15, g.max().max() + 0.03))
    fig.suptitle("Chênh Gibbs − đối thủ chính theo dataset (trung bình 3 seed; dương = Gibbs tốt hơn)", x=0.01,
                 ha="left", fontsize=10, color=INK)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.995, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(out, "fig1_gap_by_dataset.png"), dpi=160)
    plt.close(fig)


def monitors(d):
    rows = []
    for _, r in d.iterrows():
        o = dict(kv.split("=", 1) for kv in str(r.extra).split(";") if "=" in kv)
        if "monitor" in o:
            rows.append((r.dataset, r.seed, [float(v) for v in o["monitor"].split("/")]))
    return rows


def fig2(main, tabicl, out):
    sel = lambda d: d[(d.setting == "pll") & (d.level == 0.6) & (d.method == "gibbs_warm") & d.dataset.isin(COLLAPSE)]
    a, b = monitors(sel(main)), monitors(sel(tabicl))
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.0), sharey=True)
    for ax, ds in zip(axes, COLLAPSE):
        for rows, col, lab in ((a, BLUE, "TabPFN v2"), (b, ORANGE, "TabICL v2")):
            for k, (name, seed, mon) in enumerate([r for r in rows if r[0] == ds]):
                ax.plot(np.arange(1, len(mon) + 1), mon, color=col, lw=2, alpha=0.9, label=lab if k == 0 else None)
        ax.set_title(ds, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("vòng quét")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("tỉ lệ nhãn gán đúng (train)")
    axes[0].legend(frameon=False, fontsize=8, loc="center right")
    fig.suptitle("PLL q = 0,6, bốn dataset ≥ 8 lớp: chuỗi Gibbs với hai TFM (mỗi đường một seed)", x=0.01, ha="left",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_chain_monitor.png"), dpi=160)
    plt.close(fig)


def fig3(main, raw, out):
    t = load_files(raw, ["L4"], "g7_size_tfm.csv", extra_key=("n_train",))
    b = load_files(raw, ["C3"], "g7_size_base.csv", extra_key=("n_train",))
    s = pd.concat([t, b])
    ids = s.dataset_id.unique()
    f = main[main.dataset_id.isin(ids) & (((main.setting == "llp") & (main.level == 8)) |
                                          ((main.setting == "pll") & (main.level == 0.3)))]
    a = pd.concat([s, f])
    a["metric"] = np.where(a.setting == "pll", a.acc, a.auc)
    a = a[a.method.isin(["gibbs_warm", "dllplin_def", "cc_tun"])]
    a = a[~((a.setting == "llp") & (a.method == "cc_tun")) & ~((a.setting == "pll") & (a.method == "dllplin_def"))]
    a["role"] = np.where(a.method == "gibbs_warm", "gibbs", "comp")
    p = a.pivot_table(index=["setting", "dataset", "n_train", "seed"], columns="role", values="metric")
    p = (p.gibbs - p.comp).groupby(level=["setting", "dataset", "n_train"]).mean().reset_index(name="gap")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=False)
    for ax, (st, title) in zip(axes, [("llp", "LLP túi 8 — AUC: Gibbs − DLLP tuyến tính"),
                                      ("pll", "PLL q = 0,3 — accuracy: Gibbs − CC tune")]):
        q = p[p.setting == st]
        ax.axhline(0, color=MUTED, lw=1)
        for col, (ds, g) in zip([BLUE, ORANGE, AQUA], q.groupby("dataset")):
            g = g.sort_values("n_train")
            ax.plot(g.n_train, g.gap, color=col, lw=2, marker="o", ms=5, label=ds)
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xticks([500, 1000, 2000, 4000], ["500", "1.000", "2.000", "4.000"])
        ax.set_xlabel("số mẫu train (điểm cuối = toàn bộ train)")
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)
        ax.set_xlim(400, 5500)
        ax.set_ylim(bottom=min(0, q.gap.min()) - 0.005)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_size.png"), dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="../../G7/raw")
    ap.add_argument("--out", default="../figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    m = load(a.raw)
    t = load_files(a.raw, ["L6"], "g7_tabicl_*.csv")
    fig1(m, t, a.out)
    fig2(m, t, a.out)
    fig3(m, a.raw, a.out)


if __name__ == "__main__":
    main()
