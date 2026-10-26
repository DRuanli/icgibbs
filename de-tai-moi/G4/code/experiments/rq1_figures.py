"""rq1_figures.py — vẽ hình cho RQ1_REPORT.md từ các CSV trong results/rq1 (PNG, nền sáng).
Bảng màu: 3 slot đầu của bảng tham chiếu (đã kiểm định CVD); nhận diện series thêm bằng marker + nhãn trực tiếp."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "..", "results", "rq1")
FIG = os.path.join(HERE, "..", "..", "results", "figures")
os.makedirs(FIG, exist_ok=True)
S1, S2, S3, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRAY, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "lines.linewidth": 2,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "legend.frameon": False})


def fig_mc():
    d = pd.read_csv(os.path.join(R, "e1mc_convergence.csv"))
    cfgs = list(d.config.unique())
    fig, axes = plt.subplots(1, len(cfgs), figsize=(3.0 * len(cfgs), 2.8), sharey=True)
    for ax, cfg in zip(np.atleast_1d(axes), cfgs):
        x = d[d.config == cfg]
        ax.loglog(x.samples, x.tv, "o-", color=S1, ms=5, label="chuỗi (cài đặt thật)")
        ax.loglog(x.samples, x.iid_floor, "s--", color=GRAY, ms=4, lw=1.5, label="mẫu i.i.d. từ π (sàn MC)")
        ax.axhline(0.01, color=S2, lw=1, ls=":")
        names = {"PU-SCAR single-site": "PU-SCAR, Gibbs một mẫu", "LLP swap": "LLP, MH hoán đổi",
                 "SSL block + MH (K=2)": "SSL, khối + MH (K=2)", "LLP bag-block + MH (K=2)": "LLP, theo túi + MH (K=2)"}
        pri = {"cell": "prior ô", "mixture": "prior hỗn hợp"}[x.prior.iloc[0]]
        ax.set_title(f"{names.get(cfg, cfg)}\n({pri}, {x.n_states.iloc[0]} trạng thái)", fontsize=8.5, color=INK)
        ax.set_xlabel("số vòng quét S")
    np.atleast_1d(axes)[0].set_ylabel(r"TV(tần suất, $\pi$)")
    np.atleast_1d(axes)[0].legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "rq1_mc_convergence.png"), dpi=170)


def fig_thm2():
    d3 = pd.read_csv(os.path.join(R, "e3_thm2_perturbed.csv"))
    p4 = os.path.join(R, "e4_thm2_learned_pfn.csv")
    d4 = pd.read_csv(p4) if os.path.exists(p4) else None
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1))
    for ax, kind, title in zip(axes, ("single", "swap"), ("Gibbs một mẫu (PU-SCAR)", "MH hoán đổi (LLP)")):
        x = d3[d3.kernel == kind]
        ax.loglog(x.kernel_err_avg, x.tv_actual, "o", color=S1, ms=5, label="TV thật, q nhiễu")
        lim = np.array([x.kernel_err_avg.min() * 0.7, x.kernel_err_avg.max() * 1.4])
        ax.loglog(lim, x.tau.iloc[0] * lim, "-", color=S2, lw=1.5, label=rf"chặn $\tau\bar\varepsilon$ ($\tau$={x.tau.iloc[0]:.1f})")
        if d4 is not None:
            y = d4[d4.kernel == kind]
            ax.loglog(y.kernel_err_avg, y.tv_actual, "^", color=S3, ms=6, label="TV thật, PFN học được")
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel(r"$\bar\varepsilon$ = trung bình theo $\tilde\pi$ của TV($\tilde P(y,\cdot)$, $P(y,\cdot)$)")
    axes[0].set_ylabel(r"TV($\tilde\pi$, $\pi$)")
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "rq1_thm2.png"), dpi=170)


def fig_block():
    p = os.path.join(R, "e5c_block_composition.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), sharey=True)
    for ax, setting in zip(axes, ("PU", "SSL-asym")):
        x = d[d.setting == setting]
        for K, col, mk in ((20, S1, "o"), (10, S2, "s"), (5, S3, "^")):
            z = x[x.K == K]
            z = z.dropna(subset=["pred_err_all"])
            ax.loglog(z.n, np.maximum(z.pred_err_free, 1e-5), mk + "--", color=col, ms=4, lw=1.2, alpha=0.55)
            ax.loglog(z.n, np.maximum(z.pred_err_all, 1e-5), mk + "-", color=col, ms=5, label=f"K={K}")
        ax.set_ylim(5e-6, 0.2)
        ax.set_title(f"{setting}: nét liền = khối trên mọi mẫu,\nnét đứt = chỉ trên mẫu tự do (≈0 vẽ ở 1e-5)", fontsize=8.5, color=INK)
        ax.set_xlabel("n")
    axes[0].set_ylabel("|dự đoán khối − dự đoán chính xác|")
    axes[0].legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "rq1_block_bias.png"), dpi=170)


if __name__ == "__main__":
    for f in (fig_mc, fig_thm2, fig_block):
        try:
            f()
            print("ok", f.__name__)
        except FileNotFoundError as e:
            print("bỏ qua", f.__name__, e)
