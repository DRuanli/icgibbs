"""
g6_select_comparators.py — G6, việc T4: chọn đối thủ chính của từng bối cảnh trên TẬP PHÁT TRIỂN và kiểm lại
phạm vi khẳng định, theo quy tắc đã ghi (giao thức Mục 6 và Mục 11):

  • Ứng viên = mọi baseline peer-review của bối cảnh, mỗi baseline xét cả bản mặc định (_def) lẫn bản tune (_tun).
  • Đối thủ chính = ứng viên có metric chính trung bình (qua các mức chính, 6 dataset phát triển — PLL: 3 dataset —,
    3 seed) tốt nhất. Chọn trước khi chạy bất kỳ phương pháp nào trên tập đánh giá.
  • Kiểm phạm vi (cùng ngưỡng 60% đã dùng cho RQ4 và PU-SAR): Gibbs ấm phải đạt hoặc vượt CẢ ứng viên tốt nhất của
    từng ô LẪN TFM ngây thơ ở ≥ 60% ô (dataset × mức).

Cách dùng (sau lượt 2; sau lượt 3 thêm --override để bản chạy lại thay bản cũ):
    python g6_select_comparators.py --pilot_dir ../pilot --round2 "../pilot/g6r2_[lp][lu]*.csv" --override "../pilot/g6r3_*.csv"
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

SETTINGS = {
    # bối cảnh: (cột setting, mức chính, metric, lớn hơn là tốt?, ứng viên peer-review, file chứa gibbs/naive)
    "llp": ("llp", [8], "auc", True,          # mức chính chỉ còn túi 8 (giao thức §11, deviation 7, 26/10/2026)
            ["dllp_def", "dllp_tun", "dllplin_def", "dllplin_tun", "llpvat_def", "llpvat_tun",
             "easyllp_def", "easyllp_tun", "glws_def", "glws_tun"], "g4_pilot_v2.csv"),
    "pll": ("pll", [0.3, 0.6], "acc", True,
            ["proden_def", "proden_tun", "glwspll_def", "glwspll_tun", "rc_def", "rc_tun", "cc_def", "cc_tun"],
            "g4_rq4_pilot.csv"),
    # bối cảnh phụ của RQ4 (G6 lượt 3); bản mặc định của label spreading, self-training, forward là bản trong thí điểm
    "noisy": ("noisy", [0.2, 0.4], "auc", True, ["forward", "forward_def", "forward_tun", "cl_tfm"], "g4_rq4_pilot.csv"),
    "ssl": ("ssl", [0.05, 0.2], "auc", True, ["label_spreading", "ls_tun", "self_training", "st_tun"], "g4_rq4_pilot.csv"),
    "pu_sar": ("pu_sar", [0.1, 0.3], "auc", True,
               ["nnpu", "nnpu_tun", "upu", "upu_tun", "sar_em", "sar_em_tuned"], "g6_pusar_dev_merged.csv"),
    "pu_scar": ("pu_scar", [0.1, 0.3], "logloss_pm", False,
                ["nnpu", "nnpu_tun", "upu", "upu_tun"], "g4_pilot_v2.csv"),
}


def load(pilot_dir, round2_glob, gibbs_file, override_glob=""):
    # file trong --override thay các dòng cùng (dataset, setting, mức, seed, phương pháp) của các lượt trước
    frames = [pd.read_csv(f) for f in sorted(glob.glob(override_glob))] if override_glob else []
    frames += [pd.read_csv(f) for f in sorted(glob.glob(round2_glob))]
    frames.append(pd.read_csv(os.path.join(pilot_dir, gibbs_file)))
    d = pd.concat(frames, ignore_index=True)
    d = d[d.status == "ok"].copy()
    d["level"] = d["level"].astype(float)
    return d.drop_duplicates(["dataset", "setting", "level", "seed", "method"], keep="first")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot_dir", default="../pilot")
    ap.add_argument("--round2", default="../pilot/g6r2_*.csv")
    ap.add_argument("--out", default="g6_comparators.csv")
    ap.add_argument("--override", default="", help="glob các file chạy lại, ưu tiên hơn --round2 (vd. ../pilot/g6r3_*.csv)")
    a = ap.parse_args()
    rows = []
    for key, (setting, levels, met, higher, cands, gfile) in SETTINGS.items():
        d = load(a.pilot_dir, a.round2, gfile, a.override)
        d = d[(d.setting == setting) & d.level.isin([float(v) for v in levels])]
        if d.empty:
            continue
        cell = d.groupby(["level", "dataset", "method"])[met].mean().unstack("method")
        have = [c for c in cands if c in cell]
        missing = [c for c in cands if c not in cell]
        sign = 1 if higher else -1
        if not have:
            print(f"\n=== {key}: chưa có ứng viên nào trong dữ liệu ===")
            continue
        overall = cell[have].mean().sort_values(ascending=not higher)
        best = overall.index[0]
        print(f"\n=== {key} ({met}, {'cao' if higher else 'thấp'} hơn là tốt; mức {levels}) ===")
        print("trung bình trên tập phát triển:")
        print(overall.round(4).to_string())
        if missing:
            print(f"  [thiếu dữ liệu] {missing}")
        print(f"→ đối thủ chính (theo quy tắc): {best}")
        if "gibbs_warm" in cell and "naive" in cell:
            cell = cell.dropna(subset=["gibbs_warm", "naive"])
            bp = (sign * cell[have]).max(axis=1)
            ok = (sign * cell["gibbs_warm"] >= bp) & (sign * cell["gibbs_warm"] >= sign * cell["naive"])
            okc = (sign * cell["gibbs_warm"] >= sign * cell[best]) & (sign * cell["gibbs_warm"] >= sign * cell["naive"])
            n_ok, n = int(ok.sum()), len(ok)
            print(f"kiểm phạm vi: Gibbs ấm ≥ (mọi ứng viên tốt nhất từng ô) và ≥ naive ở {n_ok}/{n} ô "
                  f"({100 * n_ok / n:.0f}%; ngưỡng 60%) | so với riêng đối thủ chính và naive: {int(okc.sum())}/{n}")
            print(f"  Gibbs ấm trung bình = {cell['gibbs_warm'].mean():.4f}; naive = {cell['naive'].mean():.4f}; "
                  f"{best} = {overall[best]:.4f}")
            rows.append(dict(setting=key, metric=met, comparator=best, comparator_dev_mean=overall[best],
                             gibbs_dev_mean=cell["gibbs_warm"].mean(), naive_dev_mean=cell["naive"].mean(),
                             cells_pass=n_ok, cells=n, missing=";".join(missing)))
        else:
            rows.append(dict(setting=key, metric=met, comparator=best, comparator_dev_mean=overall[best],
                             missing=";".join(missing)))
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"\nđã ghi {a.out}")


if __name__ == "__main__":
    main()
