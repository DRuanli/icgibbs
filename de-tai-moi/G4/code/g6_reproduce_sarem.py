"""
g6_reproduce_sarem.py — G6, việc T1: tái hiện số công bố của SAR-EM (Bekker, Robberechts & Davis, ECML-PKDD 2019)
bằng CODE CHÍNH THỨC và quy trình dữ liệu của chính bài đó, rồi kiểm bản bọc của dự án (icgibbs.baselines_sar).
==============================================================================================================

Quy trình (theo README và notebook của kho github.com/ML-KULeuven/SAR-PU, commit 6e4fc3d8):
  1. Tải và tiền xử lý dữ liệu bằng notebook gốc (`notebooks/data_preprocessing/<Tên>.ipynb`; tải từ UCI).
  2. Thêm 5 thuộc tính nhân tạo (`generate_extended_data`, như notebook "Extended Data").
  3. Sinh nhãn PU với cơ chế `simple_0.2_0.8` trên k thuộc tính propensity đầu tiên trong số thuộc tính nhân tạo
     (k = 4 như hình kết quả của bài), 5 lần gán nhãn; dùng 5 lần chia 80/20 có sẵn → 25 thí nghiệm.
  4. Huấn luyện: Supervised (nhãn thật), Naive (chưa nhãn = âm), SAR-EM; mô hình phân loại và propensity là hồi
     quy logistic mặc định (C = 1, liblinear) như bài; thuộc tính phân loại = các thuộc tính gốc.
  5. Đo ROC-AUC và MSE của f trên test, MSE của e; so với số công bố trong bảng TARGETS.

Dataset: breastcancer và mushroom (nhỏ, tải trực tiếp từ UCI). Không dùng dataset nào của tập đánh giá.
Chạy trên Kaggle (CPU, Internet: On) — xem G4/README.md. Chạy thử không cần Internet:
    python g6_reproduce_sarem.py --selftest
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = "https://github.com/ML-KULeuven/SAR-PU.git"
COMMIT = "6e4fc3d8c84ac3512669a4e36ffcb5086f5b42a7"
NOTEBOOK = {"breastcancer": "BreastCancer.ipynb", "mushroom": "Mushroom.ipynb"}

# Số công bố: Hình 1 của bài (arXiv 1809.03207v4), 4 thuộc tính propensity, trung bình 25 thí nghiệm.
# Thứ tự cột trong hình được đọc là (Naive, SAR-EM, SCAR-KM2, SCAR-TIcE, Sup.) — CẦN THẦY XÁC NHẬN TRÊN HÌNH GỐC
# trước khi kết luận (xem docs/baseline_reproduction.md).
TARGETS = {
    "breastcancer": {"sar_em": {"auc": 0.983, "mse_f": 0.058}, "naive": {"auc": 0.992, "mse_f": 0.127},
                     "supervised": {"auc": 0.994, "mse_f": 0.028}},
    "mushroom": {"sar_em": {"auc": 1.000, "mse_f": 0.103}, "naive": {"auc": 1.000, "mse_f": 0.173},
                 "supervised": {"auc": 1.000, "mse_f": 0.000}},
}
TOL = 0.01   # docs/baseline_reproduction.md: ≤ 1 điểm %, hoặc trong 1 độ lệch chuẩn công bố (ngưỡng rộng hơn)


def patch_numpy2_compat():
    """Code gốc gán mảng 1 phần tử vào một ô (`atts[i] = np.random.choice(vals, 1, p=...)`), numpy ≥ 2 báo lỗi.
    Thay bằng hàm y hệt nhưng lấy phần tử [0] — cùng dãy số ngẫu nhiên, cùng kết quả như numpy cũ."""
    import sarpu.data_extending as de

    def _generate_attribute_values(y, kY, probs, seed=None):
        if seed is not None:
            np.random.seed(seed)
        vals = [-1, 1]
        atts = np.zeros_like(y, dtype=float)
        for i, k in enumerate(kY):
            atts[i] = np.random.choice(vals, 1, p=probs.loc[k, ['att-', 'att+']].values)[0]
        return atts
    de._generate_attribute_values = _generate_attribute_values


def lr_pu():
    from icgibbs.baselines_sar import _lr_pu
    return _lr_pu(1.0, 0)


def prepare_repo(path):
    if not os.path.exists(os.path.join(path, "sarpu")):
        subprocess.run(["git", "clone", "-q", REPO, path], check=True)
        subprocess.run(["git", "-C", path, "checkout", "-q", COMMIT], check=True)
    head = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"kho SAR-PU tại {path}, commit {head}", flush=True)
    return head


def notebook_to_code(path):
    """Lấy các ô code của notebook gốc; bỏ magic '%'; đổi '!mkdir -p $x' thành os.makedirs(x)."""
    import json
    import re
    nb = json.load(open(path))
    lines = []
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        for ln in "".join(c["source"]).split("\n"):
            m = re.match(r"^\s*!mkdir -p \$(\w+)\s*$", ln)
            if m:
                lines.append(f"os.makedirs({m.group(1)}, exist_ok=True)")
            elif ln.lstrip().startswith(("%", "!")):
                continue
            else:
                lines.append(ln)
    return "\n".join(lines)


def preprocess(repo, name):
    """Chạy đúng code tiền xử lý trong notebook gốc (tải từ UCI). Shim tương thích duy nhất: np.NaN (đã bị bỏ ở
    numpy 2) trỏ về np.nan, và (pandas 3) tắt kiểu chuỗi mới để cột chữ là object như pandas cũ — không đổi hành vi."""
    from sarpu.paths_and_names import data_path
    data_folder = os.path.join(repo, "Data") + "/"
    if not os.path.exists(data_path(data_folder, name)):
        nb_dir = os.path.join(repo, "notebooks", "data_preprocessing")
        code = notebook_to_code(os.path.join(nb_dir, NOTEBOOK[name]))
        if not hasattr(np, "NaN"):
            np.NaN = np.nan
        import pandas as pd
        try:                       # pandas 3: cột chữ mặc định là kiểu 'str' không nhận gán số như pandas cũ
            pd.set_option("future.infer_string", False)
        except Exception:
            pass
        cwd = os.getcwd()
        try:
            os.chdir(nb_dir)
            exec(compile(code, NOTEBOOK[name], "exec"), {"__name__": "__nb__", "os": os, "np": np})
        finally:
            os.chdir(cwd)
    return data_folder


def run_dataset(data_folder, name, n_attrs, n_labelings=5, n_partitions=5, cls_mode="orig"):
    from sklearn.metrics import roc_auc_score
    from sarpu.data_extending import generate_extended_data
    from sarpu.input_output import read_data
    from sarpu.labeling_mechanisms import label_data
    from sarpu.paths_and_names import (classlabels_path, data_path, partition_path, propensity_labeling_path,
                                       propensity_scores_path)
    from sarpu.pu_learning import pu_learn_neg, pu_learn_sar_em

    x0 = np.loadtxt(data_path(data_folder, name))
    d = x0.shape[1]
    ext = name + "_extclustering"
    if not os.path.exists(data_path(data_folder, ext)):
        patch_numpy2_compat()
        generate_extended_data(data_folder, name, 5, n_partitions)
    prop = list(range(d, d + n_attrs))
    model = label_data(data_folder, ext, "simple_0.2_0.8", prop, [1] * n_attrs, n_labelings)
    cls = np.zeros(d + 5, bool)
    if cls_mode == "orig":            # thuộc tính phân loại = thuộc tính gốc (cách đọc theo bài)
        cls[:d] = True
    elif cls_mode == "orig_plus1":    # như chuỗi "0-<d>" trong notebook Experiments của kho gốc (cận trên tính cả d)
        cls[:d + 1] = True
    elif cls_mode == "all":           # mọi thuộc tính, kể cả 5 thuộc tính nhân tạo
        cls[:] = True
    else:
        raise ValueError(cls_mode)
    rows = []
    for part in range(n_partitions):
        for lab in range(n_labelings):
            files = (data_path(data_folder, ext), classlabels_path(data_folder, ext),
                     propensity_labeling_path(data_folder, ext, model, lab),
                     propensity_scores_path(data_folder, ext, model))
            (xtr, ytr, str_, etr), (xte, yte, ste, ete) = read_data(files, partition_path(data_folder, ext, part))
            ytr, yte, str_ = ytr.astype(int), yte.astype(int), str_.astype(int)
            out = {}
            f, _ = pu_learn_neg(xtr, ytr, classification_model=lr_pu(), classification_attributes=cls)
            out["supervised"] = (f.predict_proba(xte), None)
            f, _ = pu_learn_neg(xtr, str_, classification_model=lr_pu(), classification_attributes=cls)
            out["naive"] = (f.predict_proba(xte), None)
            with np.errstate(divide="ignore", invalid="ignore"):
                f, e, info = pu_learn_sar_em(xtr, str_, prop, classification_attributes=cls,
                                             classification_model=lr_pu(), propensity_model=lr_pu())
            out["sar_em"] = (f.predict_proba(xte), e.predict_proba(xte))
            # bản bọc của dự án: phải cho đúng cùng kết quả (cùng thuật toán, cùng mô hình)
            from icgibbs.baselines_sar import SAREM
            w = SAREM(seed=0).fit(xtr, str_, prop, classification_cols=np.flatnonzero(cls))
            out["sar_em_wrapper"] = (w.predict_proba(xte)[:, 1], w.propensity(xte))
            for meth, (pf, pe) in out.items():
                r = dict(dataset=name, cls_mode=cls_mode, d=d, partition=part, labeling=lab, method=meth, n_train=len(ytr),
                         labeled=int(str_.sum()), auc=roc_auc_score(yte, pf), mse_f=float(np.mean((pf - yte) ** 2)))
                if pe is not None:
                    r["mse_e"] = float(np.mean((pe - ete) ** 2))
                rows.append(r)
            print(f"  {name} part {part} lab {lab}: " + " ".join(
                f"{m}={roc_auc_score(yte, v[0]):.3f}" for m, v in out.items()), flush=True)
    return rows


def report(df):
    print("\n========= TÁI HIỆN SAR-EM (trung bình ± độ lệch chuẩn qua 25 thí nghiệm) =========")
    keys = ["dataset", "cls_mode", "method"] if "cls_mode" in df else ["dataset", "method"]
    g = df.groupby(keys)[["auc", "mse_f", "mse_e"]].agg(["mean", "std"]).round(4)
    print(g.to_string())
    ok_all = True
    for ds, tgt in TARGETS.items():
        for meth, vals in tgt.items():
            d = df[(df.dataset == ds) & (df.method == meth)]
            if d.empty:
                continue
            for met, v in vals.items():
                mu, sd = d[met].mean(), d[met].std()
                ok = abs(mu - v) <= max(TOL, sd)
                ok_all &= ok
                print(f"  {ds:12s} {meth:10s} {met:5s}: tái hiện {mu:.4f} ± {sd:.4f} | công bố {v:.3f} | "
                      f"{'ĐẠT' if ok else 'CHÊNH'}")
    idx = [k for k in ("dataset", "cls_mode", "partition", "labeling") if k in df]
    w = df[df.method == "sar_em_wrapper"].set_index(idx).auc
    o = df[df.method == "sar_em"].set_index(idx).auc
    print(f"\nbản bọc của dự án so với gọi trực tiếp code chính thức: chênh AUC lớn nhất = {float((w - o).abs().max()):.2e}")
    print("(tiêu chí ghi trước: ≤ 1 điểm %, hoặc trong 1 độ lệch chuẩn — lấy ngưỡng rộng hơn)")
    return ok_all


def selftest():
    """Kiểm vòng lặp trên dữ liệu tổng hợp đặt đúng cấu trúc thư mục của kho gốc (không cần Internet)."""
    import tempfile
    from sarpu.paths_and_names import classlabels_path, data_path, partition_path, partitions_data_path
    rng = np.random.default_rng(0)
    folder = tempfile.mkdtemp() + "/"
    n = 600
    x = np.sign(rng.normal(size=(n, 8)))
    y = (x[:, 0] + x[:, 1] + 0.5 * rng.normal(size=n) > 0).astype(int)
    os.makedirs(partitions_data_path(folder, "toy"), exist_ok=True)
    np.savetxt(data_path(folder, "toy"), x)
    np.savetxt(classlabels_path(folder, "toy"), y, fmt="%d")
    for i in range(2):
        part = np.where(rng.uniform(size=n) < 0.8, 1, 2)
        np.savetxt(partition_path(folder, "toy", i), part, fmt="%d")
    import pandas as pd
    df = pd.DataFrame(run_dataset(folder, "toy", 4, n_labelings=2, n_partitions=2))
    report(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/kaggle/working/SAR-PU")
    ap.add_argument("--datasets", default="breastcancer,mushroom")
    ap.add_argument("--n_attrs", type=int, default=4)
    ap.add_argument("--out", default="g6_sarem_reproduction.csv")
    ap.add_argument("--cls_mode", default="orig", help="orig | orig_plus1 | all (kiểm giả thuyết về chênh lệch)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    import pandas as pd
    head = prepare_repo(a.repo)
    rows = []
    for name in a.datasets.split(","):
        folder = preprocess(a.repo, name)
        rows += run_dataset(folder, name, a.n_attrs, cls_mode=a.cls_mode)
        pd.DataFrame(rows).assign(sarpu_commit=head).to_csv(a.out, index=False)
    report(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
