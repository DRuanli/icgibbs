"""
kaggle_g2_benchmark.py — Đo thời gian chạy TFM cho đề tài A (G2, việc mở #2)
=========================================================================

Mục đích: thay giả định "~1 giây mỗi forward" trong TOPIC_CARD_A bằng số đo thật trên GPU Kaggle/Colab.

Thứ được đo
  (a) "block forward": một bước Gibbs theo khối — ngữ cảnh = (K-1)/K số hàng, truy vấn = 1/K số hàng
      (đúng phép tính mà mỗi bước Gibbs của đề tài A cần).
  (b) "final predict": dự đoán tập test (500 hàng) với ngữ cảnh đầy đủ n hàng.
  Mỗi cấu hình: 1 lần làm nóng (warm-up) + 3 lần đo, lấy trung vị. Dữ liệu là make_classification
  (chỉ để đo thời gian — thời gian phụ thuộc kích thước, không phụ thuộc nội dung).

Cách chạy trên Kaggle (Notebook → Settings → Accelerator: GPU T4 x1 hoặc P100; Internet: On)
  !pip install -q tabpfn tabicl
  # TabPFN: chấp nhận license tại https://ux.priorlabs.ai (tab License), tạo API key,
  # thêm vào Kaggle: Add-ons → Secrets → tên TABPFN_TOKEN, rồi:
  from kaggle_secrets import UserSecretsClient; import os
  os.environ["TABPFN_TOKEN"] = UserSecretsClient().get_secret("TABPFN_TOKEN")
  !python kaggle_g2_benchmark.py --backends tabpfn_v2,tabicl_v2 --out g2_benchmark.csv
  # Nếu chưa có token TabPFN: chạy riêng --backends tabicl_v2 (TabICL tải trọng số mở từ Hugging Face).

Kiểm tra trên máy không có GPU / không tải trọng số:
  python kaggle_g2_benchmark.py --dry-run --out dry.csv     (dùng LogisticRegression thay TFM)

Thời gian chạy ước tính trên T4: 20–40 phút cho cả hai backend. Script lưu CSV sau mỗi dòng,
nên nếu phiên bị ngắt vẫn giữ được phần đã đo.
"""
from __future__ import annotations

import argparse
import csv
import os
import platform
import statistics
import time
from importlib import metadata

import numpy as np
from sklearn.datasets import make_classification

K_BLOCKS = 10
N_TEST_FINAL = 500
SIZES = [500, 1000, 2000, 5000]
N_FEATURES = [10, 50]
N_CLASSES = [2, 5]
REPEATS = 3
SKIP_ABOVE_SECONDS = 120.0      # bỏ các cấu hình lớn hơn nếu một lần chạy vượt ngưỡng này


def device_info():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0), torch.__version__
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "Apple MPS", torch.__version__
        return "CPU", torch.__version__
    except ImportError:
        return "CPU (no torch)", "-"


def make_estimator(backend: str, n_estimators: int | None):
    """n_estimators=None nghĩa là cấu hình mặc định của gói."""
    if backend == "dry":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=200)
    if backend.startswith("tabpfn"):
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        version = {"tabpfn_v2": ModelVersion.V2, "tabpfn_v2_5": ModelVersion.V2_5,
                   "tabpfn_v3": ModelVersion.V3}[backend]
        overrides = {"ignore_pretraining_limits": True}
        if n_estimators is not None:
            overrides["n_estimators"] = n_estimators
        return TabPFNClassifier.create_default_for_version(version, **overrides)
    if backend == "tabicl_v2":
        from tabicl import TabICLClassifier
        kwargs = {}
        if n_estimators is not None:
            kwargs["n_estimators"] = n_estimators
        return TabICLClassifier(**kwargs)
    raise ValueError(backend)


def time_call(fn, repeats=REPEATS):
    fn()                                   # warm-up (tải trọng số, biên dịch, cấp phát bộ nhớ)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), min(ts), max(ts)


def sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def bench_one(backend, n, m, c, n_est, rng):
    X, y = make_classification(n_samples=n + N_TEST_FINAL, n_features=m, n_informative=min(m, 8),
                               n_redundant=0, n_classes=c, n_clusters_per_class=1,
                               random_state=int(rng.integers(1e9)))
    X = X.astype(np.float32)
    Xtr, ytr, Xte = X[:n], y[:n], X[n:]
    block = n // K_BLOCKS
    ctx_idx, q_idx = np.arange(block, n), np.arange(block)

    def block_forward():
        est = make_estimator(backend, n_est)
        est.fit(Xtr[ctx_idx], ytr[ctx_idx])
        est.predict_proba(Xtr[q_idx])
        sync()

    def final_predict():
        est = make_estimator(backend, n_est)
        est.fit(Xtr, ytr)
        est.predict_proba(Xte)
        sync()

    b_med, b_min, b_max = time_call(block_forward)
    f_med, f_min, f_max = time_call(final_predict)
    return dict(block_median_s=round(b_med, 4), block_min_s=round(b_min, 4), block_max_s=round(b_max, 4),
                final_median_s=round(f_med, 4), final_min_s=round(f_min, 4), final_max_s=round(f_max, 4))


def pkg_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="tabpfn_v2,tabicl_v2")
    ap.add_argument("--n_estimators", default="1,default", help="danh sách, 'default' = mặc định của gói")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="g2_benchmark.csv")
    args = ap.parse_args()

    backends = ["dry"] if args.dry_run else args.backends.split(",")
    n_ests = [None if v == "default" else int(v) for v in args.n_estimators.split(",")]
    dev, torch_v = device_info()
    rng = np.random.default_rng(0)
    fields = ["backend", "n_estimators", "n_train", "n_features", "n_classes", "k_blocks",
              "block_median_s", "block_min_s", "block_max_s", "final_median_s", "final_min_s", "final_max_s",
              "device", "torch", "tabpfn", "tabicl", "python", "status"]
    write_header = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if write_header:
            w.writeheader()
        for backend in backends:
            for n_est in n_ests:
                for m in N_FEATURES:
                    for c in N_CLASSES:
                        too_slow = False
                        for n in SIZES:
                            row = dict(backend=backend, n_estimators=n_est if n_est is not None else "default",
                                       n_train=n, n_features=m, n_classes=c, k_blocks=K_BLOCKS, device=dev,
                                       torch=torch_v, tabpfn=pkg_version("tabpfn"), tabicl=pkg_version("tabicl"),
                                       python=platform.python_version())
                            if too_slow:
                                row["status"] = "skipped (previous size too slow)"
                            else:
                                try:
                                    row.update(bench_one(backend, n, m, c, n_est, rng))
                                    row["status"] = "ok"
                                    too_slow = row["block_median_s"] > SKIP_ABOVE_SECONDS
                                except Exception as e:                     # noqa: BLE001
                                    row["status"] = f"error: {type(e).__name__}: {str(e)[:150]}"
                            w.writerow(row)
                            fh.flush()
                            print(row["backend"], row["n_estimators"], n, m, c, row.get("block_median_s"),
                                  row.get("final_median_s"), row["status"], flush=True)
    print("đã lưu", args.out)


if __name__ == "__main__":
    main()
