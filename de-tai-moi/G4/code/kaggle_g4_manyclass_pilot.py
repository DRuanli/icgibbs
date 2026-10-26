"""
kaggle_g4_manyclass_pilot.py — thí điểm PLL NHIỀU LỚP (> 10 lớp) trên dataset PHÁT TRIỂN, chuẩn bị RQ5.
=======================================================================================================

Mục đích (G5 §1.4, việc F4): các bộ PLL thật (Lost 16 lớp, MSRCv2 23, BirdSong 13) vượt giới hạn 10 lớp của
TabPFN v2. Trước khi đóng băng G5, kiểm tra trên tập PHÁT TRIỂN (không chạm dữ liệu PLL thật):
  1. In-Context Gibbs có chạy được với > 10 lớp không, tốn bao lâu mỗi ô (để lập ngân sách G6);
  2. backend nào dùng cho RQ5: TabPFN v2 bọc ECOC (ManyClassClassifier của tabpfn-extensions) hay TabICL v2
     (hỗ trợ nhiều lớp sẵn, phân loại phân cấp);
  3. lợi ích của Gibbs so với naive và PRODEN ở RQ4-PLL (≤ 10 lớp) có còn khi nhiều lớp và ít ứng viên không.
Exploratory, không phải kết quả chính.

Dataset phát triển (OpenML-CC18, > 10 lớp, ngoài mọi danh sách đánh giá của G5 vì G5 chỉ lấy 3–10 lớp):
  vowel (307, 11 lớp), letter (6, 26 lớp, lấy mẫu con phân tầng 2.000 hàng); tuỳ chọn texture (40499, 11 lớp).
PLL: mỗi nhãn sai vào tập ứng viên độc lập với xác suất q = r/(C−1), r ∈ {1, 2}
     → trung bình 2 hoặc 3 ứng viên, đúng khoảng của dữ liệu thật (Lost 2,23; BirdSong 2,18; MSRCv2 3,16).
Backend:
  tabpfn_v2_ecoc  ManyClassClassifier(TabPFN v2, alphabet_size=10, n_estimators_redundancy=4) — 8 mô hình con
  tabicl_v2       TabICLClassifier(support_many_classes=True); checkpoint nạp một lần rồi dùng lại
Phương pháp (cấu hình mặc định sau G4, giống RQ4): naive, oracle, gibbs_warm, sem_warm, hard_em_warm theo
từng backend; proden (MLP, không phụ thuộc backend) một lần mỗi ô.

Quy tắc chọn backend cho RQ5 (viết trước khi chạy):
  (i) loại backend nào có thời gian trung bình một ô gibbs_warm > 30 phút trên letter;
  (ii) trong số còn lại chọn backend có accuracy ORACLE trung bình cao hơn (đo chất lượng TFM gốc, trung lập với
       phương pháp); chênh < 0,5 điểm % thì chọn backend rẻ hơn.
Quy tắc chia dữ liệu cho RQ5-PLL thật: nếu tổng chi phí ước tính cho 3 bộ × 10 lần chia 90/10 (như văn liệu PLL)
  ≤ 6 giờ T4 thì dùng 10 × 90/10; nếu không thì 5 × 80/20.

Chạy: xem G4/README.md. Kiểm tra không cần GPU: python kaggle_g4_manyclass_pilot.py --dry-run --out dry_mc.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from icgibbs import InContextGibbs, PartialLabel, SklearnTFMConditional  # noqa: E402
from icgibbs.baselines import PRODEN, hard_em_tfm, warm_start_cross  # noqa: E402
from kaggle_g4_rq4_pilot import metrics  # noqa: E402

DEV_MANY = {307: "vowel", 6: "letter", 40499: "texture"}
DEFAULT_IDS = "307,6"
FIELDS = ["backend", "dataset_id", "dataset", "r", "q", "seed", "method", "acc", "logloss", "auc",
          "n_train", "n_test", "d", "n_classes", "n_tfm_calls", "seconds", "sec_per_call", "extra", "status"]

_TABICL_CACHE: dict = {}


def make_many(backend: str, n_est: int | None):
    """Estimator mới cho mỗi lần gọi (giống SklearnTFMConditional ở G4)."""
    if backend == "dry":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=500)
    if backend == "dry_ecoc":       # kiểm tra đường ECOC trên CPU: hồi quy logistic, bảng chữ cái 4 ký hiệu
        from sklearn.linear_model import LogisticRegression
        from tabpfn_extensions.many_class import ManyClassClassifier
        return ManyClassClassifier(estimator=LogisticRegression(max_iter=500), alphabet_size=4,
                                   n_estimators_redundancy=4, random_state=0)
    if backend == "tabpfn_v2_ecoc":
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        from tabpfn_extensions.many_class import ManyClassClassifier
        kw = {"ignore_pretraining_limits": True}
        if n_est is not None:
            kw["n_estimators"] = n_est
        base = TabPFNClassifier.create_default_for_version(ModelVersion.V2, **kw)
        return ManyClassClassifier(estimator=base, alphabet_size=10, n_estimators_redundancy=4, random_state=0)
    if backend == "tabicl_v2":
        from tabicl import TabICLClassifier

        class CachedTabICL(TabICLClassifier):
            """TabICLClassifier.fit nạp lại checkpoint từ đĩa mỗi lần; Gibbs gọi fit hàng trăm lần nên ta giữ
            mô hình (ở chế độ eval, trọng số không đổi) trong bộ nhớ và dùng lại."""

            def _load_model(self):
                key = (self.checkpoint_version, str(self.model_path))
                if key not in _TABICL_CACHE:
                    super()._load_model()
                    _TABICL_CACHE[key] = (self.model_, self.model_config_, self.model_path_)
                self.model_, self.model_config_, self.model_path_ = _TABICL_CACHE[key]

        kw = {"support_many_classes": True, "random_state": 0}
        if n_est is not None:
            kw["n_estimators"] = n_est
        return CachedTabICL(**kw)
    raise ValueError(backend)


def load_many(did, dry, max_rows):
    if dry:
        from sklearn.datasets import make_classification
        X, y = make_classification(900, 10, n_informative=8, n_classes=12, n_clusters_per_class=1,
                                   class_sep=1.5, random_state=did)
        X = X.astype(np.float32)
        return X, X, y, f"synthetic-many-{did}"
    import pandas as pd
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    b = fetch_openml(data_id=did, as_frame=True, parser="auto")
    df, target = b.data.copy(), b.target
    keep = ~pd.isna(target)
    df, target = df[keep].reset_index(drop=True), target[keep].reset_index(drop=True)
    y = pd.Categorical(target.astype(str)).codes.astype(int)
    if len(y) > max_rows:   # lấy mẫu con phân tầng, cố định cho mọi seed
        idx, _ = train_test_split(np.arange(len(y)), train_size=max_rows, stratify=y, random_state=12345)
        idx = np.sort(idx)
        df, y = df.iloc[idx].reset_index(drop=True), y[idx]
    num = df.select_dtypes(include=[np.number])
    cat = df.drop(columns=num.columns)
    num = num.fillna(num.median())
    X_tfm, X_mlp = [num.values.astype(np.float32)], [num.values.astype(np.float32)]
    if cat.shape[1]:
        codes = np.column_stack([pd.Categorical(cat[c]).codes for c in cat.columns]).astype(np.float32)
        codes[codes < 0] = np.nan
        X_tfm.append(codes)
        X_mlp.append(pd.get_dummies(cat.astype(str)).values.astype(np.float32))
    return np.hstack(X_tfm), np.hstack(X_mlp), y, b.details.get("name", str(did))


def done_keys(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {(r["backend"], r["dataset_id"], r["r"], r["seed"], r["method"])
                for r in csv.DictReader(f) if r["status"] == "ok"}


def run_cell(args, did, r, seed, data, backends, writer, fh, done):
    from sklearn.model_selection import train_test_split
    X_tfm, X_mlp, y, name = data
    C = int(y.max()) + 1
    q = r / (C - 1)
    rng = np.random.default_rng(1000 * seed + did + 11)
    itr, ite = train_test_split(np.arange(len(y)), test_size=args.test_frac, stratify=y, random_state=seed)
    Xtr, Xte, ytr, yte, Mtr, Mte = X_tfm[itr], X_tfm[ite], y[itr], y[ite], X_mlp[itr], X_mlp[ite]
    n = len(ytr)
    S = rng.uniform(size=(n, C)) < q
    S[np.arange(n), ytr] = True
    pseudo = np.array([rng.choice(np.flatnonzero(s)) for s in S])
    extra = f"avg_candidates={S.sum(1).mean():.2f}"
    base = dict(dataset_id=did, dataset=name, r=r, q=round(q, 4), seed=seed, n_train=n, n_test=len(yte),
                d=X_tfm.shape[1], n_classes=C)

    def emit(backend, method, P, t0, calls, more=""):
        P = np.asarray(P, float)
        if not np.all(np.isfinite(P)):
            raise ValueError(f"{method}: xác suất không hữu hạn")
        sec = time.time() - t0
        row = dict(base, backend=backend, method=method, seconds=round(sec, 2), n_tfm_calls=calls,
                   sec_per_call=round(sec / calls, 3) if calls else "", extra=extra + (";" + more if more else ""),
                   status="ok")
        row.update({k: (round(v, 5) if v == v else v) for k, v in metrics(yte, P).items()})
        writer.writerow(row)
        fh.flush()
        done.add((backend, str(did), str(r), str(seed), method))
        if args.preds:
            with open(args.preds, "a") as pf:
                key = "|".join([backend, str(did), str(r), str(seed), method])
                pf.write(json.dumps({"key": key, "P": np.round(P, 4).tolist(), "y": yte.tolist()}) + "\n")
        print(f"  {name:>10s} r={r} s{seed} {backend:15s} {method:13s} ACC={row['acc']:.4f} "
              f"LL={row['logloss']:.4f} {sec:7.1f}s ({calls} lần gọi)", flush=True)

    def attempt(backend, method, fn):
        if (backend, str(did), str(r), str(seed), method) in done:
            return
        try:
            fn()
        except Exception as e:
            print(f"[lỗi] {did} r={r} s{seed} {backend} {method}: {e!r}", flush=True)
            writer.writerow(dict(base, backend=backend, method=method, status=f"error: {e!r}"[:200]))
            fh.flush()

    def _proden():
        t0 = time.time()
        emit("mlp", "proden", PRODEN(epochs=args.epochs, seed=seed).fit(Mtr, S).predict_proba(Mte), t0, 0)

    attempt("mlp", "proden", _proden)
    acc_mon = lambda yy: float(np.mean(yy[:n] == ytr))
    for be in backends:
        todo = lambda m: (be, str(did), str(r), str(seed), m) not in done
        cond = SklearnTFMConditional(lambda: make_many(be, args.gibbs_est), C)
        fin = SklearnTFMConditional(lambda: make_many(be, args.final_est), C)
        attempt(be, "naive", lambda: (lambda t0: emit(be, "naive", fin.predict_proba(Xtr, pseudo, Xte), t0, 1))(
            time.time()))
        attempt(be, "oracle", lambda: (lambda t0: emit(be, "oracle", fin.predict_proba(Xtr, ytr, Xte), t0, 1))(
            time.time()))
        if not (todo("gibbs_warm") or todo("sem_warm") or todo("hard_em_warm")):
            continue
        try:
            t0w = time.time()
            P0 = warm_start_cross(cond, Xtr, pseudo, fit_mask=None, n_folds=5, seed=seed)
            t_warm = time.time() - t0w
        except Exception as e:
            print(f"[lỗi] {did} r={r} s{seed} {be} warm start: {e!r}", flush=True)
            writer.writerow(dict(base, backend=be, method="warm_start", status=f"error: {e!r}"[:200]))
            fh.flush()
            continue
        init_acc = float(np.mean(P0.argmax(1) == ytr))

        def _gibbs():
            t0 = time.time()
            cond.n_calls = 0
            g = InContextGibbs(cond, PartialLabel(S), n_blocks=args.blocks, n_sweeps=args.sweeps,
                               burn_in=args.burn_in, max_context=args.max_context, seed=seed, init=P0,
                               monitor=acc_mon).run(Xtr)
            m = g.trace_.monitor
            info = (f"init_acc={init_acc:.3f};acc_trace={m[0]:.3f}/{m[args.burn_in - 1]:.3f}/{m[-1]:.3f};"
                    f"flips_last={g.trace_.flips[-1]:.3f};warm_s={t_warm:.1f}")
            calls = cond.n_calls + 5
            if todo("gibbs_warm"):
                emit(be, "gibbs_warm", g.predict_proba(Xtr, Xte, cond=fin, n_use=args.n_use), t0 - t_warm,
                     calls + args.n_use, info)
            if todo("sem_warm"):
                emit(be, "sem_warm", g.predict_proba(Xtr, Xte, cond=fin, last_only=True), t0 - t_warm, calls + 1,
                     info)

        def _hard_em():
            t0 = time.time()
            cond.n_calls = 0
            weak = PartialLabel(S)
            init_y = weak.init_labels(np.random.default_rng(seed), P0=P0)
            P, _ = hard_em_tfm(cond, weak, Xtr, Xte, n_iter=args.em_iters, n_blocks=args.blocks, seed=seed,
                               init=init_y, cond_final=fin)
            emit(be, "hard_em_warm", P, t0, cond.n_calls + 6, f"init_acc={init_acc:.3f}")

        if todo("gibbs_warm") or todo("sem_warm"):
            attempt(be, "gibbs_warm" if todo("gibbs_warm") else "sem_warm", _gibbs)
        attempt(be, "hard_em_warm", _hard_em)


def summarize(path):
    import pandas as pd
    df = pd.read_csv(path)
    df = df[df.status == "ok"]
    if df.empty:
        return
    print("\n================ TÓM TẮT PLL NHIỀU LỚP (accuracy, trung bình theo seed) ================")
    t = df.groupby(["dataset", "r", "backend", "method"]).acc.mean().unstack("method")
    print(t.round(4).to_string())
    print("\n--- thời gian trung bình mỗi ô (giây) ---")
    print(df.groupby(["dataset", "backend", "method"]).seconds.mean().unstack("method").round(0).to_string())
    o = df[df.method == "oracle"].groupby("backend").acc.mean()
    print(f"\naccuracy oracle trung bình theo backend: {o.round(4).to_dict()}")
    g = df[df.method == "gibbs_warm"].groupby(["dataset", "backend"]).seconds.mean()
    print(f"thời gian gibbs_warm trung bình mỗi ô (phút): {(g / 60).round(1).to_dict()}")
    if set(df.backend) <= {"dry", "dry_ecoc", "mlp"}:
        print("\n(dry-run: kết quả không có ý nghĩa)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="tabpfn_v2_ecoc,tabicl_v2")
    ap.add_argument("--datasets", default=DEFAULT_IDS)
    ap.add_argument("--rs", default="1,2")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--sweeps", type=int, default=30)
    ap.add_argument("--burn_in", type=int, default=10)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--n_use", type=int, default=10)
    ap.add_argument("--em_iters", type=int, default=10)
    ap.add_argument("--gibbs_est", type=int, default=1)
    ap.add_argument("--final_est", type=lambda v: None if v == "default" else int(v), default=2)
    ap.add_argument("--max_context", type=int, default=2000)
    ap.add_argument("--max_rows", type=int, default=2000)
    ap.add_argument("--test_frac", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--preds", default="auto")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="g4_manyclass_pilot.csv")
    args = ap.parse_args()
    backends = args.backends.split(",")
    if args.dry_run:
        args.sweeps, args.burn_in, args.n_use, args.em_iters, args.epochs = 6, 2, 3, 3, 30
        backends = [b for b in ("dry", "dry_ecoc")]
    if args.preds == "auto":
        args.preds = os.path.splitext(args.out)[0] + "_preds.jsonl"
    done = done_keys(args.out)
    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        ids = [int(v) for v in args.datasets.split(",")]
        for did in ids[:1] if args.dry_run else ids:
            try:
                data = load_many(did, args.dry_run, args.max_rows)
            except Exception as e:
                print(f"[lỗi] dataset {did}: {e!r}", flush=True)
                continue
            if not args.dry_run and DEV_MANY.get(did, "?") not in data[3]:
                print(f"[cảnh báo] dataset {did}: tên '{data[3]}' khác dự kiến '{DEV_MANY.get(did)}'", flush=True)
            print(f"== {data[3]}: n={len(data[2])}, C={int(data[2].max()) + 1}, d={data[0].shape[1]}", flush=True)
            for r in [int(v) for v in args.rs.split(",")]:
                for seed in [int(v) for v in args.seeds.split(",")]:
                    run_cell(args, did, r, seed, data, backends, w, fh, done)
    summarize(args.out)


if __name__ == "__main__":
    main()
