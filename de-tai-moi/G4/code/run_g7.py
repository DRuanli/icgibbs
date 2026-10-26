"""
run_g7.py — bộ chạy thực nghiệm chính (G6 việc T6; thiết kế: G6/G7_RUNNER_DESIGN.md).

Mỗi ô (dataset × bối cảnh × mức × seed) độc lập; mỗi phương pháp một dòng CSV; chạy lại thì bỏ qua dòng đã `ok`.
Hai phần: `--part tfm` (TFM ngây thơ, oracle, Gibbs ấm, SEM, hard-EM, propensity đã biết, CL + TFM — cần GPU) và
`--part baselines` (mọi baseline peer-review, bản mặc định và bản tune — chạy CPU được).

Mặc định chạy trên TẬP PHÁT TRIỂN. Tập đánh giá chỉ chạy khi có cả `--split eval` lẫn `--eval-ok` (mở ở G7).

Ví dụ:
  python run_g7.py --part baselines --split dev --settings llp --shard 0/4 --out g7dev_llp_0.csv
  python run_g7.py --part tfm --backend dry --source synthetic --settings llp,pll --seeds 0 --out /tmp/t.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from icgibbs.g7.data import Preprocessor, Table, canonical_table, encode_tfm, load_openml, table_sha256  # noqa: E402
from icgibbs.g7.io import ResultWriter, code_version, config_hash, weak_hash  # noqa: E402
from icgibbs.g7.methods import GIBBS, TFM_METHODS, Cell, baselines_for, run_tfm_methods  # noqa: E402
from icgibbs.g7.metrics import binary_metrics, multiclass_metrics  # noqa: E402
from icgibbs.g7.registry import (DEV_BINARY, DEV_MULTICLASS, EVAL_BINARY, EVAL_MULTICLASS, RHAT_DEV,  # noqa: E402
                                 RHAT_EVAL)
from icgibbs.g7.splits import split_sha256, stratified_split  # noqa: E402
from icgibbs.g7.weakgen import make_weak, weak_rng  # noqa: E402

LEVELS = {"llp": [8, 16, 32], "pu_sar": [0.1, 0.3], "pu_scar": [0.1, 0.3], "ssl": [0.05, 0.2], "noisy": [0.2, 0.4],
          "pll": [0.3, 0.6]}
BINARY_SETTINGS = ("llp", "pu_sar", "pu_scar", "ssl", "noisy")
METRIC_FIELDS = ["auc", "pr_auc", "logloss", "brier", "acc", "ece", "logloss_pm", "brier_pm", "auc_ovr"]
FIELDS = ["backend", "dataset_id", "dataset", "setting", "level", "seed", "method", "part"] + METRIC_FIELDS + \
         ["n_train", "n_test", "d", "n_classes", "pos_rate", "n_duplicates_removed", "n_tfm_calls", "seconds",
          "peak_gpu_mb", "sha256_table", "sha256_split", "weak_hash", "code", "config", "extra", "status"]


def make_estimator(backend: str, n_estimators):
    if backend == "dry":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=300)
    if backend.startswith("tabpfn"):
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        version = {"tabpfn_v2": ModelVersion.V2, "tabpfn_v2_5": ModelVersion.V2_5}[backend]
        kw = {"ignore_pretraining_limits": True}
        if n_estimators is not None:
            kw["n_estimators"] = n_estimators
        return TabPFNClassifier.create_default_for_version(version, **kw)
    if backend == "tabicl_v2":
        from tabicl import TabICLClassifier
        return TabICLClassifier(**({} if n_estimators is None else {"n_estimators": n_estimators}))
    raise ValueError(backend)


def synthetic_table(did: int, binary: bool) -> Table:
    """Dữ liệu tổng hợp nhỏ cho kiểm thử (không cần mạng)."""
    import pandas as pd
    from sklearn.datasets import make_classification
    C = 2 if binary else 4
    X, y = make_classification(500 + 50 * (did % 4), 6, n_informative=4, n_classes=C, n_clusters_per_class=1,
                               weights=[0.65, 0.35] if binary else None, random_state=did, flip_y=0.02)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["cat"] = pd.Categorical(np.where(X[:, 0] > 0, "a", "b"))
    return canonical_table(df, pd.Series(np.array(list("ABCD"))[y]), did=did, name=f"synthetic-{did}",
                           binary=binary)


def parse_levels(spec: str):
    out = dict(LEVELS)
    for part in filter(None, spec.split(";")):
        k, v = part.split("=")
        out[k.strip()] = [float(x) if "." in x else int(x) for x in v.split(",")]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["tfm", "baselines", "rhat"], required=True)
    ap.add_argument("--split", choices=["dev", "eval"], default="dev")
    ap.add_argument("--eval-ok", action="store_true", help="xác nhận chạy trên tập đánh giá (chỉ từ G7)")
    ap.add_argument("--source", choices=["openml", "synthetic"], default="openml")
    ap.add_argument("--datasets", default="all", help="'all' hoặc danh sách ID OpenML")
    ap.add_argument("--settings", default="llp,pu_sar,pu_scar,ssl,noisy,pll")
    ap.add_argument("--levels", default="", help='ví dụ "llp=8,16;pu_sar=0.1"')
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--only", default="")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--backend", default="tabpfn_v2")
    ap.add_argument("--gibbs_est", type=int, default=1)
    ap.add_argument("--final_est", type=lambda v: None if v == "default" else int(v), default=None)
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--preds", default="auto")
    a = ap.parse_args(argv)
    if a.split == "eval" and not a.eval_ok:
        ap.error("tập đánh giá chỉ chạy ở G7: cần thêm --eval-ok")
    only = {m.strip() for m in a.only.split(",") if m.strip()}
    settings = [s.strip() for s in a.settings.split(",") if s.strip()]
    levels = parse_levels(a.levels)
    seeds = [int(s) for s in a.seeds.split(",")]
    si, sn = (int(v) for v in a.shard.split("/"))
    binary_ids = dict(EVAL_BINARY if a.split == "eval" else {k: (v, None) for k, v in DEV_BINARY.items()})
    multi_ids = dict(EVAL_MULTICLASS if a.split == "eval" else {k: (v, None) for k, v in DEV_MULTICLASS.items()})
    if a.part == "rhat" and a.datasets == "all":            # quy tắc ghi trước: 3 dataset nhỏ nhất mỗi nhóm
        rs = RHAT_EVAL if a.split == "eval" else RHAT_DEV
        a.datasets = ",".join(str(x) for x in rs["binary"] + rs["multiclass"])
    if a.datasets != "all":
        want = {int(x) for x in a.datasets.split(",")}
        binary_ids = {k: v for k, v in binary_ids.items() if k in want}
        multi_ids = {k: v for k, v in multi_ids.items() if k in want}
    preds = None if a.preds == "none" else (a.out.replace(".csv", "_preds.jsonl") if a.preds == "auto" else a.preds)
    writer = ResultWriter(a.out, FIELDS, preds)
    code = code_version()
    cfg = config_hash(dict(GIBBS=GIBBS, backend=a.backend, gibbs_est=a.gibbs_est, final_est=a.final_est))
    print(f"run_g7 · part={a.part} · split={a.split} · backend={a.backend} · code={code} · config={cfg}", flush=True)

    jobs = [(did, nm, True) for did, (nm, _) in binary_ids.items() if any(s in BINARY_SETTINGS for s in settings)] + \
           [(did, nm, False) for did, (nm, _) in multi_ids.items() if "pll" in settings]
    cell_no = -1
    for did, nm, binary in jobs:
        my_settings = [s for s in settings if (s in BINARY_SETTINGS) == binary]
        n_cells = sum(len(levels[s]) for s in my_settings) * len(seeds)
        mine = [(cell_no + 1 + j) % sn == si for j in range(n_cells)]
        if not any(mine):
            cell_no += n_cells
            continue
        try:
            t = synthetic_table(did, binary) if a.source == "synthetic" else load_openml(did, binary=binary)
        except Exception as e:
            print(f"[lỗi tải] {did}: {e!r}", flush=True)
            cell_no += n_cells
            continue
        X_all = encode_tfm(t.X)
        sha_t = table_sha256(X_all, t.y)
        print(f"\n=== {did} {t.name}: n={len(t.y)} (loại trùng {t.n_duplicates_removed}), d={X_all.shape[1]}, "
              f"C={t.n_classes}, sha256={sha_t[:12]}", flush=True)
        for setting in my_settings:
            for level in levels[setting]:
                for seed in seeds:
                    cell_no += 1
                    if cell_no % sn != si:
                        continue
                    run_cell(a, writer, t, X_all, sha_t, setting, level, seed, only, code, cfg)
    writer.close()
    return 0


def run_cell(a, writer, t, X_all, sha_t, setting, level, seed, only, code, cfg):
    from icgibbs.conditional import SklearnTFMConditional
    did, C = t.did, t.n_classes
    itr, ite = stratified_split(t.y, seed)
    prep = Preprocessor().fit(t.X.iloc[itr])
    Mtr, Mte = prep.transform(t.X.iloc[itr]), prep.transform(t.X.iloc[ite])
    Xtr, Xte, ytr, yte = X_all[itr], X_all[ite], t.y[itr], t.y[ite]
    wd = make_weak(setting, level, ytr, Xtr, weak_rng(seed, did, setting), n_classes=C)
    cell = Cell(did, t.name, setting, level, seed, C, Xtr, Xte, Mtr, Mte, wd, prior=float(ytr.mean()) if C == 2 else None)
    base = dict(backend=a.backend if a.part in ("tfm", "rhat") else "cpu", dataset_id=did, dataset=t.name, setting=setting,
                level=level, seed=seed, part=a.part, n_train=len(itr), n_test=len(ite), d=X_all.shape[1], n_classes=C,
                pos_rate=round(float(ytr.mean()), 4) if C == 2 else "", n_duplicates_removed=t.n_duplicates_removed,
                sha256_table=sha_t, sha256_split=split_sha256(itr, ite), weak_hash=weak_hash(wd), code=code,
                config=cfg)

    def todo(m):
        return (not only or m in only) and not writer.is_done(**dict(base, method=m))

    def emit(m, out, seconds):
        P = np.asarray(out.P, float)
        if not np.all(np.isfinite(P)):
            raise ValueError(f"{m}: xác suất không hữu hạn")
        met = binary_metrics(yte, P[:, 1], prior=float(ytr.mean())) if C == 2 else multiclass_metrics(yte, P)
        peak = ""
        try:
            import torch
            if torch.cuda.is_available():
                peak = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        row = dict(base, method=m, n_tfm_calls=out.n_tfm_calls, seconds=round(seconds, 2), peak_gpu_mb=peak,
                   extra=";".join(x for x in (wd.extra, out.extra) if x), status="ok",
                   **{k: round(v, 5) for k, v in met.items()})
        writer.write(row, P, yte)
        main_m = met.get("auc", met.get("acc"))
        print(f"  {t.name[:22]:>22s} {setting:7s} {str(level):>4s} s{seed} {m:16s} {main_m:.4f} {seconds:.1f}s", flush=True)

    def attempt(m, fn):
        try:
            fn()
        except Exception as e:  # lỗi của một phương pháp không làm mất các phương pháp còn lại của ô
            print(f"[lỗi] {did} {setting} {level} s{seed} {m}: {e!r}", flush=True)
            writer.write(dict(base, method=m, status=f"error: {e!r}"[:200]))

    if a.part == "rhat":
        if todo("rhat"):
            def _rh():
                t0 = time.time()
                from icgibbs.g7.methods import run_rhat
                cond = SklearnTFMConditional(lambda: make_estimator(a.backend, a.gibbs_est), C)
                st = run_rhat(cell, cond)
                writer.write(dict(base, method="rhat", seconds=round(time.time() - t0, 2), status="ok",
                                  extra=";".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in st.items())))
                print(f"  {t.name[:22]:>22s} {setting:7s} {str(level):>4s} s{seed} rhat {st}", flush=True)
            attempt("rhat", _rh)
        return
    if a.part == "tfm":
        cond = SklearnTFMConditional(lambda: make_estimator(a.backend, a.gibbs_est), C)
        fin = SklearnTFMConditional(lambda: make_estimator(a.backend, a.final_est), C)
        if not a.no_oracle and todo("oracle"):            # trần tham chiếu: nhãn sạch, không phải một phương pháp
            def _oracle():
                t0 = time.time()
                from icgibbs.g7.methods import Out
                emit("oracle", Out(fin.predict_proba(Xtr, ytr, Xte), 1), time.time() - t0)
            attempt("oracle", _oracle)
        monitor = lambda yy: float(np.mean(np.asarray(yy)[:len(ytr)] == ytr))   # chẩn đoán, không vào thuật toán
        if any(todo(m) for m in TFM_METHODS[setting]):
            run_tfm_methods(cell, cond, fin, todo, emit, monitor=monitor, guard=attempt)
    else:
        for m, fn in baselines_for(setting).items():
            if todo(m):
                def _b(m=m, fn=fn):
                    t0 = time.time()
                    out = fn(cell)
                    emit(m, out, time.time() - t0)
                attempt(m, _b)


if __name__ == "__main__":
    sys.exit(main())
