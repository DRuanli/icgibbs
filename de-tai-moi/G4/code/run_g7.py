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

from icgibbs.g7.data import (Preprocessor, Table, canonical_table, encode_tfm, load_openml,  # noqa: E402
                             load_real_pll, table_sha256)
from icgibbs.g7.io import ResultWriter, code_version, config_hash, weak_hash  # noqa: E402
from icgibbs.g7.methods import (ABLATION_VARIANTS, GIBBS, TFM_METHODS, Cell, baselines_for,  # noqa: E402
                                run_ablation, run_tfm_methods)
from icgibbs.g7.metrics import binary_metrics, multiclass_metrics  # noqa: E402
from icgibbs.g7.registry import (ABL_LARGE_DEV, ABL_LARGE_EVAL, ABL_SMALL_DEV, ABL_SMALL_EVAL,  # noqa: E402
                                 DEV_BINARY, DEV_MULTICLASS, EVAL_BINARY, EVAL_MULTICLASS, REAL_PLL, RHAT_DEV,
                                 RHAT_EVAL)
from icgibbs.g7.splits import nested_subsample, random_split, split_sha256, stratified_split  # noqa: E402
from icgibbs.g7.weakgen import make_weak, real_pll_weak, weak_rng  # noqa: E402

LEVELS = {"llp": [8, 16, 32], "pu_sar": [0.1, 0.3], "pu_scar": [0.1, 0.3], "ssl": [0.05, 0.2], "noisy": [0.2, 0.4],
          "pll": [0.3, 0.6]}
BINARY_SETTINGS = ("llp", "pu_sar", "pu_scar", "ssl", "noisy")
# Ablation (giao thức Mục 7; các điểm chưa nói rõ được định ở G7_RUNNER_DESIGN §4, giao thức §11 deviation 11).
# datasets: "all" = mọi dataset của nhóm; "small3" = 3 dataset nhỏ nhất mỗi nhóm (cùng danh sách với R̂)
ABLATIONS = {
    "init": dict(levels={"llp": [8], "pll": [0.3, 0.6]}, datasets="all", variants=["gibbs_cold"]),
    "blocks": dict(levels={"llp": [8], "pll": [0.3]}, datasets="small3", variants=["gibbs_warm_K5", "gibbs_warm_K20"]),
    "sweeps": dict(levels={"llp": [8], "pll": [0.3]}, datasets="small3", variants=["gibbs_warm_S15", "gibbs_warm_S60"]),
    "trans": dict(levels={"llp": [8]}, datasets="all", variants=["gibbs_warm_trans"]),
    "mh": dict(levels={"llp": [8], "pll": [0.3]}, datasets="small3", variants=["gibbs_warm_mh"]),
    "prior": dict(levels={"pu_sar": [0.1, 0.3]}, datasets="all", variants=["gibbs_warm_prior1", "gibbs_warm_prior4"]),
}
# Ablation cỡ dữ liệu (--n_train_max): LLP túi 8 trên 3 dataset nhị phân lớn nhất, PLL q = 0,3 trên 3 dataset đa lớp lớn
# nhất; cỡ 500, 1.000, 2.000, 4.000; cỡ ≥ toàn bộ train thì bỏ qua ô (dùng kết quả phần chính)
SIZE_LEVELS = {"llp": [8], "pll": [0.3]}
assert all(v in ABLATION_VARIANTS for r in ABLATIONS.values() for v in r["variants"])
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
    if backend == "tabicl_v2":                              # đúng cấu hình đã chọn ở F6 (giao thức §1.4)
        from tabicl import TabICLClassifier

        class CachedTabICL(TabICLClassifier):
            """TabICLClassifier.fit nạp lại checkpoint từ đĩa mỗi lần; Gibbs gọi fit hàng trăm lần nên giữ mô hình
            (chế độ eval, trọng số không đổi) trong bộ nhớ và dùng lại — như kaggle_g4_manyclass_pilot.py."""

            def _load_model(self):
                key = (self.checkpoint_version, str(self.model_path))
                if key not in _TABICL_CACHE:
                    super()._load_model()
                    _TABICL_CACHE[key] = (self.model_, self.model_config_, self.model_path_)
                self.model_, self.model_config_, self.model_path_ = _TABICL_CACHE[key]

        kw = {"support_many_classes": True, "random_state": 0}
        if n_estimators is not None:
            kw["n_estimators"] = n_estimators
        return CachedTabICL(**kw)
    raise ValueError(backend)


_TABICL_CACHE: dict = {}


def synthetic_table(did: int, binary: bool, shape=None) -> Table:
    """Dữ liệu tổng hợp cho kiểm thử (không cần mạng). shape = (n, d, C): cỡ tuỳ chọn, để đo thời gian và bộ nhớ ở cỡ
    của tập đánh giá mà không chạm vào tập đánh giá (G6 lượt 6)."""
    import pandas as pd
    from sklearn.datasets import make_classification
    n, d, C = shape if shape else (500 + 50 * (did % 4), 6, 2 if binary else 4)
    C = 2 if binary else C
    inf, red = (min(d, max(4, d // 2)), 0) if shape else (4, 2)          # không shape: đúng như bản cũ (test)
    X, y = make_classification(n, d, n_informative=inf, n_redundant=red, n_classes=C, n_clusters_per_class=1,
                               weights=[0.65, 0.35] if binary else None, random_state=did, flip_y=0.02)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(d)])
    df["cat"] = pd.Categorical(np.where(X[:, 0] > 0, "a", "b"))
    labels = np.array([f"c{k:02d}" for k in range(C)]) if shape else np.array(list("ABCD"))
    return canonical_table(df, pd.Series(labels[y]), did=did, name=f"synthetic-{did}" + (f"-{n}x{d}" if shape else ""),
                           binary=binary)


def parse_shapes(spec: str):
    """"31=4210x57;54=6430x36x6" → {31: (4210, 57, 2), 54: (6430, 36, 6)}"""
    out = {}
    for part in filter(None, spec.split(";")):
        k, v = part.split("=")
        dims = [int(x) for x in v.lower().split("x")]
        out[int(k)] = (dims[0], dims[1], dims[2] if len(dims) > 2 else 2)
    return out


def parse_levels(spec: str):
    out = dict(LEVELS)
    for part in filter(None, spec.split(";")):
        k, v = part.split("=")
        out[k.strip()] = [float(x) if "." in x else int(x) for x in v.split(",")]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["tfm", "baselines", "rhat", "ablation"], required=True)
    ap.add_argument("--ablation", choices=sorted(ABLATIONS), default="", help="với --part ablation: một ablation")
    ap.add_argument("--n_train_max", type=int, default=0,
                    help="ablation cỡ dữ liệu: lấy mẫu con lồng nhau, phân tầng của train tới cỡ này (0 = không)")
    ap.add_argument("--split", choices=["dev", "eval"], default="dev")
    ap.add_argument("--eval-ok", action="store_true", help="xác nhận chạy trên tập đánh giá (chỉ từ G7)")
    ap.add_argument("--source", choices=["openml", "synthetic"], default="openml")
    ap.add_argument("--synthetic_shape", default="", help='chỉ với --source synthetic: "ID=nxd[xC];..." (đo thời gian)')
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
    ap.add_argument("--mh-check", nargs="?", const="llp", default="",
                    help="thêm gibbs_warm_mh (khối có sửa MH, PROOFS G4-5) cho các bối cảnh liệt kê; cờ trần = chỉ LLP. "
                         "Chậm khoảng 10 lần chuỗi thường (G6 lượt 5)")
    ap.add_argument("--rq5_dir", default="", help="thư mục chứa Lost.npz, MSRCv2.npz, BirdSong.npz (việc F7)")
    ap.add_argument("--rq5_seeds", default="0,1,2,3,4,5,6,7,8,9", help="PLL thật: 10 lần chia 90/10 (giao thức §1.4)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--preds", default="auto")
    a = ap.parse_args(argv)
    a.mh_settings = {x.strip() for x in a.mh_check.split(",") if x.strip()}
    if a.backend == "tabicl_v2" and a.final_est is None:   # giao thức §1.4: TabICL 1 trong chuỗi, 2 khi dự đoán cuối
        a.final_est = 2
    shapes = parse_shapes(a.synthetic_shape)
    if shapes and a.source != "synthetic":
        ap.error("--synthetic_shape chỉ dùng với --source synthetic")
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
    if a.part == "ablation":                               # thiết lập theo quy tắc ghi trước của từng ablation
        if not a.ablation:
            ap.error("--part ablation cần --ablation")
        rule = ABLATIONS[a.ablation]
        settings[:] = list(rule["levels"])
        levels.update(rule["levels"])
        if rule["datasets"] == "small3" and a.datasets == "all":
            rs = ABL_SMALL_EVAL if a.split == "eval" else ABL_SMALL_DEV
            a.datasets = ",".join(str(x) for x in rs["binary"] + rs["multiclass"])
    if a.n_train_max:
        if a.part not in ("tfm", "baselines"):
            ap.error("--n_train_max chỉ dùng với --part tfm hoặc baselines")
        settings[:] = [s for s in settings if s in SIZE_LEVELS] or list(SIZE_LEVELS)
        levels.update(SIZE_LEVELS)
        if a.datasets == "all":
            rs = ABL_LARGE_EVAL if a.split == "eval" else ABL_LARGE_DEV
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

    if "pll_real" in settings:                               # RQ5: PLL thật (không phụ thuộc --split)
        if not a.rq5_dir:
            ap.error("pll_real cần --rq5_dir")
        if a.split != "eval" or not a.eval_ok:
            ap.error("PLL thật thuộc tập đánh giá (RQ5): cần --split eval --eval-ok")
        rseeds = [int(x) for x in a.rq5_seeds.split(",")]
        for name, (code_id, sha) in REAL_PLL.items():
            path = os.path.join(a.rq5_dir, f"{name}.npz")
            if not os.path.exists(path):
                print(f"[thiếu] {path}", flush=True)
                continue
            t, S = load_real_pll(path, name, sha)
            t.did = code_id
            X_all = t.X.to_numpy(np.float64)
            sha_t = table_sha256(X_all, t.y)
            print(f"\n=== RQ5 {name}: n={len(t.y)}, d={X_all.shape[1]}, Q={S.shape[1]}", flush=True)
            for seed in rseeds:
                itr, ite = random_split(len(t.y), seed, 0.1)
                wd = real_pll_weak(S[itr], seed, code_id)
                run_cell(a, writer, t, X_all, sha_t, "pll_real", 0.0, seed, only, code, cfg, split=(itr, ite), weak=wd)
        settings = [x for x in settings if x != "pll_real"]
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
            t = synthetic_table(did, binary, shapes.get(did)) if a.source == "synthetic" else load_openml(did, binary=binary)
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


def run_cell(a, writer, t, X_all, sha_t, setting, level, seed, only, code, cfg, split=None, weak=None):
    from icgibbs.conditional import SklearnTFMConditional
    did = t.did
    C = weak.n_classes if weak is not None else t.n_classes
    itr, ite = split if split is not None else stratified_split(t.y, seed)
    if getattr(a, "n_train_max", 0):                      # ablation cỡ dữ liệu: test không đổi
        if a.n_train_max >= len(itr):
            print(f"  [bỏ qua] {t.name} {setting} s{seed}: n_train {len(itr)} ≤ {a.n_train_max} — dùng kết quả phần chính",
                  flush=True)
            return
        itr = nested_subsample(itr, t.y, a.n_train_max, seed, did)
    prep = Preprocessor().fit(t.X.iloc[itr])
    Mtr, Mte = prep.transform(t.X.iloc[itr]), prep.transform(t.X.iloc[ite])
    Xtr, Xte, ytr, yte = X_all[itr], X_all[ite], t.y[itr], t.y[ite]
    wd = weak if weak is not None else make_weak(setting, level, ytr, Xtr, weak_rng(seed, did, setting), n_classes=C)
    cell = Cell(did, t.name, setting, level, seed, C, Xtr, Xte, Mtr, Mte, wd, prior=float(ytr.mean()) if C == 2 else None)
    base = dict(backend=a.backend if a.part in ("tfm", "rhat", "ablation") else "cpu", dataset_id=did, dataset=t.name, setting=setting,
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
                fin = SklearnTFMConditional(lambda: make_estimator(a.backend, a.final_est), C)
                st, Ps = run_rhat(cell, cond, fin)
                # metric chính trên test của từng chuỗi: sai khác do riêng ngẫu nhiên Monte Carlo (cùng dữ liệu, cùng
                # nhãn yếu, cùng điểm khởi động); chuỗi a cùng seed với gibbs_warm của phần tfm
                for nm, P in zip("ab", Ps):
                    met = binary_metrics(yte, P[:, 1], prior=float(ytr.mean())) if C == 2 else multiclass_metrics(yte, P)
                    st[f"metric_chain_{nm}"] = float(met.get("auc", met.get("acc")))
                st["metric_absdiff"] = abs(st["metric_chain_a"] - st["metric_chain_b"])
                writer.write(dict(base, method="rhat", seconds=round(time.time() - t0, 2), status="ok",
                                  extra=";".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in st.items())))
                print(f"  {t.name[:22]:>22s} {setting:7s} {str(level):>4s} s{seed} rhat {st}", flush=True)
            attempt("rhat", _rh)
        return
    if a.part == "ablation":
        cond = SklearnTFMConditional(lambda: make_estimator(a.backend, a.gibbs_est), C)
        fin = SklearnTFMConditional(lambda: make_estimator(a.backend, a.final_est), C)
        run_ablation(cell, cond, fin, ABLATIONS[a.ablation]["variants"], todo, emit, guard=attempt)
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
        mh = setting in a.mh_settings
        if any(todo(m) for m in TFM_METHODS[setting]) or (mh and todo("gibbs_warm_mh")):
            run_tfm_methods(cell, cond, fin, todo, emit, monitor=monitor, guard=attempt, mh_check=mh)
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
