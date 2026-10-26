"""
kaggle_g4_rq4_pilot.py — thí điểm RQ4 (SSL, nhãn nhiễu, nhãn một phần) trên dataset PHÁT TRIỂN.
================================================================================================

Mục đích (GATE_REVIEW_G4 §3d, R5): sau khi phạm vi RQ2 bị thu hẹp, kiểm tra sớm trên tập phát triển xem
In-Context Gibbs có triển vọng ở các loại nhãn yếu cấp mẫu của RQ4 hay không, trước khi đóng băng G5.
Exploratory, không dùng dataset đánh giá, không phải kết quả chính.

Dataset
  nhị phân (SSL, nhãn nhiễu): 6 dataset phát triển của thí điểm PU/LLP
  đa lớp (PLL): vehicle (54), segment (36), cmc (23) — từ nay xếp vào TẬP PHÁT TRIỂN, loại khỏi tập đánh giá
Bối cảnh × mức
  ssl    tỉ lệ có nhãn ∈ {0.05, 0.2} (phân tầng, ≥ 2 mẫu mỗi lớp)
  noisy  nhiễu phụ thuộc lớp: dương→âm với xác suất r, âm→dương với r/2, r ∈ {0.2, 0.4}; T đã biết
  pll    mỗi nhãn sai vào tập ứng viên độc lập với xác suất q ∈ {0.3, 0.6}
Phương pháp (cấu hình mặc định sau G4: khởi động ấm, khối trên mọi mẫu, K = 10, 30 vòng, bỏ 10)
  gibbs_warm, sem_warm, hard_em_warm   In-Context Gibbs và hai đối chứng RQ3, cùng điểm khởi động
  naive                                TFM bỏ qua mô hình quan sát (SSL: chỉ phần có nhãn; nhiễu: dùng w;
                                       PLL: một ứng viên ngẫu nhiên)
  oracle                               TFM với nhãn sạch (trần tham chiếu)
  peer-review (cài lại, chưa tune):
    ssl    label_spreading (Zhou et al., NIPS 2003), self_training (Yarowsky, ACL 1995; LR cơ sở)
    noisy  forward (Patrini et al., CVPR 2017, T đã biết), cl_tfm (Confident Learning, Northcutt et al.,
           JAIR 2021, dùng TFM — cần gói cleanlab; bỏ qua nếu không có)
    pll    proden (Lv et al., ICML 2020)
Metric chính theo RQ4 đã đóng băng: accuracy cho PLL; AUC cho SSL và nhãn nhiễu (phụ: log loss, accuracy).

Chạy: xem G4/README.md. Kiểm tra không cần GPU: python kaggle_g4_rq4_pilot.py --dry-run --out dry.csv

v5 (26/10/2026): sửa lỗi của lượt v4 — LabelSpreading (kernel kNN) trả NaN cho điểm test mà mọi láng giềng
không nhận được nhãn lan truyền; lỗi đó làm hỏng cả ô và mất luôn self_training. Nay mỗi baseline chạy trong
khối try/except riêng; hàng NaN của LabelSpreading được gán tiên nghiệm lớp và ghi số hàng vào cột extra.
Chạy lại với cùng --out để chỉ bổ sung các ô còn thiếu.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from icgibbs import SSL, InContextGibbs, NoisyLabel, PartialLabel, SklearnTFMConditional  # noqa: E402
from icgibbs.baselines import PRODEN, ForwardCorrection, hard_em_tfm, warm_start_cross  # noqa: E402
from kaggle_g4_pilot import DEV_DATASETS, load_dataset, make_estimator  # noqa: E402

DEV_MULTI = {54: "vehicle", 36: "segment", 23: "cmc"}
LEVELS = {"ssl": [0.05, 0.2], "noisy": [0.2, 0.4], "pll": [0.3, 0.6]}
FIELDS = ["backend", "dataset_id", "dataset", "setting", "level", "seed", "method", "auc", "logloss", "acc",
          "n_train", "n_test", "d", "n_classes", "n_tfm_calls", "seconds", "extra", "status"]


def load_multiclass(did, dry):
    if dry:
        from sklearn.datasets import make_classification
        X, y = make_classification(700, 8, n_informative=5, n_classes=4, n_clusters_per_class=1, random_state=did)
        return X.astype(np.float32), X.astype(np.float32), y, f"synthetic-multi-{did}"
    import pandas as pd
    from sklearn.datasets import fetch_openml
    b = fetch_openml(data_id=did, as_frame=True, parser="auto")
    df, target = b.data.copy(), b.target
    keep = ~pd.isna(target)
    df, target = df[keep], target[keep]
    y = pd.Categorical(target.astype(str)).codes.astype(int)
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


def metrics(y, P):
    from sklearn.metrics import log_loss, roc_auc_score
    P = np.clip(np.asarray(P, float), 1e-6, 1)
    P = P / P.sum(1, keepdims=True)
    C = P.shape[1]
    if C == 2:
        auc = roc_auc_score(y, P[:, 1])
    else:
        try:
            auc = roc_auc_score(y, P, multi_class="ovr", average="macro", labels=list(range(C)))
        except ValueError:
            auc = np.nan
    return dict(auc=auc, logloss=log_loss(y, P, labels=list(range(C))), acc=float(np.mean(P.argmax(1) == y)))


def done_keys(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {(r["backend"], r["dataset_id"], r["setting"], r["level"], r["seed"], r["method"])
                for r in csv.DictReader(f) if r["status"] == "ok"}


def run_cell(args, did, setting, level, seed, data, writer, fh, done):
    from sklearn.model_selection import train_test_split
    X_tfm, X_mlp, y, name = data
    C = int(y.max()) + 1
    rng = np.random.default_rng(1000 * seed + did + 7)
    itr, ite = train_test_split(np.arange(len(y)), test_size=args.test_frac, stratify=y, random_state=seed)
    if len(itr) > args.max_train:
        itr = rng.choice(itr, args.max_train, replace=False)
    Xtr, Xte, ytr, yte, Mtr, Mte = X_tfm[itr], X_tfm[ite], y[itr], y[ite], X_mlp[itr], X_mlp[ite]
    n = len(ytr)
    base = dict(backend=args.backend, dataset_id=did, dataset=name, setting=setting, level=level, seed=seed,
                n_train=n, n_test=len(yte), d=X_tfm.shape[1], n_classes=C)
    cond = SklearnTFMConditional(lambda: make_estimator(args.backend, args.gibbs_est), C)
    fin = SklearnTFMConditional(lambda: make_estimator(args.backend, args.final_est), C)

    # ---- sinh nhãn yếu ----
    if setting == "ssl":
        lab = np.zeros(n, bool)
        for k in range(C):
            idx = np.flatnonzero(ytr == k)
            m = max(2, int(round(level * len(idx))))
            lab[rng.choice(idx, min(m, len(idx)), replace=False)] = True
        y_obs = np.where(lab, ytr, -1)
        mk_weak = lambda: SSL(y_obs, C)
        pseudo, fit_mask = np.where(lab, ytr, 0), lab
        extra = f"n_labeled={int(lab.sum())}"
    elif setting == "noisy":
        r = float(level)
        T = np.array([[1 - r / 2, r / 2], [r, 1 - r]])          # T[y, w]
        w = np.array([rng.choice(2, p=T[k]) for k in ytr])
        mk_weak = lambda: NoisyLabel(w, 2, T=T)
        pseudo, fit_mask = w, None
        extra = f"noise_rate={float(np.mean(w != ytr)):.3f}"
    else:
        S = rng.uniform(size=(n, C)) < float(level)
        S[np.arange(n), ytr] = True
        mk_weak = lambda: PartialLabel(S)
        pseudo = np.array([rng.choice(np.flatnonzero(s)) for s in S])
        fit_mask = None
        extra = f"avg_candidates={S.sum(1).mean():.2f}"

    def emit(method, P, t0, calls, more=""):
        P = np.asarray(P, float)
        if P.ndim == 1:
            P = np.column_stack([1 - P, P])
        if not np.all(np.isfinite(P)):
            raise ValueError(f"{method}: {int((~np.isfinite(P)).any(1).sum())} hàng xác suất không hữu hạn")
        key = (args.backend, str(did), setting, str(level), str(seed), method)
        row = dict(base, method=method, seconds=round(time.time() - t0, 2), n_tfm_calls=calls,
                   extra=extra + (";" + more if more else ""), status="ok")
        row.update({k: (round(v, 5) if v == v else v) for k, v in metrics(yte, P).items()})
        writer.writerow(row)
        fh.flush()
        done.add(key)
        if args.preds:
            with open(args.preds, "a") as pf:
                pf.write(json.dumps({"key": "|".join(key), "P": np.round(P, 4).tolist(), "y": yte.tolist()}) + "\n")
        print(f"  {name:>24s} {setting:5s} {str(level):>4s} s{seed} {method:14s} AUC={row['auc']} "
              f"ACC={row['acc']:.4f} LL={row['logloss']:.4f} {row['seconds']:.1f}s", flush=True)

    todo = lambda m: (args.backend, str(did), setting, str(level), str(seed), m) not in done

    def attempt(method, fn):
        """Chạy một baseline; lỗi của baseline này KHÔNG làm mất các baseline còn lại của ô (sửa lỗi lượt v4)."""
        if not todo(method):
            return
        try:
            fn()
        except Exception as e:  # ghi dòng lỗi riêng cho phương pháp; lần chạy lại sẽ thử lại
            print(f"[lỗi] {did} {setting} {level} s{seed} {method}: {e!r}", flush=True)
            writer.writerow(dict(base, method=method, status=f"error: {e!r}"[:200]))
            fh.flush()
    acc_mon = lambda yy: float(np.mean(yy[:n] == ytr))

    if todo("naive"):
        t0 = time.time()
        fm = np.ones(n, bool) if fit_mask is None else fit_mask
        emit("naive", fin.predict_proba(Xtr[fm], pseudo[fm], Xte), t0, 1)
    if todo("oracle"):
        t0 = time.time()
        emit("oracle", fin.predict_proba(Xtr, ytr, Xte), t0, 1)
    P0 = None
    if todo("gibbs_warm") or todo("sem_warm") or todo("hard_em_warm") or (setting == "noisy" and todo("cl_tfm")):
        t0w = time.time()
        P0 = warm_start_cross(cond, Xtr, pseudo, fit_mask=fit_mask, n_folds=5, seed=seed)
        t_warm = time.time() - t0w
    if todo("gibbs_warm") or todo("sem_warm"):
        t0 = time.time()
        cond.n_calls = 0
        g = InContextGibbs(cond, mk_weak(), n_blocks=args.blocks, n_sweeps=args.sweeps, burn_in=args.burn_in,
                           max_context=args.max_context, seed=seed, init=P0, monitor=acc_mon).run(Xtr)
        m = g.trace_.monitor
        info = f"flips_last={g.trace_.flips[-1]:.3f};acc_trace={m[0]:.3f}/{m[args.burn_in - 1]:.3f}/{m[-1]:.3f}"
        if setting == "noisy":
            info += f";init_acc={float(np.mean(P0.argmax(1) == ytr)):.3f}"
        calls = cond.n_calls + 5
        emit("gibbs_warm", g.predict_proba(Xtr, Xte, cond=fin, n_use=args.n_use), t0 - t_warm, calls + args.n_use, info)
        emit("sem_warm", g.predict_proba(Xtr, Xte, cond=fin, last_only=True), t0 - t_warm, calls + 1, info)
    if todo("hard_em_warm"):
        t0 = time.time()
        cond.n_calls = 0
        weak = mk_weak()
        init_y = weak.init_labels(np.random.default_rng(seed), P0=P0)
        P, _ = hard_em_tfm(cond, weak, Xtr, Xte, n_iter=args.em_iters, n_blocks=args.blocks, seed=seed,
                           init=init_y, cond_final=fin)
        emit("hard_em_warm", P, t0, cond.n_calls + 6)
    # ---- baseline peer-review ----
    if setting == "ssl":
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(Mtr)
        prior = np.bincount(y_obs[y_obs >= 0], minlength=C) / max(1, int((y_obs >= 0).sum()))

        def _label_spreading():
            from sklearn.semi_supervised import LabelSpreading
            t0 = time.time()
            m = LabelSpreading(kernel="knn", n_neighbors=7, alpha=0.2, max_iter=100).fit(sc.transform(Mtr), y_obs)
            with np.errstate(invalid="ignore", divide="ignore"):
                P = m.predict_proba(sc.transform(Mte))
            # Điểm test mà cả 7 láng giềng đều không nhận được nhãn lan truyền (thành phần đồ thị kNN không chứa
            # mẫu có nhãn) cho hàng 0/0 = NaN trong sklearn. Gán tiên nghiệm lớp của phần có nhãn và ghi lại số hàng.
            bad = ~np.isfinite(P).all(1) | (np.nan_to_num(P).sum(1) <= 0)
            P = np.where(bad[:, None], prior[None, :], P)
            emit("label_spreading", P, t0, 0, f"prior_fallback_rows={int(bad.sum())}")

        def _self_training():
            from sklearn.linear_model import LogisticRegression
            from sklearn.semi_supervised import SelfTrainingClassifier
            t0 = time.time()
            m = SelfTrainingClassifier(LogisticRegression(max_iter=2000)).fit(sc.transform(Mtr), y_obs)
            emit("self_training", m.predict_proba(sc.transform(Mte)), t0, 0)

        attempt("label_spreading", _label_spreading)
        attempt("self_training", _self_training)
    elif setting == "noisy":
        def _forward():
            t0 = time.time()
            m = ForwardCorrection(T, epochs=args.epochs, seed=seed).fit(Mtr, w)
            emit("forward", m.predict_proba(Mte), t0, 0)

        def _cl_tfm():
            from cleanlab.filter import find_label_issues
            t0 = time.time()
            issues = find_label_issues(labels=w, pred_probs=P0)
            keep = ~issues
            emit("cl_tfm", fin.predict_proba(Xtr[keep], w[keep], Xte), t0, 6, f"pruned={int(issues.sum())}")

        attempt("forward", _forward)
        attempt("cl_tfm", _cl_tfm)
    else:
        def _proden():
            t0 = time.time()
            m = PRODEN(epochs=args.epochs, seed=seed).fit(Mtr, S)
            emit("proden", m.predict_proba(Mte), t0, 0)

        attempt("proden", _proden)


def summarize(path):
    import pandas as pd
    df = pd.read_csv(path)
    df = df[df.status == "ok"]
    peers = {"ssl": ["label_spreading", "self_training"], "noisy": ["forward", "cl_tfm"], "pll": ["proden"]}
    prim = {"ssl": "auc", "noisy": "auc", "pll": "acc"}
    print("\n================ TÓM TẮT RQ4 (trung bình theo seed) ================")
    for st in ("ssl", "noisy", "pll"):
        d = df[df.setting == st]
        if d.empty:
            continue
        met = prim[st]
        t = d.groupby(["dataset", "level", "method"])[met].mean().unstack("method")
        print(f"\n--- {st} ({met}) ---\n{t.round(4).to_string()}")
        for lvl in sorted(d.level.unique()):
            u = d[d.level == lvl].groupby(["dataset", "method"])[met].mean().unstack("method")
            bp = u[[p for p in peers[st] if p in u]].max(axis=1)
            for m in ("gibbs_warm",):
                ok = (u[m] >= bp) & (u[m] >= u["naive"])
                print(f"  mức {lvl}: {m} đạt hoặc vượt cả peer-review tốt nhất lẫn naive: {int(ok.sum())}/{len(ok)}"
                      f" | > hard_em_warm theo log loss: {int((d[d.level == lvl].groupby(['dataset', 'method']).logloss.mean().unstack()['gibbs_warm'] < d[d.level == lvl].groupby(['dataset', 'method']).logloss.mean().unstack()['hard_em_warm']).sum())}/{len(ok)}")
    if (df.backend == "dry").all():
        print("\n(dry-run: kết quả không có ý nghĩa)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="tabpfn_v2")
    ap.add_argument("--settings", default="ssl,noisy,pll")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--sweeps", type=int, default=30)
    ap.add_argument("--burn_in", type=int, default=10)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--n_use", type=int, default=10)
    ap.add_argument("--em_iters", type=int, default=10)
    ap.add_argument("--gibbs_est", type=int, default=1)
    ap.add_argument("--final_est", type=lambda v: None if v == "default" else int(v), default=None)
    ap.add_argument("--max_context", type=int, default=2000)
    ap.add_argument("--max_train", type=int, default=2000)
    ap.add_argument("--test_frac", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--preds", default="auto")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="g4_rq4_pilot.csv")
    args = ap.parse_args()
    if args.dry_run:
        args.backend, args.sweeps, args.burn_in, args.n_use, args.em_iters, args.epochs = "dry", 6, 2, 3, 3, 30
    if args.preds == "auto":
        args.preds = os.path.splitext(args.out)[0] + "_preds.jsonl"
    done = done_keys(args.out)
    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for setting in args.settings.split(","):
            pool = DEV_MULTI if setting == "pll" else DEV_DATASETS
            ids = list(pool)[:1] if args.dry_run else list(pool)
            for did in ids:
                try:
                    data = load_multiclass(did, args.dry_run) if setting == "pll" else load_dataset(did, args.dry_run, None)
                except Exception as e:
                    print(f"[lỗi] dataset {did}: {e!r}", flush=True)
                    continue
                if not args.dry_run and pool[did] not in data[3]:
                    print(f"[cảnh báo] dataset {did}: tên '{data[3]}' khác dự kiến '{pool[did]}'", flush=True)
                for level in LEVELS[setting]:
                    for seed in [int(v) for v in args.seeds.split(",")]:
                        try:
                            run_cell(args, did, setting, level, seed, data, w, fh, done)
                        except Exception as e:
                            print(f"[lỗi] {did} {setting} {level} s{seed}: {e!r}", flush=True)
                            w.writerow(dict(backend=args.backend, dataset_id=did, dataset=data[3], setting=setting,
                                            level=level, seed=seed, method="-", status=f"error: {e!r}"[:200]))
                            fh.flush()
    summarize(args.out)


if __name__ == "__main__":
    main()
