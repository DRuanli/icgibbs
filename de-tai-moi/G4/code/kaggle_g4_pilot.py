"""
kaggle_g4_pilot.py — thí điểm G4 (exploratory) cho đề tài A trên các dataset PHÁT TRIỂN.
==========================================================================================

Mục đích: trả lời tiêu chí dừng của RESEARCH_PLAN §5 trước khi đóng băng giao thức ở G5:
  "phương pháp vượt CẢ baseline peer-review tốt nhất LẪN TFM ngây thơ trên ≥4/6 dataset phát triển,
   ở PU hoặc LLP" — nếu thất bại ở CẢ PU và LLP → phương án B.
Đây KHÔNG phải thực nghiệm chính: không dùng dataset đánh giá; không báo cáo trong bài như kết quả chính.

Bối cảnh × mức yếu (mặc định):  PU-SCAR c ∈ {0.1, 0.3}  ·  LLP túi cỡ {8, 32}  ·  PU-SAR E[e|y=1] = 0.3
  (PU-SAR: e(x) = σ(η0 + 2 z), z = cột số đầu tiên đã chuẩn hoá; Gibbs học η với prior N(0, 2²).
   Baseline SAR chuyên biệt — SAR-EM, Bekker et al. 2019 — CHƯA có trong thí điểm; thêm ở G5–G6.)
Phương pháp
  gibbs       In-Context Gibbs (khối K=10, 30 vòng quét, burn-in 10, n_estimators=1 trong chuỗi;
              dự đoán Rao–Blackwell bằng 10 mẫu với n_estimators mặc định của gói)
  sem         cùng chuỗi, chỉ dùng mẫu cuối (stochastic EM) — ablation RQ3
  hard_em     cùng TFM, gán argmax (10 vòng) — "EM-độc-lập" của RQ3
  naive       TFM ngây thơ (PU: không nhãn = âm, xác suất hiệu chỉnh Elkan–Noto với c đã biết;
              LLP: một lần gán ngẫu nhiên khớp số đếm)
  gibbs_known In-Context Gibbs với propensity đã biết (chỉ PU-SAR; trần tham chiếu cho việc học η)
  nnpu        nnPU (Kiryo et al., NeurIPS 2017), prior lớp đã biết — PU
  upu         uPU (du Plessis et al., ICML 2015) — PU
  dllp        DLLP (Ardehaly & Culotta, ICDM 2017) — LLP
  oracle      TFM với nhãn sạch (trần tham chiếu, không tính vào so sánh)

Lượt 2 (v2, 25/10/2026) thêm: gibbs_warm / sem_warm / hard_em_warm (khởi động ấm), dllp_linear, gibbs_known_warm,
LLP túi 16, metric khớp tiên nghiệm (logloss_pm, brier_pm), lưu dự đoán test (*_preds.jsonl); --diag thêm
gibbs_trueinit (chẩn đoán). --prev <csv lượt 1>: không chạy lại các chuỗi tốn GPU đã có.
Lệnh chạy trên Kaggle: xem G4/README.md.
Script ghi CSV sau mỗi dòng và tự bỏ qua các dòng đã có (chạy lại đúng lệnh nếu phiên bị ngắt).

Kiểm tra không cần GPU / trọng số / internet:
  python kaggle_g4_pilot.py --dry-run --out dry.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import platform
import sys
import time
from importlib import metadata

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from icgibbs import LLP, PUSAR, PUSCAR, InContextGibbs, SklearnTFMConditional, make_bags  # noqa: E402
from icgibbs.baselines import DLLP, NNPU, hard_em_tfm, naive_tfm_llp, naive_tfm_pu, warm_start_pu  # noqa: E402

DEV_DATASETS = {31: "credit-g", 37: "diabetes", 1464: "blood-transfusion-service-center",
                1462: "banknote-authentication", 1480: "ilpd", 1510: "wdbc"}
LEVELS = {"pu_scar": [0.1, 0.3], "llp": [8, 16, 32], "pu_sar": [0.3]}       # v2: thêm LLP túi 16
SAR_SLOPE = 2.0          # e(x) = σ(η0 + 2·z), z = cột số đầu tiên đã chuẩn hoá; η0 chọn để E[e | y=1] = mức
FIELDS = ["backend", "dataset_id", "dataset", "setting", "level", "seed", "method", "auc", "logloss", "acc",
          "brier", "logloss_pm", "brier_pm", "n_train", "n_test", "d", "pos_rate", "n_tfm_calls", "seconds",
          "extra", "status"]
EXPENSIVE = {"gibbs", "sem", "gibbs_known", "hard_em"}       # các chuỗi tốn GPU: lấy lại từ lượt trước nếu có


# --------------------------------------------------------------------------------------------------
def make_estimator(backend: str, n_estimators: int | None):
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


def load_dataset(did: int, dry: bool, rng):
    """Trả về (X_tfm, X_mlp, y, tên). y nhị phân, 1 = lớp thiểu số. X_tfm: số + mã thứ tự cho cột phân loại;
    X_mlp: số (điền trung vị) + one-hot."""
    if dry:
        from sklearn.datasets import make_classification
        X, y = make_classification(600 + 100 * (did % 5), 8, n_informative=4, weights=[0.65],
                                   random_state=did, flip_y=0.03)
        return X.astype(np.float32), X.astype(np.float32), y, f"synthetic-{did}"
    import pandas as pd
    from sklearn.datasets import fetch_openml
    b = fetch_openml(data_id=did, as_frame=True, parser="auto")
    df, target = b.data.copy(), b.target
    keep = ~pd.isna(target)
    df, target = df[keep], target[keep]
    vals, cnt = np.unique(target.astype(str), return_counts=True)
    if len(vals) != 2:
        raise ValueError(f"dataset {did} có {len(vals)} lớp")
    y = (target.astype(str).values == vals[np.argmin(cnt)]).astype(int)
    num = df.select_dtypes(include=[np.number])
    cat = df.drop(columns=num.columns)
    num = num.fillna(num.median())
    X_tfm = [num.values.astype(np.float32)]
    X_mlp = [num.values.astype(np.float32)]
    if cat.shape[1]:
        codes = np.column_stack([pd.Categorical(cat[c]).codes for c in cat.columns]).astype(np.float32)
        codes[codes < 0] = np.nan
        X_tfm.append(codes)
        X_mlp.append(pd.get_dummies(cat.astype(str), dummy_na=False).values.astype(np.float32))
    return np.hstack(X_tfm), np.hstack(X_mlp), y, b.details.get("name", str(did))


def prior_match(p, prior):
    """Hiệu chỉnh khớp tiên nghiệm lớp (quyết định G4, áp dụng GIỐNG NHAU cho mọi phương pháp): dịch logit một hằng
    số b để trung bình xác suất dự đoán trên các điểm cần dự đoán bằng tỉ lệ lớp dương đã biết. Không dùng nhãn test;
    không đổi thứ tự nên không đổi AUC."""
    from scipy.optimize import brentq
    from scipy.special import expit, logit
    z = logit(np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6))
    f = lambda b: expit(z + b).mean() - prior
    if f(-40) > 0 or f(40) < 0:
        return np.asarray(p, float)
    return expit(z + brentq(f, -40, 40))


def metrics(y, p, prior=None):
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    out = dict(auc=roc_auc_score(y, p), logloss=log_loss(y, p, labels=[0, 1]),
               acc=float(np.mean((p > 0.5) == y)), brier=brier_score_loss(y, p))
    if prior is not None:
        q = np.clip(prior_match(p, prior), 1e-6, 1 - 1e-6)
        out.update(logloss_pm=log_loss(y, q, labels=[0, 1]), brier_pm=brier_score_loss(y, q))
    return out


def done_keys(path, only=None):
    if not path or not os.path.exists(path):
        return set()
    with open(path) as f:
        return {(r["backend"], r["dataset_id"], r["setting"], r["level"], r["seed"], r["method"])
                for r in csv.DictReader(f) if r["status"] == "ok" and (only is None or r["method"] in only)}


# --------------------------------------------------------------------------------------------------
def run_cell(args, backend, did, setting, level, seed, data, writer, fh, done):
    X_tfm, X_mlp, y, name = data
    from sklearn.model_selection import train_test_split
    rng = np.random.default_rng(1000 * seed + did)
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=args.test_frac, stratify=y, random_state=seed)
    if len(idx_tr) > args.max_train:
        idx_tr = rng.choice(idx_tr, args.max_train, replace=False)
    Xtr, Xte, ytr, yte = X_tfm[idx_tr], X_tfm[idx_te], y[idx_tr], y[idx_te]
    Mtr, Mte = X_mlp[idx_tr], X_mlp[idx_te]
    base = dict(backend=backend, dataset_id=did, dataset=name, setting=setting, level=level, seed=seed,
                n_train=len(ytr), n_test=len(yte), d=X_tfm.shape[1], pos_rate=round(float(ytr.mean()), 4))

    cond = SklearnTFMConditional(lambda: make_estimator(backend, args.gibbs_est), 2)
    cond_final = SklearnTFMConditional(lambda: make_estimator(backend, args.final_est), 2)

    if setting == "pu_scar":
        c = float(level)
        s = ((ytr == 1) & (rng.uniform(size=len(ytr)) < c)).astype(int)
        if s.sum() < 2:
            s[rng.choice(np.flatnonzero(ytr == 1), 2, replace=False)] = 1
        mk_weak = lambda: PUSCAR(s, c=c)
        extra = f"labeled_pos={int(s.sum())}"
    elif setting == "pu_sar":
        from scipy.optimize import brentq
        from scipy.special import expit
        z = np.nan_to_num(Xtr[:, 0].astype(float))
        z = (z - z.mean()) / (z.std() + 1e-8)
        c = float(level)
        b0 = brentq(lambda b: expit(b + SAR_SLOPE * z[ytr == 1]).mean() - c, -30, 30)
        e = expit(b0 + SAR_SLOPE * z)
        s = ((ytr == 1) & (rng.uniform(size=len(ytr)) < e)).astype(int)
        if s.sum() < 2:
            s[rng.choice(np.flatnonzero(ytr == 1), 2, replace=False)] = 1
        mk_weak = lambda: PUSAR(s, Z=z[:, None], prior_scale=2.0)
        mk_weak_known = lambda: PUSAR(s, Z=z[:, None], eta=[b0, SAR_SLOPE])
        extra = f"labeled_pos={int(s.sum())};eta0={b0:.3f}"
    else:
        bags, counts = make_bags(len(ytr), int(level), rng, ytr)
        mk_weak = lambda: LLP(bags, counts, len(ytr))
        extra = f"n_bags={len(bags)}"

    def emit(method, p, t0, calls, more=""):
        key = (backend, str(did), setting, str(level), str(seed), method)
        row = dict(base, method=method, seconds=round(time.time() - t0, 2), n_tfm_calls=calls,
                   extra=(extra + (";" + more if more else "")), status="ok")
        row.update({k: round(v, 5) for k, v in metrics(yte, p, prior=float(ytr.mean())).items()})
        writer.writerow(row)
        fh.flush()
        if args.preds:                                    # lưu dự đoán test để tính lại metric bất kỳ về sau
            import json
            with open(args.preds, "a") as pf:
                pf.write(json.dumps({"key": "|".join(key), "p": np.round(np.asarray(p, float), 4).tolist(),
                                     "y": yte.astype(int).tolist()}) + "\n")
        done.add(key)
        print(f"  {name:>28s} {setting:7s} {str(level):>4s} s{seed} {method:8s} AUC={row['auc']:.4f} "
              f"LL={row['logloss']:.4f} {row['seconds']:.1f}s", flush=True)

    def todo(method):
        return (backend, str(did), setting, str(level), str(seed), method) not in done

    # ---- In-Context Gibbs (+ SEM từ cùng chuỗi) ----
    if todo("gibbs") or todo("sem"):
        t0 = time.time()
        cond.n_calls = 0
        g = InContextGibbs(cond, mk_weak(), scheme="auto", n_blocks=args.blocks, n_sweeps=args.sweeps,
                           burn_in=args.burn_in, max_context=args.max_context, seed=seed).run(Xtr)
        chain_calls = cond.n_calls
        t_chain = time.time() - t0
        info = f"flips_last={g.trace_.flips[-1]:.3f};chain_s={t_chain:.1f}"
        if setting == "pu_sar":
            info += f";eta_learned={np.round(g.weak.eta, 3).tolist()}"
        cond_final.n_calls = 0
        p = g.predict_proba(Xtr, Xte, cond=cond_final, n_use=args.n_use)[:, 1]
        emit("gibbs", p, t0, chain_calls + cond_final.n_calls, info)
        t1 = time.time()
        p = g.predict_proba(Xtr, Xte, cond=cond_final, last_only=True)[:, 1]
        emit("sem", p, t1 - t_chain, chain_calls + 1, info)
    if setting == "pu_sar" and todo("gibbs_known"):
        t0 = time.time()
        cond.n_calls = 0
        g = InContextGibbs(cond, mk_weak_known(), scheme="auto", n_blocks=args.blocks, n_sweeps=args.sweeps,
                           burn_in=args.burn_in, max_context=args.max_context, seed=seed).run(Xtr)
        p = g.predict_proba(Xtr, Xte, cond=cond_final, n_use=args.n_use)[:, 1]
        emit("gibbs_known", p, t0, cond.n_calls + args.n_use)
    if todo("hard_em"):
        t0 = time.time()
        cond.n_calls = 0
        weak_em = mk_weak_known() if setting == "pu_sar" else mk_weak()   # hard-EM không học tham số phiền nhiễu
        p, _ = hard_em_tfm(cond, weak_em, Xtr, Xte, n_iter=args.em_iters, n_blocks=args.blocks, seed=seed,
                           init=(s.copy() if setting.startswith("pu") else None), cond_final=cond_final)
        emit("hard_em", p[:, 1], t0, cond.n_calls + 1)
    if todo("naive"):
        t0 = time.time()
        if setting.startswith("pu"):
            _, p = naive_tfm_pu(cond_final, Xtr, s, Xte, c=float(level))
        else:
            p = naive_tfm_llp(cond_final, Xtr, mk_weak(), Xte, rng)[:, 1]
        emit("naive", p, t0, 1)
    if todo("oracle"):
        t0 = time.time()
        emit("oracle", cond_final.predict_proba(Xtr, ytr, Xte)[:, 1], t0, 1)
    if setting.startswith("pu"):
        for meth, nn in (("nnpu", True), ("upu", False)):
            if todo(meth):
                t0 = time.time()
                m = NNPU(prior=float(ytr.mean()), nn_correction=nn, epochs=args.epochs, seed=seed).fit(Mtr, s)
                emit(meth, m.predict_proba(Mte)[:, 1], t0, 0)
    else:
        if todo("dllp"):
            t0 = time.time()
            m = DLLP(epochs=args.epochs, seed=seed).fit(Mtr, bags, counts)
            emit("dllp", m.predict_proba(Mte)[:, 1], t0, 0)

    # ================= v2: khởi động ấm (warm start) + chẩn đoán chuỗi =================
    # monitor = độ chính xác của trạng thái chuỗi so với nhãn thật của tập huấn luyện. CHỈ để chẩn đoán trên
    # dataset phát triển; thuật toán không dùng tới.
    acc_mon = lambda yy: float(np.mean(yy[:len(ytr)] == ytr))

    def chain_info(g, t_chain, more=""):
        m = g.trace_.monitor
        pick = sorted({0, max(args.burn_in - 1, 0), len(m) - 1})
        return (f"flips_last={g.trace_.flips[-1]:.3f};chain_s={t_chain:.1f};acc_trace=" +
                "/".join(f"{m[i]:.3f}" for i in pick) + (";" + more if more else ""))

    def run_chain(weak, init):
        cond.n_calls = 0
        t1 = time.time()
        g = InContextGibbs(cond, weak, scheme="auto", n_blocks=args.blocks, n_sweeps=args.sweeps,
                           burn_in=args.burn_in, max_context=args.max_context, seed=seed, init=init,
                           monitor=acc_mon).run(Xtr)
        return g, cond.n_calls, time.time() - t1

    want_warm = todo("gibbs_warm") or todo("sem_warm") or (setting == "llp" and todo("dllp_linear"))
    if want_warm:
        t0 = time.time()
        if setting.startswith("pu"):
            e_init = c if setting == "pu_scar" else mk_weak().e()     # PU-SAR học η: propensity ban đầu của chuỗi
            P0 = warm_start_pu(cond, Xtr, s, e_init, n_folds=5, seed=seed)
            warm_calls = 5
        else:
            dl = DLLP(epochs=args.epochs, hidden=(), seed=seed).fit(Mtr, bags, counts)   # DLLP tuyến tính
            P0 = dl.predict_proba(Mtr)
            warm_calls = 0
            if todo("dllp_linear"):
                emit("dllp_linear", dl.predict_proba(Mte)[:, 1], t0, 0)
        p0_acc = float(np.mean((P0[:, 1] > 0.5) == ytr))
        if todo("gibbs_warm") or todo("sem_warm"):
            g, calls, t_chain = run_chain(mk_weak(), P0)
            info = chain_info(g, t_chain, f"p0_acc={p0_acc:.3f}")
            cond_final.n_calls = 0
            p = g.predict_proba(Xtr, Xte, cond=cond_final, n_use=args.n_use)[:, 1]
            emit("gibbs_warm", p, t0, warm_calls + calls + cond_final.n_calls, info)
            p = g.predict_proba(Xtr, Xte, cond=cond_final, last_only=True)[:, 1]
            emit("sem_warm", p, t0, warm_calls + calls + 1, info)
    if todo("hard_em_warm"):                             # RQ3: hard-EM xuất phát từ CÙNG điểm khởi động ấm
        if not want_warm:
            t0 = time.time()
            if setting.startswith("pu"):
                e_init = c if setting == "pu_scar" else mk_weak().e()
                P0 = warm_start_pu(cond, Xtr, s, e_init, n_folds=5, seed=seed)
            else:
                P0 = DLLP(epochs=args.epochs, hidden=(), seed=seed).fit(Mtr, bags, counts).predict_proba(Mtr)
        t0 = time.time()
        cond.n_calls = 0
        weak_em = mk_weak_known() if setting == "pu_sar" else mk_weak()
        init_y = weak_em.init_labels(np.random.default_rng(seed), P0=P0)
        p, _ = hard_em_tfm(cond, weak_em, Xtr, Xte, n_iter=args.em_iters, n_blocks=args.blocks, seed=seed,
                           init=init_y, cond_final=cond_final)
        emit("hard_em_warm", p[:, 1], t0, cond.n_calls + 1)
    if setting == "pu_sar" and todo("gibbs_known_warm"):
        t0 = time.time()
        P0k = warm_start_pu(cond, Xtr, s, e, n_folds=5, seed=seed)
        g, calls, t_chain = run_chain(mk_weak_known(), P0k)
        p = g.predict_proba(Xtr, Xte, cond=cond_final, n_use=args.n_use)[:, 1]
        emit("gibbs_known_warm", p, t0, 5 + calls + args.n_use, chain_info(g, t_chain))
    # chẩn đoán: khởi tạo từ NHÃN THẬT. Nếu chuỗi vẫn trôi xa nhãn thật thì vấn đề là phân phối dừng của chuỗi
    # xấp xỉ (sai số của TFM bị khuếch đại, Định lý 2), không phải khởi tạo. Không phải một phương pháp.
    diag_cell = setting == "llp" or (setting == "pu_scar" and float(level) == 0.1)
    if args.diag and diag_cell and todo("gibbs_trueinit"):
        t0 = time.time()
        g, calls, t_chain = run_chain(mk_weak(), ytr.copy())
        p = g.predict_proba(Xtr, Xte, cond=cond_final, n_use=args.n_use)[:, 1]
        emit("gibbs_trueinit", p, t0, calls + args.n_use, chain_info(g, t_chain))


def summarize(path, prev=None):
    import pandas as pd
    df = pd.read_csv(path)
    if prev and os.path.exists(prev):
        old = pd.read_csv(prev)
        old = old[old.method.isin(EXPENSIVE)]
        keys = ["backend", "dataset_id", "setting", "level", "seed", "method"]
        new_keys = set(map(tuple, df[keys].astype(str).values))
        old = old[[tuple(r) not in new_keys for r in old[keys].astype(str).values]]
        df = pd.concat([df, old], ignore_index=True)
    df = df[df.status == "ok"]
    print("\n================ TÓM TẮT (AUC, trung bình theo seed) ================")
    verdict = {}
    for setting, peer in (("pu_scar", ["nnpu", "upu"]), ("pu_sar", ["nnpu", "upu"]), ("llp", ["dllp", "dllp_linear"])):
        d = df[df.setting == setting]
        if d.empty:
            continue
        t = d.groupby(["dataset", "level", "method"]).auc.mean().unstack("method")
        print(f"\n--- {setting} ---\n{t.round(4).to_string()}")
        per_ds = d.groupby(["dataset", "method"]).auc.mean().unstack("method")
        best_peer = per_ds[[p for p in peer if p in per_ds]].max(axis=1)
        win = (per_ds["gibbs"] > best_peer) & (per_ds["gibbs"] > per_ds["naive"])
        verdict[setting] = int(win.sum())
        print(f"\nGibbs thắng CẢ peer-review tốt nhất LẪN naive trên {int(win.sum())}/{len(win)} dataset "
              f"(gộp các mức): {sorted(win[win].index)}")
        for name, other in (("hard_em", "hard_em"), ("sem", "sem")):
            if other in per_ds:
                print(f"  Gibbs > {name}: {int((per_ds['gibbs'] > per_ds[other]).sum())}/{len(per_ds)}")
        # v2: theo từng mức, cho cả chuỗi lạnh (gibbs) và chuỗi ấm (gibbs_warm)
        for lvl in sorted(d.level.unique()):
            t = d[d.level == lvl].groupby(["dataset", "method"]).auc.mean().unstack("method")
            bp = t[[p for p in peer if p in t]].max(axis=1)
            for meth in ("gibbs", "gibbs_warm"):
                if meth in t:
                    w = (t[meth] > bp) & (t[meth] > t["naive"])
                    w2 = ((t[meth] - bp) > 0.01) & ((t[meth] - t["naive"]) > 0.01)
                    print(f"  mức {lvl}: {meth} thắng cả hai {int(w.sum())}/{len(w)}; với cách biệt > 0,01: {int(w2.sum())}/{len(w2)}")
    d = df[df.setting == "pu_scar"]
    if not d.empty and "logloss_pm" in d:
        for met in ("logloss", "logloss_pm"):
            t = d.groupby(["dataset", "level", "method"])[met].mean().unstack("method")
            print(f"\n--- pu_scar: {met} (thấp hơn là tốt hơn) ---\n{t.round(4).to_string()}")
    if (df.backend == "dry").all():
        print("\n(dry-run: 'TFM' là LogisticRegression và chuỗi rất ngắn — các con số trên KHÔNG có ý nghĩa)")
        return
    if verdict:
        ok = verdict.get("llp", 0) >= 4 or max(verdict.get("pu_scar", 0), verdict.get("pu_sar", 0)) >= 4
        print(f"\nTiêu chí dừng (RESEARCH_PLAN §5): {'ĐẠT — tiếp tục' if ok else 'KHÔNG ĐẠT ở cả PU và LLP → phương án B'} "
              f"{verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="tabpfn_v2")
    ap.add_argument("--datasets", default=",".join(map(str, DEV_DATASETS)))
    ap.add_argument("--settings", default="pu_scar,llp,pu_sar")
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
    ap.add_argument("--diag", action="store_true", help="thêm chuỗi khởi tạo từ nhãn thật (chẩn đoán, LLP và PU-SCAR c=0.1)")
    ap.add_argument("--prev", default=None, help="CSV của lượt trước: các chuỗi tốn GPU (gibbs, sem, gibbs_known, hard_em) "
                    "đã có ở đó thì không chạy lại; phương pháp rẻ được chạy lại để có thêm metric và dự đoán")
    ap.add_argument("--preds", default="auto", help="file .jsonl lưu dự đoán test ('auto' = cạnh --out; '' = không lưu)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="g4_pilot.csv")
    args = ap.parse_args()
    if args.dry_run:
        args.backend, args.sweeps, args.burn_in, args.n_use, args.em_iters, args.epochs = "dry", 6, 2, 3, 3, 30

    def ver(p):
        try:
            return metadata.version(p)
        except metadata.PackageNotFoundError:
            return "-"
    print(f"python {platform.python_version()} · numpy {np.__version__} · tabpfn {ver('tabpfn')} · tabicl {ver('tabicl')}"
          f" · torch {ver('torch')}", flush=True)
    if args.preds == "auto":
        args.preds = os.path.splitext(args.out)[0] + "_preds.jsonl"
    done = done_keys(args.out) | done_keys(args.prev, only=EXPENSIVE)
    if args.prev:
        print(f"lấy lại {len(done_keys(args.prev, only=EXPENSIVE))} chuỗi đã chạy từ {args.prev}", flush=True)
    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for did in [int(v) for v in args.datasets.split(",")]:
            try:
                data = load_dataset(did, args.dry_run, np.random.default_rng(did))
            except Exception as e:                                  # ghi lỗi, không dừng cả lượt chạy
                print(f"[lỗi] dataset {did}: {e!r}", flush=True)
                continue
            if not args.dry_run and did in DEV_DATASETS and DEV_DATASETS[did] not in data[3]:
                print(f"[cảnh báo] dataset {did}: tên '{data[3]}' khác dự kiến '{DEV_DATASETS[did]}'", flush=True)
            print(f"\n=== {did} {data[3]}: n={len(data[2])}, d={data[0].shape[1]}, dương={data[2].mean():.3f}", flush=True)
            for setting in args.settings.split(","):
                for level in LEVELS[setting]:
                    for seed in [int(v) for v in args.seeds.split(",")]:
                        try:
                            run_cell(args, args.backend, did, setting, level, seed, data, w, fh, done)
                        except Exception as e:
                            print(f"[lỗi] {did} {setting} {level} s{seed}: {e!r}", flush=True)
                            w.writerow(dict(backend=args.backend, dataset_id=did, dataset=data[3], setting=setting,
                                            level=level, seed=seed, method="-", status=f"error: {e!r}"[:200]))
                            fh.flush()
    summarize(args.out, args.prev)


if __name__ == "__main__":
    main()
