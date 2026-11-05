"""
v1c_chain_map.py — V2 giai đoạn 1b, lượt V1C phần C (V2/PHASE1B_PLAN.md §1C, ghi trước): ánh xạ phản ứng Φ(ρ) đo trên
chính các ô của V1, với ba cách gieo lỗi:

  uniform  như S2X / V0 (PLL: nhãn sai chọn đều trong ứng viên; LLP: đổi cặp trong túi, giữ số đếm)
  struct   chỉ PLL, như V0: P0 = khởi động ấm (TFM ngây thơ chéo 5 fold)
  chain    MỚI: P0 = Q̄ của chuỗi `warm` đã lưu ở V1 (npz). PLL: planted_struct của V0 với P0 = Q̄_warm.
           LLP: số cặp đổi như uniform; điểm dương bị đổi chọn ∝ 1 − Q̄_i(1), điểm âm ∝ 1 − Q̄_i(0).

Ô được dựng lại đúng như run_g7.py (cả lấy mẫu con S2) và phải khớp sha256_split, weak_hash, ytr của npz; không khớp →
status "mismatch". Nhãn thật chỉ dùng để gieo z và để đo — phân tích cơ chế (W1), không phải phương pháp.
Chạy:  python v1c_chain_map.py --npz_glob "/kaggle/working/v1_npz/v1_tabicl_v2_*.npz" --out v1c_chainmap_tabicl.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from icgibbs.g7.data import encode_tfm, load_openml, subsample_table  # noqa: E402
from icgibbs.g7.io import weak_hash  # noqa: E402
from icgibbs.g7.registry import S2_SUBSAMPLE  # noqa: E402
from icgibbs.g7.splits import split_sha256, stratified_split  # noqa: E402
from icgibbs.g7.weakgen import make_weak, weak_rng  # noqa: E402
from s2x_response_map import planted_labels, response  # noqa: E402
from v0_response_map import planted_llp, planted_struct, response_llp  # noqa: E402

FIELDS = ["backend", "dataset_id", "dataset", "setting", "level", "seed", "plant", "rho", "n_train", "n_classes",
          "r_actual", "phi", "phi_sd_blocks", "r_start_warm", "r_final_warm", "seconds", "status"]
PAT = re.compile(r"v1_(?P<backend>.+?)_(?P<did>\d+)_(?P<setting>llp|pll)_(?P<level>[\d.]+)_s(?P<seed>\d+)\.npz$")
PLANT_CODE = {"uniform": 0, "struct": 11, "chain": 23}
RHO = "0.2,0.35,0.5,0.65,0.8,0.9,0.97"


def planted_llp_chain(y, bags, rho, Q, rng):
    """Như planted_llp (giữ số đếm, s ~ Binomial(m, 1 − ρ) cặp mỗi túi) nhưng điểm bị đổi chọn theo độ không chắc của
    chuỗi: dương ∝ 1 − Q_i(1), âm ∝ 1 − Q_i(0)."""
    z = np.asarray(y).copy()
    Q = np.clip(np.asarray(Q, float), 0.0, 1.0)
    for b in bags:
        b = np.asarray(b)
        pos, neg = b[z[b] == 1], b[z[b] == 0]
        m = min(len(pos), len(neg))
        s = rng.binomial(m, 1.0 - rho) if m else 0
        if s:
            wp = 1.0 - Q[pos, 1] + 1e-6
            wn = 1.0 - Q[neg, 0] + 1e-6
            p = rng.choice(pos, s, replace=False, p=wp / wp.sum())
            q = rng.choice(neg, s, replace=False, p=wn / wn.sum())
            z[p], z[q] = 0, 1
    return z


def parse_name(path):
    m = PAT.search(os.path.basename(path))
    if not m:
        return None
    st = m["setting"]
    lev = float(m["level"]) if st == "pll" else int(float(m["level"]))
    return dict(backend=m["backend"], did=int(m["did"]), setting=st, level=lev, seed=int(m["seed"]))


def build_cell(info, source):
    """Dựng lại ô như run_g7.run_cell: bảng (cả lấy mẫu con S2), lần chia, nhãn yếu."""
    from run_g7 import synthetic_table
    did, st = info["did"], info["setting"]
    binary = st == "llp"
    t = synthetic_table(did, binary) if source == "synthetic" else load_openml(did, binary=binary)
    if did in S2_SUBSAMPLE:
        t = subsample_table(t, S2_SUBSAMPLE[did], seed=0)
    X_all = encode_tfm(t.X)
    itr, ite = stratified_split(t.y, info["seed"])
    Xtr, ytr = X_all[itr], t.y[itr]
    wd = make_weak(st, info["level"], ytr, Xtr, weak_rng(info["seed"], did, st), n_classes=t.n_classes)
    return t, itr, ite, Xtr, ytr, wd


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_glob", required=True)
    ap.add_argument("--plants", default="uniform,struct,chain")
    ap.add_argument("--rho", default=RHO)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--source", choices=["openml", "synthetic"], default="openml")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from icgibbs.baselines import warm_start_cross
    from icgibbs.conditional import SklearnTFMConditional
    from run_g7 import make_estimator
    si, sn = (int(x) for x in a.shard.split("/"))
    seeds = {int(s) for s in a.seeds.split(",")}
    rhos = [float(x) for x in a.rho.split(",")]
    plants = [p.strip() for p in a.plants.split(",") if p.strip()]
    files = sorted(f for f in glob.glob(a.npz_glob, recursive=True) if parse_name(f))
    files = [f for k, f in enumerate(files) if k % sn == si and parse_name(f)["seed"] in seeds]
    done = set()
    if os.path.exists(a.out):
        for r in csv.DictReader(open(a.out)):
            if r["status"] in ("ok", "mismatch"):
                done.add((r["backend"], r["dataset_id"], r["setting"], r["level"], r["seed"], r["plant"], r["rho"]))
    fh = open(a.out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if os.path.getsize(a.out) == 0:
        w.writeheader()
    print(f"{len(files)} tệp npz", flush=True)
    for f in files:
        info = parse_name(f)
        st, did, seed, be = info["setting"], info["did"], info["seed"], info["backend"]
        my = [(p, r) for p in plants for r in rhos if not (st == "llp" and p == "struct")]
        keys = {p_r: (be, str(did), st, str(info["level"]), str(seed), p_r[0], str(p_r[1])) for p_r in my}
        if all(k in done for k in keys.values()):
            continue
        z0 = np.load(f)
        base = dict(backend=be, dataset_id=did, setting=st, level=info["level"], seed=seed,
                    r_start_warm=round(float(z0["rstart_warm"]), 4),
                    r_final_warm=round(float(np.asarray(z0["monitor_warm"], float)[-5:].mean()), 4))
        try:
            t, itr, ite, Xtr, ytr, wd = build_cell(info, a.source)
        except Exception as e:
            print(f"[lỗi tải] {did}: {e!r}", flush=True)
            continue
        base.update(dataset=t.name, n_train=len(itr), n_classes=t.n_classes)
        ok = (split_sha256(itr, ite) == str(z0["sha256_split"]) and weak_hash(wd) == str(z0["weak_hash"])
              and np.array_equal(ytr, np.asarray(z0["ytr"]).astype(ytr.dtype)))
        if not ok:
            for (p, r), k in keys.items():
                if k not in done:
                    w.writerow(dict(base, plant=p, rho=r, status="mismatch"))
            fh.flush()
            print(f"[mismatch] {os.path.basename(f)}", flush=True)
            continue
        Q = np.asarray(z0["Q_warm"], float)
        cond = SklearnTFMConditional(lambda: make_estimator(be, 1), t.n_classes)
        P0 = None
        for (p, r), k in keys.items():
            if k in done:
                continue
            rng = np.random.default_rng([seed, did, int(round(float(info["level"]) * 100)), int(round(r * 100)),
                                         PLANT_CODE[p]])
            t0 = time.time()
            row = dict(base, plant=p, rho=r)
            try:
                if st == "llp":
                    z = planted_llp(ytr, wd.bags, r, rng) if p == "uniform" else planted_llp_chain(ytr, wd.bags, r, Q, rng)
                    phi, sd = response_llp(cond, Xtr, z, ytr, wd.bags, rng=rng)
                else:
                    S = np.asarray(wd.S, bool)
                    if p == "uniform":
                        z = planted_labels(ytr, S, r, rng)
                    elif p == "struct":
                        if P0 is None:
                            P0 = warm_start_cross(cond, Xtr, wd.naive_pll, n_folds=5, seed=seed)
                        z = planted_struct(ytr, S, r, P0, rng)
                    else:
                        z = planted_struct(ytr, S, r, Q, rng)
                    phi, sd = response(cond, Xtr, z, ytr, S, rng=rng)
                row.update(r_actual=round(float((z == ytr).mean()), 4), phi=round(phi, 4), phi_sd_blocks=round(sd, 4),
                           status="ok")
            except Exception as e:
                row.update(status=f"error: {e!r}"[:200])
            row["seconds"] = round(time.time() - t0, 1)
            w.writerow(row)
            fh.flush()
        print(f"xong {os.path.basename(f)}", flush=True)
    fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
