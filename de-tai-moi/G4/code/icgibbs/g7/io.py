"""Ghi kết quả G7: CSV chạy tiếp được (khoá = backend, dataset, bối cảnh, mức, seed, phương pháp), dự đoán dạng jsonl,
mã băm và phiên bản code trên mỗi dòng (R7)."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess

import numpy as np

KEY = ("backend", "dataset_id", "setting", "level", "seed", "method")


def code_version(root: str | None = None) -> str:
    """Commit git của kho (nếu có), kèm '+dirty' khi có thay đổi chưa commit; nếu không có git: mã băm các tệp .py."""
    root = root or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        c = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10)
        if c.returncode == 0:
            d = subprocess.run(["git", "-C", root, "status", "--porcelain", "--", "."], capture_output=True, text=True,
                               timeout=10)
            return c.stdout.strip() + ("+dirty" if d.stdout.strip() else "")
    except Exception:
        pass
    h = hashlib.sha256()
    for dp, _, fs in sorted(os.walk(root)):
        for f in sorted(fs):
            if f.endswith(".py"):
                with open(os.path.join(dp, f), "rb") as fh:
                    h.update(fh.read())
    return "py-" + h.hexdigest()[:12]


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def weak_hash(wd) -> str:
    h = hashlib.sha256()
    for name in ("s", "bags", "counts", "y_obs", "w", "S"):
        v = getattr(wd, name, None)
        if v is None:
            continue
        if name == "bags":
            for b in v:
                h.update(np.asarray(b, np.int64).tobytes()); h.update(b"|")
        else:
            h.update(np.ascontiguousarray(np.asarray(v)).astype(np.int64).tobytes())
    return h.hexdigest()[:16]


class ResultWriter:
    def __init__(self, path: str, fields: list[str], preds_path: str | None = None):
        self.path, self.fields, self.preds_path = path, list(fields), preds_path
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        self.done = set()
        if not new:
            with open(path) as f:
                for r in csv.DictReader(f):
                    if r.get("status") == "ok":
                        self.done.add(tuple(str(r[k]) for k in KEY))
        self.fh = open(path, "a", newline="")
        self.w = csv.DictWriter(self.fh, fieldnames=self.fields, extrasaction="ignore")
        if new:
            self.w.writeheader()

    def key(self, row) -> tuple:
        return tuple(str(row[k]) for k in KEY)

    def is_done(self, **k) -> bool:
        return tuple(str(k[x]) for x in KEY) in self.done

    def write(self, row: dict, P=None, y=None):
        self.w.writerow(row)
        self.fh.flush()
        if row.get("status") == "ok":
            self.done.add(self.key(row))
            if self.preds_path is not None and P is not None:
                with open(self.preds_path, "a") as pf:
                    pf.write(json.dumps({"key": "|".join(self.key(row)), "P": np.round(np.asarray(P, float), 4).tolist(),
                                         "y": None if y is None else np.asarray(y).astype(int).tolist()}) + "\n")

    def close(self):
        self.fh.close()
