"""
g5_verify_pll_real.py — xác minh (CHỈ ĐỌC) các bộ nhãn một phần THẬT cho RQ5 (G5 §1.4, việc F4).
=================================================================================================

Không chạy phương pháp nào trên dữ liệu này (giữ nguyên tắc không chạm tập đánh giá trước khi đóng băng G5).
Việc làm: tải về, giải nén, đọc tệp MATLAB, kiểm cấu trúc và thống kê, so với số đã công bố, áp tiêu chí chọn
viết trước, lưu bản .npz (X, S, y) và mã SHA-256 của tệp gốc để G6 dùng đúng một bản dữ liệu.

Nguồn: trang dữ liệu PLL của nhóm PALM, Đại học Đông Nam (palm.seu.edu.cn/zhangml/Resources.htm). Điều khoản
của trang: chỉ dùng cho mục đích học thuật, tự chịu rủi ro, ghi nguồn các bài gốc. Mỗi tệp có ba biến:
data (M×d), partial_target (Q×M, tập ứng viên), target (Q×M, nhãn đúng).

Số công bố để đối chiếu (Feng et al., NeurIPS 2020, Bảng 6; FG-NET theo Zhang & Yu, IJCAI 2015):
  Lost 1122×108, 16 lớp, 2,23 ứng viên | MSRCv2 1758×48, 23, 3,16 | BirdSong 4998×38, 13, 2,18
  Soccer Player 17472×279, 171, 2,09 | Yahoo! News 22991×163, 219, 1,91 | FG-NET 1002×262, 78, 7,48
Tiêu chí chọn cho RQ5 (viết trước khi đọc dữ liệu): 500 ≤ n ≤ 10.000; d ≤ 500; số lớp ≤ 26 (lớn nhất đã thử
trên tập phát triển ở kaggle_g4_manyclass_pilot.py); nhãn đúng nằm trong tập ứng viên ở ≥ 99% mẫu.

Chạy trên Kaggle (Internet: On, không cần GPU):
    !(apt-get -qq update && apt-get -qq install -y unrar) > /dev/null 2>&1 || apt-get -qq install -y p7zip-full libarchive-tools > /dev/null 2>&1
    !python g5_verify_pll_real.py --out /kaggle/working/g5_pll_real_check.csv --save_dir /kaggle/working/pll_real
Nếu máy chủ nguồn không truy cập được từ Kaggle: thầy tải các tệp .rar bằng trình duyệt, đưa vào một Kaggle
dataset, script sẽ tự tìm trong /kaggle/input (tham số --search_dirs).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import shutil
import subprocess
import urllib.request

import numpy as np

BASE = "palm.seu.edu.cn/zhangml/files/"
SOURCES = {   # tên: (tệp trên máy chủ, n, d, số lớp, số ứng viên trung bình đã công bố)
    "Lost": ("lost.rar", 1122, 108, 16, 2.23),
    "MSRCv2": ("MSRCv2.rar", 1758, 48, 23, 3.16),
    "BirdSong": ("BirdSong.rar", 4998, 38, 13, 2.18),
    "FG-NET": ("FG-NET.rar", 1002, 262, 78, 7.48),
    "Soccer Player": ("Soccer%20Player.rar", 17472, 279, 171, 2.09),
    "Yahoo! News": ("Yahoo!%20News.rar", 22991, 163, 219, 1.91),
}
MAX_CLASSES = 26


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(fname, dl_dir, search_dirs):
    plain = fname.replace("%20", " ")
    for root in search_dirs:   # tệp thầy đã tải tay lên
        hits = [p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
                if os.path.basename(p).lower() == plain.lower()]
        if hits:
            return hits[0], "local"
    os.makedirs(dl_dir, exist_ok=True)
    dst = os.path.join(dl_dir, plain)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst, "cached"
    last = None
    for scheme in ("https://", "http://"):
        try:
            req = urllib.request.Request(scheme + BASE + fname, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as f:
                shutil.copyfileobj(r, f)
            return dst, scheme + BASE + fname
        except Exception as e:  # thử giao thức kế tiếp
            last = e
    raise RuntimeError(f"không tải được {fname}: {last!r}")


def extract(archive, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    if archive.lower().endswith(".mat"):
        shutil.copy(archive, out_dir)
        return
    if archive.lower().endswith(".zip"):
        shutil.unpack_archive(archive, out_dir)
        return
    tools = [["unrar", "x", "-o+", "-inul", archive, out_dir + os.sep],
             ["7z", "x", "-y", f"-o{out_dir}", archive],
             ["7zz", "x", "-y", f"-o{out_dir}", archive],
             ["unar", "-q", "-f", "-o", out_dir, archive],
             ["bsdtar", "-xf", archive, "-C", out_dir]]
    errs = []
    for cmd in tools:
        if shutil.which(cmd[0]) is None:
            continue
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=600)
            if glob.glob(os.path.join(out_dir, "**", "*.mat"), recursive=True):
                return
        except Exception as e:
            errs.append(f"{cmd[0]}: {e!r}")
    raise RuntimeError("không giải nén được RAR; cài một công cụ: apt-get install -y unrar "
                       "(hoặc p7zip-full / libarchive-tools). " + "; ".join(errs))


def load_mat(path):
    try:
        from scipy.io import loadmat
        m = loadmat(path)
        get = lambda k: m[k]
    except NotImplementedError:   # MATLAB v7.3 = HDF5
        import h5py
        f = h5py.File(path, "r")
        get = lambda k: np.asarray(f[k]).T
    out = {}
    for k in ("data", "partial_target", "target"):
        v = get(k)
        out[k] = v.toarray() if hasattr(v, "toarray") else np.asarray(v)
    return out


def check(name, spec, args):
    fname, n_pub, d_pub, c_pub, avg_pub = spec
    archive, origin = fetch(fname, args.download_dir, args.search_dirs)
    tmp = os.path.join(args.download_dir, "_x_" + name.replace(" ", "_").replace("!", ""))
    shutil.rmtree(tmp, ignore_errors=True)
    extract(archive, tmp)
    mats = sorted(glob.glob(os.path.join(tmp, "**", "*.mat"), recursive=True))
    if not mats:
        raise RuntimeError("không thấy tệp .mat sau khi giải nén")
    m = load_mat(mats[0])
    X = m["data"].astype(np.float64)
    S, T = m["partial_target"], m["target"]
    M = X.shape[0]
    if S.shape[1] != M and S.shape[0] == M:
        S = S.T
    if T.shape[1] != M and T.shape[0] == M:
        T = T.T
    S, T = (S > 0).T, (T > 0).T          # về dạng M×Q
    Q = S.shape[1]
    n_true = T.sum(1)
    y = T.argmax(1)
    cand = S.sum(1)
    true_in = S[np.arange(M), y] & (n_true == 1)
    cls = np.bincount(y, minlength=Q)
    import pandas as pd
    dfX = pd.DataFrame(X)
    dup = int(dfX.assign(_y=y).duplicated().sum())
    conflict = int(max(dfX.duplicated(keep=False).sum() - dfX.assign(_y=y).duplicated(keep=False).sum(), 0))
    row = dict(name=name, file=os.path.basename(archive), origin=origin, sha256=sha256(archive),
               mat_file=os.path.basename(mats[0]), n=M, d=X.shape[1], n_classes=Q,
               avg_candidates=round(float(cand.mean()), 3), min_candidates=int(cand.min()),
               max_candidates=int(cand.max()), true_in_candidates=round(float(true_in.mean()), 4),
               rows_not_one_true_label=int((n_true != 1).sum()), min_class=int(cls.min()), max_class=int(cls.max()),
               empty_classes=int((cls == 0).sum()), missing=int(np.isnan(X).sum()),
               constant_features=int((X.std(0) == 0).sum()), duplicate_rows=dup, same_X_different_y=conflict,
               n_published=n_pub, d_published=d_pub, classes_published=c_pub, avg_published=avg_pub)
    row["matches_published"] = (M == n_pub and X.shape[1] == d_pub and Q == c_pub
                                and abs(row["avg_candidates"] - avg_pub) < 0.02)
    row["meets_criteria"] = (500 <= M <= 10000 and X.shape[1] <= 500 and Q <= MAX_CLASSES
                             and row["true_in_candidates"] >= 0.99)
    if args.save_dir and row["meets_criteria"]:
        os.makedirs(args.save_dir, exist_ok=True)
        np.savez_compressed(os.path.join(args.save_dir, name.replace(" ", "_").replace("!", "") + ".npz"),
                            X=X.astype(np.float32), S=S, y=y.astype(np.int16), sha256=row["sha256"])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="g5_pll_real_check.csv")
    ap.add_argument("--download_dir", default="pll_downloads")
    ap.add_argument("--save_dir", default="pll_real")
    ap.add_argument("--search_dirs", default="/kaggle/input")
    ap.add_argument("--only", default="", help="danh sách tên cách nhau bởi dấu phẩy; trống = tất cả")
    args = ap.parse_args()
    args.search_dirs = [p for p in args.search_dirs.split(",") if p and os.path.isdir(p)]
    import pandas as pd
    names = [s.strip() for s in args.only.split(",") if s.strip()] or list(SOURCES)
    rows = []
    for name in names:
        try:
            rows.append(check(name, SOURCES[name], args))
            r = rows[-1]
            print(f"{name:14s} n={r['n']:6d} d={r['d']:4d} C={r['n_classes']:3d} cand={r['avg_candidates']:.2f} "
                  f"true∈S={r['true_in_candidates']:.3f} khớp_công_bố={r['matches_published']} "
                  f"đạt_tiêu_chí={r['meets_criteria']}", flush=True)
        except Exception as e:
            rows.append(dict(name=name, error=repr(e)[:300]))
            print(f"[lỗi] {name}: {e!r}", flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nđã ghi {args.out}")


if __name__ == "__main__":
    main()
