"""
s3_dataset_census.py — Study 3 (V2), bước 1: ĐẾM dataset có thể dùng cho Study 3. Chỉ đọc METADATA OpenML (như
s2_dataset_census.py, dùng lại các hàm của nó); không tải bảng, không chạy phương pháp.

Khác S2:
  - THÊM OpenML-CC18 (study 99) vào nguồn: CC18 nằm ngoài tập huấn luyện của TabDPT (bài TabDPT giữ CC18 để đánh giá),
    và các dataset CC18 chưa dùng ở Study 1/2/tập phát triển là dữ liệu mới với ta.
  - Loại theo s3_exclude_ids.csv: mọi dataset đã dùng (Study 1, Study 2, tập phát triển, V2) và 123 dataset huấn luyện
    của TabDPT (V2/W4_TFM_REPORT §2; quy tắc thầy duyệt 06/11/2026), cùng mọi ứng viên trùng tên với chúng.
  - Thang nới tiêu chí giữ nguyên như S2 (LADDER).
Chạy (Kaggle, CPU, Internet; vài phút):  python s3_dataset_census.py --out /kaggle/working [--all_openml]
  --all_openml (V1E_PLAN E2): thêm mọi dataset OpenML đang hoạt động có 500 ≤ n ≤ 20.000, 2 ≤ C ≤ 20; gắn cờ
  artificial_flag theo mẫu tên để rà tay.
Ra: s3_census_candidates.csv, s3_census_summary.txt.
"""
from __future__ import annotations

import argparse
import csv
import os
import re

import s2_dataset_census as S2

HERE = os.path.dirname(os.path.abspath(__file__))
CC18 = 99


def load_excluded(path=os.path.join(HERE, "s3_exclude_ids.csv")):
    out = {}
    for r in csv.DictReader(open(path)):
        out[int(r["did"])] = r["reason"]
    return out


ALL_OPENML = -1                 # mã giả cho nguồn "mọi dataset OpenML đang hoạt động" (V1E_PLAN E2)
ARTIFICIAL = [r"^bng", r"fri_c", r"synthetic", r"artificial", r"random", r"^rnd", r"^monks", r"^led", r"waveform",
              r"madelon", r"^mfeat", r"^tic", r"^parity", r"^xor", r"^2dplanes", r"^puma", r"^friedman", r"^mv$",
              r"^hypothyroid_", r"_seed_", r"^autouniv", r"^sdss", r"^simulated", r"^openml_", r"^dummy"]


def list_all_openml(limit=1000, max_pages=50):
    """Mọi dataset đang hoạt động với 500 ≤ n ≤ 20.000 và 2 ≤ C ≤ 20, kèm qualities (một lần gọi mỗi trang)."""
    out = {}
    for k in range(max_pages):
        j = S2.get(f"/data/list/status/active/number_instances/500..20000/number_classes/2..20/limit/{limit}/offset/{k * limit}")
        ds = (j or {}).get("data", {}).get("dataset", [])
        for d in ds:
            q = {e["name"]: e.get("value") for e in d.get("quality", [])}
            out[int(d["did"])] = dict(name=d["name"], version=int(d.get("version", 1)), status=d.get("status"), q=q)
        if len(ds) < limit:
            break
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    ap.add_argument("--all_openml", action="store_true", help="V1E: thêm mọi dataset OpenML đang hoạt động thoả n, C")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    log = []
    P = lambda s: (print(s), log.append(s))  # noqa: E731
    excl = load_excluded()
    suites = dict(S2.KNOWN_SUITES)
    suites[CC18] = "OpenML-CC18"
    try:
        for s in S2.list_suites():
            al = (s.get("alias") or "").lower()
            if any(p in al for p in S2.ALIAS_PATTERNS):
                suites.setdefault(int(s["id"]), s.get("alias") or s.get("name"))
    except Exception as e:
        P(f"[cảnh báo] không liệt kê được suite: {e}")
    pool = {}
    for sid, label in sorted(suites.items()):
        try:
            name, alias, ids = S2.suite_data_ids(sid)
            P(f"suite {sid} ({label}; '{name}', alias '{alias}'): {len(ids)} dataset")
            for d in ids:
                pool.setdefault(d, set()).add(sid)
        except Exception as e:
            P(f"[cảnh báo] suite {sid}: {e}")
    extra_meta = {}
    if a.all_openml:
        try:
            extra_meta = list_all_openml()
            n_new = 0
            for d in extra_meta:
                if d not in pool:
                    n_new += 1
                pool.setdefault(d, set()).add(ALL_OPENML)
            P(f"mọi dataset OpenML đang hoạt động (500 ≤ n ≤ 20.000, 2 ≤ C ≤ 20): {len(extra_meta)}, mới {n_new}")
        except Exception as e:
            P(f"[cảnh báo] không liệt kê được mọi dataset OpenML: {e}")
    meta_excl = S2.qualities(sorted(excl))
    excl_names = {S2.norm(m["name"]): excl.get(d, "") for d, m in meta_excl.items()}
    cand = sorted(d for d in pool if d not in excl)
    P(f"pool: {len(pool)} dataset; sau khi bỏ {len(excl)} ID loại trừ: {len(cand)}")
    meta = S2.qualities([d for d in cand if d not in extra_meta])
    meta.update({d: extra_meta[d] for d in cand if d in extra_meta})
    rows = []
    for d in cand:
        m = meta.get(d)
        if m is None:
            continue
        q = m["q"]
        f = lambda k: float(q[k]) if q.get(k) not in (None, "", "NaN") else float("nan")  # noqa: E731
        n, nf, C = f("NumberOfInstances"), f("NumberOfFeatures"), f("NumberOfClasses")
        mino = f("MinorityClassSize") / n if n == n and n > 0 else float("nan")
        nm = S2.norm(m["name"])
        r = dict(did=d, name=m["name"], version=m["version"], status=m["status"],
                 suites=";".join(str(s) for s in sorted(pool[d])), in_cc18=CC18 in pool[d], n=n, d=nf - 1,
                 n_classes=C, minority_frac=round(mino, 4), n_missing_values=f("NumberOfMissingValues"),
                 same_name_as_excluded=nm in excl_names, same_name_reason=excl_names.get(nm, ""),
                 same_source_flag=any(re.search(p, m["name"].lower()) for p in S2.SAME_SOURCE),
                 only_all_openml=pool[d] == {ALL_OPENML},
                 artificial_flag=any(re.search(p, m["name"].lower()) for p in ARTIFICIAL))
        for L in S2.LADDER:
            ok_n = 500 <= n <= L["n_max"] and r["d"] <= L["d_max"]
            r[f"binary_step{L['step']}"] = bool(ok_n and C == 2 and mino >= L["minority_min"])
            r[f"multi_step{L['step']}"] = bool(ok_n and 3 <= C <= L["c_max"])
        rows.append(r)
    seen = {}
    for r in sorted(rows, key=lambda r: r["did"]):
        k = S2.norm(r["name"])
        r["duplicate_of"] = seen.get(k, "")
        seen.setdefault(k, r["did"])
    last = S2.LADDER[-1]["step"]
    for r in rows:
        if ((r[f"binary_step{last}"] or r[f"multi_step{last}"]) and not r["duplicate_of"] and not r["same_name_as_excluded"]
                and not r.get("artificial_flag")):
            try:
                j = S2.get(f"/data/{r['did']}")["data_set_description"]
                r["description"] = re.sub(r"\s+", " ", str(j.get("description", "")))[:400]
                r["original_url"] = j.get("original_data_url", "")
                r["md5"] = j.get("md5_checksum", "")
            except Exception as e:
                r["description"] = f"[lỗi: {e}]"
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("did", "name", "version"), k))
    with open(os.path.join(a.out, "s3_census_candidates.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    clean = [r for r in rows if not r["duplicate_of"] and not r["same_name_as_excluded"] and r["status"] == "active"]
    P("\nỨng viên SẠCH theo từng bậc — CHƯA rà tay 'dữ liệu thật' và 'cùng nguồn':")
    for L in S2.LADDER:
        s = L["step"]
        b = [r for r in clean if r[f"binary_step{s}"]]
        m = [r for r in clean if r[f"multi_step{s}"]]
        b0 = [r for r in b if not r["artificial_flag"]]
        m0 = [r for r in m if not r["artificial_flag"]]
        P(f"  bậc {s}: không cờ nhân tạo — nhị phân {len(b0)} (chỉ từ OpenML chung {sum(r['only_all_openml'] for r in b0)}), "
          f"đa lớp {len(m0)} (≥ 8 lớp {sum(r['n_classes'] >= 8 for r in m0)})")
        P(f"  bậc {s} {L}: nhị phân {len(b)} (CC18 {sum(r['in_cc18'] for r in b)}, cờ cùng nguồn "
          f"{sum(r['same_source_flag'] for r in b)}); đa lớp {len(m)} (CC18 {sum(r['in_cc18'] for r in m)}, cờ "
          f"{sum(r['same_source_flag'] for r in m)}; ≥ 8 lớp {sum(r['n_classes'] >= 8 for r in m)})")
    open(os.path.join(a.out, "s3_census_summary.txt"), "w").write("\n".join(log) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
