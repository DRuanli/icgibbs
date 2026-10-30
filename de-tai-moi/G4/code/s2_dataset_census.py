"""
s2_dataset_census.py — Nghiên cứu 2 (S2), bước 1: ĐẾM dataset mới có thể dùng cho tập đánh giá S2.
Chỉ đọc METADATA của OpenML qua REST API (số mẫu, số feature, số lớp, tỉ lệ lớp nhỏ nhất, mô tả). Không tải bảng dữ
liệu, không sinh nhãn yếu, không chạy phương pháp nào. Quy tắc chọn và thang nới tiêu chí đã ghi trước trong
de-tai-moi/S2/S2_PLAN.md §3 (viết trước khi chạy script này).

Nguồn ứng viên: các bộ chuẩn tuyển chọn KHÁC OpenML-CC18 — OpenML100 (study 14), AMLB phân loại (271), bộ của
Grinsztajn et al. 2022 (334, 337), TabZilla (379), cùng mọi suite có alias chứa "tabarena", "amlb", "tabzilla",
"tabrepo". Loại: mọi dataset của CC18 (study 99), mọi dataset của tập phát triển và tập đánh giá cũ, các bản khác
(version khác, tên trùng) của chúng, và các nhóm cùng nguồn đã biết.

Chạy (Kaggle, CPU, có Internet; khoảng 2–5 phút):
    python s2_dataset_census.py --out /kaggle/working
Ra: s2_census_candidates.csv (mọi ứng viên kèm cờ), s2_census_summary.txt (số lượng theo từng bậc của thang nới).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request

API = "https://www.openml.org/api/v1/json"
KNOWN_SUITES = {14: "OpenML100", 271: "AMLB classification", 334: "Grinsztajn clf numerical",
                337: "Grinsztajn clf categorical", 379: "TabZilla hard"}
ALIAS_PATTERNS = ("tabarena", "amlb", "tabzilla", "tabrepo")
CC18 = 99
OLD_IDS = {3, 29, 44, 50, 1049, 1489, 1494, 6332, 23381, 38, 11, 12, 28, 46, 182, 188, 458, 469, 1475, 1497, 1501,
           40966, 40975, 40982, 31, 37, 1464, 1462, 1480, 1510, 54, 36, 23, 40984}
# nhóm cùng nguồn đã biết (giao thức §1.1 C1 và loại trừ tập phát triển); ứng viên có tên khớp thì gắn cờ để rà tay
SAME_SOURCE = [r"^(pc\d|kc\d|jm1|mc\d|mw1|cm1|ar\d)$", r"^mfeat", r"^(splice|dna)$", r"breast", r"wdbc|wisconsin",
               r"credit", r"diabetes|pima", r"blood", r"banknote", r"ilpd|liver", r"vehicle", r"segment", r"^cmc$",
               r"semeion", r"optdigits|pendigits|digits", r"car(\b|_)|car-evaluation", r"balance", r"steel",
               r"satimage|landsat", r"mice", r"wall-robot", r"phoneme", r"spambase|spam", r"sick|thyroid|hypothyroid",
               r"tic-tac-toe", r"kr-vs-kp|chess", r"qsar", r"cylinder", r"dresses", r"eucalyptus", r"analcatdata",
               r"first-order"]
# Thang nới tiêu chí (S2_PLAN §3.2), áp lần lượt; bậc 0 = đúng tiêu chí của giao thức G5 §1.1
LADDER = [
    dict(step=0, n_max=10_000, d_max=500, c_max=10, minority_min=0.10),
    dict(step=1, n_max=20_000, d_max=500, c_max=10, minority_min=0.10),
    dict(step=2, n_max=20_000, d_max=500, c_max=20, minority_min=0.10),
    dict(step=3, n_max=20_000, d_max=2_000, c_max=20, minority_min=0.05),
]


def get(path, tries=6):
    for k in range(tries):
        try:
            with urllib.request.urlopen(API + path, timeout=60) as r:
                return json.loads(r.read().decode())
        except Exception as e:                                       # OpenML hay trả 503 (G7): thử lại có giãn cách
            if "412" in str(e):                                       # 412 = danh sách rỗng
                return None
            time.sleep(min(60, 5 * 2 ** k))
            last = e
    raise RuntimeError(f"{path}: {last}")


def suite_data_ids(sid):
    j = get(f"/study/{sid}")
    st = j["study"]
    ids = st.get("data", {}).get("data_id", [])
    ids = [int(x) for x in (ids if isinstance(ids, list) else [ids])]
    return st.get("name", ""), st.get("alias", ""), ids


def list_suites():
    j = get("/study/list/main_entity_type/task/limit/2000")
    return [] if j is None else j["study_list"]["study"]


def qualities(ids):
    out = {}
    for i in range(0, len(ids), 100):
        chunk = ",".join(str(x) for x in ids[i:i + 100])
        j = get(f"/data/list/data_id/{chunk}/status/all")
        for ds in (j or {}).get("data", {}).get("dataset", []):
            q = {e["name"]: e.get("value") for e in ds.get("quality", [])}
            out[int(ds["did"])] = dict(name=ds["name"], version=int(ds.get("version", 1)), status=ds.get("status"),
                                       q=q)
    return out


def norm(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = []
    P = lambda s: (print(s), log.append(s))

    suites = dict(KNOWN_SUITES)
    try:
        for s in list_suites():
            al = (s.get("alias") or "").lower()
            if any(p in al for p in ALIAS_PATTERNS):
                suites.setdefault(int(s["id"]), s.get("alias") or s.get("name"))
    except Exception as e:
        P(f"[cảnh báo] không liệt kê được suite: {e}")
    pool = {}
    for sid, label in sorted(suites.items()):
        try:
            name, alias, ids = suite_data_ids(sid)
            P(f"suite {sid} ({label}; '{name}', alias '{alias}'): {len(ids)} dataset")
            for d in ids:
                pool.setdefault(d, set()).add(sid)
        except Exception as e:
            P(f"[cảnh báo] suite {sid}: {e}")
    _, _, cc18 = suite_data_ids(CC18)
    excl = set(cc18) | OLD_IDS
    meta_excl = qualities(sorted(excl))
    excl_names = {norm(m["name"]) for m in meta_excl.values()}
    cand = sorted(d for d in pool if d not in excl)
    P(f"pool: {len(pool)} dataset; sau khi bỏ CC18 + dev/eval cũ theo ID: {len(cand)}")
    meta = qualities(cand)

    rows = []
    for d in cand:
        m = meta.get(d)
        if m is None:
            continue
        q = m["q"]
        f = lambda k: float(q[k]) if q.get(k) not in (None, "", "NaN") else float("nan")
        n, nf, C = f("NumberOfInstances"), f("NumberOfFeatures"), f("NumberOfClasses")
        mino = f("MinorityClassSize") / n if n == n and n > 0 else float("nan")
        r = dict(did=d, name=m["name"], version=m["version"], status=m["status"],
                 suites=";".join(str(s) for s in sorted(pool[d])), n=n, d=nf - 1, n_classes=C,
                 minority_frac=round(mino, 4), n_missing_values=f("NumberOfMissingValues"),
                 same_name_as_excluded=norm(m["name"]) in excl_names,
                 same_source_flag=any(re.search(p, m["name"].lower()) for p in SAME_SOURCE))
        for L in LADDER:
            ok_n = 500 <= n <= L["n_max"] and r["d"] <= L["d_max"]
            r[f"binary_step{L['step']}"] = bool(ok_n and C == 2 and mino >= L["minority_min"])
            r[f"multi_step{L['step']}"] = bool(ok_n and 3 <= C <= L["c_max"])
        rows.append(r)
    # bỏ trùng tên giữa các ứng viên (cùng dataset nhiều version / nhiều ID): giữ ID nhỏ nhất, gắn cờ phần còn lại
    seen = {}
    for r in sorted(rows, key=lambda r: r["did"]):
        k = norm(r["name"])
        r["duplicate_of"] = seen.get(k, "")
        seen.setdefault(k, r["did"])
    # mô tả ngắn cho các ứng viên qua bậc cao nhất, để rà tay "dữ liệu thật" và "cùng nguồn"
    for r in rows:
        last = LADDER[-1]["step"]
        if (r[f"binary_step{last}"] or r[f"multi_step{last}"]) and not r["duplicate_of"] and not r["same_name_as_excluded"]:
            try:
                j = get(f"/data/{r['did']}")["data_set_description"]
                r["description"] = re.sub(r"\s+", " ", str(j.get("description", "")))[:400]
                r["original_url"] = j.get("original_data_url", "")
                r["md5"] = j.get("md5_checksum", "")
            except Exception as e:
                r["description"] = f"[lỗi: {e}]"

    import csv
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("did", "name", "version"), k))
    with open(os.path.join(a.out, "s2_census_candidates.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    clean = [r for r in rows if not r["duplicate_of"] and not r["same_name_as_excluded"] and r["status"] == "active"]
    P("\nSố ứng viên SẠCH (không trùng tên, không trùng dev/eval cũ, active) theo từng bậc — CHƯA rà tay "
      "'dữ liệu thật' và 'cùng nguồn' (cờ same_source_flag):")
    for L in LADDER:
        s = L["step"]
        b = [r for r in clean if r[f"binary_step{s}"]]
        m = [r for r in clean if r[f"multi_step{s}"]]
        P(f"  bậc {s} {L}: nhị phân {len(b)} (có cờ cùng nguồn {sum(r['same_source_flag'] for r in b)}); "
          f"đa lớp {len(m)} (có cờ {sum(r['same_source_flag'] for r in m)}; ≥ 8 lớp {sum(r['n_classes'] >= 8 for r in m)})")
    open(os.path.join(a.out, "s2_census_summary.txt"), "w").write("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
