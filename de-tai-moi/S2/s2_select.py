"""
s2_select.py — S2: chọn tập đánh giá mới theo đúng quy tắc của S2_PLAN §3, từ
  (1) census/s2_census_candidates.csv  (metadata OpenML, do s2_dataset_census.py tạo; không sửa tay)
  (2) s2_review.csv                     (rà tay: dữ liệu thật hay không, cùng nguồn, cờ S1 — chỉ dựa trên metadata)
Ra: s2_selected.csv (tập chọn + danh sách dự phòng có thứ tự). Tất định, không dùng mạng.
Sau lượt S2D: `python s2_select.py --hashes raw/S2D/s2_dataset_hashes.csv` ghi thêm s2_final.csv — áp S2_PLAN §3.4:
loại dataset trượt tiêu chí sau loại trùng (pass_criteria = False); nhóm còn < 10 thì bổ sung theo đúng thứ tự dự phòng
(chỉ lấy dự phòng đạt tiêu chí sau loại trùng); dự phòng có n > 10.000 được ghi subsample = 10000.

Quy tắc:
  - Ứng viên đủ điều kiện ở bậc s: cờ của bậc s trong census, status active, không trùng tên dataset cũ, rà tay = keep.
  - Trùng tên: chỉ xét GIỮA CÁC ỨNG VIÊN ĐỦ ĐIỀU KIỆN cùng bậc, giữ ID nhỏ nhất. (Script đếm đánh dấu trùng trên mọi
    dataset của các suite, kể cả bản hồi quy không bao giờ được chọn — làm mất oan bản phân loại, ví dụ socmob 934.)
  - Chọn bậc thấp nhất cho ≥ 10 dataset mỗi nhóm; lấy hết, tối đa 20 (ID nhỏ nhất).
  - Dự phòng (§3.4): các ứng viên mới của các bậc cao hơn, theo bậc rồi theo ID tăng dần.
Chạy:  python s2_select.py
"""
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_N, MAX_N = 10, 20


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def final(sel, hashes_path):
    h = pd.read_csv(hashes_path).set_index("did")
    ok = lambda d: d in h.index and str(h.status.get(d, "")).startswith("ok") and bool(h.pass_criteria.get(d))
    rows = []
    for grp in ("binary", "multiclass"):
        chosen = sel[(sel.group == grp) & (sel.role == "chọn")]
        reserve = sel[(sel.group == grp) & (sel.role == "dự phòng")].sort_values("order")
        for r in chosen.itertuples():
            rows.append(dict(group=grp, did=r.did, name=r.name, source=f"chọn (bậc {r.step})",
                             kept=ok(r.did), reason="" if ok(r.did) else f"trượt tiêu chí sau loại trùng: n = {h.n.get(r.did)}"))
        n_kept = sum(x["kept"] for x in rows if x["group"] == grp)
        for r in reserve.itertuples():
            if n_kept >= MIN_N:
                break
            if ok(r.did):
                rows.append(dict(group=grp, did=r.did, name=r.name, source=f"dự phòng {r.order} (bậc {r.step})", kept=True,
                                 reason="bổ sung theo §3.4"))
                n_kept += 1
        if n_kept < MIN_N:
            raise SystemExit(f"{grp}: sau bổ sung vẫn < {MIN_N} — điều kiện dừng (S2_PLAN §3.2)")
    f = pd.DataFrame(rows)
    f["n_after_dedup"] = f.did.map(lambda d: h.n.get(d))
    f["subsample"] = f.n_after_dedup.map(lambda n: 10000 if n and n > 10000 else "")
    f.to_csv(os.path.join(HERE, "s2_final.csv"), index=False)
    print(f.to_string(index=False))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hashes", default="")
    args = ap.parse_args()
    c = pd.read_csv(os.path.join(HERE, "census", "s2_census_candidates.csv"))
    r = pd.read_csv(os.path.join(HERE, "s2_review.csv"))
    c = c[(c.status == "active") & (~c.same_name_as_excluded.astype(bool))]
    keep = dict(zip(r.did, r.decision == "keep"))
    s1 = dict(zip(r.did, r.s1_flag.astype(bool)))
    out = []
    for grp, pre in (("binary", "binary_step"), ("multiclass", "multi_step")):
        elig = {}
        for s in range(4):
            e = c[c[f"{pre}{s}"].astype(bool)].sort_values("did")
            e = e[~e.name.map(norm).duplicated()]
            unreviewed = [d for d in e.did if d not in keep]
            e = e[e.did.map(lambda d: keep.get(d, False))]
            elig[s] = (list(e.did), unreviewed)
        step = next((s for s in range(4) if len(elig[s][0]) >= MIN_N), None)
        if step is None:
            raise SystemExit(f"{grp}: không bậc nào đủ {MIN_N} dataset — điều kiện dừng (S2_PLAN §3.2)")
        assert not elig[step][1], f"{grp}: còn ứng viên bậc {step} chưa rà tay: {elig[step][1]}"
        chosen = elig[step][0][:MAX_N]
        seen = set(chosen)
        for d in chosen:
            out.append(dict(group=grp, role="chọn", order=len([o for o in out if o["group"] == grp]) + 1, step=step,
                            did=d))
        k = 0
        for s in range(step + 1, 4):
            for d in elig[s][0]:
                if d not in seen:
                    k += 1
                    seen.add(d)
                    out.append(dict(group=grp, role="dự phòng", order=k, step=s, did=d))
            if elig[s][1]:
                out.append(dict(group=grp, role=f"chưa rà (bậc {s})", order=0, step=s, did=-1))
    o = pd.DataFrame(out)
    meta = c.drop_duplicates("did").set_index("did")
    o["name"] = o.did.map(lambda d: meta.name.get(d, ""))
    for k in ("n", "d", "n_classes", "minority_frac", "n_missing_values"):
        o[k] = o.did.map(lambda d: meta[k].get(d))
    o["s1_flag"] = o.did.map(lambda d: s1.get(d, False))
    o.to_csv(os.path.join(HERE, "s2_selected.csv"), index=False)
    pd.set_option("display.width", 200)
    if args.hashes:
        final(o, args.hashes)
    else:
        print(o.to_string(index=False))


if __name__ == "__main__":
    main()
