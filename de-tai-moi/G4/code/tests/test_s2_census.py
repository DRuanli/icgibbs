"""Kiểm logic của s2_dataset_census.py bằng API giả (không cần mạng)."""
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import s2_dataset_census as sc  # noqa: E402


def _q(n, f, c, mino):
    return [dict(name="NumberOfInstances", value=str(n)), dict(name="NumberOfFeatures", value=str(f)),
            dict(name="NumberOfClasses", value=str(c)), dict(name="MinorityClassSize", value=str(mino)),
            dict(name="NumberOfMissingValues", value="0")]


DATA = {
    1001: ("newbin", 2000, 11, 2, 600),        # nhị phân, bậc 0
    1002: ("newbin_skew", 2000, 11, 2, 120),   # lớp nhỏ 6% → chỉ bậc 3
    1003: ("newmulti", 3000, 21, 12, 100),     # 12 lớp → từ bậc 2
    1004: ("bigbin", 15000, 11, 2, 5000),      # n 15.000 → từ bậc 1
    1005: ("pc1", 1109, 22, 2, 200),           # cùng nguồn NASA → cờ
    1006: ("credit-g", 1000, 21, 2, 300),      # trùng tên dataset cũ → loại
    1007: ("newbin", 2000, 11, 2, 600),        # trùng tên ứng viên 1001 → duplicate_of
    31: ("credit-g", 1000, 21, 2, 300),        # dev cũ
}


def fake_get(path, tries=6):
    if path.startswith("/study/list"):
        return {"study_list": {"study": [{"id": 9001, "alias": "tabarena-v0.1", "name": "TabArena"}]}}
    if path.startswith("/study/"):
        sid = int(path.split("/")[2])
        ids = {99: [31], 9001: [1001, 1002, 1003], 14: [1004, 1005, 1006, 1007, 31]}.get(sid, [])
        return {"study": {"name": f"s{sid}", "alias": "", "data": {"data_id": ids}}}
    if path.startswith("/data/list/data_id/"):
        ids = [int(x) for x in path.split("/")[4].split(",")]
        return {"data": {"dataset": [dict(did=i, name=DATA[i][0], version=1, status="active", quality=_q(*DATA[i][1:]))
                                     for i in ids if i in DATA]}}
    if path.startswith("/data/"):
        return {"data_set_description": {"description": "real data", "original_data_url": "", "md5_checksum": "x"}}
    raise AssertionError(path)


def test_census_logic(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "get", fake_get)
    monkeypatch.setattr(sys, "argv", ["x", "--out", str(tmp_path)])
    sc.main()
    rows = {int(r["did"]): r for r in csv.DictReader(open(tmp_path / "s2_census_candidates.csv"))}
    assert 31 not in rows                                            # dev/eval cũ và CC18 bị loại theo ID
    assert rows[1001]["binary_step0"] == "True"
    assert rows[1002]["binary_step0"] == "False" and rows[1002]["binary_step3"] == "True"
    assert rows[1003]["multi_step1"] == "False" and rows[1003]["multi_step2"] == "True"
    assert rows[1004]["binary_step0"] == "False" and rows[1004]["binary_step1"] == "True"
    assert rows[1005]["same_source_flag"] == "True"
    assert rows[1006]["same_name_as_excluded"] == "True"
    assert rows[1007]["duplicate_of"] == "1001"
    summary = open(tmp_path / "s2_census_summary.txt").read()
    assert "suite 9001" in summary and "bậc 0" in summary
