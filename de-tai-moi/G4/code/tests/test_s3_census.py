"""Study 3: danh sách loại trừ của bước đếm dataset (không gọi mạng)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import s3_dataset_census as S3  # noqa: E402


def test_excluded_ids_cover_used_and_tabdpt():
    ex = S3.load_excluded()
    assert len(ex) == 176
    for d in (12, 1501, 40966, 28, 54, 36, 23, 1464):          # đã dùng
        assert ex[d] == "used"
    assert ex.get(24) == "used" and 40536 in ex                 # Study 2 ∩ TabDPT: ghi là đã dùng
    assert sum(r == "tabdpt_pretrain" for r in ex.values()) == 123 - 13


def test_list_all_openml_paginates(monkeypatch):
    pages = {0: [dict(did=i, name=f"d{i}", version=1, status="active",
                      quality=[dict(name="NumberOfInstances", value="600")]) for i in range(3)],
             3: [dict(did=99, name="BNG(x)", version=1, status="active", quality=[])]}
    calls = []

    def fake_get(path):
        calls.append(path)
        off = int(path.rsplit("/offset/", 1)[1])
        return {"data": {"dataset": pages.get(off, [])}} if off in pages else None
    monkeypatch.setattr(S3.S2, "get", fake_get)
    out = S3.list_all_openml(limit=3)
    assert sorted(out) == [0, 1, 2, 99] and len(calls) == 2
    assert out[0]["q"]["NumberOfInstances"] == "600"
    import re
    assert any(re.search(p, "bng(x)") for p in S3.ARTIFICIAL)
