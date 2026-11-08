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
