# Nhật ký thực nghiệm (G7)

> Một dòng cho mỗi lần chạy, kể cả lần lỗi hoặc bị huỷ. Không xoá dòng. Kết quả thô của từng lượt ở `G7/raw/<lượt>/` (không sửa tay); mỗi dòng kết quả tự ghi mã băm dữ liệu, lần chia, nhãn yếu, code và cấu hình.

| run_id | Ngày giờ (UTC) | Commit / mã băm code | Lượt, phần | Dataset | Phương pháp | Seed | Phần cứng | Thời gian | Trạng thái (ok / lỗi / huỷ) | Kết quả thô | Ghi chú |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | |

## Môi trường
- Python 3.13 (Kaggle); thư viện chính theo `G4/code/requirements-g7.txt` (tabpfn 9.2.0, tabicl 2.2.0, torch 2.11.0, numpy 2.1.3, pandas 2.3.3, scikit-learn 1.6.1, scipy 1.16.3, cleanlab 2.9.0, sarpu commit 6e4fc3d); `pip freeze` của từng lượt lưu ở `g7_<lượt>_pip_freeze.txt`.
- PRODEN: kho chính thức, commit 637b531 (clone lúc chạy).
- GPU: Tesla T4 × 2 (15 GB mỗi GPU).

## Kiểm sau mỗi lượt
| Lượt | Số ô dự kiến | Số ô ok | Lỗi | Mã băm dữ liệu và lần chia khớp `docs/dataset_hashes.csv` | Bộ test | Ghi chú |
|---|---|---|---|---|---|---|
| | | | | | | |
