# Nhật ký thực nghiệm — nghiên cứu 2 (S2)

> Mỗi lần chạy một dòng, kể cả lần lỗi hoặc huỷ. Không xoá dòng. Kết quả thô ở `S2/raw/<lượt>/`, không sửa tay.

## Đóng băng
| Ngày giờ (UTC) | Commit | Nội dung |
|---|---|---|
| 01/11/2026 15:19 | **aaee9fd** (nhánh main) | S2_PLAN, S2_DATASETS, `s2_selected.csv`, code `G4/code` (run_g7 `--split s2`, registry S2, run_s2_lot.sh). Thầy duyệt danh sách và quy tắc S1. **Trước thời điểm này chưa tải bảng dữ liệu nào của tập đánh giá S2.** Bản code gửi thầy `s2_code_v1.zip` trùng khớp với `G4/code` ở commit này (đã so từng tệp) |

## Các lượt
| run_id | Ngày giờ (UTC) | Mã băm code | Lượt | Phần cứng | Thời gian | Trạng thái | Kết quả thô | Ghi chú |
|---|---|---|---|---|---|---|---|---|
| S2X | 01/11 17:32 – 18:35 | 70 tệp khớp commit aaee9fd | S2X: ánh xạ phản ứng (thăm dò, dataset cũ của NC1) | Kaggle 2 × T4 | 1 giờ 4 phút | ok | `S2/raw/S2X/` | bộ test 112 qua, 2 bỏ qua; 588/588 dòng ok. Báo cáo: [S2X_REPORT](S2X_REPORT.md) |
| S2D | 01/11 17:35 – 17:50 | 70 tệp khớp commit aaee9fd | S2D: tải dữ liệu S2, mã băm, kiểm tiêu chí sau loại trùng | Kaggle CPU | 14 phút | ok | `S2/raw/S2D/` | bộ test 112 qua; 35/35 dataset (cả dự phòng) tải được, phiên bản khớp lúc đếm. **Trượt n ≥ 500 sau loại trùng: irish (470), maternal_health_risk (452)** |

## Kiểm sau mỗi lượt
| Lượt | Số ô dự kiến | Số ô ok | Lỗi | Mã băm dữ liệu | Bộ test | Ghi chú |
|---|---|---|---|---|---|---|
| S2D | 35 dataset (29 chọn + 6 dự phòng) | 35 | 0 | ghi lần đầu ở `raw/S2D/s2_dataset_hashes.csv` | 112 qua | **Áp S2_PLAN §3.4 bằng `s2_select.py --hashes` (ra `s2_final.csv`):** bỏ irish, maternal_health_risk; nhị phân còn 18 (≥ 10, không bổ sung); đa lớp còn 9 nên bổ sung dự phòng 1 là gas-drift (đạt tiêu chí, n = 13.910 sau loại trùng, lấy mẫu con 10.000 hàng phân tầng, seed 0, trước khi chia). Code thực hiện quy tắc đã ghi trước (`subsample_table`, `S2_SUBSAMPLE`, danh sách cuối trong registry) ở **commit 6f89823**; không đổi gì khác. Bản `s2_code_v2.zip` dựng từ đúng commit này (`git archive`) |
