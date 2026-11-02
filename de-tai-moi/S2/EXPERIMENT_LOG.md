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
| S2T | 02/11 02:36 – 04:12 | 70 tệp khớp **commit aaee9fd** (`s2_code_v1.zip`, danh sách trước kiểm dữ liệu) | S2T: phương pháp TFM (TabICL v2) | Kaggle 2 × T4 | 1 giờ 36 phút | ok | `S2/raw/S2T/` | bộ test 112 qua; 585 dòng ok (29 dataset của danh sách cũ). **Chưa có gas-drift**, vì bản v1 chưa bổ sung |
| S2C1 | 02/11 02:36 – 13:14 | như S2T | S2C1: baseline mảnh 0–3/8 | Kaggle CPU | 10 giờ 38 phút | ok | `S2/raw/S2C1/` | 538 dòng ok |
| S2C2 | 02/11 02:36 – 11:41 | như S2T | S2C2: baseline mảnh 4–7/8 | Kaggle CPU | 9 giờ 5 phút | ok | `S2/raw/S2C2/` | 512 dòng ok |

## Kiểm sau mỗi lượt
| Lượt | Số ô dự kiến | Số ô ok | Lỗi | Mã băm dữ liệu | Bộ test | Ghi chú |
|---|---|---|---|---|---|---|
| S2D | 35 dataset (29 chọn + 6 dự phòng) | 35 | 0 | ghi lần đầu ở `raw/S2D/s2_dataset_hashes.csv` | 112 qua | **Áp S2_PLAN §3.4 bằng `s2_select.py --hashes` (ra `s2_final.csv`):** bỏ irish, maternal_health_risk; nhị phân còn 18 (≥ 10, không bổ sung); đa lớp còn 9 nên bổ sung dự phòng 1 là gas-drift (đạt tiêu chí, n = 13.910 sau loại trùng, lấy mẫu con 10.000 hàng phân tầng, seed 0, trước khi chia). Code thực hiện quy tắc đã ghi trước (`subsample_table`, `S2_SUBSAMPLE`, danh sách cuối trong registry) ở **commit 6f89823**; không đổi gì khác. Bản `s2_code_v2.zip` dựng từ đúng commit này (`git archive`) |
| S2T + S2C1 + S2C2 | 29 dataset của danh sách trước kiểm dữ liệu: TFM 117 ô × 5 = 585; baseline 57 × 10 + 60 × 8 = 1.050 | 585; 1.050 | 0 | 1.635/1.635 dòng khớp S2D (bảng và lần chia); nhãn yếu của TFM và baseline trùng nhau ở mọi ô | 112 qua | Các lượt chạy bằng **bản code đóng băng (aaee9fd)**. Code phương pháp giống hệt bản v2/v3: từ aaee9fd đến nay chỉ đổi danh sách dataset, thêm hàm lấy mẫu con và lượt S2G. Kết quả của irish và maternal_health_risk có trong thô nhưng **bị bỏ khi phân tích** theo §3.4. Thiếu gas-drift → lượt **S2G** (`s2_code_v3.zip`). 2 ô AUC đúng 0,5: `dllp_tun`, `llpvat_tun` ở eye_movements seed 1 (dự đoán hằng; kết quả hợp lệ của phương pháp, giữ nguyên) |
