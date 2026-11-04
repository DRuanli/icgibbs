# Nhật ký thực nghiệm — V2

> Mỗi lần chạy một dòng, kể cả lần lỗi hoặc huỷ. Không xoá dòng. Kết quả thô ở `V2/raw/<lượt>/`, không sửa tay.

## Ghi trước và đóng băng
| Ngày giờ (UTC) | Commit | Nội dung |
|---|---|---|
| 04/11/2026 08:00 | **1a8859f** (main) | Thầy duyệt V2_PLAN (V-Q1–V-Q5 theo đề xuất). Ghi trước P1–P4 và cổng V0 ([PHASE0_PREDICTIONS](PHASE0_PREDICTIONS.md)) **trước khi viết code và trước mọi lượt chạy** của giai đoạn 0 |
| 04/11/2026 08:28 | **b939041** (main) | Code giai đoạn 0: `run_g7.py --part v0`, `--explore`/`--trace_dir` cho R4, `v0_response_map.py`, `collect_q`/`weak_fit` trong sampler (chỉ ghi lại, không đổi chuỗi; có test), lượt V0A/V0B trong `run_s2_lot.sh`; script phân tích viết trước `V2/analysis/v0_report.py`. Bộ test 123 qua, 5 bỏ qua (máy em, CPU). Bản gửi thầy `s2_code_v6.zip` dựng bằng `git archive` từ đúng commit này |

## Các lượt
| run_id | Ngày giờ (UTC) | Mã băm code | Lượt | Phần cứng | Thời gian | Trạng thái | Kết quả thô | Ghi chú |
|---|---|---|---|---|---|---|---|---|
| V0A | — | — | ánh xạ phản ứng (TabPFN), nhiều điểm xuất phát LLP (TabPFN), R4 (TabICL) | Kaggle 2 × T4 | — | chờ thầy chạy | `V2/raw/V0A/` | |
| V0B | — | — | nhiều điểm xuất phát PLL (TabPFN, 2 GPU), rồi TabICL | Kaggle 2 × T4 | — | chờ thầy chạy | `V2/raw/V0B/` | |
