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
| V0A | 04/11 14:45 – 18:51 | 72 tệp khớp **commit b939041** (`s2_code_v6.zip`) | ánh xạ phản ứng (TabPFN), nhiều điểm xuất phát LLP (TabPFN), R4 (TabICL, 114 ô Study 2) | Kaggle 2 × T4 | 4 giờ 6 phút | ok | `V2/raw/V0A/` | bộ test 126 qua, 2 bỏ qua; 318/318 dòng ok, 0 lỗi. torch `2.11.0+cu130` (các lượt trước `+cu128`); kết quả vẫn tái tạo đúng từng số, nên không ảnh hưởng |
| V0B | 04/11 11:57 – 16:43 | 72 tệp khớp **commit b939041** | nhiều điểm xuất phát PLL (TabPFN, 2 GPU), rồi TabICL (PLL, LLP, ánh xạ) | Kaggle 2 × T4 | 4 giờ 46 phút | ok | `V2/raw/V0B/` | bộ test 126 qua, 2 bỏ qua; 312/312 dòng ok, 0 lỗi |

## Kiểm sau mỗi lượt
| Lượt | Số dòng dự kiến | Số dòng ok | Lỗi | Bảng / lần chia / nhãn yếu | Tính tất định | Ghi chú |
|---|---|---|---|---|---|---|
| V0A + V0B | 318 + 312 | 318 + 312 | 0 | V0 so với Study 1: 18/18 ô khớp; R4 so với Study 2: 114/114 ô khớp | `gibbs_init_warm` = `gibbs_warm` của Study 1 ở 44/44 ô (chênh 0); chuỗi a của R4 = `gibbs_warm` của Study 2 (chênh tối đa 5·10⁻⁵) | Chấm bằng `v0_report.py`, không sửa sau khi xem số. Kết quả: [V0_REPORT](V0_REPORT.md). P1 đúng, P2 sai, P3 đúng, P4 sai (79,2%); cổng V0 qua một phần. Phân tích thăm dò viết sau khi xem số: `v0_explore.py` |
