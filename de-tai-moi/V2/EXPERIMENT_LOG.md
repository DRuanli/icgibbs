# Nhật ký thực nghiệm — V2

> Mỗi lần chạy một dòng, kể cả lần lỗi hoặc huỷ. Không xoá dòng. Kết quả thô ở `V2/raw/<lượt>/`, không sửa tay.

## Ghi trước và đóng băng
| Ngày giờ (UTC) | Commit | Nội dung |
|---|---|---|
| 04/11/2026 08:00 | **1a8859f** (main) | Thầy duyệt V2_PLAN (V-Q1–V-Q5 theo đề xuất). Ghi trước P1–P4 và cổng V0 ([PHASE0_PREDICTIONS](PHASE0_PREDICTIONS.md)) **trước khi viết code và trước mọi lượt chạy** của giai đoạn 0 |
| 04/11/2026 08:28 | **b939041** (main) | Code giai đoạn 0: `run_g7.py --part v0`, `--explore`/`--trace_dir` cho R4, `v0_response_map.py`, `collect_q`/`weak_fit` trong sampler (chỉ ghi lại, không đổi chuỗi; có test), lượt V0A/V0B trong `run_s2_lot.sh`; script phân tích viết trước `V2/analysis/v0_report.py`. Bộ test 123 qua, 5 bỏ qua (máy em, CPU). Bản gửi thầy `s2_code_v6.zip` dựng bằng `git archive` từ đúng commit này |

| 05/11/2026 02:10 UTC | **7ec273b** (main) | Thầy duyệt V0_REPORT §6 (giai đoạn 1 và đưa R4 vào bài). Ghi trước giai đoạn 1 ([PHASE1_PLAN](PHASE1_PLAN.md)): dataset, ba bộ gộp, quy tắc chọn, P5–P7, **trước khi viết code và trước mọi lượt chạy** |

| 05/11/2026 02:34 UTC | **a8e355c** (main) | Code giai đoạn 1: `qbar()` trong sampler; `base_oof` và lưu npz trong `run_multistart`; `run_g7.py --save_dir --oof`, `--explore` cho v0 trên S2; lượt V1A/V1B; `tests/test_v1.py` (bộ test 127 qua, 5 bỏ qua trên CPU). Script chấm viết trước `V2/analysis/v1_aggregate.py` (thử trên dữ liệu giả rồi xoá). Bản gửi thầy `s2_code_v7.zip` dựng bằng `git archive` từ commit này |

## Các lượt
| run_id | Ngày giờ (UTC) | Mã băm code | Lượt | Phần cứng | Thời gian | Trạng thái | Kết quả thô | Ghi chú |
|---|---|---|---|---|---|---|---|---|
| V0A | 04/11 14:45 – 18:51 | 72 tệp khớp **commit b939041** (`s2_code_v6.zip`) | ánh xạ phản ứng (TabPFN), nhiều điểm xuất phát LLP (TabPFN), R4 (TabICL, 114 ô Study 2) | Kaggle 2 × T4 | 4 giờ 6 phút | ok | `V2/raw/V0A/` | bộ test 126 qua, 2 bỏ qua; 318/318 dòng ok, 0 lỗi. torch `2.11.0+cu130` (các lượt trước `+cu128`); kết quả vẫn tái tạo đúng từng số, nên không ảnh hưởng |
| V0B | 04/11 11:57 – 16:43 | 72 tệp khớp **commit b939041** | nhiều điểm xuất phát PLL (TabPFN, 2 GPU), rồi TabICL (PLL, LLP, ánh xạ) | Kaggle 2 × T4 | 4 giờ 46 phút | ok | `V2/raw/V0B/` | bộ test 126 qua, 2 bỏ qua; 312/312 dòng ok, 0 lỗi |
| V1A | — | — | giai đoạn 1, TabPFN v2: LLP 14 + PLL 13 dataset, 3 chuỗi, npz, ngoài fold | Kaggle 2 × T4 | — | chờ thầy chạy | `V2/raw/V1A/` | |
| V1B | — | — | giai đoạn 1, TabICL v2: như V1A thêm 28 dataset Study 2 | Kaggle 2 × T4 | — | chờ thầy chạy | `V2/raw/V1B/` | |
| W1-sim | 05/11, xong 02:53 | commit của report này | TFM giả biết Φ (`w1_drift_check.py`), 720 chuỗi | CPU (máy em) | 8 phút | ok | `G4/results/w1/` | thăm dò, không có số test thật |
| R1a-E7a | 05/11 02:49 | như trên | TC khối chính xác (`r1a_block_bias.py tc`), 80 ô | CPU | < 1 phút | ok | `G4/results/r1a/e7a_block_tc.csv` | |
| R1a-E7b | 05/11 02:55 – 02:59 | như trên | Định lý 2 với nhân khối chính xác, lưới n ≤ 1000 | CPU | 4 phút | **huỷ** | — | n = 500–1000 quá chậm và chỉ ghi CSV ở cuối; dừng, sửa thành ghi sau mỗi ô và bỏ n = 1000 |
| R1a-E7b | 05/11 02:59 – 03:08 | như trên | như trên, n ∈ {50, 100, 200, 500} | CPU | khoảng 9 phút | ok | `G4/results/r1a/e7b_*` | lần chạy trước đó trong cùng phiên dùng nhân tham chiếu sai (bỏ mẫu có nhãn trong khối khỏi ngữ cảnh, không bất biến với π); đã sửa trước lượt này, kiểm bất biến ≤ 1e-12 |
| R1a-E7c, E7d | 05/11 03:08 – 03:10 | như trên | mô-men của π̃ so với π; phân rã chặn | CPU | < 2 phút | ok | `G4/results/r1a/e7c_*`, `e7d_*` | thăm dò |
| W1-delta | 05/11 | như trên | δ trên TFM thật, chỉ đọc dữ liệu V0 (`V2/analysis/w1_delta.py`) | CPU | < 1 phút | ok | `V2/results/w1_delta.csv` | thăm dò |

## Kiểm sau mỗi lượt
| Lượt | Số dòng dự kiến | Số dòng ok | Lỗi | Bảng / lần chia / nhãn yếu | Tính tất định | Ghi chú |
|---|---|---|---|---|---|---|
| V0A + V0B | 318 + 312 | 318 + 312 | 0 | V0 so với Study 1: 18/18 ô khớp; R4 so với Study 2: 114/114 ô khớp | `gibbs_init_warm` = `gibbs_warm` của Study 1 ở 44/44 ô (chênh 0); chuỗi a của R4 = `gibbs_warm` của Study 2 (chênh tối đa 5·10⁻⁵) | Chấm bằng `v0_report.py`, không sửa sau khi xem số. Kết quả: [V0_REPORT](V0_REPORT.md). P1 đúng, P2 sai, P3 đúng, P4 sai (79,2%); cổng V0 qua một phần. Phân tích thăm dò viết sau khi xem số: `v0_explore.py` |
