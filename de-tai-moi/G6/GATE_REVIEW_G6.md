# Biên bản kiểm duyệt cổng G6

Cổng: **G6 — Code đúng, baseline đã kiểm** · Dự án: đề tài A, *In-Context Gibbs* · Ngày lập: 29/10/2026

- Người tạo sản phẩm: Claude (code, tài liệu), thầy (chạy 9 lượt trên Kaggle)
- Người kiểm duyệt (không tham gia tạo sản phẩm): **thầy** ☐ Người ☐ Agent AI. Phần toán đã có kiểm duyệt độc lập riêng ([PROOFS_REVIEW_G6](PROOFS_REVIEW_G6.md)).

Mục 3a là **tự rà soát của người tạo**, không thay kiểm duyệt độc lập. Người kiểm duyệt điền cột cuối của Mục 2, Mục 3b và Mục 4.

## 1. Sản phẩm được kiểm
| File | Ghi chú |
|---|---|
| `G4/code/` (kho git `DRuanli/icgibbs`, nhánh main) | bộ chạy G7 `run_g7.py`, `icgibbs/g7/`, baseline, 105 test |
| [docs/baseline_reproduction.md](../docs/baseline_reproduction.md) | số công bố so với số tái hiện, xếp loại |
| [docs/primary_comparators.md](../docs/primary_comparators.md) | đối thủ chính, đã chốt mọi bối cảnh |
| [docs/tuning_spaces.md](../docs/tuning_spaces.md), [docs/dataset_hashes.csv](../docs/dataset_hashes.csv) | không gian tune; mã băm 33 dataset |
| [G7_RUNNER_DESIGN](G7_RUNNER_DESIGN.md) | thiết kế bộ chạy, ablation đã định rõ |
| Báo cáo lượt 2–8: [ROUND2](G6_ROUND2_REPORT.md), [ROUND45](G6_ROUND45_REPORT.md), [ROUND6](G6_ROUND6_REPORT.md), [ROUND7](G6_ROUND7_REPORT.md), [ROUND8](G6_ROUND8_REPORT.md); [T1 SAR-EM](G6_T1_SAREM_REPORT.md) | kết quả trên tập phát triển |
| `G4/code/requirements-g7.txt` | phiên bản thư viện đã chốt |

## 2. Tiêu chí qua cổng (02_QUY_TRINH_CONG_KIEM_DUYET §G6)
| Tiêu chí | Người tạo tự đánh giá | Người kiểm duyệt |
|---|---|---|
| Mọi test qua | **Đạt**: 105 qua, 2 bỏ qua (cần kho ngoài) | ☐ |
| Mỗi baseline tái hiện trong sai lệch cho phép, hoặc chênh lệch được giải thích và xếp loại | **Đạt có điều kiện.** Đạt: PRODEN 8/8; RC, CC ở Texture, Har; SAR-EM ở Mushroom. Lệch, xếp loại (b): SAR-EM Breast Cancer; RC, CC ở Yeast, Dermatology. Chưa tái hiện số, xếp loại (a), thay bằng đối chiếu code: GLWS/ILL, LLP-VAT, DLLP, EasyLLP và các baseline phụ — số công bố chỉ có trên ảnh hoặc trong hình. Xem Phát hiện 1, 2 | ☐ |
| Người kiểm duyệt chạy được từ clone sạch | **Chưa** — đề xuất làm ngay đầu G7: lượt G7 đầu tiên chạy bộ test từ bản code mới tải lên Kaggle, môi trường mới; dừng nếu có test hỏng (Phát hiện 3) | ☐ |
| Phương pháp không sai trên ví dụ đồ chơi (điều kiện dừng) | Đạt: RQ1 (G4), kiểm số E1–E6, kiểm duyệt toán G6 | ☐ |

## 3a. Phát hiện — tự rà soát
| # | Mức | Mô tả | Đề xuất | Trạng thái |
|---|---|---|---|---|
| 1 | Lớn | CC, đối thủ chính của H4, lệch số công bố ở 2/4 bộ UCI (−3 tới −5 điểm), xếp loại (b) | Ghi rõ trong bài; báo cáo thêm H4 so với từng baseline PLL còn lại (phân tích phụ đã có trong giao thức Mục 6) | Đã ghi |
| 2 | Lớn | Đối thủ chính của H2-LLP (DLLP tuyến tính) và GLWS/ILL không có số công bố tái hiện được trên dữ liệu bảng | Đối chiếu code đã làm (loss chép nguyên văn, E-step khớp vét cạn); bài nêu là giới hạn | Đã ghi |
| 3 | Nhỏ | Chưa chạy từ clone sạch bởi người khác | Lượt G7 đầu tiên chạy bộ test trước khi chạy bất kỳ ô nào | Đề xuất |
| 4 | Lớn | Dưới xử lý đúng giao thức, lợi thế LLP túi 8 so với DLLP tuyến tính chỉ +0,003 AUC | Thầy đã quyết giữ là khẳng định chính, kỳ vọng ghi trước; tiêu chí thành công viết lại (H2-LLP hoặc H4-PLL) | Đã ghi (giao thức §11) |
| 5 | Nhỏ | LLP-Bench chưa có dữ liệu (Criteo) | Hoãn sang cuối G7; không có dữ liệu thì bỏ và ghi deviation | Đề xuất ([G7_PLAN](../G7/G7_PLAN.md)) |
| 6 | Nhỏ | Lượt 8, 9 gửi về không kèm thư mục code đã chạy nên không `cmp` được | Mỗi lượt G7 ghi mã băm code (`code=` trong từng dòng) và kèm thư mục code | Đã có trong bộ chạy |

## 3b. Phát hiện — người kiểm duyệt
| # | Mức | Mô tả | Đề xuất | Trạng thái |
|---|---|---|---|---|
| | | | | |

## 4. Quyết định
☐ **Qua** ☐ **Sửa** (quay lại G__) ☐ **Dừng**

Chữ ký người kiểm duyệt: ____________ · Ngày: ____________
