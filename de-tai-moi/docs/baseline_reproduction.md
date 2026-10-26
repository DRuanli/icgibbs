# Tái hiện số công bố của baseline (G6)

Theo [02 §G6](../../research-handbook/02_QUY_TRINH_CONG_KIEM_DUYET.md) và [G6_PLAN](../G6/G6_PLAN.md) Mục 3.

**Tiêu chí (ghi trước khi chạy, 26/10/2026):** đạt nếu số tái hiện cách số công bố ≤ 1 điểm % (accuracy hoặc AUC), hoặc nằm trong 1 độ lệch chuẩn mà bài công bố hay đo được qua các lần lặp — lấy ngưỡng rộng hơn. Không đạt thì xếp loại: (a) có nguyên nhân rõ; (b) chưa rõ nguyên nhân.

**Ngoại lệ về dữ liệu:** nếu dataset gốc của baseline trùng tập đánh giá của dự án (Lost, MSRCv2, BirdSong) thì tái hiện trên một dataset khác trong cùng bài.

## Bảng tổng hợp

| Baseline | Code | Dataset tái hiện | Metric | Công bố | Tái hiện | Kết luận |
|---|---|---|---|---|---|---|
| SAR-EM | chính thức (`sarpu`, commit 6e4fc3d8) | Breast Cancer, Mushroom (bài gốc, 4 thuộc tính propensity, 25 thí nghiệm) | ROC-AUC, MSE của f | Mushroom 1,000 / 0,103; Breast Cancer 0,983 / 0,058 | Mushroom 1,000 / 0,098; Breast Cancer 0,994 / 0,096 | Mushroom **đạt**; Breast Cancer **lệch**, xếp loại (a) tạm thời: Naive (code gốc) lệch cùng chiều, nghi do thiết lập dữ liệu. Bản bọc trùng code gốc tuyệt đối (50/50 thí nghiệm) |

## SAR-EM (Bekker, Robberechts & Davis, ECML-PKDD 2019)

- **Script:** `G4/code/g6_reproduce_sarem.py`. Dùng đúng code tiền xử lý trong notebook của kho gốc, đúng `generate_extended_data`, `label_data` (cơ chế `simple_0.2_0.8`) và `pu_learn_sar_em`.
- **Thích ứng để chạy được trên thư viện mới.** Không đổi hành vi thuật toán:
  1. `LogisticRegressionPU` của gói gọi `LogisticRegression(multi_class=...)`, tham số đã bị bỏ khỏi scikit-learn mới. Dự án truyền một lớp tương đương vào `pu_learn_sar_em`: cùng cách đánh trọng số propensity của `BasePU`, cùng `C = 1`, solver liblinear.
  2. `generate_attributes` gán mảng 1 phần tử vào một ô; numpy ≥ 2 báo lỗi. Dự án thay bằng hàm y hệt nhưng lấy phần tử `[0]`, nên cùng dãy số ngẫu nhiên.
  3. Notebook dùng `np.NaN` (đã bị bỏ ở numpy 2) và gán số vào cột chữ (pandas 3 không cho). Dự án đặt `np.NaN = np.nan` và tắt kiểu chuỗi mới của pandas.
- **Kiểm bản bọc của dự án** (`icgibbs.baselines_sar.SAREM`): trên cùng dữ liệu, bản bọc cho đúng cùng AUC với lời gọi trực tiếp code chính thức (chênh 0 trên dữ liệu thử). Script in lại chênh lệch này trên dữ liệu thật.
- **Kiểm trên dữ liệu tổng hợp** (`tests/test_baselines_g6.py`):
  - Khi thuộc tính propensity tách khỏi thuộc tính phân loại (thiết lập của bài): AUC của SAR-EM cách oracle ≤ 0,01 và MSE của propensity < 0,02.
  - Khi z vừa là feature vừa quyết định propensity (thiết lập PU-SAR của dự án): SAR-EM hơn bộ ngây thơ > 0,05 AUC nhưng không đạt oracle (0,82 so với 0,85). Đây là giới hạn định danh đã biết của mô hình SAR khi hai phần dùng chung thuộc tính.
- **Số công bố cần so** (Hình 1 của arXiv 1809.03207v4, 4 thuộc tính propensity):

  | Dataset | Phương pháp | ROC-AUC | MSE của f |
  |---|---|---|---|
  | Breast Cancer | SAR-EM | 0,983 | 0,058 |
  | Breast Cancer | Naive | 0,992 | 0,127 |
  | Breast Cancer | Supervised | 0,994 | 0,028 |
  | Mushroom | SAR-EM | 1,000 | 0,103 |
  | Mushroom | Naive | 1,000 | 0,173 |
  | Mushroom | Supervised | 1,000 | 0,000 |

  Các số này lấy từ bản trích văn bản của hình. Cách gán cột theo thứ tự (Naive, SAR-EM, KM2, TIcE, Sup.) là suy luận của em, vì Supervised phải tốt nhất. **Cần thầy xác nhận trên hình gốc** trước khi kết luận.
- **Kết quả (26/10/2026, Kaggle, `G4/pilot/g6_sarem_reproduction.csv`):** chi tiết ở [G6_T1_SAREM_REPORT](../G6/G6_T1_SAREM_REPORT.md) Mục 2.
  - Bản bọc của dự án cho đúng cùng kết quả với code chính thức (chênh AUC = 0 trên 50/50 thí nghiệm).
  - Mushroom đạt ở mọi metric của SAR-EM.
  - Breast Cancer: SAR-EM có AUC 0,994 so với 0,983 công bố (cao hơn), MSE 0,096 so với 0,058. Naive cũng lệch cùng chiều về MSE ở cả hai dataset.
  - Giả thuyết đang kiểm: (1) tập thuộc tính của bộ phân loại — chạy lại với `--cls_mode orig_plus1` và `all`; (2) cách gán cột trong Hình 1 — cần thầy xác nhận trên hình gốc.
