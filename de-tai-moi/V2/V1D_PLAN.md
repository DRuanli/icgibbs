# Quyết định sau V1C và lượt V1D (ghi trước, 08/11/2026)

- **Thầy duyệt toàn bộ đề xuất ngày 08/11/2026** ("duyệt toàn bộ và thực hiện tiếp"). Căn cứ: [V1C_REPORT](V1C_REPORT.md) §6–§8.
- **Thứ tự:** tài liệu này được commit trước code của lượt V1D (`s3_dataset_census.py` và lượt V1D trong `run_s2_lot.sh`). Chưa có số nào của V1D.

## 1. Quyết định

| # | Quyết định | Căn cứ |
|---|---|---|
| Q1 | **v2 = A1:** 3 điểm xuất phát (warm, base, cold), mỗi chuỗi 30 vòng quét, gộp đều. Không có bản rẻ hơn | P12 sai |
| Q2 | **v2-đa-TFM:** gộp đều A1 của 4 TFM (TabPFN v2, TabICL v2, TabPFN-3, TabDPT). Vào Study 3 như một biến thể chính của v2, cùng H7 | P13 đúng |
| Q3 | **Cơ chế nhãn yếu của Study 3:** LLP túi ngẫu nhiên 8 và 32; PLL iid q = 0,6; PLL phụ thuộc mẫu với q chọn ở V1D (§2) | V1C phần G |
| Q4 | **Bỏ túi gom theo k-means.** Không thêm túi theo feature phân loại, vì cần chọn feature riêng cho từng dataset, ngoài phạm vi | V1C phần G: dễ hơn túi iid |
| Q5 | **H8 viết lại (H8′).** Bỏ yêu cầu "đo trước khi chạy chuỗi"; đó trở thành giới hạn của bài. Tín hiệu mới c3 = TV trung bình giữa Q̄_warm và Q̄_base (trên train, không cần nhãn, không tốn thêm lượt gọi TFM). H8′: AUC(c3 → T2) ≥ 0,8 ở Study 3, với T2 = v1 thua max(A1, L) quá 0,02 | P15 sai. Thăm dò `h8_explore.py` trên 276 ô cũ: AUC 0,85. Thăm dò, chưa phải bằng chứng |
| Q6 | **Nguồn dataset của Study 3:** như S2, thêm OpenML-CC18. Loại mọi dataset đã dùng và 123 dataset huấn luyện của TabDPT (`G4/code/s3_exclude_ids.csv`, 176 ID) | Nguồn của S2 sau khi loại chỉ còn ≤ 13 nhị phân và ≤ 7 đa lớp, dưới mục tiêu của Study 3 |

## 2. Lượt V1D (Kaggle 2 × T4, khoảng 2–3 giờ)

**V1D-1: đếm dataset** (`s3_dataset_census.py`, CPU). Chỉ đọc metadata OpenML. Số liệu dùng để viết protocol Study 3 trước cổng V2.

**V1D-2: hiệu chỉnh q cho PLL phụ thuộc mẫu.**
- **Cấu hình:** `--weak_variant inst`, q ∈ {0,2; 0,3; 0,4}, trên tập phát triển PLL (54, 36, 23), seed 0–2, A1 với 3 điểm xuất phát, cấu hình V1.
- **TFM:** TabICL v2 (GPU 1), TabPFN-3 (GPU 0).
- **Quy tắc chọn q, ghi trước, theo TabICL v2:**
  - Chọn q **lớn nhất** trong {0,2; 0,3; 0,4} sao cho trung bình theo dataset của (A1_inst(q) − A1 iid q 0,6 của V1) **≥ −0,10**.
  - Nếu không mức nào đạt, chọn q = 0,2.
- **Lý do:** cần mức khó nhưng còn học được. Ở q = 0,6, dataset 36 rơi từ 0,97 xuống 0,52.
- **TabPFN-3:** chỉ báo cáo.
- **Chấm:** tại đây, bằng tay theo đúng quy tắc trên. Bảng tính bằng `V2/analysis/v1c_report.py` (phần G đọc thư mục npz).

## 3. Mở rộng lượt V1D (ghi trước, 08/11/2026 tối, trước code mở rộng; V1D chưa chạy)

- **Lý do:** thầy yêu cầu V1D làm kỹ hơn và dùng GPU cho nhiều việc hơn trong một lần chạy.
- **Mục tiêu:** có đủ bằng chứng trên tập phát triển cho các quyết định Q2 (v2-đa-TFM) và Q5 (H8′), và ước lượng chi phí thật của Study 3, trước khi đóng băng protocol.
- **Cơ sở:** mọi phần dưới đây chạy trên **dữ liệu mới**, tức các cặp (TFM, ô) chưa chạy lần nào. Các dự đoán chỉ chấm trên dữ liệu đó.

### 3.1 Hạ tầng: bộ nhớ đệm cho learner L (không đổi kết quả)

- **Vấn đề:** learner khởi động L (đã tune, chạy trên CPU) cho cùng một ô thì giống hệt nhau với mọi TFM. V1 đã kiểm: 69/69 ô. Tuy vậy, mỗi TFM lại tính lại L. Ở V1C, phần này chiếm khoảng 6 giờ của GPU 1.
- **Cách làm:**
  - Thêm `run_g7.py --l_cache DIR`. Kết quả của L (xác suất trên train, trên test và chuỗi `extra`) được lưu theo khoá: mã băm của (dataset, setting, mức, seed, nhãn yếu, ma trận đầu vào train và test, cấu hình learner).
  - Thêm `--l_only`: một tiến trình CPU tính trước L cho mọi ô, song song với GPU.
- **Kiểm:** test chạy khô so khớp dự đoán và chỉ số khi trúng và khi trượt bộ nhớ đệm. Trong lượt, một số ô của V1C phần E được chạy lại; kết quả phải khớp từng số.

### 3.2 Việc thêm vào V1D

**V1D-3: TabPFN-3 và TabDPT trên các ô chung của V1.**
- **Ô:** 23 dataset có cả TabPFN v2 và TabICL ở V1, gồm LLP túi 8 (10 dataset) và PLL iid q 0,6 (13 dataset), seed 0–2, tức 69 ô.
- **Cách chạy:** 3 điểm xuất phát, cấu hình V1, chạy cho mỗi TFM. Cộng với V1 là đủ 4 TFM trên 69 ô.
- **Dự đoán P16 (v2-đa-TFM, Q2):**
  - **Gộp:** trên mỗi dataset, X4 = trung bình đều dự đoán A1 của 4 TFM, rồi lấy trung bình chỉ số theo 3 seed.
  - **So với TFM tốt nhất của từng dataset (tiên tri):** trung bình (X4 − tiên tri) **≥ −0,01**, và X4 thua quá 0,02 ở **≤ 3/23** dataset.
  - **So với từng TFM đơn lẻ:** X4 ≥ A1 của mỗi TFM, tính trung bình theo dataset.
  - **Dự đoán:** đúng.
- **Dự đoán P17 (A1 với TFM mới, H7):** với TabPFN-3 và với TabDPT, trên 23 dataset:
  - **Lợi so với v1:** trung bình (A1 − v1 của chính TFM đó) **> 0**;
  - **Thụt lùi:** số dataset có A1 < max(v1, L) − 0,02 là **≤ 2**.
  - **Dự đoán:** đúng.
- **Dự đoán P18 (H8′, Q5):**
  - **Phạm vi:** trên 138 ô mới (2 TFM × 69).
  - **Điều kiện:** AUC(c3 → T2) **≥ 0,80**, với c3 = TV trung bình giữa Q̄_warm và Q̄_base, và T2 = v1 thua max(A1, L) quá 0,02.
  - **Nếu T2 có dưới 5 ô dương tính:** báo "không đủ ô", không chấm.
  - **Dự đoán:** đúng, độ tin vừa.
- **Báo cáo kèm:**
  - thời gian GPU trung bình mỗi ô cho từng TFM, cả có và không có bộ nhớ đệm L, dùng làm cơ sở ước lượng chi phí Study 3;
  - so khớp từng số giữa các ô seed 0 ở đây và ở V1C phần E.

**V1D-4: bối cảnh khó của Study 3 với cả 4 TFM.**
- **Thứ tự:** chạy sau V1D-2, với q* đã chọn tự động trong lượt theo quy tắc §2 (script `v1d_choose_q.py`; không chọn nếu V1D-2 chưa đủ 27 ô).
- **Cấu hình:**
  - LLP iid túi 32 trên 6 dataset LLP phát triển;
  - PLL phụ thuộc mẫu ở q* trên 54, 36, 23;
  - seed 0–2, với TabPFN v2, TabPFN-3 và TabDPT. TabICL đã có: túi 32 từ V1C G, q* từ V1D-2.
- **Không có dự đoán.** Báo cáo:
  - v1, A1, L, X4;
  - các ô mà v1 có AUC < 0,5, tức chuỗi hội tụ về nhãn đảo;
  - chỉ báo cho việc có giữ các bối cảnh này ở Study 3 không.

### 3.3 Thứ tự chạy và chi phí

- **CPU:**
  - đếm dataset (vài phút);
  - rồi `--l_only` tính trước L cho các ô của V1D-3 và V1D-4. V1D-4 dùng q* nếu đã có.
- **GPU 1:** V1D-2 (TabICL, 27 ô) → chọn q* → V1D-3 TabDPT → V1D-4 TabDPT.
- **GPU 0:** V1D-3 TabPFN-3 → V1D-4 TabPFN-3 → V1D-4 TabPFN v2.
- **Ước lượng:** khoảng 18–22 giờ GPU, tức 2 phiên. Phiên 2 chạy tiếp như V1C.
- **Chấm:** `V2/analysis/v1d_report.py`, viết và commit trước khi có số.
