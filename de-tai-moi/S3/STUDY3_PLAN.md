# Study 3 (bản gọn): xác nhận v2 trên dataset mới — protocol ghi trước (10/11/2026)

- **Trạng thái:** bản đề xuất, chờ thầy duyệt. Thầy chạy lượt S3 tức là duyệt và **đóng băng** protocol này.
- **Thời điểm ghi:** tài liệu này được commit trước code của lượt S3, trước script chấm, và trước khi tải bất kỳ bảng dữ liệu nào của Study 3.
- **Phạm vi:** thầy chọn bản gọn ngày 10/11/2026: dừng phát triển ở V1D, bỏ lượt V1E.
- **Tính chất:** **xác nhận**. Không cấu hình nào của phương pháp được chỉnh trên các dataset này.

## 1. Câu hỏi

Trên dataset chưa từng dùng để phát triển, có ba câu hỏi:
1. v2 (trung bình đều 3 chuỗi) có hơn v1 (1 chuỗi) không?
2. Gộp nhiều TFM có hơn việc dùng một TFM không?
3. Tín hiệu chẩn đoán H8′ có nhận ra được các ô mà v1 thất bại không?

## 2. Dataset: 20 dataset, đóng băng

- **Nguồn đếm:** `V2/raw/V1D/s3_census_candidates.csv`. Nguồn này gồm các suite như S2, thêm OpenML-CC18, và chỉ đọc metadata.
- **Loại trước khi đếm (176 ID):**
  - mọi dataset đã dùng ở Study 1, Study 2, tập phát triển và V2;
  - 123 dataset huấn luyện của TabDPT.
- **Tiêu chí:** các bậc 0–3 của S2_PLAN §3.2, lấy mọi ứng viên đạt tới bậc 3.
- **Rà tay:** `S3/s3_review.csv`, theo đúng quy tắc của S2:
  - phải là dữ liệu thật;
  - cùng nguồn với dataset đã dùng thì loại;
  - nhiều ứng viên cùng nguồn thì giữ ID nhỏ nhất.
  
  Việc rà chỉ dựa trên metadata và mô tả.
- **Kết quả rà:** 43 ứng viên, giữ 20, loại 23:
  - 8 nhân tạo;
  - 14 cùng nguồn với dataset đã dùng;
  - 1 trùng nguồn giữa các ứng viên.

| Nhóm | Bối cảnh | Dataset (ID OpenML) |
|---|---|---|
| Nhị phân, 12 | LLP, túi ngẫu nhiên cỡ 8 | 1038 gina_agnostic, 1046 mozilla4, 1467 climate-model-simulation-crashes, 1487 ozone-level-8hr, 4134 Bioresponse, 46916 coil2000, 46924 E-Commerce shipping, 46935 HR analytics, 46937 in-vehicle coupon, 46947 online shoppers, 46950 polish bankruptcy, 46956 seismic-bumps |
| Đa lớp, 8 | PLL, ứng viên gieo đều, q = 0,6 | 42 soybean, 307 vowel, 1468 cnae-9, 1478 har, 1515 micro-mass, 4538 GesturePhaseSegmentation, 40499 texture, 46933 hiva_agnostic |

- **Lấy mẫu con:** dataset có n > 10.000 (1046, 46924, 46935, 46937, 46947, 1478) được lấy mẫu con 10.000 hàng, phân tầng, seed 0, sau khi loại trùng và trước khi chia. Cách làm như S2.
- **Kiểm sau khi tải** (tự động, ở đầu lượt chạy, `g6_dataset_hashes.py --split s3`):
  - Dataset nào trượt tiêu chí bậc 3 sau khi loại trùng (n < 500, lớp nhỏ nhất nhị phân < 5%, …) thì **bị loại khỏi phân tích**, có ghi lý do.
  - Không bổ sung dataset thay thế.
- **Cờ S1** (cấu trúc nhóm hoặc chuỗi): 307, 1467, 1478, 1487, 4538. Mọi kiểm định chính được chạy lại khi bỏ 5 dataset này. Nếu kết luận khác nhau thì báo cáo cả hai, và lấy kết luận yếu hơn.
- **Giới hạn đã biết:**
  - Nhóm đa lớp chỉ có 8 dataset, dưới mức 10 của S2. Vì vậy các kiểm định chính **gộp cả hai nhóm**, n = 20; kết quả theo từng nhóm chỉ là mô tả.
  - 42, 1515 và 46933 từng nằm trong danh sách dự phòng của S2. Bảng của chúng đã được tính mã băm, nhưng chưa chạy phương pháp nào.
  - Không kiểm được các dataset TabArena có được dùng khi phát triển TabPFN-3 hay TabICL không.

## 3. Phương pháp và cấu hình (không đổi so với V1)

- **Cấu hình chuỗi:** K = 10 khối, 30 vòng quét, burn-in 10, dùng 10 mẫu, ngữ cảnh tối đa 2.000.
- **Số vòng lặp:** 3 seed (0, 1, 2) cho lần chia và nhãn yếu.
- **v1:** một chuỗi, điểm xuất phát `warm`.
- **Learner L:** learner khởi động đã tune, là điểm xuất phát `base`. L dùng làm mốc.
- **v2 (A1):** trung bình đều dự đoán của 3 chuỗi `warm`, `base`, `cold`, cùng một TFM.
- **v2-đa-TFM (X):** trung bình đều dự đoán A1 của mọi TFM chạy được trên dataset đó.
- **TFM và giới hạn công bố:**

| TFM | Chạy trên |
|---|---|
| TabICL v2, TabPFN-3, TabDPT | cả 20 dataset |
| TabPFN v2 | chỉ dataset có d ≤ 500 và C ≤ 10: 10 nhị phân (trừ 1038, 4134) và 1 đa lớp (4538) |

- **Thất bại khi chạy:** một cặp (TFM, dataset) được tính nếu đủ cả 3 seed không lỗi. Cặp lỗi bị bỏ khỏi mọi phép tính của dataset đó và được báo cáo. Dataset còn dưới 2 TFM thì bị loại khỏi H2.
- **Chỉ số:** AUC cho LLP, accuracy cho PLL, trên tập test (30%), lấy trung bình theo seed.

## 4. Giả thuyết và kiểm định

**Chính** (α = 0,05, hiệu chỉnh Holm trên H1 và H2; Wilcoxon dấu hạng một phía trên dataset):
- **H1 (v2 hơn v1).** Với mỗi dataset, δ₁ = trung bình trên các TFM chạy được của (A1 − v1). Giả thuyết: δ₁ > 0.
- **H2 (đa-TFM hơn một TFM).** Với mỗi dataset, δ₂ = X − trung bình trên các TFM chạy được của A1. Giả thuyết: δ₂ > 0. Số trừ là kỳ vọng khi chọn ngẫu nhiên một TFM.

**Tiêu chí ghi trước, không dùng p:**
- **H3 (H8′).** AUC(c3 → T2) **≥ 0,80**, gộp mọi ô (TFM, dataset, seed).
  - c3 = TV trung bình giữa Q̄ của chuỗi `warm` và của chuỗi `base` trên train.
  - T2 = v1 thua max(A1, L) quá 0,02 trên ô đó.
  - Báo cáo kèm khoảng tin cậy bootstrap theo dataset.
  - Nếu T2 có dưới 5 ô dương tính: kết luận "không đủ ô".
- **C4 (thụt lùi của v2).** Tỉ lệ cặp (TFM, dataset) có A1 < max(v1, L) − 0,02 là **≤ 10%**. Mốc từ tập phát triển: 2/78 ở V1 và 4/46 ở V1D.
- **C5 (đa-TFM so với tiên tri).** Trung bình (X − A1 của TFM tốt nhất trên dataset) **≥ −0,01**, và X thua quá 0,02 ở **≤ 3/20** dataset. Tiêu chí như P16.

**Báo cáo kèm:**
- chênh trung bình, khoảng tin cậy bootstrap 95%, số thắng/hoà/thua (hoà khi |chênh| < 0,001);
- kết quả theo từng TFM và từng nhóm;
- A1 so với L (Wilcoxon hai phía, chỉ mô tả);
- phân tích độ nhạy S1;
- số ô v1 có AUC < 0,5 (nhãn đảo);
- thời gian chạy.

**Dự đoán của em** (ghi trước, không ảnh hưởng tới kiểm định):
- H1 đúng, độ tin vừa: lợi trung bình trên tập phát triển chỉ +0,003 đến +0,012.
- H2 đúng.
- H3 đúng, độ tin vừa.
- C4 đạt; C5 đạt.

## 5. Điều không làm

Các việc sau thành **future work** trong bài:
- PLL phụ thuộc mẫu, và LLP túi 32;
- so với baseline ngoài L (đã làm ở Study 1–2 cho v1);
- chẩn đoán đo trước khi chạy chuỗi;
- mở rộng Mệnh đề 5 sang PU và PLL;
- đếm dataset trên toàn bộ OpenML.

## 6. Cách chạy và chấm

- **Lượt S3** (Kaggle 2 × T4, ước 2–3 phiên, chạy tiếp được):
  - **CPU:** kiểm mã băm dataset, rồi tính trước learner L.
  - **GPU:** 4 TFM, mỗi TFM 3 chuỗi cho mỗi ô.
- **Chấm:** `S3/s3_report.py`, viết và commit trước khi có số, đã thử trên dữ liệu giả.
- **Sau khi chạy:**
  - không chạy lại để "sửa" kết quả;
  - mọi sai lệch so với protocol này phải ghi thành deviation có ngày.
