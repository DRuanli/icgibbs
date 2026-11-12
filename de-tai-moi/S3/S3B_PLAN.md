# Lượt bổ sung S3B trên dataset của Study 3 — kế hoạch ghi trước (12/11/2026)

- **Trạng thái:** thầy đã duyệt ngày 12/11/2026 ("đồng ý hết với các phản biện, chạy cả CPU và GPU").
- **Thời điểm ghi:** tài liệu này và script chấm `S3/s3b_report.py` được commit **trước** khi chạy lượt S3B.
- **Tính chất:** **bổ sung, ngoài protocol của Study 3**. Các dataset này đã được dùng để chấm Study 3 và em đã thấy mọi số của Study 3. Vì vậy:
  - kết quả S3B **không** thay đổi kết luận nào của Study 3 (H1″, H2″, H3″, C4″, C5″);
  - trong bài, S3B được ghi là "supplementary analysis, registered after Study 3 was scored";
  - phương pháp của bài **không** được chỉnh theo kết quả S3B.
- **Lý do:** các biên bản `G9/reviews/REVIEW_V2_NEURIPS_01.md` và `REVIEW_V2_ROUND2.md` nêu ba thiếu hụt mà reviewer thật sẽ hỏi.

## 1. Ba câu hỏi

| Mã | Câu hỏi | Phần chạy |
|---|---|---|
| B1 | Bản gộp nhiều TFM so với baseline **không dùng mạng nơ-ron** thì sao? | CPU |
| B2 | Trên dataset của Study 3, chuỗi đơn so với TFM ngây thơ, hard-EM và oracle thế nào? | GPU |
| B3 | Chạy chuỗi dài hơn (100 vòng quét) có sửa được các ô chuỗi kẹt không? | GPU |

## 2. Dữ liệu và cấu hình chung

- Đúng 20 dataset, cùng lần chia, cùng nhãn yếu, seed 0, 1, 2 của Study 3 (STUDY3_PLAN §2–§3). Script chấm kiểm `sha256_split` và `weak_hash` khớp với lượt S3; ô không khớp bị loại và báo cáo.
- Chỉ số: AUC cho LLP túi 8, accuracy cho PLL q = 0,6; thêm log loss ở B2.
- Số của single chain (v1), multi-start (A1), multi-TFM (X) và L lấy **nguyên** từ lượt S3 (`S3/results/`), không chạy lại.

## 3. B1 — baseline CPU (cài đặt riêng, cấu hình cố định, không tune)

| Baseline | Bối cảnh | Mô tả |
|---|---|---|
| PL-kNN | PLL | bỏ phiếu k = 10 láng giềng gần nhất trên tập ứng viên (Hüllermeier và Beringer, 2006) |
| GBDT tự huấn luyện | PLL | HistGradientBoosting (100 cây, tốc độ học 0,1); nhãn giả bị ràng trong tập ứng viên; 3 vòng gán lại bằng dự đoán ngoài fold |
| GBDT tự huấn luyện | LLP | điểm số đầu là hồi quy tỉ lệ dương của túi; nhãn giả của mỗi túi là c điểm có điểm số cao nhất; 3 vòng |

- **Không tune:** cả ba chạy đúng một cấu hình, viết trước khi có số. Đây là giới hạn của so sánh và sẽ được nêu trong bài.
- **Kiểm định** (mô tả, α = 0,05, Wilcoxon dấu hạng **hai phía** trên dataset, Holm trên 3 phép so):
  1. X so với GBDT tự huấn luyện, gộp LLP và PLL (20 dataset);
  2. X so với PL-kNN (8 dataset PLL);
  3. L so với baseline CPU tốt hơn của từng bối cảnh (20 dataset), để biết baseline nơ-ron đã tune có yếu hơn baseline cây không.
- **Báo cáo kèm:** chênh trung bình, khoảng tin cậy bootstrap, thắng/hoà/thua (hoà khi |chênh| < 0,001), số dataset X kém quá 0,02; cùng các số đó cho chuỗi đơn trung bình trên TFM.

## 4. B2 — TFM ngây thơ, hard-EM, oracle

- Bốn TFM, đúng phạm vi của Study 3 (TabPFN v2 chỉ trên 11 dataset trong giới hạn công bố).
- Phương pháp: `naive`, `hard_em_warm`, `oracle` của `run_g7.py --part tfm`, cấu hình như Study 1–2.
- **Kiểm định** (một phép, hai phía): chuỗi đơn so với hard-EM về **log loss**, theo dataset (trung bình trên TFM chạy được). Đây là hướng của H3 ở Study 1–2.
- **Mô tả:** theo từng TFM và từng bối cảnh: chuỗi đơn − naive; chuỗi đơn − hard-EM (chỉ số chính); khoảng cách tới oracle của chuỗi đơn, của A1 và của X (X so với oracle tốt nhất trong các TFM).

## 5. B3 — chuỗi dài

- PLL, 8 dataset × 3 seed, với TabPFN-3, TabDPT và TabICL v2 (72 ô): một chuỗi `warm`, **100 vòng quét**, burn-in 33 (quy ước `--sweeps` của V1C), các thiết lập khác không đổi. TabPFN v2 chỉ có một dataset PLL nên không chạy.
- **Ô kẹt** = ô có T2 = 1 ở lượt S3 (chuỗi đơn 30 vòng kém max(A1, L) quá 0,02).
- **Dự đoán ghi trước của em (P22):**
  - trong các ô kẹt, chuỗi 100 vòng "sửa được" (đạt trong vòng 0,02 của max(A1, L)) ở **dưới 25%** số ô;
  - trên cả 72 ô, lợi trung bình của 100 vòng so với 30 vòng **dưới 0,03**.
  
  Nếu P22 sai thì kết luận của bài về "lưu vực" phải viết yếu đi, và câu "more sweeps do not repair a start in the wrong basin" phải sửa.
- **Mô tả:** độ đúng nhãn cuối ở vòng 30 và vòng 100 của cùng chuỗi; số ô còn đang tăng (chênh trung bình 5 vòng cuối so với 5 vòng quanh vòng 30 lớn hơn 0,03).
- Nếu lượt chạy không kịp hết 72 ô: chấm trên các TFM đã đủ 24 ô; TFM chưa đủ được báo là "chưa chạy".

## 6. Điều không làm trong S3B

- Ánh xạ phản ứng và ước lượng δ trên Study 3.
- Mức q khác 0,6, cỡ túi khác 8, thêm dataset.
- Tune baseline CPU; IPAL; ensemble của L; fine-tuning TFM.

Các việc này giữ ở phần hạn chế của bài.

## 7. Cách chạy và chấm

- Lượt `S3B` của `run_s2_lot.sh` (Kaggle 2 × T4, chạy tiếp được, ước 2 phiên): CPU chạy B1 song song với hai hàng GPU chạy B2 rồi B3.
- Chấm bằng `S3/s3b_report.py`, đúng một lần trên dữ liệu đủ. Xem tạm giữa chừng (nếu có) phải ghi lại như ở Study 3.
- Mọi sai lệch so với kế hoạch này ghi thành deviation có ngày.
