# S2E: R̂ và khối có sửa MH với TabICL v2 (thăm dò; 04/11/2026)

- **Mục đích:** G9_PLAN O3 (R̂ với TabICL) và O4 (khối có sửa MH với TabICL). Thầy duyệt D3 ngày 03/11.
- **Phạm vi:** chỉ trên dataset cũ của NC1, tức tập phát triển của S2; không chạm tập đánh giá S2. **Đây là kết quả thăm dò, không phải kiểm định.**
- **Script phân tích:** `S2/analysis/s2e_report.py`, viết trước khi có kết quả (commit 7d7b5ad). Kết quả ở `S2/results/s2e_*`.
- **Lượt chạy:** S2E, 03/11, Kaggle 2 × T4.
  - Code khớp commit 7d7b5ad (70/70 tệp).
  - Bộ test: 113 qua, 2 bỏ qua.
  - 71 ô ok, 0 lỗi.
  - **Thiếu 7/12 ô** của phần "khối có sửa MH, ≥ 8 lớp, q 0,6" (Mục 3).

## 1. R̂ (O3): TabICL cũng chưa trộn hết ở mức nhãn, nhưng dự đoán của hai chuỗi gần nhau

Mỗi ô chạy hai chuỗi Gibbs ấm, khác seed lấy mẫu. Đặt cạnh số của TabPFN v2 ở NC1, cùng 3 dataset nhỏ mỗi nhóm.

| Nhóm, bối cảnh | Ô | Mẫu có R̂ > 1,1 (trung vị; lớn nhất): TabICL | Như trên: TabPFN (NC1) | Chênh metric test giữa 2 chuỗi (trung vị; lớn nhất): TabICL | Như trên: TabPFN |
|---|---|---|---|---|---|
| 3 dataset nhỏ, LLP túi 8 | 9 | **28%** (35%) | 11% | **0,018** (0,098) | 0,041 |
| 3 dataset nhỏ, PLL q 0,3 | 9 | 6% (11%) | 6% | 0,018 (0,032) | 0,006 |
| 3 dataset nhỏ, PLL q 0,6 | 9 | **31%** (46%) | 21% | 0,021 (0,030) | 0,016 |
| 4 dataset ≥ 8 lớp, PLL q 0,6 | 12 | 22% (58%, MiceProtein) | — | **0,002** (0,040) | — |

- **Cách đọc mốc:** khi trộn hoàn hảo, tỉ lệ mẫu có R̂ > 1,1 không quá khoảng 4,5% (hiệu chỉnh ở G6).
- **TabICL chưa trộn hết ở mức nhãn từng mẫu.** Ở LLP và PLL q = 0,6, tỉ lệ còn cao hơn TabPFN.
- **Dự đoán của hai chuỗi vẫn rất gần nhau:** trung vị chênh ≤ 0,021. Trên bốn dataset ≥ 8 lớp, trung vị chỉ 0,002.
- **Kết luận giống NC1:** 30 vòng quét cho dự đoán ổn định, chứ không cho mẫu từ posterior đã hội tụ ở mức nhãn. Câu này trong bài nay có số cho cả hai TFM.

## 2. Khối có sửa MH (O4): bản "chính xác" không cho dự đoán tốt hơn, và tốn hơn nhiều

So ghép cặp với `gibbs_warm` TabICL v2 của lượt L6 (NC1): cùng ô, cùng seed, cùng điểm khởi động, cùng 30 vòng quét. Chênh dương nghĩa là bản có sửa MH tốt hơn.

| Nhóm, bối cảnh | Ô (dataset) | MH − xấp xỉ, metric chính | Tốt hơn / hoà / kém | Log loss: MH tốt hơn | Tỉ lệ chấp nhận MH trung bình |
|---|---|---|---|---|---|
| 3 dataset nhỏ, LLP túi 8 (AUC) | 9 (3) | +0,033 (−0,018 tới +0,133) | 6/0/3 | −0,064 (kém hơn) | 0,68 |
| 3 dataset nhỏ, PLL q 0,3 (accuracy) | 9 (3) | −0,009 | 3/0/6 | +0,006 | 0,78 |
| 3 dataset nhỏ, PLL q 0,6 | 9 (3) | −0,021 (eucalyptus −0,05 tới −0,10) | 2/1/6 | −0,199 | 0,72 |
| ≥ 8 lớp, PLL q 0,6 | 5 (2) | −0,016 (optdigits −0,047, −0,022) | 0/0/5 | −0,284 | 0,47 (optdigits 0,12–0,17) |

- **Không có lợi thế hệ thống cho bản chính xác.**
  - LLP nhỉnh hơn về AUC nhưng kém về log loss. Phần nhỉnh chủ yếu đến từ dresses-sales, nơi chuỗi xấp xỉ có AUC 0,47–0,60, gần ngẫu nhiên.
  - PLL kém hơn, rõ nhất ở q = 0,6.
  - Với TabPFN ở NC1, kết luận tương tự: +0,010 AUC (4/5) và +0,006 accuracy.
- **Cách hiểu.** Ở cùng 30 vòng quét, bản MH từ chối nhiều đề xuất khối. Trên optdigits chỉ 12–17% đề xuất được chấp nhận, nên chuỗi di chuyển chậm hơn. Có hai hiệu ứng không tách được bằng phép so này:
  - (a) độ lệch của bản xấp xỉ;
  - (b) độ trộn chậm hơn của bản chính xác.

  Thêm nữa, "chính xác" chỉ có nghĩa dưới (A1). Với một TFM thật, các xác suất có điều kiện không tương thích, nên tỉ số MH tính qua bổ đề Brook phụ thuộc đường đi, và chuỗi MH cũng không nhắm đúng một posterior nào. Điều này khớp với ablation chuyển nạp của NC1: bản Bayes chính xác (theo prior của TFM) kém bản quy nạp.
- **Chi phí:** bản MH chậm hơn bản xấp xỉ trung vị 7,2 lần. Trên mfeat-factors chậm hơn 22 lần (45 phút so với 2 phút); trên optdigits chậm hơn 200 lần (khoảng 4 giờ so với 71 giây).
- **Kết luận thực hành:**
  - Bộ lấy mẫu khối không sửa là lựa chọn đúng cho thực hành: không kém về dự đoán, rẻ hơn một đến hai bậc độ lớn.
  - Khoảng trống lý thuyết vẫn còn: cấu hình mặc định không có chặn khác rỗng (PROOFS G4-5). Bài đã nói rõ điều này.

## 3. Ô thiếu

- **Phần "khối có sửa MH, ≥ 8 lớp, q 0,6" chỉ xong 5/12 ô:**
  - xong: mfeat-factors 3 seed, optdigits seed 0 và 1;
  - thiếu: optdigits seed 2, semeion 3 seed, MiceProtein 3 seed.
- **Lý do, suy từ thời gian trong log:** lượt bắt đầu 05:16 UTC. Hai ô optdigits mỗi ô mất khoảng 4 giờ, nên lượt dừng quanh mốc 12 giờ của phiên Kaggle. Log không có dòng "kết thúc", và không ô nào báo lỗi.
- **Đề xuất: không chạy bù.**
  - Bảy ô còn lại cần khoảng 7–10 giờ GPU.
  - Năm ô đã có cùng chiều với các nhóm khác (0/5 tốt hơn).
  - Đây là phân tích thăm dò, không có tiêu chí đạt hay không đạt.
  - Bài báo cáo đúng 5 ô và ghi rõ thiếu 7 ô vì giới hạn thời gian.
  - Nếu thầy muốn đủ: chạy lại lượt S2E bằng `s2_code_v5.zip` và chép kết quả cũ vào; các ô đã xong được bỏ qua.
