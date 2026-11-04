# Giai đoạn 0 của V2: dự đoán ghi trước (04/11/2026)

- **Căn cứ:** [V2_PLAN §6](V2_PLAN.md). Thầy duyệt toàn bộ kế hoạch ngày 04/11/2026, gồm V-Q1–V-Q5 theo đề xuất.
- **Tính chất:** đây là phân tích **thăm dò**. Dữ liệu dùng ở đây đều đã được thấy trong Study 1, nên giai đoạn này không xác nhận gì. Phép xác nhận là Study 3.
- **Thời điểm ghi:** tài liệu được commit **trước** khi viết code và trước mọi lượt chạy của giai đoạn 0. Mọi thay đổi về sau phải ghi thành deviation có ngày.

## 1. Ô thí nghiệm

- **Dữ liệu:** dataset, lần chia, nhãn yếu và seed (0, 1, 2) giống hệt Study 1 (`--split eval`). Chỉ tính các ô dưới đây.
  - **PLL q = 0,6:** 12 mfeat-factors, 28 optdigits, 40966 MiceProtein (cho P1); 1501 semeion (cho P2).
  - **LLP túi 8:** 50 tic-tac-toe, 6332 cylinder-bands (cho P3).
- **TFM:**
  - TabPFN v2 cho P1–P3.
  - TabICL v2 chạy trên cùng các ô. Kết quả này chỉ dùng cho P4 và để xem v2 có làm hại một TFM tốt không. Phần này không có dự đoán riêng.
- **Chuỗi:** cấu hình giống hệt v1: K = 10, T = 30, burn-in 10, S = 10, ngữ cảnh tối đa 2.000, cùng seed với `gibbs_warm`. Chỉ điểm xuất phát khác nhau.

## 2. Các điểm xuất phát

Mọi điểm xuất phát đều chỉ dùng nhãn yếu.

| Tên | PLL | LLP |
|---|---|---|
| `warm` (v1) | TFM ngây thơ, tính chéo 5 fold trên một ứng viên ngẫu nhiên | DLLP tuyến tính |
| `base` | CC đã tune (20 cấu hình, validation chỉ có nhãn yếu), fit trong cùng lượt chạy | DLLP-MLP đã tune, như trên |
| `cold` | y_i lấy đều trong tập ứng viên | Hoán vị ngẫu nhiên khớp số đếm |

- **Nhãn ban đầu:** y_i ~ P0_i · ℓ_i. Với LLP, mỗi túi được lấy mẫu có điều kiện theo số đếm của túi.
- **"Learner khởi động" của `base`:** chính mô hình đã fit trong lượt chạy, đánh giá trên test (phương thức `base_start`).
- Với TabPFN ở PLL, chỉ chạy `warm` và `base`, vì chuỗi TabPFN đắt. Trong các ô sụp, `cold` gần trùng với `warm`.

## 3. Đại lượng đo

- **r_final:** tỉ lệ nhãn huấn luyện của chuỗi trùng nhãn thật, ở vòng quét cuối. Đây là chẩn đoán, thuật toán không thấy nhãn thật. Đại lượng này đã có trong `monitor` của bộ chạy.
- **r_start:** tỉ lệ nhãn ban đầu đúng.
- **WPLL (độ khớp nhãn yếu, không cần nhãn thật).**
  - Gọi Q̄_i là trung bình, qua các vòng quét sau burn-in, của dự đoán ngoài khối q(· | x_i, D_{−B(i)}) mà chuỗi đã tính sẵn. Không tốn thêm forward pass nào.
  - PLL: WPLL = trung bình qua các điểm tự do của log Σ_{k ∈ S_i} Q̄_i(k).
  - LLP: WPLL = trung bình qua các túi của log P(số đếm của túi | Q̄ của các phần tử) chia cho cỡ túi. Xác suất này tính đúng bằng quy hoạch động Poisson–binomial.
- **Chuỗi được chọn (`gibbs_ms`):** điểm xuất phát có WPLL lớn nhất trong cùng ô.
- **Ánh xạ phản ứng.** Đo như S2X, với lỗi gieo đều.
  - PLL: thêm ρ ∈ {0,85; 0,90; 0,95} trên 4 dataset PLL ở q = 0,6.
  - LLP: gieo lỗi bằng cách đổi nhãn khác lớp trong cùng túi để giữ số đếm, với ρ ∈ {0,1; 0,2; 0,3; 0,45; 0,6; 0,8; 0,9}. Φ là xác suất một vòng quét bag-block gán đúng.
  - Gain(r) = Φ(r) − r, lấy trung bình qua 3 seed tại mỗi ρ.
- Ánh xạ trên lỗi có cấu trúc (Φ_struct) cũng được đo trong lượt này, phục vụ W1, nhưng **không** dùng để chấm P1–P4.

## 4. Dự đoán

**P1 (khởi động từ baseline cứu được chuỗi khi TFM có lưu vực trên).** TabPFN v2, PLL q = 0,6, điểm xuất phát `base`, trên mfeat-factors, optdigits và MiceProtein. P1 đúng khi cả hai điều sau đều đúng:
- (a) r_final ≥ 0,80 ở cả 9/9 lần chạy;
- (b) trên từng dataset, accuracy test trung bình 3 seed của `gibbs_init_base` ≥ accuracy test trung bình của `base_start` − 0,02.

**P2 (semeion, quy tắc có điều kiện theo ánh xạ phản ứng).** Gọi g85, g90, g95 là gain đo được tại ρ = 0,85; 0,90; 0,95 với TabPFN v2.
- Nếu cả ba đều < 0: dự đoán **vẫn sụp** dù khởi động từ `base`, nghĩa là r_final < 0,50 ở ít nhất 2/3 seed.
- Nếu có ít nhất một mức có gain > 0 với r đo được ≤ r_start trung bình của `base`: dự đoán **không sụp**, nghĩa là r_final ≥ 0,80 ở ít nhất 2/3 seed.
- Các trường hợp khác: không có dự đoán, P2 không được chấm.

**P3 (LLP, quy tắc có điều kiện).** TabPFN v2, túi 8, trên tic-tac-toe và cylinder-bands, điểm xuất phát `base`. Với từng dataset, gọi g* là gain tại mức ρ có r đo được gần r_start trung bình nhất, và chỉ xét khi khoảng cách ≤ 0,10.
- Nếu g* > 0: dự đoán AUC trung bình 3 seed của `gibbs_init_base` ≥ AUC của `base_start` − 0,01.
- Nếu g* < 0: dự đoán AUC của `gibbs_init_base` < AUC của `base_start` − 0,01.
- Nếu không có mức ρ nào gần đủ: không có dự đoán.

**P4 (WPLL chọn đúng điểm xuất phát).**
- Xét mọi ô có nhiều điểm xuất phát trong giai đoạn 0, cả hai TFM, cả PLL và LLP.
- Chỉ tính các ô mà chênh lệch giữa chỉ số test cao nhất và thấp nhất của các điểm xuất phát > 0,01. Chỉ số test là AUC cho LLP, accuracy cho PLL.
- Một ô được tính là chọn đúng khi chỉ số test của chuỗi được chọn cách chỉ số cao nhất không quá 0,005.
- P4 đúng khi tỉ lệ ô chọn đúng ≥ 80%.

## 5. Cổng V0

- **P1 và P4 đều đúng:** lấy nhiều điểm xuất phát cùng chọn bằng WPLL làm mặc định của v2 (W3), rồi đi tiếp giai đoạn 1.
- **P1 sai:** cơ chế ở F1 sai một phần. Quay lại W1 trước khi thiết kế Study 3.
- **P4 sai, P1 đúng:** giữ nhiều điểm xuất phát, nhưng tìm tiêu chí chọn khác trên tập phát triển trước Study 3.
- P2, P3 và Φ_struct không quyết định cổng V0. Chúng là đầu vào cho W1.
- Mọi kết quả đều được báo cáo, kể cả khi dự đoán sai.

## 6. Phân tích

Script phân tích được viết và commit **trước** khi có số: `V2/analysis/v0_report.py`. Script đọc CSV kết quả của các lượt V0A và V0B, rồi in kết luận cho P1–P4. Nó cũng tính R4 theo đúng REVIEW_RESPONSE_PLAN §3.R4, các mục (a)–(d).
