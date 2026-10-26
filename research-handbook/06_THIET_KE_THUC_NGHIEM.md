# 06 · Thiết kế thực nghiệm và phân tích (G5–G8)

[← Về README](README.md)

Mọi quyết định trong file này được ghi vào [EXPERIMENT_PROTOCOL.md](templates/EXPERIMENT_PROTOCOL.md) và **đóng băng trước khi chạy thực nghiệm chính**.

---

## 1. Dữ liệu

### 1.1 Số lượng và tính độc lập
- **Tối thiểu 10 bộ dữ liệu độc lập** (tiêu chí C1). Hai bộ là độc lập khi khác *nguồn thu thập*. Các phiên bản, mẫu con, hay biến thể target của cùng một nguồn chỉ tính là một.
- **Đa dạng có chủ đích:** trải theo kích thước (n), số chiều (d), lĩnh vực, tỉ lệ lớp, tỉ lệ thiếu dữ liệu, kiểu feature. Ghi bảng đặc trưng dataset trong bài.
- **Chọn dataset trước khi chạy**, theo tiêu chí viết sẵn (ví dụ: "mọi bộ trong kho Z có n ≤ 10.000 và d ≤ 500"). Không thêm hay bớt dataset sau khi thấy kết quả. Nếu phải bớt (lỗi dữ liệu, giấy phép) thì ghi vào *Deviations*.
- **Ưu tiên bộ benchmark chuẩn có ID cố định** (OpenML task ID, Mulan, UCI, SPMF), để split và tiền xử lý so sánh được với tài liệu.

### 1.2 Dữ liệu thật và dữ liệu tổng hợp
- **Bằng chứng chính phải đến từ dữ liệu thật.**
- Dữ liệu tổng hợp chỉ dùng cho: (a) kiểm chứng cơ chế khi cần biết đáp án đúng (ví dụ posterior chính xác, cấu trúc phụ thuộc đã biết); (b) quét có điều khiển một yếu tố (kích thước, nhiễu, mật độ); (c) huấn luyện trước nếu chính phương pháp đòi hỏi (ví dụ prior của PFN).
- Mọi kết quả từ dữ liệu tổng hợp được dán nhãn rõ trong bảng và hình.

### 1.3 Trùng lặp và nhiễm bẩn
- Loại bản ghi trùng *trong* và *giữa* train và test (rò rỉ L1.4).
- Với mô hình nền tảng được huấn luyện trước (LLM, tabular foundation model): kiểm tra dataset đánh giá có nằm trong dữ liệu huấn luyện trước hay tập phát triển của mô hình không. Nếu có, báo cáo riêng.

### 1.4 Tài liệu hoá
Mỗi dataset có [DATASET_CARD.md](templates/DATASET_CARD.md): nguồn, phiên bản hoặc ID, giấy phép, đơn vị quan sát, target, cách đo, thiếu dữ liệu, rủi ro rò rỉ, tiền xử lý.

---

## 2. Chia dữ liệu và chọn mô hình

| Cấu trúc dữ liệu | Cách chia |
|---|---|
| i.i.d., phân loại | Stratified k-fold (k = 5 hoặc 10), lặp lại với nhiều seed |
| Có nhóm (học sinh trong trường, bệnh nhân, người dùng) | **Group k-fold**: cả nhóm nằm hoàn toàn trong train hoặc test (chống L3.2) |
| Theo thời gian | Chia theo thời gian, train trước test; không xáo trộn (chống L3.1) |
| Benchmark có split chuẩn | Dùng đúng split chuẩn (ví dụ OpenML task splits) |

- **Mọi phương pháp dùng chung một bộ split.**
- **Chọn siêu tham số bằng nested cross-validation** (vòng trong để tune, vòng ngoài để đánh giá) hoặc tập validation tách riêng *bên trong* train. Không bao giờ tune trên test. Cawley & Talbot (2010) và Varma & Simon (2006) chỉ ra rằng chọn mô hình trên chính dữ liệu đánh giá gây ra thiên lệch lạc quan.
- **Pipeline:** mọi bước fit (scaler, imputer, encoder, chọn feature, giảm chiều, oversampling) đặt trong một pipeline và fit lại ở *từng* fold train.

---

## 3. Baseline: mạnh, công bằng, tái hiện được

1. **Số lượng:** ≥2 baseline đã qua peer-review cho đúng bài toán (tiêu chí C2), cộng:
   - **một baseline đơn giản** (dummy, hồi quy tuyến tính hoặc logistic, kNN), để thấy bài toán có khó thật không;
   - **baseline mạnh nhất hiện tại**, kể cả khi nó chưa qua peer-review (ví dụ preprint mạnh).

   Phản biện luôn hỏi đến baseline mạnh nhất.
2. **Dùng code chính thức.** Nếu tự cài đặt thì ghi rõ và kiểm tra tính đúng.
3. **Tái hiện trước khi so** (G6): chạy baseline trên dataset và giao thức gốc của nó, so với số đã công bố, ghi sai lệch vào `baseline_reproduction.md`.
4. **Cùng ngân sách tune:** cùng số cấu hình thử hoặc cùng thời gian, cùng không gian tìm kiếm hợp lý cho mỗi phương pháp. Melis et al. (2018) và Musgrave et al. (2020) cho thấy nhiều "cải tiến" biến mất khi baseline được tune đúng mức. Báo cáo cả bản *mặc định* lẫn bản *tune* nếu có thể. Dodge et al. (2019) khuyến nghị báo cáo hiệu năng kỳ vọng theo ngân sách tune.
5. **Cùng phần cứng và cùng dữ liệu đầu vào** khi so thời gian và bộ nhớ.

---

## 4. Rò rỉ dữ liệu: checklist 8 loại (Kapoor & Narayanan 2023)

Người kiểm duyệt G5 đánh dấu từng dòng.

| Mã | Loại rò rỉ | Kiểm tra |
|---|---|---|
| L1.1 | Không có tập test riêng | Có test tách biệt, không dùng để huấn luyện hay tune |
| L1.2 | Tiền xử lý trên toàn bộ dữ liệu | Impute, scale, oversample chỉ fit trên train của từng fold |
| L1.3 | Chọn feature trên toàn bộ dữ liệu | Chọn feature nằm trong vòng train |
| L1.4 | Bản ghi trùng giữa train và test | Đã loại trùng; báo cáo số bản ghi bị loại |
| L2 | Feature không hợp lệ | Feature nào cũng có sẵn tại thời điểm dự đoán và không phải proxy của target |
| L3.1 | Rò rỉ thời gian | Test không chứa dữ liệu trước thời điểm của train |
| L3.2 | Train và test không độc lập | Cùng nhóm (người, trường) không nằm ở cả hai phía |
| L3.3 | Lệch mẫu trong phân phối test | Test đại diện cho quần thể mà kết luận nói tới |

---

## 5. Metric

- **Chọn trước**, theo đúng câu hỏi nghiên cứu. Ghi metric chính (dùng cho khẳng định chính) và metric phụ.
- Phân loại: metric không phụ thuộc ngưỡng (ROC AUC, PR AUC khi lớp mất cân bằng), metric xác suất có tính chất *proper scoring rule* (log loss, Brier), metric phụ thuộc ngưỡng (accuracy, F1) kèm cách chọn ngưỡng, và hiệu chỉnh xác suất (calibration).
- Hồi quy: RMSE, MAE, R²; với dự đoán phân phối thì dùng NLL, CRPS.
- Bài toán đặc thù dùng metric chuẩn của cộng đồng đó (ví dụ Hamming loss, subset accuracy, F1 vi mô và vĩ mô cho multi-label; aRRMSE cho multi-target regression).
- Hiệu năng tính toán: thời gian huấn luyện, thời gian suy luận mỗi mẫu, bộ nhớ đỉnh, phần cứng (NeurIPS Q8).
- **Báo cáo điểm thô**, không chỉ điểm chuẩn hoá. Điểm chuẩn hoá theo dataset phụ thuộc vào tập phương pháp đem so.

---

## 6. Độ biến thiên và phân tích thống kê

> Kiểm định thống kê ở đây là **công cụ đánh giá bắt buộc**, không phải chủ đề nghiên cứu (tiêu chí C5 chỉ loại nó như là chủ đề).

### 6.1 Nguồn biến thiên (Bouthillier et al. 2021)
Kết quả dao động theo: cách chia dữ liệu, khởi tạo trọng số, thứ tự dữ liệu, augmentation, dropout, và quá trình tune. Ngẫu nhiên hoá **càng nhiều nguồn càng tốt**. Tăng số split thường có ích hơn chỉ tăng số seed. Báo cáo rõ khoảng biến thiên (error bar) phản ánh nguồn nào, và là std hay standard error (NeurIPS Q7).

### 6.2 Chọn kiểm định

| Tình huống | Khuyến nghị |
|---|---|
| 2 phương pháp trên N dataset | **Wilcoxon signed-rank** trên điểm từng dataset (Demšar 2006). Báo cáo thêm thắng/hoà/thua |
| k > 2 phương pháp trên N dataset | Friedman, sau đó **so sánh cặp bằng Wilcoxon kèm hiệu chỉnh Holm**. Benavoli et al. (2016) cảnh báo: post-hoc dựa trên mean-rank (như Nemenyi) cho kết luận về A và B phụ thuộc vào các phương pháp khác có mặt trong so sánh. Biểu đồ critical difference chỉ dùng để minh hoạ |
| Muốn biết xác suất "thực tế tương đương" | Bayesian signed-rank với ROPE (vùng tương đương thực tế, ví dụ ±0,01 AUC) (Benavoli et al. 2017) |
| 2 phương pháp trên **một** dataset | Corrected resampled t-test cho lặp CV (Nadeau & Bengio 2003) hoặc 5×2cv (Dietterich 1998). Không dùng t-test thường trên các fold vì các fold không độc lập |

### 6.3 Quy tắc
- Viết kế hoạch kiểm định ở G5: kiểm định nào, cho khẳng định nào, mức α, hiệu chỉnh đa so sánh.
- Báo cáo **effect size** (chênh lệch trung bình và khoảng tin cậy), không chỉ p-value.
- p > 0,05 **không** có nghĩa là "tương đương". Muốn khẳng định tương đương thì dùng ROPE hoặc kiểm định tương đương.
- Với ít dataset (N < 10), Wilcoxon có lực yếu: thêm dataset quan trọng hơn thêm kiểm định.

---

## 7. Ablation, độ nhạy, phân tích lỗi

- **Ablation:** bỏ từng thành phần của phương pháp, đo thay đổi. Mỗi khẳng định "thành phần X giúp cải thiện" cần một dòng ablation.
- **Độ nhạy siêu tham số:** quét từng siêu tham số quan trọng trên một dải, báo cáo phương pháp có ổn định không.
- **Khả năng mở rộng:** theo n, d, và tham số đặc thù.
- **Phân tích lỗi:** phương pháp thua ở dataset nào, vì sao (đặc trưng dataset, giả định bị vi phạm). Đây là phần phản biện đánh giá cao, và là nguồn của mục Limitations.
- **Kết quả âm:** báo cáo dataset và cấu hình mà phương pháp thua.

---

## 8. Mối đe doạ tính hợp lệ (Wohlin et al. 2012)

Mục bắt buộc trong bài:

| Loại | Câu hỏi |
|---|---|
| Hợp lệ nội tại (internal) | Cải thiện có thật do phương pháp không, hay do tune nhiều hơn, cài đặt khác, rò rỉ? |
| Hợp lệ ngoại tại (external) | Kết quả tổng quát ra ngoài các dataset đã dùng tới đâu? |
| Hợp lệ cấu trúc (construct) | Metric có đo đúng điều khẳng định không? |
| Hợp lệ kết luận (conclusion) | Kiểm định đúng chưa, đủ lực chưa, có đa so sánh không? |

---

## 9. Ghi chép trong lúc chạy (G7)

- Mỗi lần chạy lưu: `run_id`, commit hash, file config, seed, môi trường (phiên bản thư viện), phần cứng, thời gian bắt đầu và kết thúc, đường dẫn kết quả thô.
- Kết quả thô ở mức (dataset, phương pháp, fold, seed) lưu dạng CSV hoặc Parquet trong `results/raw/`. **Không sửa tay.**
- Bảng và hình trong bài được sinh bằng script từ `results/raw/`. Không copy số bằng tay.
- Lần chạy lỗi hay bị huỷ vẫn được ghi trong [EXPERIMENT_LOG.md](templates/EXPERIMENT_LOG.md), kèm lý do.
