# G2 · Các công trình gần nhất và khác biệt

Ma trận đầy đủ: [related_work_matrix.csv](related_work_matrix.csv) (36 công trình; 22 đã peer-review). Đọc theo phương pháp ba lượt (Keshav 2007): 3 bài ở lượt 3, 8 bài ở lượt 2.

## 1. Bảng định vị

| | Mô hình có sẵn, không huấn luyện lại | Dữ liệu bảng | Nhiều loại nhãn yếu | Suy luận trên nhãn ẩn có lý thuyết hợp lệ | Mô hình hoá phụ thuộc giữa các nhãn ẩn |
|---|---|---|---|---|---|
| In-Context PU (arXiv 2026) | ✗ (pretrain PFN riêng) | ✓ | ✗ (chỉ PU, SCAR) | ✗ (dự đoán trực tiếp) | — |
| GLWS (ICML 2024) | ✗ (huấn luyện gradient) | ✗ (chỉ ảnh) | ✓ (11 loại) | Một phần (EM, posterior chính xác qua NFA) | ✗ (giả định độc lập có điều kiện) |
| ILL (NeurIPS 2024) | ✗ | ✗ | ✓ (PLL, SSL, NLL, kết hợp) | Một phần (EM) | ✗ |
| Count-based WSL (NeurIPS 2023) | ✗ | — | ✓ (các loại nhãn dạng đếm) | Một phần | ✗ |
| Sepin (*Stats* 2026) | ✓ (TabPFN) | ✓ | ✗ (chỉ outcome thiếu, MCAR/MAR) | ✗ (tác giả nói rõ không có tuyên bố hợp lệ) | ✗ (một lượt, không lặp) |
| Noise Immunity (arXiv 2026) | ✓ | ✓ | ✗ (lật nhãn đối xứng) | ✗ (chỉ đo độ bền) | — |
| In-Context SSL (arXiv 2025) | ✗ (pretrain) | ✗ | ✗ (chỉ SSL) | ✗ | — |
| **Đề tài A** | **✓** | **✓** | **✓ (LLP, PU-SAR, PLL, nhiễu, SSL)** | **✓ (định lý tính đúng và độ bền)** | **✓ (qua ngữ cảnh của TFM)** |

## 2. Đọc sâu và câu "khác ở đâu"

### In-Context PU (Liu et al., arXiv 2605.05591), lượt 3
- **Làm gì:** pretrain từ đầu một PFN kiểu TabPFN (khoảng 1,23 triệu tham số, 6 khối, khoảng 15 giờ trên 2 GPU H100, khoảng 4,8 triệu dataset tổng hợp) trên prior PU. Mẫu dương có nhãn dùng embedding nhãn; mẫu không nhãn dùng chung một token học được.
- **Giả định:** SCAR (mẫu dương được gán nhãn hoàn toàn ngẫu nhiên); ≤20 feature; tỉ lệ âm π trong [0,1; 0,9].
- **Kết quả:** AUC trung bình 0,882, so với 0,841 của baseline tốt nhất, trên 20 tác vụ PU bán tổng hợp.
- **Chỗ hở:**
  - Không so với TabPFN có sẵn (ví dụ coi mẫu không nhãn là âm) hay với EM dùng TFM có sẵn.
  - Future work của chính họ: cơ chế chọn khác SCAR, và **"mở rộng sang các dạng nhãn yếu khác"**.
- **Khác ở đâu:** ta không pretrain. Ta dùng TFM tổng quát có sẵn làm xác suất có điều kiện trong suy luận Bayes trên nhãn ẩn, nên SAR chỉ là đổi mô hình quan sát $p(w\mid y,x)$ chứ không phải pretrain lại. Ta phủ 5 loại nhãn yếu bằng một cơ chế.
- **Rủi ro:** nhóm này có thể ra bản mở rộng. **Đây là đối thủ cạnh tranh trực tiếp nhất.**

### GLWS (Chen et al., ICML 2024), lượt 3
- **Làm gì:** EM thống nhất cho 11 loại nhãn yếu. E-step tính posterior trên các cách gán nhãn bằng forward-backward trên một automaton (NFA) mô tả nhãn yếu, đưa độ phức tạp về tuyến tính.
- **Giả định:** dự đoán độc lập có điều kiện khi biết input (Assumption 3.1).
- **Thực nghiệm:** chỉ ảnh (CIFAR, STL, ImageNet-100), huấn luyện bằng gradient.
- **Chỗ hở:** nhãn yếu có nhiễu để làm sau; không có dữ liệu bảng.
- **Khác ở đâu:** E-step của GLWS dùng dự đoán từng mẫu *độc lập*. Ở ta, xác suất có điều kiện của mỗi nhãn ẩn phụ thuộc nhãn ẩn của các mẫu khác qua ngữ cảnh của TFM. Đây là Gibbs trên phân phối đồng thời, không phải EM với posterior tách rời. Hơn nữa ta không có M-step huấn luyện mạng. **Cơ chế NFA của GLWS có thể dùng lại** để tính phần ràng buộc tổ hợp (ví dụ đếm trong túi LLP) trong bước Gibbs theo khối.

### TabPFN (Hollmann et al., *Nature* 2025), lượt 3 (khóa học)
- **Vai trò:** xương sống. Định lý PFN: TFM tối ưu chính là posterior predictive. Đây là giả định trung tâm của định lý (i) trong đề tài A.
- **Giới hạn liên quan:** ≤10.000 mẫu và ≤10 lớp (v2); ensemble và các phép tiền xử lý khi suy luận làm dự đoán thực tế lệch khỏi posterior predictive của prior. Đây là động cơ của định lý (ii).

### Sepin (*Stats* 2026), lượt 2
- Rút độc lập từ phân phối dự đoán của TabPFN để multiple imputation outcome; một lượt, không lặp. Tác giả viết rõ không có tuyên bố hợp lệ hình thức cho Rubin's rules.
- **Khác ở đâu:** ta lặp (Gibbs), điều kiện hoá trên nhãn đã gán của các mẫu khác, đưa mô hình nhãn yếu vào, và chứng minh tính hợp lệ.
- **Nên trích dẫn** như bằng chứng rằng "rút nhãn từ PPD của TabPFN" đã được dùng trong thống kê ứng dụng.

### ILL (Chen et al., NeurIPS 2024), lượt 2
- EM với tổng chính xác trên các lớp, không lấy mẫu nhãn ẩn; ảnh và văn bản; nhiễu không phụ thuộc mẫu.
- **Khác ở đâu:** giống phần GLWS. Thêm: ILL không có LLP và PU.

### Focused PU (Zavitsanos & Paliouras, arXiv 2605.14467), lượt 2
- Ghi rõ: "tích hợp tabular foundation model là future work, vì mô hình đông cứng không có rủi ro thực nghiệm (empirical risk) riêng cho từng tác vụ". **Đây là bằng chứng độc lập cho khoảng trống.** Cách tiếp cận Bayes của ta không cần risk estimator.
- Cung cấp 14 dataset PU mất cân bằng và một danh sách baseline PU cho dữ liệu bảng (uPU, nnPU, NNIF, PUHDT, PUHRF, LBE, SAR-EM).

## 3. Hệ quả cho thiết kế (chuyển sang G3 và G5)
1. **Baseline bắt buộc bổ sung:**
   - In-Context PU (đối thủ trực tiếp, nếu code công khai);
   - "TabPFN ngây thơ": PU coi không nhãn là âm; LLP gán nhãn đa số của túi; PLL chọn ngẫu nhiên trong tập ứng viên;
   - "TabPFN + hard-EM": phân biệt đóng góp của *lấy mẫu Bayes* với của *lặp nhãn giả*.
2. **Ablation bắt buộc:** Gibbs (phụ thuộc giữa các nhãn qua ngữ cảnh) so với EM với posterior từng mẫu độc lập kiểu GLWS. Đây là phép thử trực tiếp cho đóng góp khác biệt với GLWS.
3. **Nguồn nhãn yếu thật** (kiểm chứng ngoại tại): LLP-Bench (Criteo), ClinVar (nhãn nhiễu thật, theo bài 2609.32976), dữ liệu báo cáo tài chính trễ (PU thật, theo Focused PU; cần kiểm tra quyền truy cập).
