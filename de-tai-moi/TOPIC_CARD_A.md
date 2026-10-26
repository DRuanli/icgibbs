# Thẻ đề tài A (G1): ứng viên được đề xuất

**Tên tạm:** *In-Context Gibbs: học từ nhãn yếu bằng mô hình nền tảng cho dữ liệu bảng, không cần huấn luyện lại*
Tiếng Anh: *Weakly Supervised In-Context Learning: Bayesian Inference from Weak Labels with Tabular Foundation Models*

Ngày: 24/10/2026 · Trạng thái: chờ kiểm duyệt G1

---

## 0. Ý tưởng trong ba câu
Nhiều bộ dữ liệu bảng không có nhãn sạch cho từng mẫu. Có thể chỉ có tỉ lệ nhãn theo nhóm (LLP), chỉ có một phần mẫu dương được ghi nhận (PU), mỗi mẫu có một tập nhãn ứng viên (partial label), nhãn bị nhiễu, hoặc nhiều mẫu không có nhãn.

Một tabular foundation model (TFM) như TabPFN xấp xỉ posterior predictive $p(y\mid x,\mathcal D)$ khi ngữ cảnh có nhãn sạch. Dự đoán của nó cho một mẫu $i$, khi ngữ cảnh là mọi mẫu còn lại, chính là xác suất có điều kiện $p(y_i \mid y_{-i}, X)$ mà một bộ lấy mẫu Gibbs trên các nhãn ẩn cần.

Ghép xác suất này với mô hình quan sát của nhãn yếu $p(w_i\mid y_i)$, ta được suy luận Bayes cho học từ nhãn yếu, dùng một TFM có sẵn và **không cập nhật một trọng số nào**.

## 1. Phát biểu bài toán
- Dữ liệu train: $\{(x_i, w_i)\}_{i=1}^n$, với $w_i$ là quan sát yếu về nhãn ẩn $y_i$.
- **Nhãn yếu cấp mẫu** (instance-level), theo một mô hình quan sát đã biết hoặc có tham số $p(w_i \mid y_i, x_i)$:
  - PU: $w_i=1 \Rightarrow y_i=1$.
  - Partial label: $y_i\in S_i$.
  - Nhãn nhiễu: $w_i\sim T(\cdot\mid y_i)$.
  - Không nhãn: $w_i=\varnothing$.
- **Nhãn yếu cấp nhóm** (aggregate): $\sum_{i\in B} \mathbb 1[y_i=c]=k_{B,c}$, tức LLP.
- **Mục tiêu:** $p(y_*\mid x_*, X, W)=\sum_{y_{1:n}} p(y_*\mid x_*, X, y_{1:n})\,p(y_{1:n}\mid X, W)$.

## 2. Đóng góp dự kiến
1. **Phương pháp:** bộ lấy mẫu Gibbs dùng TFM làm xác suất có điều kiện, cho cả nhãn yếu cấp mẫu và cấp nhóm. Với LLP, bước Metropolis–Hastings hoán đổi nhãn giữ nguyên số đếm của nhóm.
2. **Lý thuyết:**
   - (i) **Tính đúng.** Nếu TFM trả đúng posterior predictive của một prior trao đổi được (exchangeable), chuỗi có phân phối dừng là $p(y_{1:n}\mid X,W)$, và dự đoán trung bình là posterior predictive Bayes dưới nhãn yếu.
   - (ii) **Độ bền.** Khi TFM chỉ xấp xỉ, sai lệch của phân phối dừng bị chặn theo sai số của các xác suất có điều kiện, dựa trên lý thuyết nhiễu loạn chuỗi Markov (Mitrophanov 2005). Cần xử lý cả trường hợp các xác suất có điều kiện không tương thích với nhau, như trong văn liệu imputation lặp (Liu et al. 2014).
   - (iii) **Định danh.** Điều kiện để nhãn yếu đủ xác định nhãn ẩn, tuỳ theo loại nhãn yếu.
3. **Hiệu quả tính toán:** cập nhật theo khối (fold) và cache, để chi phí mỗi vòng quét là $O(K)$ lần forward thay vì $O(n)$, kèm phân tích sai số của xấp xỉ khối.
4. **Thực nghiệm:** 5 loại nhãn yếu (LLP, PU, partial label, nhãn nhiễu, bán giám sát) × ≥10 dataset độc lập, so với baseline đã peer-review của từng loại và với các khung thống nhất GLWS và ILL.

## 3. Tám câu Heilmeier
1. **Đang cố làm gì?** Dạy máy dự đoán tốt khi người làm dữ liệu chỉ có nhãn không đầy đủ hoặc không chính xác, mà không cần huấn luyện mô hình mới cho từng bài toán.
2. **Hiện nay làm thế nào, giới hạn gì?** Mỗi loại nhãn yếu có họ phương pháp riêng: DLLP và BP cho LLP, nnPU và SAR-EM cho PU, PRODEN cho partial label. GLWS và ILL thống nhất chúng bằng EM. Tất cả **huấn luyện mạng bằng gradient**, nên yếu trên dữ liệu bảng nhỏ, cần tune nhiều, và mỗi bài toán phải huấn luyện lại. TFM thì mạnh trên bảng nhỏ nhưng **đòi nhãn sạch** trong ngữ cảnh. Chỉ có In-Context PU (2026) đưa nhãn yếu vào TFM, bằng cách pretrain một mô hình riêng cho đúng một loại nhãn yếu (PU) dưới giả định SCAR.
3. **Điều gì mới, vì sao sẽ thành công?** Dùng chính TFM có sẵn làm xác suất có điều kiện trong suy luận Bayes trên nhãn ẩn. Một cơ chế duy nhất cho mọi loại nhãn yếu có mô hình quan sát. Lý do tin nó sẽ thành công: định lý PFN (Chương 7 của khóa học) nói TFM tối ưu chính là posterior predictive; Gibbs với xác suất có điều kiện đúng là kỹ thuật chuẩn; TFM đã vượt GBDT trên bảng nhỏ khi có nhãn sạch.
4. **Ai quan tâm?** Bất kỳ ai có dữ liệu bảng với nhãn tổng hợp vì quyền riêng tư (chỉ công bố tỉ lệ theo nhóm), nhãn chỉ ghi nhận ca dương (hồ sơ bỏ học, chẩn đoán, gian lận), hoặc nhãn từ nhiều người gán không đồng thuận.
5. **Rủi ro:** xem Mục 7.
6. **Tốn bao nhiêu?** Chỉ cần suy luận TFM: GPU miễn phí (Colab, Kaggle) cho thực nghiệm, MacBook M1 cho phát triển. Không pretrain.
7. **Mất bao lâu?** Khoảng 6,5 tháng đến NeurIPS 2027 (hạn ước tính 21/05/2027).
8. **Bài kiểm tra giữa kỳ và cuối kỳ?** Mục 6.

## 4. Tiêu chí bắt buộc

| Mã | Đạt? | Bằng chứng |
|---|---|---|
| C1 ≥10 dataset độc lập | **Đạt**, có lưu ý | ≥10 bộ phân loại độc lập từ OpenML/TabArena, n ≤ 10.000 (khớp giới hạn của TFM). Nhãn yếu sinh từ nhãn thật theo cơ chế công bố kèm, đây là chuẩn của văn liệu nhãn yếu. **Lưu ý:** theo sổ tay, nhãn yếu sinh theo cơ chế là *bán tổng hợp*. Bổ sung dữ liệu nhãn yếu thật làm kiểm chứng ngoại tại: LLP-Bench (Criteo; dù 70 bộ nhưng chỉ tính là 1 nguồn) và các bộ partial label thật (Lost, MSRCv2, BirdSong, Soccer Player, Yahoo! News, Mirflickr; nhiều bộ có hơn 10 lớp nên cần TabPFN-3 hỗ trợ 160 lớp, hoặc TabICLv2) |
| C2 ≥2 baseline peer-review | **Đạt** | LLP: DLLP (ICDM 2017), Easy-LLP (NeurIPS 2023), BP-LLP (ICLR 2024). PU: nnPU (NeurIPS 2017), SAR-EM (ECML-PKDD 2019). Partial label: PRODEN (ICML 2020), IPAL (IJCAI 2015). Nhãn nhiễu: Confident Learning (JAIR 2021), forward correction (CVPR 2017). Thống nhất: **GLWS (ICML 2024)**, **ILL (NeurIPS 2024)**. Thêm baseline TFM ngây thơ (dùng thẳng nhãn yếu, hard-EM) và In-Context PU (preprint) |
| C3 Venue đích | **Đạt** | **NeurIPS 2027** (hạn ước tính 21/05/2027). Dự phòng: ICLR 2028, ICML 2028. GLWS, ILL, BP-LLP, nnPU, PRODEN đều đăng ở ICML, NeurIPS, ICLR |
| C4 Mảng cần đi sâu | **Đạt** | **Deep learning** (TFM, in-context learning của transformer) + **data-centric AI** (chất lượng và chi phí nhãn, nhãn yếu), đúng hai mảng thầy chọn. Cần thêm: MCMC (Gibbs, Metropolis–Hastings, chẩn đoán hội tụ), khoảng 2–3 tuần |
| C5 Loại trừ | **Đạt** | Đóng góp thuộc weak supervision. Không phải nhân quả, fairness, hay kiểm định |
| C6 Khả thi (cập nhật 24/10/2026 theo số đo Kaggle T4, xem [G2/05_COMPUTE_BENCHMARK.md](G2/05_COMPUTE_BENCHMARK.md)) | **Đạt với giao thức rút gọn** | Số đo thật: giao thức đầy đủ bên dưới cần khoảng 700 giờ (TabPFN) hay 505 giờ (TabICL), tức 17–23 tuần, **không vừa**. Giao thức rút gọn (3 seed, 30 vòng quét, ngữ cảnh Gibbs ≤ 2.000 hàng) cần khoảng 250–264 giờ, tức 8–9 tuần; thí điểm khoảng 3 tuần. Ước tính ban đầu để tham khảo: |
| C6 (ước tính ban đầu, đã thay) | Đạt, nhưng sát | Chỉ suy luận, không pretrain. Ước tính thô, giả định khoảng 1 giây mỗi forward trên T4 với n ≈ 2.000 (**phải đo lại ở G5**):<br>• Mỗi lần chạy: 50 vòng quét × 10 khối = 500 forward, khoảng 8 phút.<br>• Giao thức: 12 dataset × 5 loại nhãn yếu × 3 mức độ yếu × 5 seed = 900 lần chạy, khoảng 120 giờ.<br>• Baseline học sâu: khoảng 50–100 giờ.<br>• Tổng khoảng 200–250 giờ GPU; nhân 2 dự phòng là 400–500 giờ.<br>Với quota GPU miễn phí của Kaggle (khoảng 30 giờ/tuần, cần kiểm tra quota hiện hành) cộng Colab và M1, cần khoảng 3–4 tháng chạy. Sát với mốc NeurIPS 2027, nên thí điểm dùng ít seed và chỉ 2 loại nhãn yếu |
| C7 Đạo đức, pháp lý | **Đạt** | Dữ liệu công khai |

## 5. Chấm điểm
| Tiêu chí | Trọng số | Điểm | Lý do |
|---|---|---|---|
| Tính mới | 0,25 | 4 | Chỗ giao còn trống; trừ 1 điểm vì EM thống nhất (GLWS, ILL) và In-Context PU đều ở gần |
| Tầm quan trọng | 0,20 | 4 | Nhãn yếu phổ biến; giải pháp không cần huấn luyện cho bảng nhỏ là khoảng trống thực tế |
| Đóng góp lý thuyết | 0,15 | 5 | Tính đúng, chặn độ bền, định danh |
| Khớp kỹ năng | 0,15 | 5 | Nền PFN và Bayes (khóa TabPFN), kinh nghiệm với nhãn không đầy đủ |
| Rủi ro bị đi trước (5 = thấp) | 0,10 | 3 | Mảng TFM nóng; nhóm In-Context PU có thể mở rộng sang SAR |
| Phương án dự phòng | 0,10 | 4 | Bài benchmark "TFM học từ nhãn yếu tốt đến đâu" vẫn đăng được |
| Khớp venue | 0,05 | 5 | Các baseline thống nhất đều ở ICML, NeurIPS |
| **Tổng** | | **4,25** | (0,25×4 + 0,20×4 + 0,15×5 + 0,15×5 + 0,10×3 + 0,10×4 + 0,05×5) |

## 6. Mốc kiểm tra và tiêu chí dừng
- **Mốc 1 (G2, tuần 2):** quét đầy đủ. **Dừng hoặc thu hẹp** nếu tìm thấy công trình dùng TFM có sẵn để học từ ≥2 loại nhãn yếu bằng suy luận trên nhãn ẩn. Khi đó thu hẹp sang loại nhãn yếu chưa được làm, hoặc sang phần lý thuyết.
- **Mốc 2 (thí điểm, tuần 5–6):** LLP và PU trên 10 dataset. **Chuyển phương án B** nếu phương pháp không vượt *cả* baseline peer-review tốt nhất *lẫn* baseline TFM ngây thơ trên ít nhất 6/10 dataset theo metric chính (ROC AUC).
- **Mốc 3 (tuần 10):** định lý (i) có chứng minh hoàn chỉnh, qua kiểm chứng số. Định lý (ii) có ít nhất một chặn cho trường hợp các xác suất có điều kiện không tương thích.
- **Cuối kỳ:** bản thảo NeurIPS qua G10 trước 07/05/2027.

**Phương án B:** bài benchmark có hệ thống về TFM với 5 loại nhãn yếu (đích: NeurIPS Datasets & Benchmarks hoặc TMLR).

## 7. Rủi ro chính (tự phản biện)
1. **Xấp xỉ của TFM:** TFM không trả posterior predictive chính xác, và các phép biến đổi khi suy luận (ensemble, preprocessing) làm các xác suất có điều kiện không còn tương thích. Gibbs có thể trôi. Giảm rủi ro: định lý (ii), chẩn đoán hội tụ ($\hat R$, nhiều chuỗi), và so với trường hợp prior GP có đáp án chính xác (như thí nghiệm GP trong khóa học).
2. **Chi phí:** cập nhật từng mẫu một tốn $O(n)$ forward mỗi vòng quét. Giải pháp khối làm phương pháp bớt chính xác; cần đo đánh đổi.
3. **"Chỉ là EM với TabPFN":** phản biện có thể coi là phép ghép. Lý thuyết (i)–(iii) và độ phủ 5 loại nhãn yếu phải là trọng tâm, không chỉ bảng điểm.
4. **Dữ liệu bán tổng hợp:** nhãn yếu sinh từ nhãn thật. Đây là chuẩn của mảng, nhưng cần ít nhất 2 nguồn nhãn yếu thật (LLP-Bench, partial label thật) để kiểm chứng ngoại tại.
5. **Giới hạn của TFM:** khoảng ≤10.000 mẫu và ≤10 lớp với TabPFN v2. Cần TabPFN-3 (160 lớp) hoặc TabICLv2 cho partial label nhiều lớp. Giấy phép trọng số của TabPFN-3 là phi thương mại, vẫn dùng được cho nghiên cứu.
