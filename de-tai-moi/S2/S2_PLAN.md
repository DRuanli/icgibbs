# Nghiên cứu 2 (S2): xác nhận In-Context Gibbs với TabICL v2 trên dữ liệu mới — kế hoạch (bản nháp 30/10/2026)

**Quyết định (thầy, 30/10/2026):** tập trung vào bài phương pháp, tức hướng ban đầu: In-Context Gibbs, nhắm NeurIPS 2027.

**Lý do cần một nghiên cứu mới:**
- Nghiên cứu 1 (NC1, G5–G8) dùng TabPFN v2 làm TFM chính và không đạt H2-LLP, H4-PLL ([G8_REPORT](../G8/G8_REPORT.md)).
- Ablation TFM dùng TabICL v2 đạt mọi kiểm định. Nhưng đây là kết quả chọn sau khi đã xem số, nên chỉ là thăm dò (giao thức §11, deviation 16).
- S2 kiểm đúng phát hiện thăm dò đó, đăng ký trước, trên dataset chưa từng dùng.

**Trạng thái:** bản nháp. Bản này đóng băng bằng git tag `s2-frozen` **trước khi tải bất kỳ bảng dữ liệu nào của tập đánh giá S2**. Bước đếm dataset (§3) chỉ đọc metadata.

## 0. Nguyên tắc
1. **Bài báo cáo cả hai nghiên cứu.** NC1 báo cáo đầy đủ, kể cả phần không đạt và phân tích lỗi. Không có kết quả nào của NC1 bị bỏ.
2. **Mọi lựa chọn của S2 viết ra trước khi chạm dữ liệu mới.** Gồm dataset, phương pháp, đối thủ, kiểm định, ngưỡng. Sau khi đã thấy kết quả S2 thì không đổi gì.
3. **33 dataset cũ của NC1** (6 + 3 dataset phát triển, 24 dataset đánh giá) trở thành **tập phát triển của S2**. Mọi việc dò, thử, phát triển phương pháp hay lý thuyết đều làm trên đây. Tập này có đủ dataset ≥ 8 lớp, lấp được điểm mù của NC1 (GATE_REVIEW_G8, phát hiện 3).
4. Tài liệu và code của S2 nằm ở `de-tai-moi/S2/`. Bộ chạy dùng lại `G4/code/` (chỉ thêm, không sửa hành vi đã đóng băng của NC1).

## 1. Giả thuyết: giữ nguyên NC1, chỉ đổi TFM

| | Phát biểu | Kiểm định | Tiêu chí |
|---|---|---|---|
| **H2-LLP′** | LLP túi 8: Gibbs ấm (TabICL v2) vượt baseline peer-review tốt nhất và TFM ngây thơ (TabICL v2) | Wilcoxon hai phía, AUC trung bình 3 seed mỗi dataset; so với DLLP tuyến tính mặc định và TFM ngây thơ; Holm trên 2 | cả hai có ý nghĩa sau Holm theo hướng Gibbs tốt hơn, **và** Gibbs thắng ≥ 70% dataset trước baseline có AUC trung bình cao nhất trên tập đánh giá S2 |
| **H4-PLL′** | PLL q = 0,3 và 0,6: như trên | Wilcoxon hai phía, accuracy; so với CC tune và TFM ngây thơ ở hai mức; Holm trên 4 | cả 4 có ý nghĩa sau Holm, **và** ở **mỗi mức**, Gibbs ≥ baseline peer-review có accuracy trung bình cao nhất ở mức đó trên ≥ 60% dataset. Cách đọc này viết rõ để tránh chỗ mơ hồ của NC1 (deviation 15); các cách đọc khác chỉ báo cáo kèm |
| **H3′** | Gibbs tốt hơn hard-EM (cùng TFM, cùng điểm khởi động) | Bayesian signed-rank, ROPE ±0,005 log loss, đơn vị là ô (dataset × mức) của LLP túi 8 và PLL | như NC1: ≥ 70% ô; bác bỏ nếu P(ROPE) > 0,9 hoặc ρ ≤ 0 |

- **Thành công của S2:** (H2-LLP′ hoặc H4-PLL′) cùng H3′. Thành công mạnh: cả hai.
- **Độ nhạy kiểu S1:** khi rà tay (§3.3), gắn cờ các dataset có cấu trúc nhóm/chuỗi hoặc nhãn sinh từ quy tắc. Mọi kiểm định chạy hai lần, có và không có các dataset này. Nếu kết luận khác nhau, dùng kết luận yếu hơn.
- **Không chạy** PU, SSL, nhãn nhiễu. NC1 đã trả lời các bối cảnh này (D3). Bỏ chúng giảm chi phí và số phép so sánh.
- Đối thủ chính giữ nguyên NC1 (`dllplin_def`, `cc_tun`). Hai đối thủ này được chọn bằng quy tắc trên tập phát triển cũ, không dựa vào kết quả.

## 2. Phương pháp và baseline: không đổi gì so với lượt L6 của NC1
- **TFM:** TabICL v2. Checkpoint `tabicl-classifier-v2-20260212.ckpt`, tabicl 2.2.0, `support_many_classes=True`, `random_state=0`. 1 bộ ước lượng trong chuỗi, 2 khi dự đoán cuối (cấu hình F6).
- **Chuỗi:**
  - Khởi động ấm: DLLP tuyến tính cho LLP; TFM cross-fit cho PLL.
  - K = 10, 30 vòng quét, bỏ 10 vòng đầu, Rao–Blackwell 10 mẫu.
  - Ngữ cảnh ≤ 2.000, dự đoán quy nạp.
- **Các phương pháp TFM chạy cùng:** TFM ngây thơ, hard-EM, SEM, oracle.
- **Baseline:** mọi baseline LLP và PLL của NC1, bản mặc định và bản tune, cùng không gian tune và ngân sách (docs/tuning_spaces.md), cùng quy tắc `refit_best`.
- Tiền xử lý, loại trùng, chia 80/20 phân tầng, seed 0–2, sinh nhãn yếu: giữ nguyên giao thức NC1 §2–§3.
- **Không đưa cải tiến nào vào phép thử chính.** Phép thử chính kiểm đúng cấu hình đã tạo ra phát hiện thăm dò. Các cải tiến ở §5 chỉ là nhánh phụ, và nhánh nào được đưa vào S2 phải ghi ở đây trước khi đóng băng.

## 3. Tập đánh giá mới

### 3.1 Nguồn và loại trừ
- **Nguồn:** các bộ chuẩn tuyển chọn ngoài CC18:
  - OpenML100 (study 14);
  - AMLB phân loại (271);
  - Grinsztajn et al. 2022 (334, 337);
  - TabZilla (379);
  - mọi suite trên OpenML có alias chứa "tabarena", "amlb", "tabzilla", "tabrepo".
- **Loại trừ:**
  - mọi dataset của CC18;
  - 33 dataset của NC1 cùng mọi version hay bản trùng tên;
  - các dataset cùng nguồn với NC1 hoặc với nhau (giao thức NC1 §1.1, C1: mỗi nhóm nguồn chỉ lấy một, là dataset có ID nhỏ nhất).

### 3.2 Tiêu chí và thang nới (ghi trước khi chạy script đếm)

| Bậc | n (trước loại trùng) | d | Đa lớp: số lớp | Nhị phân: lớp nhỏ nhất |
|---|---|---|---|---|
| **0** (đúng NC1 §1.1) | 500–10.000 | ≤ 500 | 3–10 | ≥ 10% |
| 1 | 500–20.000 (lấy mẫu con 10.000 hàng, phân tầng, seed cố định, trước khi chia) | ≤ 500 | 3–10 | ≥ 10% |
| 2 | như bậc 1 | ≤ 500 | 3–20 | ≥ 10% |
| 3 | như bậc 1 | ≤ 2.000 | 3–20 | ≥ 5% |

- Mọi bậc đều yêu cầu dữ liệu thật, không phải bài toán nhân tạo.
- **Quy tắc chọn bậc:** dùng bậc thấp nhất cho được **≥ 10 dataset nhị phân và ≥ 10 dataset đa lớp** sau khi rà tay. Nhị phân và đa lớp có thể dừng ở hai bậc khác nhau.
- **Lấy những dataset nào:** lấy mọi dataset đạt tiêu chí. Nếu nhiều hơn 20 thì lấy 20 dataset có ID OpenML nhỏ nhất (quy tắc trung lập, giống cách NC1 chọn đại diện nhóm nguồn).
- **Điều kiện dừng:** nếu tới bậc 3 vẫn không đủ 10 mỗi loại, **S2 không chạy theo thiết kế này**. Em báo thầy và đề xuất thiết kế thay thế. Ví dụ kiểm định Bayes phân cấp trên seed, hoặc chỉ một bối cảnh. Thiết kế thay thế cũng phải quyết trước khi chạm dữ liệu.
- **Lý do không giảm xuống 5 dataset:** với n = 5, p nhỏ nhất của Wilcoxon hai phía là 0,063, nên kiểm định không thể đạt. Cần n ≥ 8 nếu thắng mọi dataset; n ≥ 10 nếu muốn chịu được 1–2 lần thua.

### 3.3 Rà tay (chỉ dựa trên metadata và mô tả)
1. Script `G4/code/s2_dataset_census.py` liệt kê ứng viên kèm cờ: trùng tên, cùng nguồn, mô tả.
2. Em đề xuất cho từng ứng viên: dữ liệu thật hay không, nhóm nguồn, cờ S1.
3. **Thầy duyệt.**
4. Kết quả ghi vào `S2/S2_DATASETS.md` rồi đóng băng.

Bước 1–4 không tải bảng dữ liệu nào.

### 3.4 Kiểm sau đóng băng
- Một lượt Kaggle CPU tải dữ liệu, ghi mã băm (bảng, lần chia) vào `S2/s2_dataset_hashes.csv` và kiểm lại tiêu chí sau khi loại trùng.
- Dataset nào trượt tiêu chí sau loại trùng (ví dụ n < 500) thì bị loại theo đúng tiêu chí, ghi lý do. Không thay bằng dataset khác.

## 4. Thống kê, báo cáo, chạy
- Thống kê như NC1 Mục 6: α = 0,05, Holm, chênh trung bình kèm CI bootstrap 95%, thắng/hoà/thua, điểm thô theo (dataset, mức, seed). Phân tích phụ: so với từng baseline, Holm trên mọi cặp.
- **Script phân tích đã có**: `G8/analysis/g8_confirmatory.py`. Chỉ thêm tham số thư mục S2 và tên TFM, không đổi logic.
- **Ước tính chi phí:**
  - TFM (TabICL): chuỗi Gibbs trung vị 41 giây ở NC1, tổng khoảng 5–8 giờ GPU.
  - Baseline LLP túi 8 và PLL hai mức: khoảng 25–35 giờ CPU, tức 2–3 phiên Kaggle song song trong khoảng 1 ngày.
- Lượt chạy, nhật ký, kiểm sau lượt: như G7 (`S2/EXPERIMENT_LOG.md`). Bộ test chạy đầu mỗi lượt.

## 5. Làm bài mạnh hơn ngoài phép xác nhận (thăm dò, chỉ trên tập phát triển S2)

**5.1 Lý thuyết về kiểu sụp.** Đây là phần đóng góp mới đáng giá nhất.
- Hiện tượng (NC1, PLL q = 0,6, ≥ 8 lớp): chuỗi với TabPFN kẹt ở trạng thái "nhãn gần ngẫu nhiên". Trên optdigits có hai lưu vực: cùng điểm xuất phát, 2/3 seed thoát, 1/3 kẹt.
- Ghi chú định hướng: mô hình **tuyến tính** cho phản ứng của TFM với nhãn ngữ cảnh chỉ có một điểm bất động ổn định, nên không giải thích được hai lưu vực. Hai lưu vực đòi hỏi phản ứng **phi tuyến**: TFM tin ngữ cảnh nhiều hơn khi nhãn nhất quán hơn.
- Kế hoạch:
  - (a) Đo đường phản ứng g(r) của mỗi TFM trên tập phát triển S2: gieo các cách gán có tỉ lệ đúng r, đo xác suất TFM gán cho lớp thật.
  - (b) Phân tích ánh xạ Gibbs r′ = Φ(g(r)): điểm bất động, độ ổn định, ngưỡng thoát theo số mẫu mỗi lớp, số lớp, q.
  - (c) Phát biểu thành mệnh đề, nối với Định lý 2.
  - (d) Dựng **phép chẩn đoán không cần nhãn thật**, chạy trước chuỗi, để dự báo chuỗi có sụp không.
- *Tuỳ chọn, quyết trước khi đóng băng:* ghi trước dự báo của phép chẩn đoán cho TabPFN trên tập đánh giá S2, rồi kiểm. Nếu đúng, đây là bằng chứng mạnh cho lý thuyết. Cái giá là thêm khoảng 10–15 giờ GPU.

**5.2 Làm phương pháp bền hơn với TFM yếu** (TabPFN). Chỉ thử trên tập phát triển S2.
- Các cách thử:
  - nhiều chuỗi, chọn chuỗi theo độ khớp nhãn yếu (không cần nhãn thật);
  - tôi luyện ở các vòng đầu, cuối chuỗi trả về nhiệt độ 1 để Định lý 1 vẫn đúng;
  - khởi động từ mô hình rẻ học trên likelihood của tập ứng viên.
- Cách nào giúp trên tập phát triển thì có thể vào S2 như **nhánh phụ** đã ghi trước, không phải phép thử chính.

## 6. Lịch dự kiến

| Thời gian | Việc |
|---|---|
| 31/10 – 02/11 | thầy chạy script đếm (Kaggle CPU, vài phút); chạy lại L5 (21 ô) |
| 03 – 08/11 | rà tay ứng viên, thầy duyệt; đóng băng S2 (`s2-frozen`); lượt kiểm dữ liệu và mã băm |
| 09 – 20/11 | chạy S2: TFM (GPU) và baseline (CPU) song song |
| 21 – 30/11 | phân tích S2 bằng script có sẵn; lập GATE_REVIEW_S2 |
| 11 – 12/2026 | lý thuyết 5.1, nhánh 5.2 (song song với S2) |
| 12/2026 – 04/2027 | viết bài (G9), kiểm duyệt nội bộ (G10) |
| tháng 5/2027 | nộp NeurIPS 2027 |

## 7. Rủi ro
1. **Không đủ dataset:** thang nới tiêu chí và điều kiện dừng ở §3.2.
2. **Lợi thế của TabICL không lặp lại:** bài trở về phương án B (benchmark) với dữ liệu của cả hai nghiên cứu. Thành công tối thiểu không bị hạ để bù.
3. **Phản biện nói "đổi TFM tới khi đạt":** bài trình bày đủ NC1, phát hiện thăm dò, và việc S2 được đăng ký trước có dấu thời gian (git tag; có thể thêm OSF).
4. **OpenML gián đoạn (như G7):** lượt chạy tiếp được, các ô đã xong được bỏ qua.
5. **TabICL đổi phiên bản:** cố định phiên bản gói và mã băm checkpoint.
