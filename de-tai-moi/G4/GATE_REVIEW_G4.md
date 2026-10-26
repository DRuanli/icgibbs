# Biên bản kiểm duyệt cổng G4 (phát triển phương pháp và lý thuyết)

Đề tài A · Ngày lập: 24/10/2026
- **Người tạo sản phẩm:** Claude (mã, chứng minh, kiểm chứng số), thầy (chủ sở hữu)
- **Người kiểm duyệt:** thầy ☑ · Ngày: 26/10/2026. Ghi chú: phần toán chưa có kiểm duyệt độc lập riêng; việc này chuyển sang G6 (G6_PLAN, T7) và phải xong trước khi viết bài.
- Đã có một lượt **tự rà soát phản biện phần toán**: [SELF_REVIEW_MATH.md](SELF_REVIEW_MATH.md) (không thay kiểm duyệt độc lập).

## 1. Sản phẩm
| File | Nội dung |
|---|---|
| [METHOD_SPEC.md](METHOD_SPEC.md) | Đặc tả phương pháp, chi phí, 9 quyết định thiết kế có căn cứ, baseline, rủi ro |
| [PROOFS.md](PROOFS.md) | Chứng minh đầy đủ (thay THEORY_DRAFT của G3) |
| [RQ1_REPORT.md](RQ1_REPORT.md) | Kiểm chứng số E1–E5 |
| `code/` | Thư viện `icgibbs`, 24 kiểm thử (xanh), script thí điểm Kaggle, script kiểm chứng E1–E6 |
| G3/RESEARCH_PLAN.md §9, G3/CLAIM_EVIDENCE_MATRIX.md | Ghi 3 thay đổi sau đóng băng (lý thuyết, không đổi RQ/metric/dataset) |

## 2. Tiêu chí qua cổng G4
| Tiêu chí | Đạt? | Bằng chứng hoặc ghi chú |
|---|---|---|
| Mỗi định lý có giả định đầy đủ và chứng minh hoàn chỉnh | ☑ Đạt | Bổ đề 1, Định lý 1(a,b,c), Mệnh đề 1.4, Mệnh đề 3, Định lý 2, Mệnh đề 4(a): mức L4 theo tự rà soát. Mệnh đề 4(b): chặn đủ (L3), ghi rõ giới hạn. PU-SAR: chỉ nhận xét (việc mở G4-1) |
| Mỗi bổ đề đã qua kiểm chứng số | ☑ Đạt | RQ1_REPORT §1–§4: 4/4 dự đoán đã đăng ký của RQ1 đạt (dự đoán về khối: đạt trên chuỗi gộp, vượt ở bài nhỏ có feature mang thông tin) |
| Phương pháp cho đáp án đúng trên ví dụ đồ chơi | ☑ Đạt | E1 (phân phối dừng chính xác), E1mc (cài đặt thật), E6 (PFN thật, dữ liệu trong prior: Gibbs có log loss thấp nhất như Mệnh đề 3 dự đoán), kiểm thử |
| Thí điểm trên tập phát triển — tiêu chí dừng 2 của RESEARCH_PLAN §5 | ☑ Đạt, phạm vi thu hẹp | Lượt 1 (24/10) và lượt 2 (25/10): không chạm điều kiện chuyển phương án B. Theo quy tắc đọc kết quả đã ghi trước, phạm vi khẳng định thu hẹp về chế độ nhãn yếu đủ thông tin. Chi tiết: [pilot/PILOT_V1_REPORT.md](pilot/PILOT_V1_REPORT.md), [pilot/PILOT_V2_REPORT.md](pilot/PILOT_V2_REPORT.md) |

## 3a. Tự rà soát của người tạo
| # | Mức | Phát hiện | Đề xuất | Trạng thái |
|---|---|---|---|---|
| 1 | Lớn | **Định lý 1 ở G3 khẳng định quá mức**: dự đoán Rao–Blackwell quy nạp chỉ trùng posterior predictive Bayes khi prior tách được; TabPFN dùng prior SCM (feature mang thông tin) nên điều kiện này không đúng | Đã sửa chứng minh, thêm chế độ chuyển nạp (chính xác), ghi RESEARCH_PLAN §9, thu hẹp khẳng định #3, #10. Việc mở G4-2: chọn chế độ cho thực nghiệm chính ở G5 bằng tập phát triển | Đã sửa, **thầy xác nhận** |
| 2 | Lớn | **PU-SCAR và AUC (Elkan & Noto 2008).** Dưới SCAR, xếp hạng của TFM ngây thơ đã có cùng thứ tự với posterior thật. Dự đoán của RQ2 "hơn TFM ngây thơ ≥ 0,03 AUC ở mức yếu nhất" có thể không đạt được ở PU-SCAR vì lý do lý thuyết, không phải vì phương pháp kém. **Đã có bằng chứng (E6, 40 dataset trong prior, mini-TabPFN):** PU-SCAR AUC Gibbs − naive = +0,007 [−0,005; +0,018], trong khi log loss Gibbs − naive = −0,060 [−0,113; −0,017] | **Thầy quyết trước khi chạy thí điểm** (để quyết định không bị ảnh hưởng bởi kết quả): (a) giữ AUC; (b) đổi metric chính của PU-SCAR sang log loss (hoặc accuracy) và giữ AUC cho PU-SAR, LLP. Nếu (b): ghi RESEARCH_PLAN §9 với lý do lý thuyết. Thí điểm ghi mọi metric nên phương án nào cũng kiểm được | **Đã quyết 25/10/2026** (thầy uỷ quyền): xem mục 3c, quyết định Q1 |
| 3 | Lớn | **Cách hiện thực RQ3.** "EM với posterior từng mẫu độc lập (kiểu GLWS)" không làm trực tiếp được với TFM (không huấn luyện được trên nhãn mềm). Đề xuất: hard-EM (gán argmax) cùng TFM; thêm SEM (mẫu cuối) để tách hai thành phần lợi ích | Thầy xác nhận cách hiểu; ghi vào EXPERIMENT_PROTOCOL ở G5 | **Đã quyết 25/10/2026** (thầy uỷ quyền): xem mục 3c, quyết định Q2 |
| 4 | Lớn | **Lệch do cập nhật khối.** Chia khối chỉ trên mẫu tự do gây lệch thành phần ngữ cảnh (PU, n = 500, K = 10: lệch dự đoán 0,031). Chia trên mọi mẫu giảm còn 0,0003 (E5c). Phần sai còn lại do phụ thuộc trong khối; ở n ≤ 10 với prior hỗn hợp vẫn lớn (E5a) | Đã đổi mặc định (D1). Bản sửa MH chính xác nhưng tốn hơn — giữ làm phân tích độ nhạy | Đã sửa |
| 5 | Nhỏ | Tiêu chí RQ1 "K = 10: TV tăng < 0,02" suy biến khi n ≤ 14 (K = 10 ⇒ khối cỡ 1, chính xác) | Báo cáo cả n ≤ 14 với khối cỡ 2–8 và chuỗi gộp n tới 2000 (RQ1_REPORT §5); không đổi tiêu chí | Ghi nhận |
| 6 | Nhỏ | nnPU, uPU, DLLP trong thí điểm là bản cài lại, chưa tune; tiêu chí dừng 2 so với chúng nên chỉ có tính chỉ báo | G5–G6: dùng cài đặt chính thức, ngân sách tune bằng nhau, tái hiện số công bố | Ghi nhận |
| 7 | Nhỏ | Việc mở G4-1 (hội tụ PU-SAR), G4-3 (trích dẫn bản in Liu et al., Mitrophanov) | Làm trước G9 | Mở |
| 8 | **Lớn** | **Thí điểm lượt 1:** thất bại ở chế độ nhãn yếu ít thông tin (LLP túi 32: thua DLLP 5/6, chuỗi không hội tụ; PU-SCAR c = 0,1: log loss kém naive). Hai giả thuyết: (a) khởi tạo/trộn chậm — sửa được bằng khởi động ấm; (b) sai số TFM bị khuếch đại khi chuỗi trộn chậm (Định lý 2) — không sửa được, phải thu hẹp phạm vi | Lượt 2 với khởi động ấm + chẩn đoán khởi tạo từ nhãn thật; quy tắc đọc kết quả đã ghi trước (PILOT_V1_REPORT §4) | **Đã giải quyết ở lượt 2:** áp quy tắc (b) — thu hẹp phạm vi; khởi động ấm thành mặc định (PILOT_V2_REPORT §5) |
| 9 | Lớn | Câu hỏi metric PU-SCAR (#2) chưa được trả lời trước khi chạy lượt 1. Giờ mọi quyết định đều đã thấy số liệu phát triển: lượt 1 cho thấy AUC ≈ naive (đúng lý thuyết) **và** log loss của Gibbs kém naive ở c = 0,1 | Vẫn được quyết (tập phát triển), nhưng phải ghi ở RESEARCH_PLAN §9 là "quyết sau khi xem thí điểm phát triển", kèm lý do lý thuyết. Đề xuất: chờ lượt 2 rồi quyết một lần | **Đã quyết** cùng Q1; đã ghi RESEARCH_PLAN §9 |

## 3c. Quyết định ngày 25/10/2026

Thầy đồng ý các đề xuất ở mục 3a (trừ #2 và #3) và đồng ý chạy thí điểm lượt 2. Hai quyết định #2 và #3 thầy giao cho Claude tự quan sát và quyết; chúng được đưa ra **sau khi đã thấy thí điểm lượt 1 trên tập phát triển** và được ghi đúng như vậy ở RESEARCH_PLAN §9.

**Q1 — Metric chính của PU-SCAR: log loss sau hiệu chỉnh khớp tiên nghiệm lớp.**
- *Định nghĩa:* với mọi phương pháp như nhau, dịch logit của xác suất dự đoán một hằng số sao cho trung bình xác suất trên các điểm cần dự đoán bằng tỉ lệ lớp dương đã biết, rồi tính log loss. Bước này không dùng nhãn test, dùng đúng thông tin mọi phương pháp đã được cho (nnPU, uPU dùng tỉ lệ lớp; naive dùng c), và không đổi AUC.
- *Phụ:* AUC (kiểm định **không kém** TFM ngây thơ, ngưỡng 0,01), log loss thô, Brier.
- *PU-SAR và LLP giữ AUC là metric chính*, như đã đóng băng.
- *Lý do:*
  1. Dưới SCAR, xếp hạng của bộ phân loại ngây thơ trùng thứ tự với posterior thật (Elkan & Noto 2008), nên AUC không đo được đóng góp của bất kỳ phương pháp PU nào so với naive. Điều này đúng về lý thuyết, đã thấy trên prior (E6) và trên lượt 1.
  2. Phương pháp nhắm tới posterior predictive, nên quy tắc chấm điểm proper mới là thước đo khớp với Mệnh đề 3.
  3. Hiệu chỉnh khớp tiên nghiệm bỏ đi lợi thế không công bằng trước các baseline không xuất xác suất hiệu chỉnh (uPU có log loss thô 2–3 ở lượt 1).
- *Công khai:* ở lượt 1, log loss **thô** của Gibbs **kém** naive ở c = 0,1, nên quyết định này không chọn metric có lợi cho phương pháp. Bản khớp tiên nghiệm chưa được đo ở lượt 1; lượt 2 sẽ đo.
- *Hệ quả cho H2 ở PU-SCAR:* giữ tiêu chí thắng (≥ 7/10, Wilcoxon + Holm) nhưng tính trên log loss khớp tiên nghiệm. Hai dự đoán độ lớn theo AUC (+0,02 so với peer-review, +0,03 so với naive) không còn áp dụng cho PU-SCAR, thay bằng điều kiện AUC không kém naive quá 0,01.

**Q2 — Hiện thực RQ3: hard-EM cùng TFM, cùng điểm khởi động; SEM là phân tích phụ.**
- *"EM với posterior từng mẫu độc lập"* := **hard-EM** (classification EM; Celeux & Govaert 1992) với cùng TFM, cùng cách chia khối, **cùng điểm khởi động ấm** như Gibbs (`hard_em_warm`). Với LLP, bước argmax có ràng buộc số đếm (top-k theo log-odds, chính xác với nhị phân).
- *Phân tích phụ:* SEM (stochastic EM; Celeux & Diebolt 1985) = cùng chuỗi Gibbs nhưng chỉ dùng mẫu cuối. SEM so với hard-EM tách tác dụng của *lấy mẫu*; Gibbs so với SEM tách tác dụng của *lấy trung bình Bayes*.
- *Metric:* log loss (đúng như đã đóng băng), AUC phụ. Với PU-SCAR dùng log loss khớp tiên nghiệm theo Q1.
- *Lý do:* TFM không huấn luyện được trên nhãn mềm nên E-step kiểu GLWS không làm trực tiếp được; hard-EM là dạng tất định gần nhất. Bản "EM mềm với mạng được huấn luyện" đã có mặt qua chính GLWS (ICML 2024), một baseline peer-review của RQ2/RQ4.
- *Sửa so với lượt 1:* hard-EM ở lượt 1 khởi động khác Gibbs (PU từ lời giải ngây thơ, LLP ngẫu nhiên). Lượt 2 chạy `hard_em_warm` cùng điểm khởi động để so sánh công bằng.

## 3d. Kết quả thí điểm lượt 2 và các quyết định (25/10/2026)

Chi tiết ở [pilot/PILOT_V2_REPORT.md](pilot/PILOT_V2_REPORT.md).

| # | Nội dung | Căn cứ | Loại |
|---|---|---|---|
| R1 | **Khởi động ấm thành cấu hình mặc định** (PU: TFM ngây thơ tính chéo 5 fold, hiệu chỉnh theo propensity; LLP: DLLP tuyến tính) | Xoá các ca sụp (wdbc LLP: 0,63 → 0,99), giúp ở LLP túi 16 và PU-SCAR c = 0,1, không hại ở LLP túi 8 | Áp quy tắc đã ghi trước (thầy đã đồng ý) |
| R2 | **Thu hẹp phạm vi khẳng định chính về chế độ nhãn yếu đủ thông tin:** LLP túi ≤ 16 là phạm vi chính, túi 32 là bài kiểm tra áp lực, báo cáo trong phần giới hạn | Chuỗi khởi tạo từ nhãn thật cũng trôi; ở túi 16–32 kết quả phụ thuộc khởi tạo và đa mode | Áp quy tắc (b) đã ghi trước (thầy đã đồng ý) |
| R3 | Đưa PU-SCAR ra khỏi khẳng định chính của RQ2, báo cáo như kết quả phụ (TFM ngây thơ + Elkan–Noto gần tối ưu) | Theo metric chính đã chốt (Q1): Gibbs vượt cả hai chỉ 1–2/6, kể cả khi khởi tạo từ nhãn thật | **Thầy xác nhận 26/10/2026.** Ghi ở RESEARCH_PLAN §9, CLAIM_EVIDENCE_MATRIX #6, G5 Mục 0 |
| R4 | RQ3 (Q2) được ủng hộ trên tập phát triển: Gibbs tốt hơn hard-EM ở 30/36 ô (83% ≥ 70%), Spearman dương | | Ghi nhận; kiểm chứng thật ở G7 trên tập đánh giá |
| R5 | **Đề xuất trước khi đóng băng G5:** thí điểm RQ4 (SSL, nhãn nhiễu, PLL) trên tập phát triển; thêm baseline SAR-EM; kiểm tra phân bố cỡ túi của LLP-Bench | Phạm vi RQ2 hẹp lại nên sức nặng của bài dựa nhiều hơn vào RQ4 và RQ5 | Việc của G5 |
| R6 | **Kết quả thí điểm RQ4 (v4 + v5, 26/10/2026, đã đủ):** PLL 6/6 ô đạt tiêu chí (≥ cả peer-review tốt nhất lẫn naive) → chính; nhãn nhiễu 4/12, SSL 3/12 → phụ. PLL: Gibbs vượt PRODEN 0,011–0,064 accuracy, vượt naive 0,016–0,44. RQ3: Gibbs có log loss tốt hơn hard-EM ở 27/30 ô của RQ4 (gộp v2: 57/66 = 86%). Thí điểm many-class: Gibbs chạy được với 11–26 lớp, lợi ích PLL giữ nguyên; chọn TabICL v2 cho RQ5 | Ngưỡng 60% và quy tắc chọn backend đã viết trong script trước khi chạy | Ghi ở G5 Mục 0, 1.4; RESEARCH_PLAN §9; CLAIM #8, #9; [PILOT_RQ4_REPORT](pilot/PILOT_RQ4_REPORT.md), [PILOT_MANYCLASS_REPORT](pilot/PILOT_MANYCLASS_REPORT.md) |

## 3b. Phát hiện của người kiểm duyệt
| # | Mức | Mô tả | Đề xuất | Trạng thái |
|---|---|---|---|---|
| | | | | |

## 4. Quyết định
☑ **Qua** → G5  ☐ **Sửa** (giữ ở G4)  ☐ **Dừng / phương án B** (tiêu chí dừng 2)

Đề xuất của người tạo (cập nhật 25/10/2026, sau lượt 2): **Qua → G5**, với phạm vi đã thu hẹp (R2, R3) và cấu hình mặc định mới (R1). Lý thuyết và kiểm chứng RQ1 đạt; tiêu chí dừng không bị chạm sau hai lượt thí điểm. Còn mở chuyển sang G5: R5 (thí điểm RQ4 phần lớn xong — R6), các việc G4-1/G4-3/G4-4 trong PROOFS. R3 đã được thầy xác nhận 26/10/2026.

Chữ ký người kiểm duyệt: thầy — xác nhận "đã qua G4" qua tin nhắn ngày 26/10/2026. Điều kiện kèm theo: kiểm duyệt độc lập phần toán (Bổ đề 1, Định lý 1, 2, Mệnh đề 3, 4) ở G6.
