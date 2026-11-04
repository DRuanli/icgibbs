# G9: kế hoạch viết bài (lập 03/11/2026)

**Điều kiện bắt đầu:** cổng S2 qua (thầy duyệt toàn bộ [GATE_REVIEW_S2](../S2/GATE_REVIEW_S2.md), 03/11/2026).
**Venue:** NeurIPS 2027, main track. Hạn ước tính: abstract khoảng 14/05/2027, bài khoảng 21/05/2027; xác minh khi có call for papers. Dự phòng: ICLR 2028, ICML 2028.
**Quy tắc** (handbook 09):
- Mỗi câu khẳng định trong abstract, giới thiệu và kết luận trỏ tới một dòng của [CLAIM_EVIDENCE_MATRIX](../G3/CLAIM_EVIDENCE_MATRIX.md).
- Viết theo thứ tự: hình và bảng → kết quả → phương pháp → thực nghiệm → giới thiệu → kết luận → abstract → tiêu đề.

## 1. Thông điệp chính và giới hạn của nó

**Một câu:** một TFM bảng có sẵn, không huấn luyện lại, có thể làm xác suất có điều kiện cho một bộ lấy mẫu Gibbs trên nhãn ẩn. Cách này học được từ nhãn yếu cấp mẫu và cấp nhóm. Nó vượt baseline chuyên biệt khi TFM bền với nhãn ngữ cảnh nhiễu. Điều đó được xác nhận trong một nghiên cứu đăng ký trước, trên 28 dataset mới.

**Được viết** (ma trận #2–#13):
- Phương pháp và tính đúng khi TFM là PPD. Chỉ áp cho sơ đồ một mẫu, hoán đổi và khối có sửa MH.
- Độ bền khi TFM xấp xỉ, kể cả khi các xác suất có điều kiện không tương thích (Định lý 2).
- **LLP túi 8:** +0,052 AUC so với DLLP tuyến tính, 16/18 dataset.
- **PLL:** +1,2 điểm accuracy so với CC tune ở cả hai mức, không sụp ở ≥ 8 lớp.
- Gibbs tốt hơn hard-EM (cơ chế qua ngữ cảnh).
- Khi nào bộ ngây thơ đã đủ (Mệnh đề 5–6).
- Nhãn yếu thật (H5).
- Không huấn luyện lại, chạy trên một T4.

**Không được viết:**
- "Với mọi TFM". Với TabPFN v2, nghiên cứu 1 không đạt.
- "TabICL tốt hơn TabPFN" như một thí nghiệm đối chứng cùng dữ liệu.
- "Chính xác nên tốt hơn". Bản chuyển nạp chính xác kém bản quy nạp.
- "Có đảm bảo lý thuyết cho cấu hình mặc định". Khối K = 10 xấp xỉ và bag-block chưa có chặn khác rỗng.
- "Vượt xa ở PLL".
- Khẳng định cho PU, SSL, nhãn nhiễu, LLP túi 16 hoặc 32. Các bối cảnh này là kết quả phụ hoặc giới hạn của nghiên cứu 1.

**Trình bày hai nghiên cứu** (thầy duyệt ở D1, 03/11):
- Cả hai nằm trong bài chính, theo thứ tự thời gian.
  - **Study 1:** đăng ký trước, TabPFN v2, 24 dataset. Không đạt hai khẳng định chính; phân tích lỗi tìm ra kiểu sụp.
  - **Study 2:** đăng ký trước, đổi duy nhất TFM, 28 dataset mới. Đạt.
- Lý do: phản biện sẽ hỏi "đổi TFM tới khi đạt". Câu trả lời mạnh nhất là trình bày thẳng cả hai nghiên cứu, kèm dấu thời gian đóng băng.

## 2. Dàn ý và ngân sách trang (bài chính khoảng 9 trang; xác minh theo call for papers 2027)

| Mục | Trang | Nội dung | Nguồn |
|---|---|---|---|
| Abstract | — | bài toán → khoảng trống → phương pháp → con số của Study 2 → ý nghĩa | ma trận #1, #2, #6, #8 |
| 1. Introduction | 1,25 | nhãn yếu trên dữ liệu bảng; TFM có sẵn nhưng chưa ai dùng làm xác suất có điều kiện cho suy luận trên nhãn ẩn; đóng góp C1–C4 có trỏ mục; Hình 1 | G2 (cập nhật), ma trận #1–#4, #6–#8, #13 |
| 2. Related work | 0,75 | học từ nhãn yếu (PLL, LLP, PU; các khung thống nhất như GLWS); TFM bảng; imputation lặp (chained equations), Liu et al. 2014; **Bảng 1** so với 3–6 công trình gần nhất | G2/02_CLOSEST_WORKS, related_work_matrix.csv |
| 3. Problem setup | 0,5 | prior trao đổi được, mô hình quan sát $\ell(w\mid y,x)$, PPD; các bối cảnh như trường hợp riêng | PROOFS §0, METHOD_SPEC §1, §3 |
| 4. In-Context Gibbs | 1,0 | thuật toán (mã giả), khối, khởi động ấm, Rao–Blackwell, quy nạp và chuyển nạp, chi phí | METHOD_SPEC §2, §4, §5 |
| 5. Theory | 1,25 | Định lý 1 (tính đúng), Mệnh đề 3 (Bayes-optimal theo prior), Định lý 2 (độ bền, xác suất có điều kiện không tương thích), Mệnh đề 5–6 (khi nào bộ ngây thơ đủ); nêu rõ phạm vi | PROOFS §1–§5, §7 |
| 6. Experiments | 3,5 | 6.1 thiết lập và đăng ký trước; 6.2 tính đúng trên bài giải tích (RQ1); 6.3 Study 1; 6.4 **Study 2 (kết quả chính)**; 6.5 cơ chế: Gibbs so với hard-EM, ánh xạ phản ứng, khi nào Gibbs giúp; 6.6 nhãn yếu thật; 6.7 ablation và chi phí | G4/RQ1_REPORT, G8_REPORT, S2_RESULTS, S2X_REPORT |
| 7. Limitations | 0,5 | phụ thuộc TFM; không có đảm bảo cho cấu hình mặc định; R̂ chưa trộn hết ở mức nhãn; PLL lợi thế khiêm tốn; bối cảnh không giúp (PU-SCAR, nhiễu, SSL); n ≤ 10.000 | ma trận cột "Phạm vi đúng" |
| 8. Conclusion | 0,25 | | |
| Phụ lục | — | chứng minh đầy đủ; giao thức và mốc thời gian đăng ký trước; danh sách sai lệch; dataset card; không gian tune; bảng theo dataset của cả hai nghiên cứu; bối cảnh phụ; ablation đầy đủ; NeurIPS Paper Checklist | |

**Đóng góp dự kiến:**
- **C1 — Phương pháp:** In-Context Gibbs, một thủ tục cho nhãn yếu cấp mẫu và cấp nhóm, không huấn luyện lại.
- **C2 — Lý thuyết:** tính đúng khi TFM là PPD; độ bền dưới xác suất có điều kiện xấp xỉ và không tương thích; khi nào bộ ngây thơ đã đủ.
- **C3 — Bằng chứng:** hai nghiên cứu đăng ký trước. Nghiên cứu xác nhận trên 28 dataset mới đạt cả LLP và PLL.
- **C4 — Cơ chế:** độ bền của TFM với nhãn ngữ cảnh nhiễu quyết định thành công, thể hiện qua ánh xạ phản ứng và kiểu sụp.

## 3. Hình và bảng

Mọi hình và bảng sinh bằng script trong `paper/figures/` từ kết quả thô. Nhãn tiếng Anh, PDF vector, đọc được khi in đen trắng, chú thích ghi rõ error bar.

| # | Nội dung | Bài chính / phụ lục | Nguồn | Trạng thái |
|---|---|---|---|---|
| Hình 1 | Sơ đồ phương pháp: nhãn yếu → chuỗi Gibbs theo khối, TFM làm xác suất có điều kiện → dự đoán | chính | mới | cần vẽ |
| Bảng 1 | So sánh với công trình gần nhất (không huấn luyện lại, nhãn yếu cấp nhóm, suy luận Bayes, đảm bảo) | chính | G2 | cần làm |
| Bảng 2 | Kết quả chính: Study 2 (H2′, H4′) đặt cạnh Study 1 | chính | S2_RESULTS §9, G8_REPORT §2 | **xong 03/11** (`paper/tables/main_results.tex`) |
| Hình 2 | Chênh theo dataset ở Study 2 (LLP, PLL) | chính | `S2/analysis/s2_figures.py` | **xong 03/11** (`paper/figures/s2_gaps.pdf`) |
| Hình 3 | Cơ chế: (a) ánh xạ phản ứng TabPFN / TabICL ở q 0,6; (b) vết chuỗi sụp và thoát | chính | S2X, `G8/analysis/g8_figures.py` (fig2) | **xong 03/11** (`paper/figures/mechanism.pdf`) |
| Bảng 3 | Gibbs so với hard-EM (H3, H3′) và ablation chính | chính | G8_REPORT §3, §6; S2_RESULTS §3 | **phần H3 xong 03/11** (`paper/tables/h3.tex`); ablation chờ S2E |
| Bảng 4 | Nhãn yếu thật: Lost, MSRCv2, BirdSong | chính hoặc phụ lục | G8_REPORT §4 | có số |
| Phụ lục | RQ1 (TV theo số mẫu, chặn của Định lý 2), cỡ dữ liệu, R̂, bối cảnh phụ D1–D3, bảng theo dataset, so với từng baseline | phụ lục | G4, G8, S2 | có số |

## 4. Lịch

| Thời gian | Việc | Sản phẩm |
|---|---|---|
| 04 – 15/11/2026 | ~~thầy chốt D1–D5~~ (xong 03/11); dựng `paper/` (LaTeX, template NeurIPS hiện hành, đổi sang 2027 khi có); hình và bảng tiếng Anh | Hình 1–3, Bảng 2–3 |
| 16/11 – 06/12 | mục 6 (kết quả), mục 4 (phương pháp) | bản nháp mục 4, 6 |
| 16/11 – 20/12 | song song: lý thuyết ngưỡng sụp (S2_PLAN §5.1); quyết định đưa vào bài hay không trước 20/12 (D2) | mệnh đề, hoặc ghi chú cho Discussion |
| 07/12 – 31/12 | mục 5 (lý thuyết) và chứng minh ở phụ lục; mục 3; đối chiếu trích dẫn trong chứng minh (việc mở G4-6) | bản nháp mục 3, 5, phụ lục A |
| 01 – 31/01/2027 | cập nhật tìm kiếm tính mới (G2); mục 2, mục 1, mục 7, 8; abstract, tiêu đề; Paper Checklist | **bản nháp đầy đủ v1** |
| 02 – 03/2027 | G10: một người hoặc agent chưa tham gia đọc như reviewer; tái lập một bảng chính từ clone sạch; đối chiếu từng trích dẫn | GATE_REVIEW_G10 |
| 04/2027 | sửa theo G10; cập nhật tìm kiếm tính mới trong 30 ngày trước khi nộp; chuẩn bị kho code ẩn danh | bản nộp |
| 05/2027 | G11: nộp abstract, rồi nộp bài | |

## 5. Việc mở cần xử lý trong G9

| # | Việc | Vì sao | Cách |
|---|---|---|---|
| O1 | Phạm vi của lý thuyết so với cấu hình thật | Khối K = 10 và bag-block xấp xỉ không có chặn khác rỗng (PROOFS §5, G4-5) | Mục 5 và mục 7 viết rõ. Bằng chứng thực nghiệm: so với khối có sửa MH, chênh ≤ 0,01 (NC1). Tuỳ chọn O4 |
| O2 | Trích dẫn trong chứng minh và phần liên quan | Việc mở G4-6: đối chiếu số trang Schweitzer, Meyer, Seneta, Cho & Meyer, Liu et al. | Đối chiếu bản gốc khi viết mục 5 |
| O3 | R̂ với TabICL | Mọi con số về trộn chuỗi hiện là của TabPFN (GATE_REVIEW_S2, phát hiện 9) | **Xong 04/11** (lượt S2E, [S2E_REPORT](../S2/S2E_REPORT.md)): TabICL cũng chưa trộn hết ở mức nhãn, dự đoán hai chuỗi gần nhau; đã đưa vào mục 6.7 và Bảng 5 (phụ lục) |
| O4 | Khối có sửa MH với TabICL | Nối lý thuyết với cấu hình dùng trong Study 2 | **Xong 04/11** (thiếu 7/12 ô ≥ 8 lớp vì hết giờ): bản chính xác không dự đoán tốt hơn, chậm hơn trung vị 7 lần; mục 5 (Scope), 6.7 và 7 đã viết lại cho khớp |
| O5 | Template NeurIPS 2027 | Chưa công bố (thường khoảng tháng 3) | Tạm dùng `paper/neurips_placeholder.sty` (mô phỏng khổ trang); đổi sang style chính thức khi có |
| O6 | Ẩn danh | Double-blind: kho code, tên file, lời cảm ơn | Kho ẩn danh khi nộp; bản công khai mở sau |
| O7 | Khai báo dùng AI | Chính sách NeurIPS | Cập nhật `docs/AI_USAGE.md` theo từng cổng (đã thêm G7, G8, S2, G9) |
| O8 | **Nhận xét của reviewer ngoài 01** (04/11): lý thuyết không phủ bộ mặc định; chưa tách độ lệch của xấp xỉ khỏi độ trộn; MCSE; PLL và LLP thật; TFM chưa đối xứng; ngữ cảnh; riêng tư | Kết luận của reviewer: "promising, major revision" | Đối chiếu từng ý và kế hoạch R1–R10 ở [REVIEW_RESPONSE_PLAN](REVIEW_RESPONSE_PLAN.md). R9, R10 đã sửa. R1–R8 chờ thầy chọn (Q1–Q5) |

## 6. Quyết định của thầy

**Thầy duyệt toàn bộ D1–D5 theo đề xuất ngày 03/11/2026** ("duyệt toàn bộ và hãy tiếp tục").

| # | Câu hỏi | Mặc định đề xuất |
|---|---|---|
| D1 | Trình bày Study 1 ở đâu? | **Cả hai nghiên cứu trong bài chính** (Study 1 ngắn gọn, đủ con số; chi tiết ở phụ lục). Không đẩy Study 1 hoàn toàn xuống phụ lục |
| D2 | Lý thuyết ngưỡng sụp có thành đóng góp không? | Làm song song tới 20/12. Có mệnh đề gọn và kiểm được thì vào mục 5 hoặc 6.5; không thì để ở Discussion như quan sát thăm dò |
| D3 | Có chạy thăm dò O3 + O4 không? | **Có**, một lượt GPU khoảng 2–3 giờ trên dataset cũ của NC1. Ghi nhãn thăm dò |
| D4 | Tác giả, cơ quan, tài trợ, CRediT | Thầy điền; AI không phải tác giả |
| D5 | Nơi viết LaTeX | Viết trong kho, thư mục `paper/` (em biên dịch kiểm tra được); thầy có thể đồng bộ sang Overleaf |

## 7. Qua cổng G9 khi (handbook 02 §G9)
- [x] Bản thảo LaTeX (khổ trang tạm, đổi sang style NeurIPS 2027 khi có). Phần chính 9 trang; 24 trang kể cả phụ lục và checklist.
- [~] Có Data & Code Availability, Compute, Broader impacts, khai báo dùng LLM (phụ lục J). Còn thiếu: tài trợ, xung đột lợi ích, CRediT (D4, điền ở bản camera-ready vì bài ẩn danh); đối chiếu chính sách LLM của NeurIPS 2027.
- [x] Abstract và giới thiệu không khẳng định điều gì vượt ma trận; mỗi khẳng định có trỏ mục, bảng, hình hoặc định lý (dấu `\claim{}` trong bản nháp).
- [x] Limitations có nội dung thật: phụ thuộc TFM, lý thuyết so với bộ lấy mẫu mặc định, độ trộn, phạm vi và độ lớn hiệu ứng.
- [~] NeurIPS Paper Checklist: bản nháp đủ 16 câu. Mục 12 (giấy phép) để `\pending{}` cho tới khi xác nhận checkpoint TabICL v2, dữ liệu OpenML và PALM; bản final không biên dịch được khi còn `\pending`.
- [ ] Trả lời các ý P0 và P1 của reviewer ngoài 01 ([REVIEW_RESPONSE_PLAN](REVIEW_RESPONSE_PLAN.md)), hoặc ghi rõ vì sao không làm.

## 8. Tiến độ
| Ngày | Việc |
|---|---|
| 03/11/2026 | Thầy duyệt D1–D5. Dựng `paper/`: khung LaTeX biên dịch được, khổ trang tạm; mọi con số lấy từ `numbers.tex` do `paper/scripts/make_tables.py` sinh. Xong Bảng 2, bảng H3, bảng trung bình của Study 2 (phụ lục), Hình 2, Hình 3. Bản nháp mục 6.3 (Study 1), 6.4 (Study 2), 6.5 (cơ chế). `refs.bib` là bản khung, mọi mục chờ đối chiếu. Chuẩn bị lượt S2E (O3, O4) |
| 03/11/2026 (chiều) | Bản nháp mục 3 (thiết lập), 4 (phương pháp, Thuật toán 1), 5 (lý thuyết: Bổ đề 1, Định lý 1–2, Mệnh đề 1–3, có đoạn "Phạm vi" nói rõ cấu hình mặc định không có đảm bảo), 6.1 (thiết kế), 6.2 (RQ1), 6.6 (nhãn yếu thật, Bảng 3 sinh bằng script), 6.7 (ablation, R̂, chi phí), 7 (giới hạn); Phụ lục A (chứng minh đầy đủ, dịch từ PROOFS.md). Mọi số mới cũng sinh bằng `make_tables.py` (189 macro). **Độ dài:** phần chính đã khoảng 9 trang khi chưa có giới thiệu, công trình liên quan, abstract → phải cắt khoảng 2,5 trang (dự kiến: chuyển Bảng 3, chi tiết RQ1 và mã giả sang phụ lục, rút gọn mục 5). Số liệu RQ1 trong mục 6.2 còn gõ tay từ RQ1_REPORT; phải sinh bằng script trước khi nộp |
| 04/11/2026 | Nhận kết quả S2E (thăm dò): R̂ và khối có sửa MH với TabICL ([S2E_REPORT](../S2/S2E_REPORT.md)). Sửa câu "bản chính xác cho dự đoán tương tự" thành "không dự đoán tốt hơn ở cùng số vòng quét, chậm hơn nhiều, và không tách được độ lệch của bản xấp xỉ với độ trộn chậm". Thêm Bảng 5 (phụ lục). Hết các `\todo` phụ thuộc S2E |
| 04/11/2026 | Theo yêu cầu của thầy ("viết sao cho hay, đừng viết theo hướng kết quả âm"): viết mục 1 (giới thiệu), 2 (công trình liên quan, Bảng 1 định vị), abstract, kết luận; dựng lại mục 6 theo mạch tích cực — 6.2 kết quả chính (nghiên cứu xác nhận), 6.3 nguồn gốc lợi ích, 6.4 lợi ích phụ thuộc vào gì (Study 1 trình bày như nghiên cứu tìm ra điều kiện, đủ số liệu), 6.5 nhãn yếu thật, 6.6 tính chính xác, độ trộn, chi phí. Không bỏ số liệu nào của Study 1 (D1). **Phần chính vừa 9 trang** (kết luận kết thúc ở trang 9): chuyển bảng H3, bảng nhãn yếu thật, Mệnh đề về khối và chi tiết RQ1 sang phụ lục; rút gọn mục 3–5, 7 |
| 04/11/2026 (tiếp) | Hoàn thành phụ lục bằng script `paper/scripts/make_appendix.py` (12 bảng: dòng thời gian đăng ký trước, Study 1 với TabICL, độ nhạy và mọi baseline của Study 2, kết quả theo dataset, RQ1, ablation, dataset của hai nghiên cứu, không gian tune, bối cảnh phụ). Số RQ1 nay sinh từ `G4/results/rq1/*.csv` (độ lỏng của chặn Định lý 2 theo từng cấu hình là 1,9–15 lần, trung vị 3,9; số 2–12 lần trước đây là trung bình theo seed). Viết checklist NeurIPS và các phần khai báo. **Đối chiếu 32 tài liệu tham khảo** ([REFERENCE_CHECK](REFERENCE_CHECK.md)): sửa 9 sai lệch, trong đó TabICLv2 đã đăng ICML 2026, DLLP là ICDM Workshops, và Hệ quả 12 của Agarwal chỉ phát biểu cho ước lượng xác suất lớp. Còn 3 `\todo`: ngưỡng sụp (D2, hạn 20/12), chính sách LLM của NeurIPS 2027, xác nhận giấy phép |
| 04/11/2026 (tối) | **Nhận xét của reviewer ngoài 01**, lưu nguyên văn ở `G9/reviews/`. Đối chiếu từng ý ([REVIEW_RESPONSE_PLAN](REVIEW_RESPONSE_PLAN.md)). **Đã sửa trong bài:** checklist giấy phép đổi thành `\pending{}`; câu "π̃ nói chung không phải posterior" sau Định lý 2; giới thiệu và kết luận nói "các biến thể chính xác" thay vì cả bộ lấy mẫu; §6.6 nói rõ chênh hai chuỗi đo trên 3 dataset nhỏ nhất và kiểm định xác nhận đã gồm biến thiên Monte Carlo; Limitations nêu lấy mẫu con ngữ cảnh ở 14/28 dataset của Study 2; Broader impacts nói rủi ro riêng tư chưa được đo. **Sửa một lỗi:** ở Lost là ba (không phải bốn) phương pháp đầu cách nhau < 0,003; nay sinh bằng macro. Phần chính vẫn vừa 9 trang. Thêm [HANDOFF.md](../HANDOFF.md) để làm tiếp ở chat khác |
