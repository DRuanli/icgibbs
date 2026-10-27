# Kế hoạch nghiên cứu, đăng ký trước nội bộ (G3)

> Commit file này vào git **trước** khi chạy thực nghiệm chính. Ngày commit là bằng chứng. Mọi thay đổi sau đó ghi ở Mục 9.

**Đề tài:** *In-Context Gibbs: học từ nhãn yếu bằng tabular foundation model, không huấn luyện lại*
**Bản nháp:** 24/10/2026 (Claude soạn, thầy và người kiểm duyệt duyệt) · **Ngày đóng băng:** 24/10/2026 (sau kiểm duyệt G3) · **Commit:** ghi khi đưa vào repo git của dự án
**Venue đích:** NeurIPS 2027 (hạn ước tính: abstract 14/05/2027, bài 21/05/2027; xác minh khi có call for papers). Dự phòng: ICLR 2028, ICML 2028.

## 1. Đóng góp dự kiến
1. **(Chính) Phương pháp:** *In-Context Gibbs*, suy luận Bayes trên nhãn ẩn dùng một TFM có sẵn làm xác suất có điều kiện đầy đủ. Gồm Gibbs một mẫu cho nhãn yếu cấp mẫu (PU, PLL, nhãn nhiễu, SSL), Metropolis–Hastings hoán đổi cho nhãn cấp nhóm (LLP), và phiên bản theo khối cho dữ liệu lớn. Không cập nhật trọng số nào.
2. **(Chính) Lý thuyết** ([THEORY_DRAFT.md](THEORY_DRAFT.md)):
   - Định lý 1: tính đúng khi TFM là PPD.
   - Định lý 2: độ bền khi TFM xấp xỉ, kể cả khi các xác suất có điều kiện không tương thích.
   - Mệnh đề 3: Bayes-optimal hữu hạn mẫu theo prior.
3. **(Phụ) Thực nghiệm:** ≥10 dataset bảng độc lập × 5 loại nhãn yếu, so với baseline đã peer-review của từng loại và với GLWS, ILL; cộng ≥2 nguồn nhãn yếu thật.

## 2. Dữ liệu: tách tập phát triển và tập đánh giá
Học theo cách làm của bài TabPFN: **mọi lựa chọn siêu tham số** (số vòng quét, K, số chuỗi, n_estimators, giới hạn ngữ cảnh, prior trên propensity) **chỉ được chọn trên ≥6 dataset phát triển**, rời với tập đánh giá. Danh sách cụ thể chốt ở G5 (EXPERIMENT_PROTOCOL).
- **Tập đánh giá:** ≥10 dataset độc lập (C1), n ≤ 10.000, ≤500 feature, lấy từ OpenML/TabArena theo tiêu chí viết trước.
- **Nhãn yếu sinh theo cơ chế công bố kèm** (đã được thầy chấp nhận ở G1), với 2–3 mức độ yếu mỗi loại:
  - LLP: kích thước túi;
  - PU: tỉ lệ dương được gán nhãn, SCAR và SAR;
  - PLL: số nhãn ứng viên phụ;
  - nhãn nhiễu: tỉ lệ nhiễu, đối xứng và phụ thuộc lớp;
  - SSL: tỉ lệ có nhãn.
- **Nhãn yếu thật (bắt buộc ≥2 nguồn):** LLP-Bench (Criteo; tính là 1–2 nguồn), cùng ít nhất một trong: ClinVar (nhiễu thật), bộ partial label thật (Lost, MSRCv2, BirdSong…). Chốt ở G5 sau khi kiểm tra quyền truy cập và giấy phép.

## 3. Câu hỏi nghiên cứu, giả thuyết, dự đoán

| RQ | Câu hỏi | Giả thuyết | Metric chính | Kết quả dự đoán | Kết quả nào **bác bỏ** |
|---|---|---|---|---|---|
| **RQ1** Tính đúng | Khi biết PPD chính xác, In-Context Gibbs có khôi phục đúng posterior predictive Bayes dưới nhãn yếu không? Khi dùng TFM xấp xỉ và khi chia khối, sai bao nhiêu? | H1: với PPD giải tích, sai lệch → 0 theo số mẫu; với PFN học được, sai lệch nằm dưới chặn của Định lý 2; khối K ≤ 10 chỉ làm tăng sai lệch ít | TV giữa phân phối của chuỗi và posterior chính xác (liệt kê, n ≤ 14) | PPD giải tích: TV < 0,01 ở 10⁵ mẫu. PFN học được: TV < 0,05. Khối K = 10: TV tăng < 0,02 so với một mẫu | TV không giảm theo số mẫu với PPD giải tích (lỗi lý thuyết hoặc cài đặt); hoặc TV với PFN vượt chặn |
| **RQ2** Hiệu quả trên PU và LLP | Trên ≥10 dataset, phương pháp có vượt baseline peer-review tốt nhất *và* baseline TFM ngây thơ không? | H2: thắng baseline peer-review tốt nhất của mỗi bối cảnh (PU-SCAR, PU-SAR, LLP) trên ≥7/10 dataset; Wilcoxon kèm hiệu chỉnh Holm, p < 0,05 | ROC AUC (phụ: log loss, PR-AUC cho PU mất cân bằng) | AUC trung bình hơn baseline peer-review tốt nhất ≥ 0,02 khi n ≤ 2.000; mức hơn giảm khi n tăng; hơn TFM ngây thơ ≥ 0,03 ở mức yếu nhất | < 6/10 dataset thắng, hoặc không có ý nghĩa sau Holm, ở cả PU và LLP → **phương án B** |
| **RQ3** Cơ chế | Lợi ích có đến từ việc mô hình hoá phụ thuộc giữa các nhãn ẩn qua ngữ cảnh không? | H3: Gibbs tốt hơn EM với posterior từng mẫu độc lập (E-step kiểu GLWS, cùng TFM), và khoảng cách tăng khi nhãn yếu hơn | Log loss (phụ: AUC) | Gibbs tốt hơn EM-độc-lập trên ≥70% ô (dataset × mức); tương quan Spearman giữa mức yếu và khoảng cách > 0 | Không có khác biệt (Bayesian signed-rank: xác suất nằm trong ROPE ±0,005 log loss > 0,9), hoặc tương quan ≤ 0 |
| **RQ4** Tính tổng quát | Cùng một thủ tục, không tune riêng theo loại, có đạt hay vượt baseline peer-review tốt nhất của PLL, nhãn nhiễu, SSL không? | H4: đạt hoặc vượt trên ≥60% dataset của mỗi loại | Accuracy (PLL), AUC hoặc accuracy (nhiễu, SSL) | Mạnh nhất ở SSL và nhiễu phụ thuộc lớp; yếu nhất ở PLL nhiều lớp | < 50% ở một loại → loại đó ra khỏi khẳng định chính, bài thu hẹp về các loại còn lại |
| **RQ5** Kiểm chứng ngoại tại | Trên ≥2 nguồn nhãn yếu thật, thứ hạng ở RQ2 và RQ4 có giữ không? | H5: phương pháp nằm trong top 2 ở mỗi nguồn | Metric chính của từng loại | Top 2 ở cả hai nguồn | Ngoài top 3 ở cả hai nguồn → giới hạn khẳng định về dữ liệu bán tổng hợp, ghi trong Limitations |

**Phân tích bắt buộc kèm theo (không phải RQ riêng):**
- Chi phí so với độ chính xác: số vòng quét, K, n_estimators, giới hạn ngữ cảnh.
- Độ nhạy theo prior của propensity (PU-SAR).
- Chẩn đoán hội tụ ($\hat R$ trên nhiều chuỗi).
- Phân tích lỗi: phương pháp thua ở đâu.

## 4. Định lý dự kiến
| Định lý | Phát biểu nháp | Giả định | Kiểm chứng số |
|---|---|---|---|
| Bổ đề 1 | Dự đoán của PFN với ngữ cảnh $D_{-i}$ là xác suất có điều kiện đầy đủ $p(y_i\mid y_{-i},X)$ | Prior trao đổi được | Prior ô–Beta–Bernoulli (giải tích) |
| Định lý 1 | Gibbs (cấp mẫu) và MH hoán đổi (LLP) có phân phối dừng $\pi=p(y\mid X,W)$; ước lượng Rao–Blackwell hội tụ về $p(y_*\mid x_*,X,W)$ | (A1)–(A3) | Liệt kê với n ≤ 14 |
| Định lý 2 | $\|\tilde\pi-\pi\|_{TV}\le c(\text{mixing})\cdot 2n\kappa\varepsilon$; tồn tại và duy nhất $\tilde\pi$ dù các xác suất có điều kiện không tương thích | (B1)–(B3) | PFN nhỏ huấn luyện trên prior giải tích |
| Mệnh đề 3 | Bayes-optimal hữu hạn mẫu theo prior | (A1) | — (hệ quả) |
| Mệnh đề 4 | Sai lệch của cập nhật theo khối | Ổn định $\beta_n$ | Thực nghiệm nếu không chứng minh được |

## 5. Tiêu chí thành công và tiêu chí dừng
- **Thành công tối thiểu (đủ viết bài NeurIPS):** Định lý 1 có chứng minh hoàn chỉnh và qua RQ1; ~~H2 đúng cho PU *hoặc* LLP~~ **H2 đúng cho LLP túi 8 *hoặc* H4 đúng cho PLL** *(sửa 27/10/2026, trước G7, thầy chọn — xem §9)*; H3 đúng.
- **Thành công mạnh:** ~~thêm H2 cho cả hai; H4 cho ≥2 loại~~ **cả H2-LLP lẫn H4-PLL** *(sửa 27/10/2026)*; Định lý 2 chứng minh xong; H5.
- **Dừng hoặc chuyển phương án B:**
  1. RQ1 thất bại với PPD giải tích sau khi đã loại trừ lỗi cài đặt → lỗi lý thuyết; dừng và xem lại Bổ đề 1.
  2. Thí điểm trên **tập phát triển** (≥6 dataset) cho thấy phương pháp không vượt *cả* baseline peer-review tốt nhất *lẫn* TFM ngây thơ trên ít nhất 4/6 dataset, ở cả PU và LLP → phương án B: bài benchmark có hệ thống về TFM với nhãn yếu (đích NeurIPS Datasets & Benchmarks hoặc TMLR). *Tiêu chí này thay cho Mốc 2 của TOPIC_CARD_A, vốn thí điểm trên 10 dataset. Lý do đổi: không được chạm vào tập đánh giá trước khi đóng băng giao thức ở G5.*
  3. Một công trình mới công bố trùng đóng góp chính → quay lại G2, thu hẹp.
- **Preprint sớm:** khi Định lý 1 xong và thí điểm xác nhận H2 cho ít nhất một bối cảnh, đưa lên arXiv (mục tiêu: trước 28/02/2027) vì rủi ro bị đi trước.

## 6. Mốc thời gian
| Cổng | Việc chính | Ngày dự kiến |
|---|---|---|
| G3 | Đóng băng kế hoạch này (sau kiểm duyệt) | 31/10/2026 |
| G4 | Chứng minh Định lý 1; kiểm chứng số RQ1 trên CPU; cài prototype In-Context Gibbs; thí điểm PU và LLP trên **dataset phát triển** (exploratory) | 01/11 – 12/12/2026 |
| G5 | Đóng băng giao thức: dataset đánh giá, baseline, metric, split, ngân sách tune, kế hoạch thống kê, cấu hình rút gọn | 19/12/2026 — **xong 26/10/2026** (sớm khoảng 7 tuần) |
| G6 | Code qua test; tái hiện baseline; đo lại thời gian thật (estimator dùng lại) | 20/12/2026 – 31/01/2027 |
| Preprint | arXiv: phương pháp + Định lý 1 + kết quả PU/LLP | trước 28/02/2027 |
| G7–G8 | Thực nghiệm chính (khoảng 8–9 tuần GPU theo giao thức rút gọn), phân tích | 01/02 – 15/04/2027 |
| G9 | Viết bản thảo (song song từ tháng 3) | 01/03 – 30/04/2027 |
| G10 | Kiểm duyệt nội bộ độc lập, tái lập từ clone sạch | 01 – 10/05/2027 |
| G11 | Nộp NeurIPS 2027 | abstract khoảng 14/05, bài khoảng 21/05/2027 (ước tính) |

Rủi ro lịch: G7–G8 chỉ có khoảng 10 tuần cho 8–9 tuần GPU. Không có dư địa; mọi trễ ở G4–G6 phải được bù bằng cách thu hẹp RQ4 (bớt loại nhãn yếu), không phải bằng cắt seed hay dataset của RQ2.

## 7. Tác giả và vai trò (CRediT)
| Tác giả | Thứ tự | Vai trò dự kiến |
|---|---|---|
| (thầy điền) | 1 | Conceptualization, Methodology, Software, Investigation, Formal analysis, Writing – original draft |
| (người hướng dẫn) | | Supervision, Validation, Writing – review & editing |
| (đồng tác giả) | | Validation (kiểm duyệt cổng), Formal analysis (đọc chéo chứng minh), Writing – review & editing |

Ghi chú: AI (Claude) không phải tác giả. Mọi phần do AI hỗ trợ được ghi vào `docs/AI_USAGE.md` để khai báo theo chính sách của NeurIPS.

## 8. Phương án B (đã viết trước)
Bài benchmark "TFM học từ nhãn yếu tốt đến đâu", gồm các TFM có sẵn kết hợp với cách xử lý ngây thơ, hard-EM và In-Context Gibbs, trên 5 loại nhãn yếu × ≥10 dataset, kèm phân tích khi nào TFM thua baseline chuyên biệt. Đích: NeurIPS Datasets & Benchmarks track hoặc TMLR.

## 9. Thay đổi sau khi đóng băng
| Ngày | Thay đổi | Lý do | Ảnh hưởng tới khẳng định nào |
|---|---|---|---|
| 24/10/2026 | **Sửa phát biểu Định lý 1 (phần dự đoán).** Ước lượng Rao–Blackwell quy nạp hội tụ về $\bar p=\sum_y\pi(y)p(y_*\mid x_*,X,y)$, chỉ trùng $p(y_*\mid x_*,X,W)$ khi prior tách được (feature không mang thông tin về cơ chế sinh nhãn). Thêm chế độ chuyển nạp (đưa $X_*$ vào chuỗi), hội tụ về đúng posterior predictive Bayes | Lỗi lý thuyết ở bản nháp G3, phát hiện khi chứng minh ở G4; phản ví dụ số E2 (chênh tới 0,11 ở $n=8$). Xem G4/PROOFS.md §2(d) | #3, #10 trong CLAIM_EVIDENCE_MATRIX: thu hẹp phát biểu. Không đổi RQ, metric, dataset |
| 24/10/2026 | **Dạng chặn của Định lý 2** đổi thành $\mathrm{TV}(\tilde\pi,\pi)\le\tau(P)\,\bar\varepsilon$ (hệ số Dobrushin, sai số trung bình theo $\tilde\pi$); dạng $2n\kappa\varepsilon$ của G3 thành hệ quả | Chứng minh tự chứa, chặt hơn chặn Mitrophanov 2–3 lần trên các ví dụ (E3) | #4: điều kiện kiểm "TV với PFN nằm dưới chặn" giữ nguyên, chặn được tính theo dạng mới |
| 24/10/2026 | **Chi tiết phương pháp:** cập nhật khối chia trên mọi mẫu (mẫu có nhãn cố định trong khối cũng bị bỏ khỏi ngữ cảnh) | Chia chỉ trên mẫu tự do làm ngữ cảnh lệch thành phần, gây lệch dự đoán ~0,03 ở PU với K = 10 (E5b) | #5: phát biểu "sai lệch nhỏ" kiểm trên cách chia mới |
| 25/10/2026 | **Metric chính của PU-SCAR** đổi từ ROC AUC sang **log loss sau hiệu chỉnh khớp tiên nghiệm lớp** (cùng một bước cho mọi phương pháp, không dùng nhãn test); AUC thành metric phụ với kiểm định không kém naive (ngưỡng 0,01). PU-SAR và LLP giữ AUC | Dưới SCAR, AUC không phân biệt được phương pháp với TFM ngây thơ (Elkan & Noto 2008); phương pháp nhắm posterior predictive (Mệnh đề 3). **Quyết sau khi xem thí điểm lượt 1 trên tập phát triển**, do Claude quyết theo uỷ quyền của thầy; ở lượt 1, log loss thô của phương pháp kém naive ở c = 0,1. Chi tiết: G4/GATE_REVIEW_G4 §3c, Q1 | #6 (phần PU-SCAR): dự đoán độ lớn theo AUC bỏ cho PU-SCAR; tiêu chí thắng giữ nguyên, tính trên log loss khớp tiên nghiệm |
| 25/10/2026 | **Hiện thực RQ3:** "EM với posterior từng mẫu độc lập" = hard-EM cùng TFM, cùng điểm khởi động với Gibbs; SEM là phân tích phụ | TFM không huấn luyện được trên nhãn mềm. Chi tiết: G4/GATE_REVIEW_G4 §3c, Q2 | #7: phát biểu giữ nguyên, nói rõ đối chứng là hard-EM |
| 25/10/2026 | **Thu hẹp phạm vi khẳng định chính về chế độ nhãn yếu đủ thông tin:** LLP túi ≤ 16; túi 32 thành bài kiểm tra áp lực (báo cáo như giới hạn). **Khởi động ấm** thành cấu hình mặc định | Thí điểm lượt 2 trên tập phát triển; áp quy tắc đọc kết quả đã ghi trước khi chạy (G4/pilot/PILOT_V1_REPORT §4), thầy đã đồng ý quy tắc này. Chi tiết: G4/pilot/PILOT_V2_REPORT | #6: phạm vi LLP; #5: cấu hình mặc định |
| 25/10/2026 | **Đề xuất (chờ thầy xác nhận ở G5):** PU-SCAR ra khỏi khẳng định chính của RQ2, thành kết quả phụ | Theo metric chính đã chốt, Gibbs vượt cả naive lẫn peer-review chỉ 1–2/6 trên tập phát triển, kể cả khi khởi tạo từ nhãn thật | #6: thu hẹp về PU-SAR và LLP (điều khoản thu hẹp đã có) |
| 26/10/2026 | **Thầy xác nhận:** PU-SCAR là kết quả phụ của RQ2 | Như dòng trên | #6 |
| 26/10/2026 | **RQ4 thu hẹp về PLL** (đã chốt sau lượt v5). SSL và nhãn nhiễu phụ thuộc lớp thành bối cảnh phụ, vẫn chạy và báo cáo đầy đủ | Thí điểm RQ4 trên tập phát triển: PLL 6/6, nhãn nhiễu 4/12, SSL 3/12 ô đạt tiêu chí 60% đã viết trước (bản ghi đầu ghi SSL ≤ 2/12 là sai cách đếm; đính chính trong PILOT_RQ4_REPORT §3). Giải thích dự kiến: ở PU-SCAR và nhiễu phụ thuộc lớp, $p(w\mid x)$ là hàm affine tăng của $p(y\mid x)$ nên xếp hạng của TFM ngây thơ đã đúng; Gibbs giúp khi nhãn yếu ở dạng tập, tổng hoặc phụ thuộc mẫu | #8: thu hẹp về PLL (điều khoản "chỉ các loại đạt ≥ 60%" đã có) |
| 26/10/2026 | **RQ5 — nguồn thứ hai:** PLL thật dạng bảng (Lost, MSRCv2, BirdSong; tiêu chí chọn viết trước); backend > 10 lớp chọn bằng thí điểm many-class trên tập phát triển. LLP-Bench giữ vai trò kiểm tra áp lực | Đúng loại nhãn yếu còn triển vọng; có số công bố để đối chiếu | #9 |
| 26/10/2026 | **RQ5 — backend và cách chia:** TabICL v2 (nhiều lớp sẵn có) cho mọi phương pháp dựa trên TFM ở RQ5-PLL; 10 lần chia 90/10 như văn liệu | Quy tắc đã viết trước trong `kaggle_g4_manyclass_pilot.py`: accuracy oracle TabICL 0,942 so với TabPFN v2 + ECOC 0,930; rẻ hơn khoảng 20 lần | #9 |
| 26/10/2026 | **Phân tích độ nhạy S1** (G5 §1.5): H4 kiểm định thêm sau khi bỏ MiceProtein, wall-robot-navigation (cấu trúc nhóm hoặc chuỗi) và balance-scale, car (sinh từ quy tắc) | Phát hiện khi viết dataset card, trước khi chạy trên tập đánh giá; rò rỉ do mẫu gần trùng có thể có lợi cho phương pháp dựa trên ngữ cảnh | #8 |
| 26/10/2026 | **Danh sách dataset đánh giá:** churn (nhân tạo) bị loại, thay bằng sick theo danh sách dự phòng cố định; dùng cả 14 dataset đa lớp cho PLL | Xác minh chỉ đọc bằng `g5_verify_datasets.py`; chưa chạy phương pháp nào trên tập đánh giá | #6, #8 |
| 26/10/2026 | **Làm rõ cách kiểm định H2 (và H4):** hai đối thủ chính — baseline peer-review tốt nhất của bối cảnh (chọn trên 6 dataset phát triển sau khi tune ở G6, trước khi chạy tập đánh giá) và TFM ngây thơ; Wilcoxon hai phía, Holm trên 4 phép so sánh (2 đối thủ × 2 mức); vẫn đòi thắng ≥ 7/10 dataset trước baseline mạnh nhất của từng mức. So sánh với từng baseline còn lại là phân tích phụ, luôn báo cáo | Khớp phát biểu H2 của G3; bản nháp G5 đã viết chặt hơn G3 ("thắng mọi baseline sau Holm" trên khoảng 10 cặp), mà với n = 10 thì p Wilcoxon nhỏ nhất là 0,0020 so với ngưỡng Holm đầu 0,005 — gần như không đạt được dù lợi ích có thật. Quyết định trước khi chạy tập đánh giá (thầy, F8) | #6, #8 |
| 26/10/2026 | **G5 đóng băng.** Thầy duyệt GATE_REVIEW_G5; F7 xác nhận Lost, MSRCv2, BirdSong khớp số công bố và đạt tiêu chí | Hoàn tất mọi việc trước khi đóng băng (giao thức Mục 12) | #9: nguồn RQ5 đã chốt |
| 26/10/2026 | **PU-SAR thành bối cảnh phụ**; khẳng định chính của RQ2 chỉ còn LLP túi ≤ 16. Quy tắc chọn đối thủ chính xét cả bản mặc định lẫn bản tune; thêm RC, CC cho PLL tổng hợp (giao thức Mục 11) | G6 trên tập phát triển: với SAR-EM, Gibbs đạt 3/12 ô ở PU-SAR; khi biết propensity đạt 8/12 → nút thắt là học propensity. "Thành công tối thiểu" (H2 cho PU hoặc LLP) vẫn khả thi qua LLP | #6: thu hẹp về LLP |
| 26/10/2026 | **Đính chính cơ chế giải thích (G6 T7, PROOFS §7):** TFM ngây thơ nhất quán về xếp hạng ở cả PLL và LLP (không chỉ PU-SCAR, nhiễu); Gibbs giúp nhờ dùng thông tin mà bộ ngây thơ bỏ đi. Thêm phân tích thăm dò D1–D3 (giao thức §11) | Nhận xét cũ sai với cơ chế sinh nhãn của giao thức; cách giải thích mới khớp mẫu hình trên tập phát triển | #12 (mới) |
| 26/10/2026 | **LLP: khẳng định chính chỉ còn túi 8** (thầy chọn); túi 16 thành phụ. Đối thủ chính đã chốt: LLP — DLLP tuyến tính mặc định; PLL — GLWS/ILL tune (tạm thời, chờ PRODEN có Adam); PU — nnPU (phụ) | G6 lượt 2: với mọi baseline đã tune, túi 8 đạt 5/6 ô, túi 16 đạt 2/6; PLL đạt 6/6. "Thành công tối thiểu" (H2 cho PU hoặc LLP) vẫn khả thi qua LLP túi 8 | #6 |
| 27/10/2026 | **Giữ LLP túi 8 là khẳng định chính** (thầy chọn) dù quy tắc 60% áp trên dữ liệu đúng giao thức chỉ còn 3/6 ô; ghi trước kỳ vọng hiệu ứng nhỏ so với DLLP tuyến tính (+0,003 AUC trên tập phát triển) | G6 lượt 5: bộ chạy G7 (loại trùng, tiền xử lý fit trên train, không cắt train) làm lợi thế của Gibbs co từ +0,013 xuống +0,003 AUC; giao thức §11 ghi chú 27/10/2026 | #6: không đổi phát biểu; thêm rủi ro H2-LLP không đạt so với baseline mạnh nhất |
| 27/10/2026 | **Tiêu chí thành công (§5) viết lại theo phạm vi sau G6** (thầy chọn): tối thiểu = Định lý 1 + RQ1 + (H2-LLP túi 8 *hoặc* H4-PLL) + H3; mạnh = cả H2-LLP lẫn H4-PLL + Định lý 2 + H5. Kiểm định của H2 và H4 không đổi (giao thức Mục 6) | Tiêu chí cũ ("H2 đúng cho PU hoặc LLP"; "H4 cho ≥ 2 loại") viết ở G3, trước các lần thu hẹp phạm vi đã ghi ở trên: PU-SAR và SSL, nhãn nhiễu thành phụ; PLL thành khẳng định chính với cùng cấu trúc kiểm định (hai đối thủ chính, Holm). Tiêu chí mới giữ ý định cũ: phương pháp vượt baseline peer-review mạnh nhất và TFM ngây thơ ở ít nhất một loại nhãn yếu chính. **Động cơ có đến từ tập phát triển** (LLP yếu đi, PLL vững — G6 lượt 5), nên ghi rõ ở đây và ở giao thức §11; quyết trước khi chạy bất kỳ phương pháp nào trên tập đánh giá | Tiêu chí dừng / phương án B: chỉ chuyển phương án B khi cả H2-LLP lẫn H4-PLL đều không đạt |
