# Bài v2: ba vòng phản biện đầu và cách xử lý (12/11/2026)

- **Biên bản:** [toán](reviews/REVIEW_V2_MATH_01.md), [kiểm toán số liệu](reviews/REVIEW_V2_AUDIT_01.md), [kiểu NeurIPS](reviews/REVIEW_V2_NEURIPS_01.md). Cả ba do agent chưa thấy quá trình viết thực hiện.
- **Bản được phản biện:** bản nháp 1, commit `0278914`.
- **Trạng thái:** đã sửa trong bản nháp 2; vòng 2 ([biên bản](reviews/REVIEW_V2_ROUND2.md)) đã kiểm việc sửa, các phát hiện mới của vòng 2 đã sửa trong bản nháp 3 (xem §5).

## 1. Kết quả ba vòng

| Vòng | Kết luận |
|---|---|
| Toán | 0 lỗi chặn, 4 lỗi lớn, 16 lỗi nhỏ. Phát biểu và chứng minh ở phụ lục **đúng** (đã dẫn lại đại số và kiểm số bằng ma trận chuyển chính xác). Cả 4 lỗi lớn nằm ở cách thân bài *dùng* và *mô tả* kết quả |
| Kiểm toán | 1 lỗi chặn (dấu nháp trong PDF), 5 lỗi lớn (đều là câu chữ nói sai nghĩa con số), 21 lỗi nhỏ. Mọi số của Study 3 tái lập được từ 213 tệp npz bằng script độc lập |
| NeurIPS | Overall 4/10 cho bản nháp 1; ước 6/10 nếu sửa các điểm chính mà không chạy thêm GPU |

## 2. Lỗi lớn và cách xử lý

| Nguồn | Phát hiện | Xử lý trong bản 2 |
|---|---|---|
| Toán M1; NeurIPS 2 | §6.3 viện dẫn Định lý 3(b) cho TabPFN v2 trong khi nửa trên của giả định không đúng | Viết lại §6.3. Bỏ câu viện dẫn. Nêu độ đúng nhãn cuối của từng dataset (bảng mới ở phụ lục). Nói rõ ánh xạ gieo đều không mô tả trung thành chuỗi: chuỗi kết thúc cao hơn điểm bất động của nó ở 96% số ô |
| Toán M2; NeurIPS 5 | Abstract, introduction, theory nói quá: "for the sampler as run with a real TFM", "the start decides", "exponential escape time", "thirty sweeps cannot repair" | Mọi câu nay có điều kiện "under a homogeneity assumption". Thời gian thoát viết là cận dưới của *kỳ vọng*. Thêm câu: ở cỡ dữ liệu của bài, định lý không chứng nhận được gì (cần biên gain 0,17–0,30 trong khi gain đo được ≤ 0,15) |
| Toán M3 | Câu tóm tắt Mệnh đề độ lệch dự đoán sai ba chỗ | Sửa: chỉ *số đếm nhãn ẩn* có trung bình hậu nghiệm của prior phóng đại; độ lệch cỡ 1/((K−1)N) theo số điểm **có nhãn** N; đúng tuyệt đối chỉ với khối ngẫu nhiên mới mỗi bước, còn lịch quét của thuật toán chỉ đúng tới số hạng chính |
| Toán M4 | Giả định 1 lấy sup trên mọi trạng thái và mọi khối; phép "kiểm" trong bài kiểm một thứ khác | Giới hạn giả định cho khối cỡ ≤ ⌈n/K⌉ có F ⊄ B. Ghi rõ đây là đòi hỏi mạnh, coi như mô hình của TFM. Câu thực nghiệm đổi thành: kiểm một *hệ quả* dọc chuỗi, không kiểm bất đẳng thức |
| Kiểm toán M1 | "With no trained model": sai, khởi động ấm của LLP là DLLP tuyến tính có huấn luyện | Đổi thành "which does not use L", kèm tách LLP 12/0/0 và PLL 6/1/1. Mục Method nói thẳng các điểm xuất phát có huấn luyện. Bảng định vị có chú thích |
| Kiểm toán M2 | "Fell behind a tuned learner in 14,1%": mốc thật là cái tốt hơn giữa chuỗi đơn và L | Sửa ở abstract, introduction, §6.4. Limitations dùng con số thua riêng L: 9/71 |
| Kiểm toán M3 | Bảng dự đoán giai đoạn phát triển thiếu P1–P4 | Thêm 4 dòng (sinh từ `v0_summary.json`). Con số đúng: 7 sai trong 17. Sửa ba ngưỡng ghi sai (P7, P13, P17) |
| Kiểm toán M4; NeurIPS 5 | "+0,052 so với DLLP" gắn với chữ "mạnh nhất" | Abstract và introduction nêu cả hai: +0,052 so với comparator ghi trước, +0,041 so với baseline mạnh nhất |
| Kiểm toán M5 | Câu về bản MH ghi ngược nghĩa | "Not consistently better", ghi rõ quy ước dấu (MH trừ mặc định) |
| Kiểm toán B1; NeurIPS must-1 | Dấu nháp trong PDF | Bỏ hai TODO; nhãn [C#] ẩn mặc định. Mục "Pending" về giấy phép **giữ có chủ ý** (xem §4) |
| NeurIPS 1 | Phương pháp khuyến nghị không còn "training-free" | Không còn tuyên bố này cho bản nhiều điểm xuất phát. Abstract ghi "one of them a tuned specialised learner" và trung vị +0,003 |
| NeurIPS 3 | Tính mới của Định lý 2 là sơ cấp | Thu hẹp: trích dependency networks và Chen–Ip 2015 (đã đối chiếu); chỉ giữ phần cận sai số. Định lý 1 và 2 được rút gọn |
| NeurIPS 3 | "Gain comes from sampling jointly" không có bằng chứng | Thu hẹp thành "sampling beats arg max"; nêu số của stochastic EM (0,843 so với 0,851) |
| NeurIPS must-9 | Khoảng tin cậy PLL trùng nhau ở hai mức q | Trùng do làm tròn, không phải lỗi. Thêm ghi chú và số 4 chữ số ở phụ lục |

## 3. Lỗi nhỏ đã sửa (chọn lọc)

- **Toán:** quy ước cho Φ̲, Φ̄ ngoài [0, 1]; max{0, 1 − p} trong chứng minh; bản Hoeffding có khoảng dự báo được; đổi tên biến trùng (N_k → Ξ_k); Mệnh đề mô-men nêu đúng giả thiết theo từng điểm; Mệnh đề độ lệch phát biểu theo cỡ khối M; sửa câu về phép kiểm số (15 cấu hình đại số, 9 cấu hình ma trận); nêu tham số của con số "lỏng 10 lần".
- **Kiểm toán:** định nghĩa cell, pair, ngưỡng hoà 0,001; mục tiêu của H3″ và số ô dương tính 66/213; "0,000" thay cho "+0,000"; chi phí ghi rõ 9 hoặc 12 chuỗi và chưa tính tune L; câu "one-sided" nay đúng cho cả hai TFM; "hard for every TFM" sửa lại cho PLL phụ thuộc mẫu (chỉ chạy TabICL); bảng thụt lùi lấy số từ bảng cặp để khỏi làm tròn hai lần.

## 4. Chưa xử lý, và lý do

| Ý | Lý do |
|---|---|
| Đưa mã giả, hình quỹ đạo và hình theo dataset về thân bài | Không đủ chỗ trong 9 trang. Thân bài trỏ rõ tới Phụ lục B, C, F |
| Baseline dạng cây hoặc kNN cho PLL/LLP; so với phương pháp belief propagation | Cần chạy mới. Ghi trong Limitations ("the baselines are neural learners") |
| Study 3 với q khác 0,6, túi khác 8, thêm dataset đa lớp | Cần chạy mới; protocol đã đóng băng. Ghi trong Limitations |
| Ước lượng δ của Giả định 1 trên TFM thật | Dữ liệu đã lưu chỉ cho phần dư trung bình theo vòng quét, tức cận dưới của δ. Bài nay nói thẳng giả định chỉ được kiểm qua hệ quả |
| Mục "Pending" về giấy phép trong checklist | Phải đối chiếu giấy phép từng dataset OpenML, các checkpoint và bộ PALM. Bản final cố ý không biên dịch được khi mục này còn mở |
| Style chính thức của NeurIPS 2027 | Chưa phát hành |

## 5. Vòng 2 (kiểm việc sửa, 12/11/2026)

- **Người rà:** một agent mới, không tham gia vòng 1 và không tham gia việc sửa.
- **Kết quả trên 75 mục của vòng 1:** 34 đã sửa, 23 sửa một phần, 16 chưa sửa, 2 chỗ việc sửa sinh lỗi mới. Phần "chưa sửa" chủ yếu là lỗi nhỏ, lỗi trình bày và các ý cần thí nghiệm mới.
- **Số liệu:** mọi con số mới được tính lại từ dữ liệu thô đều khớp.
- **Điểm kiểu NeurIPS:** Overall từ 4 lên **5** (dao động 4–6); Soundness 2 → 3, Contribution 2 → 3, Presentation giữ 2 (sát 3).

| Phát hiện mới | Mức | Xử lý trong bản nháp 3 |
|---|---|---|
| N1. Câu "naive labels carry less Fisher information … exactly for PLL and LLP" sai với LLP: nhãn ngây thơ khớp số đếm túi nên thông tin Fisher bằng nhau; con số 3% là thông tin Godambe của hợp lý từng điểm. Lỗi này có từ bài gốc | lớn | Sửa cả đoạn ở §5 và phát biểu (b) của mệnh đề ở phụ lục theo đúng `G4/PROOFS.md`: với LLP, mất mát nằm ở *learner* coi nhãn ngây thơ là nhãn độc lập |
| N2. Câu "n_F từ 756 đến 3.934 cần biên 0,17–0,30" gắn sai số | lớn | Sinh lại theo từng dataset: 3 dataset trong phạm vi định lý (n_F 756–1.395) cần biên 0,25–0,35, gain đo được ≤ 0,14; optdigits nằm ngoài phạm vi |
| N3. "Ba trong bốn dataset leo qua vùng gain âm" chưa chính xác | nhỏ | Chỉ còn nói về optdigits (2/3 seed) |
| N4, N5. Chú thích hình quỹ đạo mâu thuẫn với §6.3; cột "start" của bảng chuỗi thực ra là sau vòng quét 1 | nhỏ | Sửa chú thích và tên cột |
| N6. Phần dư của Mệnh đề độ lệch thiếu thừa số | nhỏ | Viết công thức chính xác dạng tích ba thừa số |
| N7. Quy tắc hoà 0,001 phát biểu chung, trong khi Bảng 1 đếm hoà chính xác | nhỏ | Ghi rõ: Study 3 dùng ngưỡng 0,001; Study 1–2 chỉ đếm hoà chính xác (như đã ghi trước) |
| N15. Nhãn [C#] ẩn để lại khoảng trắng trước dấu câu | nhỏ | Thêm `\unskip` vào macro |
| N16. "38 cells" trái định nghĩa "cell"; "stochastic EM", "oracle" chưa định nghĩa | nhỏ | Đổi thành "(dataset, setting) comparisons"; thêm định nghĩa |
| N18. Bảng 2 in "gap +0.000" | nhỏ | In "0.000" |
| Gợi ý: bảng "cấu hình – mức bằng chứng – chi phí" | — | Thêm Bảng ở Phụ lục F |

**Còn lại sau vòng 2** (chưa làm trong đợt này):
- các lỗi nhỏ và lỗi trình bày chưa sửa trong biên bản vòng 2;
- ước lượng δ của Giả định 1;
- baseline chạy bằng CPU trên các lần chia đã lưu của Study 3 (PL-kNN hoặc IPAL, cây tăng cường với loss PLL/LLP). Người rà coi đây là cải thiện giá trị nhất không cần GPU. Việc này là **thí nghiệm mới ngoài protocol**, nên cần thầy quyết định và phải ghi trước;
- các thí nghiệm cần GPU mà reviewer sẽ đòi: naive TFM, hard-EM, oracle ở Study 3; chạy dài T ≥ 100 trên ô kẹt; ánh xạ phản ứng trên Study 3.


## 6. Vòng 3 (13/11/2026): kiểm toán số liệu S3B và phản biện theo tiêu chí TMLR

- **Biên bản:** `G9/reviews/REVIEW_V2_AUDIT_R3.md` (0 nghiêm trọng, 4 lớn, 17 nhỏ) và `G9/reviews/REVIEW_V2_TMLR_R3.md` (khuyến nghị sửa lớn; 0 nghiêm trọng, 10 lớn, 12 nhỏ, 4 trình bày).
- **Bản sau khi sửa:** `paper/main.pdf` (42 trang; chính văn và Broader Impact Statement trong 12 trang, tài liệu tham khảo từ trang 13), gói nguồn ẩn danh `paper/_submission/source.zip`, phụ lục bổ sung `paper/_submission/supplement.zip` (cả hai sinh bằng `make submission`).
- **Phân tích mới, đều là thăm dò (exploratory), viết sau khi đã chấm:**
  - `S3/s3_round3_explore.py` → `S3/results/s3_round3_explore.json`: tách H1″, H2″ theo bối cảnh; B1 theo bối cảnh; tỉ lệ lớp đông nhất; độ lệch chuẩn theo seed; bộ so sánh cùng thông tin; độ đúng nhãn LLP; chuỗi 100 vòng (giá trị đúng vòng 30/100, độ lặp lại của TabDPT); số lượt gọi TFM; lần xem tạm B3.
  - `V2/analysis/w1_delta_chain.py` → `V2/results/w1_delta_chain.csv`: cận dưới ước lượng của δ.

### 6.1 Kiểm toán số liệu (L1–L4, N1–N16)

| Mục | Xử lý |
|---|---|
| L1 chi phí hard-EM gõ tay sai | Cột chi phí của bảng khuyến nghị đọc `n_tfm_calls` từ CSV thô: naive 1, hard-EM 101/106, chuỗi 310/315. §6.5–6.6 nói hard-EM tốn khoảng 1/3 số lượt |
| L2 abstract, intro chỉ nêu vế đúng của P22 | Cả hai vế ở abstract và intro: sửa được 5/35, nhưng trái dự đoán, độ chính xác vẫn tăng +0,053 |
| L3 định nghĩa "repaired" | theory.tex và §6.5: "came within 0.02 of the better of the multi-start average and the tuned learner" |
| L4 "L không phải baseline mạnh nhất" chỉ đúng ở LLP | Macro theo bối cảnh: LLP, L thua GBDT 9/12 (−0,013); PLL, L hơn PL-kNN 7/8 (+0,097). Câu kết luận giới hạn vào LLP |
| N1 r30/r100 là trung bình cửa sổ | Chú thích Bảng B3 nói rõ cửa sổ 28–32 và 96–100, kèm giá trị đúng vòng 30 và 100 |
| N2 "still rising" | Thêm cột "rose > 0.03 after 30" (5/10/4, tổng 19); bỏ chữ "therefore" |
| N3 burn-in 33, TabDPT không tất định | Ghi ở §6.5 và chú thích: 12/24 chuỗi TabDPT khác ngay trong 30 vòng đầu (tới 0,14) |
| N4 baseline tốt hơn chọn theo dataset | Ghi rõ trong §6.5; Phụ lục F, sai lệch (ii): chọn theo bối cảnh cho +0,031, 9/1/10, p_Holm 0,522 |
| N5 số phép kiểm định | "three comparisons registered as descriptive, two-sided, Holm-adjusted together" |
| N6 hard-EM theo bối cảnh, TFM | LLP +0,001; PLL +0,050 (5/1/2); TabDPT LLP là ngoại lệ (0,814 so với 0,786) |
| N7 giải thích PL-kNN | "three datasets on which it stalled (exploratory; median +0.029, 5/0/3)" |
| N8 "on average" | Thêm, kèm "more than 0.02 below on 12 of 20" |
| N9 learner tự nó | "multi-start 0.650, chain from L 0.692; the learner itself 0.717 (exploratory)" |
| N10 deviation thật | Phụ lục F có đoạn "Deviations" (i)–(iii) đúng như `S3B_REPORT.md` §7 |
| N11 "whatever the missing cells gave" | "could have held only if the missing cells had lost more than 0.32 accuracy each" (tính từ commit phiên 1) |
| N12 cấu hình GBDT | "100 iterations, learning rate 0.1, no early stopping, 2-fold out-of-fold relabelling" |
| N13 kết luận | "did not clearly do so" |
| N14 số gõ tay | "twelve" → macro; "5 of" → macro; chi phí từ dữ liệu; ngày giờ chỉ còn trong bảng mốc thời gian |
| N15 "Trained parts" | "tuned L; LLP: also linear DLLP" |
| N16 lỗ đánh số dự đoán | Phụ lục G: P19–P21 thuộc lượt V1E đã huỷ, chưa chạy; P22 ở Phụ lục F |

### 6.2 Phản biện TMLR (F1–F26)

| Mục | Xử lý | Còn lại |
|---|---|---|
| F1 Định lý 3 ngoài phạm vi | Abstract, contributions, intro, related, method, limitations đều nói rõ: nhãn yếu cấp mẫu, context không bị cắt, "with high probability", "certify nothing at our sample sizes". Bỏ chữ "invariant" | — |
| F2 bảo đảm của biến thể chính xác bị gán cho sampler mặc định | Related: bảo đảm chỉ cho biến thể chính xác; Bảng định vị ghi "(exact variants only)"; intro, conclusion: "for an exact TFM … approximately so for a real one"; §6.7 "and the context is not subsampled"; Định lý 1 tách giới hạn quy nạp và chuyển nạp (câu thay thế N12 vòng 2) | — |
| F3 "therefore a mixture …" | "Our data cannot separate a wrong basin from slow mixing; the theorem describes only the former, and only in a model of the TFM" | — |
| F4 phép kiểm chain-seeded vòng tròn | Câu mới: trạng thái gieo gần trạng thái cuối nên gần dừng; chỉ cho thấy vị trí lỗi quan trọng | **Không** làm phép kiểm chéo giữa các chuỗi |
| F5 "robust TFM", "specialised learners" | Tiêu đề §6.2: "Study 2: one chain with TabICL v2 outperformed specialised neural and linear learners"; bỏ "robust"; conclusion nói thêm chuỗi đơn ngang GBDT ở LLP Study 3 | — |
| F6 H2″ chỉ so với TFM ngẫu nhiên | Abstract, intro, Bảng 2 ("a TFM drawn at random"), §6.4 ("a weak benchmark"), limitations; nêu không hơn riêng TabICL v2 | — |
| F7 GBDT-PLL suy biến; thiết kế khi đã biết kết quả | Chọn phương án (b): abstract chỉ nêu LLP 12/12 (trung vị +0,017 AUC) và PL-kNN 8/8; §6.5 nói GBDT ở 3/8 dataset PLL nằm trong 0,02 của tỉ lệ lớp đông nhất (ở 1 trong đó X cũng vậy); bảng B1 thêm cột lớp đông nhất; ghi "designed with its results known", "two-sided, Holm over three"; trích Hüllermeier & Beringer cho PL-kNN; nêu IPAL, InvCal, ∝SVM chưa chạy | **Không** chạy lại GBDT-PLL có tune (cần dữ liệu OpenML, tức một lượt Kaggle CPU) |
| F8 bảng "What to run" | Gộp Bảng 4 vào bảng khuyến nghị; xếp theo chi phí; hàng chuỗi đơn ghi "confirmatory with TabICL v2 only (Study 2)"; thêm 5 hàng thăm dò: chuỗi từ L, chuỗi đơn trung bình qua TFM, chuỗi đơn qua TFM + L, chuỗi từ L trung bình qua TFM, multi-start TabICL v2. Bộ so sánh cùng thông tin: chuỗi đơn mọi TFM + dự đoán L hơn L 20/0/0 nhưng kém X (−0,027, 4/0/16); chuỗi từ L trung bình qua TFM ngang X (−0,001, 5/7/8) với 1/3 chi phí | **Không** chạy ensemble của L (cần Kaggle); bài nói rõ |
| F9 tác hại ở PU, nhiễu, SSL | Limitations, theory, Phụ lục M: kém naive ở 6/6 mức (tới −0,054 AUC); D3 cũng thoả khi phương pháp kém hơn; nêu khả năng do sai lệch của khối không sửa | — |
| F10 ghi trước không kiểm được | `supplement.zip`: README, `freeze_log.csv` (giờ commit và SHA-256 của mọi file đóng băng; không ghi mã commit vì kho công khai), script chấm (đã thay tên định danh), SHA-256 của 95 file kết quả thô. Bài nói rõ giờ commit do tác giả đặt; chỉ Study 3 và S3B có script chấm commit trước kết quả; Study 1, 2 thì script commit cùng kết quả đầu tiên. "Reviewers" → "Internal reviews" | Chưa có kho ẩn danh (anonymous.4open.science) — cần thầy quyết |
| F11 abstract | Thêm đơn vị gộp, "+0.041 over the strongest baseline", PLL không qua với chuỗi thứ hai, cả hai vế P22 | — |
| F12 thống kê | Quy tắc hoà (chỉ cho W/T/L); "two-sided" cho B1 và cho A1 − L; bảng tách theo bối cảnh (Bảng `tab:s3-split`); độ lệch chuẩn theo seed (trung vị 0,013–0,027) | — |
| F13 Mệnh đề 5 | Thân bài dùng câu thay thế N10; MĐ 5(a) thêm M_kk > b_k, "binary … T00 + T11 > 1", "same probability q"; "in every case we computed"; intro "computed for a logistic model" | — |
| F14 lượng từ, δ | Giả thiết 1: khối cỡ ⌊n/K⌋ hoặc ⌈n/K⌉; cận dưới ước lượng δ ≥ 0,05 (trung vị, 78 ô PLL trong phạm vi, tới 0,14) ở §6.3, Remark (iv), Limitations | — |
| F15 Remark (iii), A.5(iii) | Nêu ánh xạ Φ = 0,45 + 0,5r, dải bị cắt ở 1, h_max; thêm K = 5 (tới 0,13) | — |
| F16 bằng chứng trong phạm vi | Dẫn semeion (0,17 → 0,34–0,37 trong vùng gain âm) cạnh optdigits; tỉ lệ "above the fixed point" riêng PLL 94% (108 ô) | — |
| F17 Phụ lục E, Bảng 23 | Câu thay thế m3 (9/10 cặp L hơn multi-start; cặp còn lại chỉ kém chuỗi đơn của chính nó; 4 ô AUC 0,30–0,49); P12 trên 145/234 ô, P13 trên 13/15 dataset | — |
| F18 PLL thật | Nêu lý do đăng ký TabICL v2 (> 10 lớp); "by small margins on two of them" | — |
| F19 sampling vs arg max | "in log loss … in accuracy only at q = 0.6" | — |
| F20 Broader impact | Độ đúng nhãn LLP khôi phục từ số đếm: trung vị 90% (61–96%), so với 72% nếu rút ngẫu nhiên khớp số đếm | — |
| F21 sai lệch Study 1 | Bảng mốc thời gian nêu quy tắc 60% của H4 và vì sao không đổi phán quyết; định nghĩa H1–H5 | — |
| F22 văn liệu | Thêm 8 tài liệu đã đối chiếu (`REFERENCE_CHECK.md`, mục vòng 3): Hogwild/asynchronous Gibbs, SEM, data augmentation, PL-kNN, IPAL, InvCal, ∝SVM | Celeux & Diebolt: chỉ đối chiếu qua danh mục tham khảo của một bài khác |
| F23 thuật ngữ | free, H1–H5, giới hạn TabPFN v2 (≤ 500 đặc trưng, ≤ 10 lớp), tên các suite, quy tắc chọn đối thủ chính, cold start, TabPFN-3.5, PU, "far from the truth" | — |
| F24 ký hiệu | TFM nay là p̃ (q chỉ còn là tỉ lệ distractor); Δ₁, Δ₂; Dobrushin ϑ; Λ-Lipschitz; đoạn "Notation" ở đầu Phụ lục A liệt kê các chữ còn dùng lại | H, τ, λ, a, S vẫn dùng lại, có định nghĩa tại chỗ |
| F25 bảng, hình | Bảng tuning dùng cột xuống dòng, không thu nhỏ; bảng dataset in đủ tên; gộp Bảng 4 vào Bảng 5; Hình 1 có dải min–max theo 3 seed | — |
| F26 câu chữ | Đã sửa mọi mục liệt kê; trích dẫn arXiv không còn "In"; tabimpute ghi v4; Balakrishnan trích bản Annals of Statistics 2017 | "+0.110" ba chữ số (c2): giữ, vì mọi khoảng tin cậy trong bài đều ba chữ số |

### 6.3 Mục của vòng 2 còn treo

- N17 (chuỗi MiceProtein còn tăng ở vòng 30): chú thích Bảng `tab:s1chains` liệt kê các chuỗi tăng > 0,03 giữa vòng 16–20 và 26–30.
- Các mục khác của vòng 2 được xử lý qua F1–F26 ở trên.

### 6.4 Chưa làm, và lý do

1. Chạy lại GBDT-PLL có tune, chạy ensemble của L, thêm IPAL/InvCal/∝SVM: cần dữ liệu OpenML, tức một lượt Kaggle CPU có ghi trước. Bài đã nói rõ các giới hạn này.
2. Phép kiểm response map không vòng tròn (dự đoán chuỗi B bằng map của chuỗi A) và response map trên dataset Study 3: chưa làm; bài đã bỏ khẳng định dự báo.
3. Kho ẩn danh truy cập được cho người phản biện: cần thầy quyết. **Lưu ý:** hai kho `DRuanli/icgibbs` và `DRuanli/neur-v3` đang **công khai** (kiểm bằng GitHub API ngày 13/11/2026), có toàn văn bản thảo và tên tài khoản. Khi nộp bản ẩn danh, nên chuyển hai kho sang riêng tư trong thời gian phản biện.
4. Study 3 với q và cỡ túi khác: ngoài phạm vi; Limitations giữ phạm vi "bags of 8, q ≤ 0.6".
