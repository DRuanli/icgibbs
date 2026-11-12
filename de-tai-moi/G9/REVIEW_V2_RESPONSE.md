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

