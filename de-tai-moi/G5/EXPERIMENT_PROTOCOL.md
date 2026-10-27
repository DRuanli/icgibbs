# Giao thức thực nghiệm (G5) — đề tài A: In-Context Gibbs

> **ĐÃ ĐÓNG BĂNG 26/10/2026** sau khi thầy duyệt [GATE_REVIEW_G5](GATE_REVIEW_G5.md); mọi việc ở Mục 12 đã xong. Dataset card: [DATASET_CARDS](DATASET_CARDS.md). Mã băm của bản đóng băng: [FREEZE_RECORD](FREEZE_RECORD.md). **Từ nay mọi thay đổi ghi ở Mục 11**, kèm lý do và việc kết quả liên quan có phải gắn nhãn exploratory hay không.
> Nguồn quyết định: [RESEARCH_PLAN](../G3/RESEARCH_PLAN.md) (đã đóng băng, kèm §9), [GATE_REVIEW_G4](../G4/GATE_REVIEW_G4.md) §3c–3d, [PILOT_V2_REPORT](../G4/pilot/PILOT_V2_REPORT.md). Quy tắc chung: [06_THIET_KE_THUC_NGHIEM](../../research-handbook/06_THIET_KE_THUC_NGHIEM.md).

Đề tài: *In-Context Gibbs: học từ nhãn yếu bằng tabular foundation model, không huấn luyện lại* · Ngày đóng băng: **26/10/2026** · Commit: `c4da1f4` (kho git riêng tư của đề tài; SHA-256 từng file trong [FREEZE_RECORD](FREEZE_RECORD.md))

## 0. Phạm vi khẳng định sau G4

| Bối cảnh | Vai trò | RQ | Metric chính |
|---|---|---|---|
| LLP, túi 8 và 16 | **Chính** | RQ2, RQ3 | ROC AUC |
| PU-SAR, E[e \| y=1] ∈ {0,1; 0,3} | **Chính** | RQ2, RQ3 | ROC AUC |
| PLL, q ∈ {0,3; 0,6} | **Chính** — thí điểm RQ4 đạt 6/6 ô (26/10/2026) | RQ4, RQ3 | accuracy |
| SSL, tỉ lệ có nhãn ∈ {0,05; 0,2} | **Phụ** — thí điểm RQ4 không đạt ngưỡng 60% (3/12 ô; 26/10/2026) | RQ4 (phụ), RQ3 | ROC AUC |
| Nhãn nhiễu phụ thuộc lớp, r ∈ {0,2; 0,4} | **Phụ** — thí điểm RQ4 không đạt ngưỡng 60% (4/12 ô; 26/10/2026) | RQ4 (phụ), RQ3 | ROC AUC |
| LLP túi 32 | Kiểm tra áp lực, báo cáo như giới hạn | RQ2 (phụ) | ROC AUC |
| PU-SCAR, c ∈ {0,1; 0,3} | **Phụ** (R3 — thầy xác nhận 26/10/2026) | RQ2 (phụ), RQ3 | log loss khớp tiên nghiệm |
| Nhãn yếu thật: PLL thật (Lost, MSRCv2, BirdSong); LLP-Bench | Kiểm chứng ngoại tại; LLP-Bench là kiểm tra áp lực | RQ5 | accuracy (PLL); ROC AUC (LLP) |

Bối cảnh phụ vẫn chạy đầy đủ trên tập đánh giá và báo cáo trung thực (kể cả khi thua); chúng không vào abstract. Quy tắc phân vai cho RQ4 (viết trong `kaggle_g4_rq4_pilot.py` trước khi chạy): một loại nhãn yếu ở lại khẳng định chính nếu trên tập phát triển In-Context Gibbs đạt hoặc vượt **cả** baseline peer-review tốt nhất **lẫn** TFM ngây thơ ở ≥ 60% ô (dataset × mức) — cùng ngưỡng với H4.

*Nhận xét (giả thuyết, sẽ viết thành mệnh đề ở G6):* hai bối cảnh mà Gibbs không giúp được về AUC — PU-SCAR và nhãn nhiễu phụ thuộc lớp — có chung một tính chất: $p(w=1\mid x)$ là hàm affine tăng của $p(y=1\mid x)$ (PU-SCAR: $c\,p(y{=}1\mid x)$; nhiễu: $\rho_{01}+(1-\rho_{10}-\rho_{01})\,p(y{=}1\mid x)$ khi $\rho_{10}+\rho_{01}<1$), nên xếp hạng của TFM ngây thơ đã trùng xếp hạng Bayes. Gibbs có chỗ để giúp khi nhãn yếu ở dạng tập (PLL), tổng (LLP) hoặc phụ thuộc mẫu (PU-SAR) — nơi bộ ngây thơ không nhất quán.

## 1. Dataset

### 1.1 Tiêu chí chọn (viết trước khi xem bất kỳ kết quả nào trên tập đánh giá)
- **Nguồn:** bộ OpenML-CC18 (Bischl et al., NeurIPS Datasets & Benchmarks 2021), ID cố định.
- **Nhị phân** (LLP, PU, SSL, nhãn nhiễu): mọi dataset nhị phân của CC18 thoả 500 ≤ n ≤ 10.000, d ≤ 500, lớp nhỏ nhất ≥ 10%, dữ liệu thật (không phải bài toán nhân tạo).
- **Đa lớp** (PLL): mọi dataset của CC18 có 3–10 lớp (giới hạn của TabPFN v2), 500 ≤ n ≤ 10.000, d ≤ 500.
- **Độc lập (C1):** mỗi nhóm nguồn chỉ lấy một dataset — dataset có ID nhỏ nhất trong nhóm. Các nhóm đã biết: NASA MDP (pc1, pc3, pc4, kc1, kc2, jm1); mfeat-*; splice và dna.
- **Loại trừ tập phát triển và mọi dataset cùng nguồn với nó:**
  - nhị phân: credit-g, diabetes, blood-transfusion, banknote, ilpd, wdbc; thêm breast-w vì cùng nguồn Wisconsin với wdbc;
  - đa lớp: vehicle, segment (cả ID 36 lẫn 40984), cmc.
- **Lớp dương:** lớp thiểu số. Đa lớp: mã hoá lớp theo thứ tự chữ cái của nhãn gốc.

### 1.2 Danh sách áp tiêu chí (đã xác minh bằng `g5_verify_datasets.py` trên Kaggle, 25–26/10/2026; kết quả: `G4/pilot/g5_dataset_check.csv`)

Xác minh chỉ đọc metadata và dữ liệu, không chạy phương pháp nào. Mọi ID khớp tên; giấy phép đều Public. n, d, tỉ lệ lớp nhỏ nhất dưới đây là số đo thật. "Trùng" = hàng trùng hoàn toàn (X và y); "xung đột" = số hàng thuộc nhóm cùng X khác y.

**Nhị phân (10):**

| # | Tên | OpenML ID | Nhóm nguồn | n | d | Lớp nhỏ nhất | Thiếu | Trùng | Xung đột | Ghi chú |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | kr-vs-kp | 3 | cờ vua (UCI) | 3196 | 36 | 47,8% | 0 | 0 | 0 | toàn bộ là feature phân loại |
| 2 | credit-approval | 29 | tín dụng Úc (Quinlan) | 690 | 15 | 44,5% | 0,7% | 0 | 0 | |
| 3 | spambase | 44 | email (HP Labs) | 4601 | 57 | 39,4% | 0 | 391 | 6 | n sau loại trùng 4210 |
| 4 | tic-tac-toe | 50 | trò chơi (UCI) | 958 | 9 | 34,7% | 0 | 0 | 0 | phân loại |
| 5 | pc4 | 1049 | NASA MDP | 1458 | 37 | 12,2% | 0 | 114 | 1 | đại diện nhóm NASA; n sau loại trùng 1344 |
| 6 | phoneme | 1489 | âm vị (ELENA) | 5404 | 5 | 29,4% | 0 | 9 | 0 | |
| 7 | qsar-biodeg | 1494 | hoá học | 1055 | 41 | 33,7% | 0 | 3 | 0 | |
| 8 | cylinder-bands | 6332 | công nghiệp in | 540 | 37 | 42,2% | 5,0% | 0 | 0 | |
| 9 | dresses-sales | 23381 | bán lẻ | 500 | 12 | 42,0% | 13,9% | 0 | 2 | ở biên n = 500 |
| 10 | **sick** (thay churn) | 38 | tuyến giáp (Garavan) | 3772 | 29 | 6,1% | 5,5% | 61 | 0 | dự phòng số 1, ngưỡng lớp nhỏ nhất 5%; n sau loại trùng 3711 |

**churn (40701) bị loại** (26/10/2026): tệp nguồn gốc của bộ này (kho MLC++ của SGI, cùng kích thước 5000 × 20) ghi rằng dữ liệu là "artificial based on claims similar to real world" (tài liệu của gói R `modeldata`, mục `mlc_churn`), vi phạm tiêu chí "dữ liệu thật" của Mục 1.1. Lý do loại không liên quan tới kết quả (chưa chạy phương pháp nào trên nó). Thay bằng dataset đầu tiên của danh sách dự phòng theo thứ tự cố định: sick (38). Lưu ý cho phân tích: sick có lớp nhỏ nhất 6,1%, thấp hơn ngưỡng 10% của danh sách chính (đúng như điều khoản dự phòng); kết quả trên sick sẽ được báo cáo kèm chú thích này.

Bị loại theo tiêu chí: sick (6% < 10%), ozone-level-8hr (6%), wilt (5%), climate-model-simulation-crashes (8,5%, mô phỏng), madelon (nhân tạo), Bioresponse và Internet-Advertisements (d > 500), PhishingWebsites, jm1, bank-marketing, adult, electricity, nomao, numerai (n > 10.000).

**Danh sách dự phòng** (thứ tự cố định; chỉ dùng khi một dataset ở trên bị loại vì lý do ghi ở Mục 11, không vì kết quả): sick (38), ozone-level-8hr (1487), wilt (40983) — các dataset nhị phân nới ngưỡng lớp nhỏ nhất xuống 5%.

**Đa lớp cho PLL — cả 14 ứng viên đều đạt tiêu chí sau xác minh; dùng cả 14** (quyết định 26/10/2026, trước khi chạy bất kỳ phương pháp nào trên chúng; điều khoản cũ là "≥ 10 đầu tiên", dùng hết để tăng lực kiểm định, chi phí thêm khoảng 1 giờ T4):

| Tên | ID | n | d | Số lớp | Lớp nhỏ nhất | Trùng | Xung đột | Ghi chú |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| balance-scale | 11 | 625 | 4 | 3 | 7,8% | 0 | 0 | |
| mfeat-factors | 12 | 2000 | 216 | 10 | 10,0% | 6 | 0 | đại diện nhóm mfeat |
| optdigits | 28 | 5620 | 64 | 10 | 9,9% | 0 | 0 | |
| splice | 46 | 3190 | 60 | 3 | 24,0% | 184 | 2 | |
| satimage | 182 | 6430 | 36 | 6 | 9,7% | 0 | 0 | |
| eucalyptus | 188 | 736 | 19 | 5 | 14,3% | 0 | 0 | thiếu 3,2% |
| analcatdata_authorship | 458 | 841 | 70 | 4 | 6,5% | 0 | 0 | |
| analcatdata_dmft | 469 | 797 | 4 | 6 | 15,4% | 241 | 329 | n sau loại trùng 556; nhãn rất nhiễu (41% hàng thuộc nhóm cùng X khác y) — báo cáo kèm chú thích |
| first-order-theorem-proving | 1475 | 6118 | 51 | 6 | 7,9% | 482 | 209 | n sau loại trùng 5636 |
| wall-robot-navigation | 1497 | 5456 | 24 | 4 | 6,0% | 0 | 0 | |
| semeion | 1501 | 1593 | 256 | 10 | 9,7% | 0 | 0 | |
| MiceProtein | 40966 | 1080 | 77 | 8 | 9,7% | 0 | 0 | thiếu 1,7% |
| car | 40975 | 1728 | 6 | 4 | 3,8% | 0 | 0 | |
| steel-plates-fault | 40982 | 1941 | 27 | 7 | 2,8% | 0 | 0 | |

Hàng trùng được loại trước khi chia (Mục 2); hàng "cùng X khác y" được giữ nguyên vì đó là nhiễu sẵn có của dữ liệu.

### 1.3 Nhiễm bẩn dữ liệu huấn luyện trước (06 §1.3)
TabPFN v2 và TabICL được huấn luyện trước hoàn toàn trên dữ liệu tổng hợp. Việc phải làm ở G6: đối chiếu danh sách trên với các dataset mà nhóm TabPFN và TabICL dùng để **phát triển** (chọn prior, chọn checkpoint). Dataset nào trùng thì vẫn giữ (tiêu chí đã viết trước), nhưng báo cáo riêng.

### 1.4 Nhãn yếu thật (RQ5)
- **LLP-Bench** (Brahmbhatt et al., NeurIPS 2023 D&B; dữ liệu Criteo): túi ngẫu nhiên cỡ 64–512, túi theo feature trung bình 150–500 phần tử. Cỡ túi này **nằm ngoài phạm vi khẳng định** sau G4 (túi ≤ 16). Đề xuất: dùng như kiểm tra áp lực ngoại tại, kèm dự đoán đã viết trước là phương pháp yếu ở đây. Giữ H5 nhưng ghi rõ điều này.
- **Nguồn thứ hai — đã chọn (26/10/2026, thầy giao Claude chọn): các bộ PLL thật dạng bảng** của nhóm PALM (palm.seu.edu.cn/zhangml/Resources.htm; chỉ dùng cho học thuật, ghi nguồn bài gốc). Lý do chọn: (1) PLL là loại duy nhất của RQ4 còn triển vọng trên tập phát triển, nên nguồn thật nên kiểm chứng đúng loại đó; (2) đây là bộ chuẩn của văn liệu PLL (PRODEN, Feng et al. 2020, LWS, …) nên có số công bố để đối chiếu; (3) dữ liệu dạng bảng, feature đã trích sẵn, n vừa với TFM. Phương án (c) (nhãn nhiễu thật hoặc PU thật) bị loại vì cả hai loại đều đã thành bối cảnh phụ.
  - **Tiêu chí chọn bộ (viết trước khi đọc dữ liệu):** 500 ≤ n ≤ 10.000; d ≤ 500; số lớp ≤ 26 (lớn nhất đã thử trên tập phát triển); nhãn đúng nằm trong tập ứng viên ở ≥ 99% mẫu. Theo số công bố, đạt: **Lost** (1122 × 108, 16 lớp, 2,23 ứng viên), **MSRCv2** (1758 × 48, 23 lớp, 3,16), **BirdSong** (4998 × 38, 13 lớp, 2,18). Không đạt: Soccer Player (n = 17.472; 171 lớp), Yahoo! News (n = 22.991; 219 lớp), FG-NET (78 lớp), Mirflickr (d = 1536). Xác minh chỉ đọc bằng `g5_verify_pll_real.py` (việc F7), không chạy phương pháp nào.
  - **Kết quả F7 (26/10/2026):** cả 6 bộ tải được và khớp số công bố (n, d, số lớp, số ứng viên trung bình); nhãn đúng nằm trong tập ứng viên ở 100% mẫu, mỗi mẫu đúng một nhãn đúng. Đạt tiêu chí: **Lost, MSRCv2, BirdSong**; không đạt: FG-NET, Soccer Player, Yahoo! News. Ba chi tiết ràng buộc cách cài ở G6: (1) Lost có 2 trong 16 lớp không bao giờ là nhãn đúng (chỉ xuất hiện như ứng viên sai) → không gian lớp lấy từ số hàng của `partial_target` (Q), **không** suy từ nhãn; (2) MSRCv2 có lớp nhỏ nhất 3 mẫu và 5 hàng trùng (loại theo Mục 2 → 1753 hàng); (3) BirdSong lớp nhỏ nhất 32 mẫu. SHA-256 của tệp gốc ở DATASET_CARDS Mục D; kết quả: `G4/pilot/g5_pll_real_check.csv`.
  - **Vượt giới hạn 10 lớp của TabPFN v2 — đã chọn: TabICL v2** (26/10/2026; `support_many_classes=True`, checkpoint `tabicl-classifier-v2-20260212.ckpt`, `n_estimators = 1` trong chuỗi, 2 khi dự đoán cuối). Chọn bằng thí điểm many-class trên tập phát triển (việc F6, [PILOT_MANYCLASS_REPORT](../G4/pilot/PILOT_MANYCLASS_REPORT.md)) theo quy tắc viết trong script trước khi chạy: không backend nào vượt 30 phút mỗi ô (TabPFN v2 + ECOC 17,9 phút, TabICL 0,9 phút trên letter); accuracy oracle trung bình TabICL 0,942 so với 0,930 (chênh 1,2 điểm % > 0,5) → TabICL.
  - **Chia dữ liệu cho RQ5-PLL — đã chọn: 10 lần chia ngẫu nhiên 90/10, không phân tầng** (seed 0–9), như văn liệu (Feng et al. 2020), vì chi phí ước tính với TabICL khoảng 2 giờ T4 ≤ 6 giờ (quy tắc đã viết trước). Baseline: PRODEN, và các phương pháp PLL peer-review ở Mục 4; số của chúng tự chạy, số công bố chỉ để đối chiếu.
  - **Hai nguồn của RQ5 và H5:** PLL thật (3 bộ) và LLP-Bench (kiểm tra áp lực). H5 đánh giá trên PLL thật; LLP-Bench báo cáo kèm dự đoán đã viết trước (phương pháp yếu ở túi cỡ ≥ 64).

### 1.5 Dataset có cấu trúc đặc biệt — phân tích độ nhạy đăng ký trước
Phát hiện khi viết dataset card (26/10/2026, trước khi chạy bất kỳ phương pháp nào trên tập đánh giá):

| Nhóm | Dataset | Vấn đề | Xử lý |
|---|---|---|---|
| Nhóm / chuỗi | MiceProtein (40966) | khoảng 15 phép đo lặp mỗi con chuột; chia ngẫu nhiên đặt cùng một con ở cả train lẫn test (L3.2) | giữ chia ngẫu nhiên (thông lệ CC18); vào phân tích độ nhạy S1 |
| Nhóm / chuỗi | wall-robot-navigation (1497) | đọc cảm biến liên tiếp khi robot di chuyển; mẫu kề nhau gần trùng (L3.1) | như trên |
| Nhóm / chuỗi | BirdSong (RQ5) | nhiều âm tiết từ cùng một bản ghi | chia ngẫu nhiên như văn liệu PLL để so được với số công bố; ghi ở phần giới hạn |
| Sinh từ quy tắc đã biết | balance-scale (11), car (40975) | liệt kê đủ tổ hợp thuộc tính, nhãn sinh tất định từ một mô hình (không có nhiễu đo) | tiêu chí đa lớp ở Mục 1.1 không có điều khoản "dữ liệu thật", nên giữ; vào phân tích độ nhạy S1 |

**S1 (đăng ký trước):** mọi kiểm định của H4 (PLL) chạy hai lần — trên cả 14 dataset và trên 10 dataset còn lại sau khi bỏ bốn dataset trên. Báo cáo cả hai. Nếu kết luận khác nhau, bài ghi rõ và dùng kết luận yếu hơn trong abstract. Lý do riêng cho đề tài: rò rỉ do mẫu gần trùng có thể có lợi cho phương pháp dựa trên ngữ cảnh (TFM chép nhãn từ láng giềng gần trùng) hơn cho baseline MLP, nên không thể giả định nó tác động đều lên mọi phương pháp.

## 2. Tiền xử lý (06 §2: mọi bước fit trong train của từng lần chia)

| Bước | Áp cho | Fit trên | Tham số |
|---|---|---|---|
| Mã hoá feature phân loại thành mã thứ tự (giá trị thiếu = NaN) | TFM | không cần fit | — |
| Tiền xử lý nội bộ của TFM | TFM | ngữ cảnh | mặc định của gói |
| Điền giá trị thiếu bằng trung vị, chuẩn hoá z | MLP baseline | train | — |
| One-hot feature phân loại | MLP baseline | train | bỏ mức chưa gặp |
| Loại hàng trùng giữa train và test | mọi phương pháp | toàn bộ, trước khi chia | báo cáo số hàng bị loại |

## 3. Chia dữ liệu
- Mỗi dataset: **3 lần chia** train/test phân tầng 70/30, seed 0, 1, 2 (giao thức rút gọn của G2). Nhãn yếu sinh trên train từ cùng seed theo cơ chế ở Mục 3.1. Mọi phương pháp dùng chung các lần chia và cùng nhãn yếu.
- Train có thể tới 7.000 hàng. In-Context Gibbs dùng ngữ cảnh ≤ 2.000 hàng (lấy mẫu con đều mỗi lần gọi, D4).
- **Chọn siêu tham số khi chỉ có nhãn yếu:** không có nhãn sạch trong train. Baseline nào cần tune thì tách 20% train làm validation **có nhãn yếu**, chấm bằng loss của chính baseline trên nhãn yếu (LLP: loss tỉ lệ túi; PU: rủi ro PU không chệch; nhiễu: loss đã hiệu chỉnh; PLL: loss trên tập ứng viên; SSL: loss trên phần có nhãn). In-Context Gibbs **không tune theo dataset**; cấu hình chốt trên tập phát triển (Mục 4).

### 3.1 Cơ chế sinh nhãn yếu và mức
| Loại | Cơ chế | Mức |
|---|---|---|
| LLP | chia train ngẫu nhiên thành túi cỡ b, biết số đếm nhãn từng túi | b ∈ {8, 16} chính; 32 áp lực |
| PU-SAR | $e(x)=\sigma(\eta_0+2z)$, z = cột đầu tiên của ma trận feature đã mã hoá, chuẩn hoá; η₀ chọn để E[e \| y=1] bằng mức | {0,1; 0,3} |
| PU-SCAR | mỗi dương được gán nhãn với xác suất c | c ∈ {0,1; 0,3} |
| SSL | tỉ lệ có nhãn phân tầng, ≥ 2 mẫu mỗi lớp | {0,05; 0,2} |
| Nhãn nhiễu | phụ thuộc lớp: dương→âm với r, âm→dương với r/2; T đã biết | r ∈ {0,2; 0,4} |
| PLL | mỗi nhãn sai vào tập ứng viên độc lập với xác suất q | q ∈ {0,3; 0,6} |

## 4. Phương pháp so sánh

**Phương pháp đề xuất — cấu hình cố định chọn trên tập phát triển (G4):** khởi động ấm (D6), khối trên mọi mẫu (D1), K = 10, 30 vòng quét, bỏ 10, Rao–Blackwell với 10 mẫu, `n_estimators = 1` trong chuỗi và mặc định khi dự đoán cuối, ngữ cảnh ≤ 2.000. TFM chính: TabPFN v2; TFM phụ: TabICL v2 (kiểm tra độ bền theo TFM). Với PLL thật > 10 lớp (RQ5): TabICL v2 (chọn ở F6, Mục 1.4), dùng chung cho mọi phương pháp dựa trên TFM. Dự đoán quy nạp (đề xuất cho việc mở G4-2: quy nạp cho thực nghiệm chính vì rẻ và đã được thí điểm; chuyển nạp — bản Bayes chính xác theo Định lý 1c — là ablation ở Mục 7).

| Phương pháp | Loại | Venue | Code | Tune | Bối cảnh |
|---|---|---|---|---|---|
| In-Context Gibbs | đề xuất | — | `icgibbs` | không (cấu hình chốt ở G4) | tất cả |
| TFM ngây thơ | đơn giản (cùng TFM) | — | `icgibbs` | không | tất cả |
| hard-EM cùng TFM, cùng điểm khởi động | ablation RQ3 | (Celeux & Govaert 1992) | `icgibbs` | không | tất cả |
| SEM (mẫu cuối) | ablation RQ3 | (Celeux & Diebolt 1985) | `icgibbs` | không | tất cả |
| GLWS | peer-review | ICML 2024 | github.com/Hhhhhhao/General-Framework-Weak-Supervision | 20 cấu hình | LLP, PU, PLL |
| DLLP (MLP và tuyến tính) | peer-review | ICDM 2017 | cài lại (đã kiểm ở G4); tìm bản chính thức ở G6 | 20 cấu hình | LLP |
| LLP-VAT | peer-review | ECML-PKDD 2020 | github.com/kevinorjohn/LLP-VAT | 20 cấu hình | LLP |
| Easy-LLP | peer-review | NeurIPS 2023 | **xác minh ở G6** | 20 cấu hình | LLP |
| SAR-EM | peer-review | ECML-PKDD 2019 (Bekker et al.) | `pysarpu` (MIT, bên thứ ba); **tìm bản của tác giả ở G6** | 20 cấu hình | PU-SAR |
| nnPU, uPU | peer-review | NeurIPS 2017; ICML 2015 | bản PyTorch của nnPU; **xác minh ở G6** | 20 cấu hình | PU |
| VPU | peer-review | NeurIPS 2020 | **xác minh ở G6** | 20 cấu hình | PU-SCAR |
| In-Context PU | preprint mạnh nhất (gần nhất) | arXiv 2605.05591 | **xác minh ở G6** | mặc định | PU-SCAR |
| Label Spreading; self-training | peer-review (cổ điển) | NIPS 2003; ACL 1995 | scikit-learn | 20 cấu hình | SSL |
| VIME | peer-review | NeurIPS 2020 | **xác minh ở G6** | 20 cấu hình | SSL |
| Forward correction | peer-review | CVPR 2017 | cài lại (T đã biết) | 20 cấu hình | nhãn nhiễu |
| Confident Learning (TFM sau khi lọc) | peer-review | JAIR 2021 | cleanlab | mặc định | nhãn nhiễu |
| PRODEN | peer-review | ICML 2020 | **xác minh ở G6** | 20 cấu hình | PLL (cả PLL thật của RQ5) |
| RC, CC (Feng et al.) | peer-review | NeurIPS 2020 | **xác minh ở G6** | 20 cấu hình | PLL thật (RQ5; có số công bố trên Lost, MSRCv2, BirdSong để đối chiếu) |
| ILL | peer-review | NeurIPS 2024 | **xác minh ở G6** | 20 cấu hình | PLL, nhãn nhiễu, SSL |
| TFM với nhãn sạch | trần tham chiếu | — | — | không | tất cả (không tính vào so sánh) |

**Quy tắc công bằng:**
1. Mỗi baseline cần tune nhận **cùng ngân sách: 20 cấu hình random search** trên validation có nhãn yếu (Mục 3). Báo cáo cả bản mặc định lẫn bản tune.
2. In-Context Gibbs không tune theo dataset. Toàn bộ lựa chọn của nó đến từ tập phát triển, nên lợi thế duy nhất nó có là việc chọn cấu hình chung trên tập phát triển; baseline được phép tune theo từng dataset.
3. Mọi phương pháp nhận cùng thông tin phụ: tỉ lệ lớp (PU), ma trận T (nhiễu), thuộc tính propensity (SAR).

## 5. Metric
- **Chính:** theo bảng ở Mục 0. Log loss khớp tiên nghiệm cho PU-SCAR: định nghĩa ở GATE_REVIEW_G4 §3c, Q1.
- **Phụ:** log loss, Brier, accuracy tại ngưỡng 0,5, PR-AUC (PU), lỗi hiệu chỉnh (ECE).
- **Tính toán:** số lần forward của TFM, thời gian thực, GPU, bộ nhớ đỉnh.
- **Chẩn đoán chuỗi** (G6, mọi lần chạy): tỉ lệ nhãn đổi mỗi vòng; $\hat R$ từ 2 chuỗi trên mẫu con 3 dataset.
- Báo cáo điểm thô theo (dataset, mức, seed); không chỉ báo cáo điểm chuẩn hoá.

## 6. Kế hoạch thống kê

| Khẳng định | Đơn vị | Kiểm định | α | Hiệu chỉnh | Effect size | Tiêu chí (đã đăng ký ở G3, cập nhật ở RESEARCH_PLAN §9) |
|---|---|---|---|---|---|---|
| H2 — LLP, PU-SAR (chính) | trung bình 3 seed mỗi dataset, mỗi mức | Wilcoxon signed-rank hai phía, Gibbs so với **hai đối thủ chính**: (i) baseline peer-review tốt nhất của bối cảnh, xác định theo quy tắc ở dưới; (ii) TFM ngây thơ | 0,05 | Holm trên 4 phép so sánh của một bối cảnh (2 đối thủ × 2 mức) | chênh AUC trung bình, CI bootstrap 95% | cả 4 phép so sánh có ý nghĩa sau Holm theo hướng Gibbs tốt hơn, **và** thắng ≥ 7/10 dataset so với baseline peer-review mạnh nhất trên tập đánh giá của từng mức |
| H2 — PU-SCAR (phụ) | như trên | như trên (hai đối thủ chính), với log loss khớp tiên nghiệm | 0,05 | Holm trên 4 phép so sánh | chênh log loss | như trên; thêm AUC không kém naive quá 0,01 (kiểm định không kém một phía) |
| H3 (RQ3) | ô (dataset × mức) | Bayesian signed-rank, ROPE ±0,005 log loss; Spearman giữa mức yếu và khoảng cách | — | — | P(Gibbs tốt hơn), P(ROPE) | Gibbs tốt hơn hard-EM ở ≥ 70% ô; bác bỏ nếu P(ROPE) > 0,9 hoặc ρ ≤ 0 |
| H4 (RQ4) | trung bình seed mỗi dataset | Wilcoxon hai phía với hai đối thủ chính như H2; thắng/hoà/thua | 0,05 | Holm trên 4 phép so sánh của một loại (2 đối thủ × 2 mức) | chênh accuracy (PLL), CI bootstrap 95% | đạt hoặc vượt baseline mạnh nhất trên ≥ 60% dataset mỗi loại. Khẳng định chính chỉ cho PLL (nếu F1 xác nhận); SSL và nhãn nhiễu kiểm định như nhau nhưng báo cáo là bối cảnh phụ (Mục 0) |
| H5 (RQ5) | nguồn | thứ hạng trung bình qua các lần chia | — | — | | top 2 trên PLL thật (Lost, MSRCv2, BirdSong); LLP-Bench báo cáo riêng như kiểm tra áp lực |

**Đối thủ chính và lý do (quyết định 26/10/2026, F8).**
- *Quy tắc chọn baseline peer-review tốt nhất của một bối cảnh* (viết trước khi chạy tập đánh giá): baseline peer-review có metric chính trung bình (qua các mức chính của bối cảnh) cao nhất trên 6 dataset phát triển, **sau khi tune ở G6** theo đúng ngân sách ở Mục 4. Kết quả chọn ghi vào `docs/primary_comparators.md` trước khi bắt đầu G7. Quy tắc này chọn đối thủ mạnh nhất, trung lập với phương pháp đề xuất. Tập phát triển của PLL là vehicle, segment, cmc.
- *Phân tích phụ, luôn báo cáo:* Gibbs so với **từng** baseline peer-review còn lại (Wilcoxon, Holm trên mọi cặp baseline × mức), bảng thắng/hoà/thua, và kết luận theo tiêu chí chặt của bản nháp G5 trước đó ("thắng mọi baseline sau Holm").
- *Lý do:* (1) đây đúng là phát biểu H2 đã đóng băng ở G3 ("thắng baseline peer-review tốt nhất của mỗi bối cảnh … trên ≥ 7/10 dataset; Wilcoxon kèm Holm") và câu hỏi RQ2 ("vượt baseline peer-review tốt nhất *và* TFM ngây thơ"); bản nháp G5 trước đó đã vô tình viết chặt hơn G3. (2) Lực kiểm định: với n = 10, p hai phía nhỏ nhất của Wilcoxon là 0,0020; Holm trên khoảng 10 cặp đòi p ≤ 0,005 cho phép so sánh đầu, tức gần như phải thắng mọi dataset trước mọi baseline (GATE_REVIEW_G5, Phát hiện 1). Với 4 phép so sánh, ngưỡng đầu là 0,0125, tức tổng hạng của các dataset thua phải ≤ 3 (ví dụ thua một dataset có chênh lệch nhỏ thứ ba trở xuống).

Không khẳng định "tương đương" từ p > 0,05; khẳng định tương đương dùng ROPE.

## 7. Ablation và độ nhạy (trên tập đánh giá, báo cáo đầy đủ)

| Thí nghiệm | Thành phần | Dải |
|---|---|---|
| RQ3 | Gibbs / SEM / hard-EM | — |
| Khởi tạo | ấm / lạnh | — |
| Số khối K | 5, 10, 20 | 3 dataset, 1 mức |
| Số vòng quét | 15, 30, 60 | 3 dataset, 1 mức |
| Dự đoán | quy nạp / chuyển nạp (G4-2) | toàn bộ LLP túi 8 |
| Khối có sửa MH (chính xác) | bật / tắt | 3 dataset nhỏ |
| Propensity | học η / đã biết; prior η ∈ {N(0,1), N(0,2²), N(0,4²)} | PU-SAR |
| TFM | TabPFN v2 / TabICL v2 | toàn bộ bối cảnh chính |
| Cỡ dữ liệu | n_train ∈ {500, 1000, 2000, 4000} | 3 dataset lớn nhất |

## 8. Checklist rò rỉ (người kiểm duyệt đánh dấu)
- [ ] L1.1 có test riêng — test 30% không bao giờ vào ngữ cảnh hay validation
- [ ] L1.2 tiền xử lý trong lần chia — MLP: scaler và imputer fit trên train; TFM: tiền xử lý nội bộ trên ngữ cảnh
- [ ] L1.3 không chọn feature
- [ ] L1.4 đã loại trùng — `g5_verify_datasets.py` báo số hàng trùng; loại trước khi chia
- [ ] L2 không có feature bất hợp lệ — dataset chuẩn CC18; đọc dataset card
- [ ] L3.1 yếu tố thời gian — không dataset nào là bài toán dự báo; riêng wall-robot-navigation là chuỗi đọc cảm biến liên tiếp (mẫu kề nhau gần giống nhau) → xử lý theo Mục 1.5
- [ ] L3.2 cấu trúc nhóm — MiceProtein (khoảng 15 phép đo mỗi con chuột), BirdSong (nhiều âm tiết mỗi bản ghi) có nhóm; chia ngẫu nhiên theo thông lệ của CC18 và văn liệu PLL → xử lý theo Mục 1.5
- [ ] L3.3 test đại diện — chia phân tầng
- [ ] Riêng cho đề tài: nhãn sạch của train chỉ dùng để sinh nhãn yếu và để chẩn đoán; không phương pháp nào (kể cả bước khởi động ấm) đọc nhãn sạch

## 9. Compute dự kiến
Phần cứng: Kaggle GPU T4 (30 giờ/tuần). Đo ở G4: một chuỗi khoảng 55 giây với n_train ≤ 960; hard-EM khoảng 17 giây.

| Khối việc | Số lần chạy | Ước tính |
|---|---:|---:|
| Bối cảnh chính + phụ (LLP 3 mức, PU-SAR 2, PU-SCAR 2, SSL 2, nhiễu 2) × 10 dataset × 3 seed | 330 | khoảng 2 phút mỗi lần (n lớn hơn tập phát triển) → 11 giờ |
| PLL 2 mức × 14 × 3 | 84 | 3 giờ |
| Ablation RQ3 (hard-EM, SEM đi kèm) | | +4 giờ |
| Baseline (20 cấu hình × 18 phương pháp-bối cảnh) | | khoảng 25 giờ, phần lớn chạy CPU được |
| Ablation khác, TabICL | | khoảng 10 giờ |
| RQ5: PLL thật 3 bộ × 10 lần chia (TabICL v2) + LLP-Bench | | khoảng 2 + 3 giờ |
| **Tổng** | | khoảng 58 giờ; **× 2 dự phòng ≈ 115 giờ** ≈ 4 tuần quota Kaggle |

*Cập nhật ước tính sau G6 lượt 5 (27/10/2026; chỉ là ước tính, không đổi thiết kế):* đo trên tập phát triển với bộ chạy G7 (không cắt train ở 2.000 hàng), một ô (Gibbs + hard-EM + TFM ngây thơ + oracle) mất 55–290 giây với n_train 370–1.460; ngoại suy theo n_train và d cho tập đánh giá (n_train tới 4.500, d tới 256) được khoảng **42 giờ T4** cho hai dòng đầu bảng (thay vì 14 + 4 giờ), độ bất định lớn vì phải ngoại suy quá n_train 1.460. Khối có sửa MH chậm khoảng 10 lần chuỗi thường (ablation Mục 7 chỉ chạy trên 3 dataset nhỏ). Phần GPU dự kiến khoảng 60–65 giờ T4 (42 + ablation 10–15 + RQ5 khoảng 5 + R̂ khoảng 2), tức 2–2,5 tuần quota. **Baseline (CPU) bị đánh giá thấp nhiều nhất:** đo trên tập phát triển (0,17–1,4 giây cho mỗi mẫu train mỗi ô, tuỳ bối cảnh) và ngoại suy tuyến tính theo n_train cho khoảng **140 giờ CPU** (LLP 47, PU-SAR 35, PU-SCAR 36, PLL 13, SSL 5, nhiễu 4), thay vì 25 giờ; phải chia mảnh và chạy song song nhiều phiên CPU.

Nằm trong khung G7–G8 (01/02 – 15/04/2027).

## 10. Ký duyệt
Người kiểm duyệt: thầy (người hướng dẫn đề tài; không soạn giao thức, có tham gia quyết định F3 và F8) · Ngày: 26/10/2026 · ☑ Đạt ☐ Sửa · Biên bản: [GATE_REVIEW_G5](GATE_REVIEW_G5.md)

## 11. Deviations (ghi sau khi đóng băng)
| Ngày | Thay đổi | Lý do | Kết quả liên quan được gắn nhãn exploratory? |
|---|---|---|---|
| 26/10/2026 | Sửa thông tin trích dẫn ở bảng Mục 4: LLP-VAT là **ACML 2020** (Tsai & Lin), không phải ECML-PKDD 2020; DLLP là **ICDM Workshops 2017** (Ardehaly & Culotta) | Phát hiện khi kiểm code chính thức đầu G6 (README của kho LLP-VAT); chỉ sửa trích dẫn, không đổi thiết kế | Không (chưa có kết quả nào trên tập đánh giá) |
| 26/10/2026 | **PU-SAR thành bối cảnh phụ** (Mục 0): báo cáo đầy đủ, không vào abstract, không thuộc H2 chính. Thêm phân tích phụ: Gibbs với propensity đã biết, so với SAR-E (cùng propensity đã biết, code chính thức) | Tập phát triển, sau khi có baseline chuyên cho SAR (G6, [G6_T1_SAREM_REPORT](../G6/G6_T1_SAREM_REPORT.md)): Gibbs đạt cả baseline tốt nhất lẫn naive ở 3/12 ô (< 60%, cùng ngưỡng đã dùng để phân vai RQ4). Nút thắt là việc học propensity: khi biết propensity Gibbs đạt 8/12. Quyết định trước khi chạy bất kỳ phương pháp nào trên tập đánh giá | Không (chỉ dùng tập phát triển) |
| 26/10/2026 | **Làm rõ quy tắc chọn đối thủ chính** (Mục 6): ứng viên gồm **cả bản mặc định lẫn bản tune** của mỗi baseline peer-review; chọn ứng viên có metric chính trung bình tốt nhất trên tập phát triển | Bản tune (chấm bằng loss trên nhãn yếu) có thể kém bản mặc định (SAR-EM, PU-SAR 0,1: 0,671 so với 0,694); cách này khắt khe hơn với phương pháp đề xuất | Không |
| 26/10/2026 | **Thêm RC và CC** (Feng et al., NeurIPS 2020) vào baseline của PLL tổng hợp (trước chỉ dùng cho PLL thật) | Thêm đối thủ peer-review mạnh, khắt khe hơn; quyết trước khi chạy tập đánh giá | Không |
| 26/10/2026 | **GLWS (LLP): E-step dùng hậu nghiệm chính xác** khi biết số đếm của túi (đúng như bài GLWS mô tả) thay cho hàm `create_proportion_graph` của kho chính thức | Kiểm ở G6: log P(đếm = c \| đếm ≤ c) của hàm chính thức khớp số với cách tính của dự án, nhưng nhãn giả của hàm đó không phải hậu nghiệm chính xác (tổng theo túi khác c; ví dụ túi 5 mẫu, c = 1: tổng 0,45). Dùng bản chính xác là bản mạnh hơn của baseline; ghi ở docs/baseline_reproduction.md | Không |
| 26/10/2026 | **Thay nhận xét cuối Mục 0 bằng [PROOFS §7](../G4/PROOFS.md)** và **thêm phân tích thăm dò E-T7** (không đổi H1–H5, không đổi đối thủ hay metric). Với cơ chế sinh nhãn của giao thức, TFM ngây thơ ở PLL và LLP cũng nhất quán về xếp hạng (Mệnh đề 5); khác biệt với PU-SCAR, nhiễu, SSL nằm ở lượng thông tin bộ ngây thơ bỏ đi (Mệnh đề 6, Quan sát 6(b)). Dự đoán ghi trước cho tập đánh giá, kiểm bằng chênh lệch Gibbs ấm − TFM ngây thơ theo dataset: **(D1)** PLL: chênh lệch accuracy ở q = 0,6 lớn hơn ở q = 0,3 (Wilcoxon một phía, ghép cặp theo dataset); **(D2)** LLP: chênh lệch AUC ở túi 16 lớn hơn ở túi 8 (như D1); **(D3)** PU-SCAR, nhiễu phụ thuộc lớp, SSL: chênh lệch AUC trung bình ≤ +0,02 ở mọi mức (không có lợi ích AUC có hệ thống) | G6 T7: nhận xét cũ ("bộ ngây thơ không nhất quán ở PLL, LLP") sai với cơ chế của giao thức. Trên tập phát triển (hậu kiểm, PROOFS §7): khớp cả ba dự đoán về trung bình; theo dataset, D1 đúng ở 3/3, D2 ở 4/6. Ghi trước khi chạy bất kỳ phương pháp nào trên tập đánh giá | **Có** — D1–D3 là phân tích thăm dò |
| 26/10/2026 | **LLP: khẳng định chính chỉ còn túi 8** (thầy chọn). Túi 16 thành bối cảnh phụ (chạy và báo cáo đầy đủ trên tập đánh giá), túi 32 vẫn là kiểm tra áp lực. H2-LLP: Wilcoxon hai phía, Gibbs so với hai đối thủ chính (DLLP tuyến tính mặc định; TFM ngây thơ) ở túi 8, **Holm trên 2 phép so sánh**; vẫn đòi thắng ≥ 7/10 dataset trước baseline mạnh nhất. Đối thủ chính chọn lại theo đúng quy tắc Mục 6 trên mức chính mới: `dllplin_def` ([docs/primary_comparators.md](../docs/primary_comparators.md)). Dự đoán thăm dò D2 (deviation 6) giữ nguyên | G6 lượt 2, chỉ trên tập phát triển ([G6_ROUND2_REPORT](../G6/G6_ROUND2_REPORT.md)): với mọi baseline đã tune, quy tắc phạm vi 60% cho 7/12 ô khi gộp hai mức (thiếu một ô); tách theo mức: túi 8 đạt 5/6, túi 16 đạt 2/6 (chuỗi trôi về cách gán nhãn tệ hơn trên blood-transfusion và credit-g). Cùng kiểu thu hẹp theo mức như lần bỏ túi 32 (25/10/2026). Quyết định trước khi chạy tập đánh giá | Không (khẳng định chính, kiểm định xác nhận); túi 16 báo cáo như kết quả phụ |
| 26/10/2026 | **Baseline: PRODEN tune thêm bộ tối ưu ∈ {SGD momentum 0,9 (mặc định, như kho chính thức), Adam}; thêm bản tune cho label spreading, self-training, forward correction** (bối cảnh phụ SSL, nhãn nhiễu), theo không gian đã ghi ở docs/tuning_spaces.md. Đối thủ chính của PLL là tạm thời tới khi chạy lại PRODEN tune (G6 lượt 3) | PRODEN bản dùng loss chính thức với SGD chỉ đạt 0,573 (mặc định) và 0,716 (tune) trên tập phát triển, thấp hơn bản Adam cũ của dự án (0,740): chưa hội tụ trên dữ liệu bảng nhỏ. Làm baseline mạnh hơn, khắt khe hơn với phương pháp đề xuất | Không |
| 26/10/2026 | **Baseline của các bối cảnh phụ: cách chuyển sang dữ liệu bảng.** GLWS (PU), ILL (nhãn nhiễu, SSL): một góc nhìn thay cho hai phép tăng cường ảnh; GLWS cho PU dùng E-step chính xác như deviation 5. ILL nhãn nhiễu được tune thêm "T đã biết" và "bỏ cân bằng entropy" (docs/tuning_spaces.md). VIME cài lại theo bài. In-Context PU: không có code truy cập được → ghi là chưa tái hiện được | Các baseline này có trong bảng Mục 4 nhưng code gốc viết cho ảnh hoặc TF1; ghi rõ cách chuyển trước khi chạy. Chỉ ảnh hưởng bối cảnh phụ | Không |
| 27/10/2026 | **Baseline có tune: quy tắc dự phòng khi fit lại hỏng.** Sau khi chọn cấu hình theo loss trên validation nhãn yếu, mô hình được fit lại trên toàn bộ train; nếu mô hình đó cho xác suất không hữu hạn trên chính X train (huấn luyện phân kỳ), dùng cấu hình xếp hạng kế tiếp, tối đa 5 cấu hình; ghi thứ hạng đã dùng (`icgibbs/tuning.py`, `refit_best`). Áp cho mọi baseline có bản tune | G6 lượt 5 (tập phát triển): `proden_tun` ở segment, q = 0,3, seed 2 cho xác suất NaN sau khi fit lại (cấu hình tốt nhất trên 80% train phân kỳ trên 100%) — 1 ô lỗi trên 324. Quy tắc chỉ nhìn X train, không nhìn nhãn hay test; khi không có lỗi kết quả trùng hoàn toàn với trước (có test). Quyết trước khi chạy tập đánh giá | Không |

## 12. Việc phải xong trước khi đóng băng (đã xong hết, trừ F5 thuộc G6)
| # | Việc | Ai | Ghi chú |
|---|---|---|---|
| F1 | Chạy `kaggle_g4_rq4_pilot.py` (thí điểm RQ4 trên tập phát triển) | thầy (Kaggle) | **Xong 26/10** (v4 + v5): PLL 6/6 → chính; nhãn nhiễu 4/12, SSL 3/12 → phụ ([PILOT_RQ4_REPORT](../G4/pilot/PILOT_RQ4_REPORT.md)) |
| F2 | Chạy `g5_verify_datasets.py` | thầy (CPU, có Internet) | **Xong 26/10:** mọi ID khớp, giấy phép Public; churn bị loại (nhân tạo) → sick; dùng cả 14 dataset đa lớp; số hàng trùng ghi ở Mục 1.2 |
| F3 | Xác nhận R3 (PU-SCAR thành kết quả phụ) | thầy | **Xong 26/10:** thầy xác nhận |
| F4 | Chọn nguồn thứ hai cho RQ5 (Mục 1.4) | thầy giao Claude | **Xong 26/10:** PLL thật (Lost, MSRCv2, BirdSong) |
| F6 | Thí điểm many-class trên tập phát triển (`kaggle_g4_manyclass_pilot.py`) | thầy (Kaggle) | **Xong 26/10:** TabICL v2; 10 × 90/10 ([PILOT_MANYCLASS_REPORT](../G4/pilot/PILOT_MANYCLASS_REPORT.md)) |
| F7 | Xác minh chỉ đọc bộ PLL thật (`g5_verify_pll_real.py`) | thầy (Kaggle, CPU) | **Xong 26/10:** Lost, MSRCv2, BirdSong đạt; cả 6 bộ khớp số công bố; SHA-256 đã ghi; bản .npz giữ trên Kaggle cho G6 |
| F8 | Quyết định cách kiểm định H2 | thầy | **Xong 26/10:** thầy chọn phương án B — hai đối thủ chính (baseline peer-review tốt nhất theo quy tắc trên tập phát triển, và TFM ngây thơ), Holm trên 4 phép so sánh; so sánh với từng baseline là phân tích phụ (Mục 6). Ghi RESEARCH_PLAN §9 |
| F5 | Tìm code chính thức của các baseline đánh dấu "xác minh" | Claude, đầu G6 | Nếu không có thì cài lại và kiểm bằng tái hiện số công bố (G6) |
