# Dataset card — tập đánh giá của đề tài A (G5)

Theo mẫu [DATASET_CARD](../../research-handbook/templates/DATASET_CARD.md) (Datasheets for Datasets, Gebru et al. 2021). Một file gồm phần chung cho mọi dataset, rồi phần riêng của từng dataset. Bản 26/10/2026, viết **trước khi chạy bất kỳ phương pháp nào** trên các dataset này. Số đo n, d, lớp, thiếu, trùng lấy từ `G4/pilot/g5_dataset_check.csv` (chạy trên Kaggle 25–26/10/2026). Chỗ ghi *[kiểm ở G6]* là thông tin em chưa đọc trực tiếp từ tài liệu gốc, phải xác minh trước khi viết vào bài.

## A. Phần chung

**A1. Nguồn và tải.** 24 dataset OpenML thuộc bộ OpenML-CC18 (Bischl et al., NeurIPS Datasets & Benchmarks 2021), tải bằng `sklearn.datasets.fetch_openml(data_id=..., parser="auto")`. Giấy phép trên OpenML: Public (mọi dataset). Phiên bản OpenML ghi trong bảng. **SHA-256:** ~~chưa tính~~ — *cập nhật G6, 27/10/2026:* đã ghi ở [`docs/dataset_hashes.csv`](../docs/dataset_hashes.csv) cho 33 dataset (24 của tập đánh giá, 9 của tập phát triển): `md5_checksum` của OpenML, SHA-256 của bảng (X, y) sau khi loại trùng và mã hoá như bộ chạy G7 (`icgibbs/g7/data.py`), SHA-256 của chỉ số train/test của 3 lần chia (seed 0–2). Chạy ở G6 lượt 4 bằng `g6_dataset_hashes.py`. Đối chiếu với bảng dưới (từ `g5_dataset_check.csv`): **33/33 dataset khớp** n gốc, d, số lớp, số hàng trùng bị loại. Tỉ lệ lớp nhỏ nhất trong file mã băm tính **sau** loại trùng nên khác bảng dưới ở 11 dataset (lệch lớn nhất: blood-transfusion 23,8% → 27,9%; splice 24,0% → 22,4%). Hai phiên Kaggle độc lập (lượt 4 và lượt 5) cho cùng mã băm bảng và lần chia ở cả 27 cặp (dataset, seed) của tập phát triển đã chạy. Ba bộ PLL thật (Mục D) tải từ trang của nhóm PALM, có SHA-256 của tệp gốc do `g5_verify_pll_real.py` ghi (việc F7).

**A2. Tiền xử lý trong dự án** (G5/EXPERIMENT_PROTOCOL Mục 2; áp như nhau cho mọi dataset):

| Bước | Lý do | Fit trên |
|---|---|---|
| Bỏ hàng có target thiếu | không đánh giá được | không cần fit |
| Loại hàng trùng hoàn toàn (X và y), giữ bản đầu | chống L1.4 | toàn bộ, trước khi chia |
| Mã hoá feature phân loại thành mã thứ tự, giá trị thiếu = NaN | đầu vào của TFM | không cần fit |
| Điền trung vị, chuẩn hoá z, one-hot | đầu vào của baseline MLP | train của từng lần chia |
| Lớp dương (nhị phân) = lớp thiểu số; đa lớp: mã theo thứ tự chữ cái của nhãn gốc | quy ước cố định | — |

**A3. Rủi ro rò rỉ chung.** Mọi dataset là bảng tĩnh, không phải bài toán dự báo theo thời gian. Ngoại lệ (cấu trúc nhóm hoặc chuỗi) ghi ở từng card và xử lý bằng phân tích độ nhạy S1 (giao thức Mục 1.5).

**A4. Trùng với dữ liệu huấn luyện trước.** TabPFN v2 và TabICL v2 được huấn luyện trước hoàn toàn trên dữ liệu tổng hợp. Tuy vậy, nhóm tác giả của chúng có dùng benchmark thật (gồm OpenML-CC18) để đánh giá và chọn mô hình. Việc của G6: lập danh sách dataset mỗi nhóm dùng khi phát triển; dataset nào trùng vẫn giữ, nhưng báo cáo riêng (giao thức Mục 1.3).

**A5. Mục đích sử dụng.** Mọi dataset dùng để *sinh nhãn yếu tổng hợp* từ nhãn sạch (giao thức Mục 3.1) rồi so sánh phương pháp. Nhãn sạch của train chỉ dùng để sinh nhãn yếu và để chẩn đoán. Không dùng để rút kết luận về chính lĩnh vực của dataset.

## B. Nhị phân (LLP, PU, SSL, nhãn nhiễu) — 10 dataset

| ID | Tên (phiên bản) | Nguồn gốc, trích dẫn | Một hàng là | n | d (phân loại) | Lớp dương (thiểu số) | Thiếu | Trùng / xung đột |
|---:|---|---|---|---:|---|---:|---:|---|
| 3 | kr-vs-kp (1) | UCI Chess (King-Rook vs King-Pawn), Shapiro 1983 | một thế cờ tàn | 3196 | 36 (36) | 47,8% | 0 | 0 / 0 |
| 29 | credit-approval (1) | UCI Credit Approval, Quinlan | một hồ sơ xin cấp tín dụng | 690 | 15 (9) | 44,5% | 0,7% | 0 / 0 |
| 44 | spambase (1) | UCI Spambase, HP Labs (Hopkins, Reeber, Forman, Suermondt 1999) | một email | 4601 | 57 (0) | 39,4% | 0 | 391 / 6 |
| 50 | tic-tac-toe (1) | UCI Tic-Tac-Toe Endgame, Aha 1991 | một thế cờ cuối ván | 958 | 9 (9) | 34,7% | 0 | 0 / 0 |
| 1049 | pc4 (1) | NASA Metrics Data Program, kho PROMISE | một module phần mềm | 1458 | 37 (0) | 12,2% | 0 | 114 / 1 |
| 1489 | phoneme (1) | dự án ELENA (Alinat) | một đoạn âm vị | 5404 | 5 (0) | 29,4% | 0 | 9 / 0 |
| 1494 | qsar-biodeg (1) | UCI QSAR biodegradation, Mansouri et al. 2013 | một hợp chất hoá học | 1055 | 41 (0) | 33,7% | 0 | 3 / 0 |
| 6332 | cylinder-bands (2) | UCI Cylinder Bands, Evans & Fisher 1994 | một lần chạy máy in ống đồng | 540 | 37 (19) | 42,2% | 5,0% | 0 / 0 |
| 23381 | dresses-sales (2) | UCI Dresses Attribute Sales | một mẫu váy | 500 | 12 (11) | 42,0% | 13,9% | 0 / 2 |
| 38 | sick (1) — thay churn | UCI Thyroid Disease, Garavan Institute (Quinlan) | một hồ sơ bệnh nhân | 3772 | 29 (22) | 6,1% | 5,5% | 61 / 0 |

Ghi chú riêng:
- **kr-vs-kp, tic-tac-toe:** nhãn tính tất định từ luật chơi (thắng hay không khi chơi tối ưu) trên các thế cờ thật của trò chơi. Không có nhiễu đo. Xem là "dữ liệu thật" theo nghĩa của Mục 1.1 (không phải dữ liệu mô phỏng ngẫu nhiên). Người kiểm duyệt cân nhắc có đưa vào S1 không (Phát hiện 3 của GATE_REVIEW_G5).
- **credit-approval:** tên và giá trị thuộc tính đã được ẩn danh hoá ở nguồn. Là dữ liệu tài chính cá nhân đã ẩn danh.
- **spambase:** 391 hàng trùng (8,5%), lớn nhất trong nhóm nhị phân; sau khi loại trùng còn 4210.
- **pc4:** đại diện duy nhất của nhóm NASA MDP (điều kiện độc lập C1); 114 hàng trùng.
- **dresses-sales:** ở biên n = 500; thiếu 13,9%. Nhãn là "có được đề xuất hay không", độ tin cậy nhãn thấp *[kiểm ở G6]*.
- **sick:** dữ liệu sức khoẻ đã ẩn danh hoá (tuổi, giới tính, chỉ số hormone tuyến giáp). Lớp dương 6,1%, dưới ngưỡng 10% của danh sách chính (điều khoản dự phòng cho phép ≥ 5%). Báo cáo kèm chú thích này.
- **churn (40701) bị loại:** dữ liệu nhân tạo (giao thức Mục 1.2).

## C. Đa lớp (PLL) — 14 dataset

| ID | Tên (phiên bản) | Nguồn gốc, trích dẫn | Một hàng là | n | d (phân loại) | Số lớp | Lớp nhỏ nhất | Thiếu | Trùng / xung đột |
|---:|---|---|---|---:|---|---:|---:|---:|---|
| 11 | balance-scale (1) | UCI Balance Scale, Siegler 1976 | một cấu hình cân | 625 | 4 (0) | 3 | 7,8% | 0 | 0 / 0 |
| 12 | mfeat-factors (1) | UCI Multiple Features, Duin | một chữ số viết tay (bản đồ Hà Lan) | 2000 | 216 (0) | 10 | 10,0% | 0 | 6 / 0 |
| 28 | optdigits (1) | UCI Optical Recognition of Handwritten Digits, Alpaydin & Kaynak 1998 | một chữ số viết tay | 5620 | 64 (0) | 10 | 9,9% | 0 | 0 / 0 |
| 46 | splice (1) | UCI Splice-junction Gene Sequences, Towell, Noordewier & Shavlik | một đoạn DNA 60 nucleotide | 3190 | 60 (60) | 3 | 24,0% | 0 | 184 / 2 |
| 182 | satimage (1) | UCI Statlog Landsat Satellite, Srinivasan 1993 | lân cận 3×3 pixel của ảnh vệ tinh | 6430 | 36 (0) | 6 | 9,7% | 0 | 0 / 0 |
| 188 | eucalyptus (1) | Bulloch 1991, khảo sát lô giống bạch đàn ở New Zealand *[kiểm ở G6]* | một lô giống tại một điểm trồng | 736 | 19 (5) | 5 | 14,3% | 3,2% | 0 / 0 |
| 458 | analcatdata_authorship (1) | Simonoff 2003 (*Analyzing Categorical Data*) | một đoạn văn | 841 | 70 (0) | 4 | 6,5% | 0 | 0 / 0 |
| 469 | analcatdata_dmft (1) | Simonoff 2003; chỉ số răng sâu-mất-trám (DMFT) *[kiểm ở G6]* | *[kiểm ở G6]* | 797 | 4 (4) | 6 | 15,4% | 0 | 241 / 329 |
| 1475 | first-order-theorem-proving (1) | UCI, Bridge, Holden & Paulson 2014 | một bài toán chứng minh định lý | 6118 | 51 (0) | 6 | 7,9% | 0 | 482 / 209 |
| 1497 | wall-robot-navigation (1) | UCI, Freire et al. 2009 | một lần đọc 24 cảm biến siêu âm | 5456 | 24 (0) | 4 | 6,0% | 0 | 0 / 0 |
| 1501 | semeion (1) | UCI Semeion Handwritten Digit, Semeion Research Center | một chữ số viết tay 16×16 | 1593 | 256 (0) | 10 | 9,7% | 0 | 0 / 0 |
| 40966 | MiceProtein (4) | UCI Mice Protein Expression, Higuera, Gardiner & Cios 2015 | một phép đo 77 protein | 1080 | 77 (0) | 8 | 9,7% | 1,7% | 0 / 0 |
| 40975 | car (3) | UCI Car Evaluation, Bohanec & Rajkovič 1988 | một tổ hợp thuộc tính xe | 1728 | 6 (6) | 4 | 3,8% | 0 | 0 / 0 |
| 40982 | steel-plates-fault (3) | UCI Steel Plates Faults, Semeion Research Center | một tấm thép có lỗi | 1941 | 27 (0) | 7 | 2,8% | 0 | 0 / 0 |

Ghi chú riêng:
- **balance-scale, car — sinh từ quy tắc đã biết:** liệt kê đủ mọi tổ hợp thuộc tính (625 = 5⁴; 1728 = 4·4·4·3·3·3), nhãn tính tất định từ một quy tắc (cân bằng mô-men; mô hình quyết định phân cấp). Không có nhiễu đo. Vào phân tích độ nhạy S1.
- **MiceProtein — cấu trúc nhóm (L3.2):** 72 con chuột × 15 phép đo lặp = 1080 hàng. Chia ngẫu nhiên đặt phép đo của cùng một con ở cả train lẫn test. Vào S1.
- **wall-robot-navigation — chuỗi (L3.1):** đọc cảm biến liên tiếp khi robot đi men tường (4 vòng); mẫu kề nhau gần trùng. Vào S1.
- **satimage, optdigits, semeion — cấu trúc yếu:** lân cận pixel chồng lấn (satimage); nhiều chữ số của cùng người viết (optdigits, semeion). Theo thông lệ CC18, không đưa vào S1; ghi ở phần giới hạn.
- **analcatdata_dmft:** nhãn rất nhiễu (41% hàng thuộc nhóm cùng X khác y; chỉ 4 feature phân loại). Sau loại trùng còn 556 hàng. Có thể chứa thuộc tính nhạy cảm (giới tính, dân tộc) và dữ liệu trẻ em *[kiểm ở G6]*.
- **eucalyptus:** theo mô tả gốc, mức hữu dụng được đánh giá từ chính các phép đo sinh trưởng có trong feature (chiều cao, tỉ lệ sống, …) *[kiểm ở G6]*. Không phải rò rỉ theo nghĩa của bài toán đánh giá phương pháp, nhưng ghi lại.
- **first-order-theorem-proving:** 482 hàng trùng, 209 hàng xung đột. Nhãn là heuristic chứng minh tốt nhất trong 5 heuristic, hoặc "từ chối".
- **mfeat-factors:** đại diện duy nhất của nhóm mfeat (C1).
- **splice:** đại diện duy nhất của cặp splice/dna (C1).

## D. PLL thật (RQ5) — 3 bộ

Nguồn: trang dữ liệu PLL của nhóm PALM, Đại học Đông Nam (palm.seu.edu.cn/zhangml/Resources.htm). Điều khoản: chỉ dùng cho học thuật, tự chịu rủi ro, ghi nguồn các bài gốc. Định dạng: tệp MATLAB với `data` (M×d), `partial_target` (Q×M), `target` (Q×M). Bảng đầu là **số công bố** (Feng et al., NeurIPS 2020, Bảng 6); bảng thứ hai là **số đo thật** (F7, tải ngày 26/10/2026, `G4/pilot/g5_pll_real_check.csv`) — khớp số công bố ở cả ba bộ.

| Bộ | Bài gốc | Một hàng là | Tập ứng viên đến từ | n | d | Lớp | Ứng viên TB |
|---|---|---|---|---:|---:|---:|---:|
| Lost | Cour, Sapp & Taskar, JMLR 2011 | một khuôn mặt trong phim truyền hình *Lost* (feature đã trích) | tên nhân vật xuất hiện trong kịch bản hoặc phụ đề của cảnh | 1122 | 108 | 16 | 2,23 |
| MSRCv2 | Liu & Dietterich, NeurIPS 2012 | một vùng ảnh (MSRC v2) | nhãn cấp ảnh của ảnh chứa vùng đó | 1758 | 48 | 23 | 3,16 |
| BirdSong | Briggs, Fern & Raich, KDD 2012 | một âm tiết tiếng chim | loài có mặt trong bản ghi 10 giây chứa âm tiết | 4998 | 38 | 13 | 2,18 |

Số đo thật và định danh tệp (F7):

| Bộ | Tệp | SHA-256 của tệp gốc | n | d | Lớp (Q) | Ứng viên TB (min–max) | Nhãn đúng ∈ ứng viên | Lớp nhỏ nhất / lớn nhất (theo nhãn đúng) | Lớp không bao giờ là nhãn đúng | Trùng | Thiếu |
|---|---|---|---:|---:|---:|---|---:|---|---:|---:|---:|
| Lost | lost.rar → lost.mat | `ce4c1e0d5ad117e1b48f9f83805cba46f2a955ca5bdf75e578026ae4f6c7a960` | 1122 | 108 | 16 | 2,23 (1–3) | 100% | 0 / 204 | **2** | 0 | 0 |
| MSRCv2 | MSRCv2.rar → MSRCv2.mat | `8734cbd7b2f96623bfa8a0ea037daa4111fbce3954f077bf6c53d23dfb5e7a3c` | 1758 | 48 | 23 | 3,16 (1–7) | 100% | 3 / 255 | 0 | 5 | 0 |
| BirdSong | BirdSong.rar → BirdSong.mat | `8c0c9b1828e523b6bd499242a94a360a55326c326fe31b788138b3537bc1c33f` | 4998 | 38 | 13 | 2,18 (1–4) | 100% | 32 / 1280 | 0 | 0 | 0 |

Ba bộ không đạt tiêu chí (cũng khớp số công bố; chỉ để ghi nhận): FG-NET (1002 × 262, 78 lớp, 15 lớp không bao giờ là nhãn đúng), Soccer Player (17.472 × 279, 171 lớp, 1545 hàng trùng), Yahoo! News (22.991 × 163, 219 lớp).

Ghi chú:
- **Lost:** feature trích từ ảnh khuôn mặt diễn viên (người của công chúng); dự án chỉ dùng vector feature, không dùng ảnh.
- **BirdSong — cấu trúc nhóm:** nhiều âm tiết từ cùng bản ghi. Chia ngẫu nhiên như văn liệu để so được với số công bố; ghi ở phần giới hạn.
- Tập ứng viên ở đây là **nhãn yếu thật**, không sinh tổng hợp. Nhãn đúng (`target`) chỉ dùng để đánh giá.
- **Lost — lớp rỗng:** 2 trong 16 lớp chỉ xuất hiện như ứng viên sai, không bao giờ là nhãn đúng. Cài đặt phải lấy số lớp từ `partial_target` (Q = 16), không suy từ nhãn đúng.
- **MSRCv2:** lớp nhỏ nhất chỉ 3 mẫu; 5 hàng trùng, loại trước khi chia (còn 1753). Văn liệu không loại trùng; ảnh hưởng tới so sánh với số công bố là không đáng kể nhưng sẽ ghi ở phụ lục.
- Chia: 10 lần ngẫu nhiên 90/10, không phân tầng (giao thức Mục 1.4). Backend: TabICL v2.
