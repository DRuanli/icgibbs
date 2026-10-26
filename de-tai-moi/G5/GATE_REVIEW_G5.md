# Biên bản kiểm duyệt cổng G5

Cổng: **G5 — Thiết kế và đóng băng giao thức thực nghiệm** · Dự án: đề tài A, *In-Context Gibbs* · Ngày lập: 26/10/2026

- Người tạo sản phẩm: Claude (soạn), thầy (chạy thí điểm và xác minh trên Kaggle, quyết định F3)
- Người kiểm duyệt (không tham gia tạo sản phẩm): **thầy** ☑ Người ☐ Agent AI. Ghi chú về tính độc lập: thầy không soạn giao thức và dataset card, nhưng có chạy thí điểm và quyết định F3 (PU-SCAR) và F8 (cách kiểm định H2).

Mục 3a dưới đây là **tự rà soát của người tạo**, không thay kiểm duyệt độc lập. Người kiểm duyệt điền Mục 2 và 3b.

## 1. Sản phẩm được kiểm
| File | Ghi chú |
|---|---|
| [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) | bản nháp 26/10/2026 |
| [DATASET_CARDS.md](DATASET_CARDS.md) | 10 nhị phân, 14 đa lớp, 3 PLL thật |
| [../G4/pilot/PILOT_RQ4_REPORT.md](../G4/pilot/PILOT_RQ4_REPORT.md), [../G4/pilot/PILOT_MANYCLASS_REPORT.md](../G4/pilot/PILOT_MANYCLASS_REPORT.md) | căn cứ của các quyết định phạm vi và backend |
| `G4/pilot/g5_dataset_check.csv` | xác minh danh sách dataset (F2) |
| `G4/code/g5_verify_pll_real.py` → `G4/pilot/g5_pll_real_check.csv` | xong 26/10/2026 (F7): Lost, MSRCv2, BirdSong đạt; cả 6 bộ khớp số công bố |

## 2. Tiêu chí qua cổng (chép từ 02_QUY_TRINH_CONG_KIEM_DUYET §G5 và 06_THIET_KE_THUC_NGHIEM)

| Tiêu chí | Người tạo tự đánh giá | Người kiểm duyệt |
|---|---|---|
| ≥ 10 dataset độc lập, có dataset card | Đạt: 10 nhị phân, 14 đa lớp, độc lập theo C1; card ở DATASET_CARDS (còn mục *[kiểm ở G6]*) | ☑ |
| ≥ 2 baseline peer-review mỗi bối cảnh, cộng baseline đơn giản và baseline mạnh nhất (kể cả preprint) | Đạt trên giấy (giao thức Mục 4); nhiều baseline còn "xác minh ở G6" | ☑ |
| Metric, cách chia, ngân sách tune cho mọi phương pháp | Đạt (Mục 0, 3, 4, 5) | ☑ |
| Số seed, kế hoạch thống kê, ablation, độ nhạy | Đạt (Mục 3, 6, 7, 1.5); xem Phát hiện 1 | ☑ |
| Phần cứng và ngân sách tính toán | Đạt (Mục 9: khoảng 58 giờ, × 2 dự phòng ≈ 115 giờ T4) | ☑ |
| **Checklist rò rỉ 8 loại** (giao thức Mục 8) | Đạt có điều kiện: L3.1, L3.2 xử lý bằng S1 (Mục 1.5) | ☑ |
| **Checklist công bằng baseline** (bảng dưới) | Đạt trên giấy; kiểm thực tế ở G6 | ☑ |
| Ngày đóng băng và commit được ghi | Ngày 26/10/2026; định danh bằng SHA-256 ([FREEZE_RECORD](FREEZE_RECORD.md)) vì dự án chưa dùng git | ☑ |

**Checklist công bằng baseline** (06 §3):

| # | Mục | Trong giao thức | Người kiểm duyệt |
|---|---|---|---|
| B1 | Mọi phương pháp dùng chung lần chia và cùng nhãn yếu (cùng seed) | Mục 3 | ☑ |
| B2 | Cùng thông tin phụ: tỉ lệ lớp (PU), ma trận T (nhiễu), thuộc tính propensity (PU-SAR) | Mục 4, quy tắc 3 | ☑ |
| B3 | Cùng ngân sách tune: 20 cấu hình random search, validation 20% có nhãn yếu, chấm bằng loss trên nhãn yếu | Mục 3, Mục 4 quy tắc 1 | ☑ |
| B4 | Báo cáo cả bản mặc định lẫn bản tune | Mục 4 quy tắc 1 | ☑ |
| B5 | Phương pháp đề xuất không tune theo dataset; cấu hình chốt trên tập phát triển | Mục 4 quy tắc 2 | ☑ |
| B6 | Code chính thức; nếu cài lại thì kiểm bằng tái hiện số công bố (G6, `baseline_reproduction.md`) | Mục 4, F5 | ☑ |
| B7 | Có baseline mạnh nhất kể cả preprint (In-Context PU) và baseline đơn giản (TFM ngây thơ) | Mục 4 | ☑ |
| B8 | So thời gian trên cùng phần cứng | Mục 5, 9 | ☑ |

## 3a. Tự rà soát của người tạo (26/10/2026)

| # | Mức | Mô tả | Vị trí | Đề xuất | Trạng thái |
|---|---|---|---|---|---|
| 1 | **Lớn** | **Lực kiểm định của H2 rất thấp theo cách viết hiện tại.** H2 đòi thắng *mọi* baseline peer-review sau Holm trên mọi cặp (baseline × mức) của một bối cảnh — khoảng 8–10 phép so sánh — với Wilcoxon trên n = 10 dataset. Với n = 10, p hai phía nhỏ nhất có thể là 0,0020 (thắng cả 10); thua 1 dataset ở hạng 1, 2, 3 cho p = 0,0039; 0,0059; 0,0098. Ngưỡng Holm đầu tiên với m = 10 là 0,005, và nửa số phép so sánh cần p ≤ khoảng 0,008. Nghĩa là H2 chỉ đạt khi phương pháp thắng gần như *mọi* dataset trước *mọi* baseline. Đây là tính chất của thiết kế, không phụ thuộc kết quả; nhưng trên tập phát triển Gibbs chỉ vượt cả peer-review tốt nhất lẫn naive ở 3–4/6 dataset (LLP túi 16, PU-SAR), nên theo cách viết hiện tại H2 gần như chắc chắn không đạt ở hai bối cảnh này, kể cả khi lợi ích là thật nhưng không đồng đều. (Với PLL, n = 14 nên dễ hơn: p nhỏ nhất 0,00012.) | Giao thức Mục 6; RESEARCH_PLAN H2 (đã đóng băng ở G3) | Thầy chọn: **(A)** giữ nguyên (nghiêm, chấp nhận H2 có thể không đạt và báo cáo trung thực); **(B, em đề xuất)** mỗi bối cảnh có *một* đối thủ chính, xác định bằng quy tắc viết ngay bây giờ: baseline peer-review có metric chính trung bình tốt nhất trên 6 dataset phát triển *sau khi tune ở G6* (chọn trên tập phát triển, trước khi chạy tập đánh giá, trung lập với phương pháp đề xuất và chọn đối thủ mạnh nhất chứ không phải yếu nhất); Holm chỉ trên các mức; các baseline khác báo cáo thắng/hoà/thua và p đã hiệu chỉnh như phân tích phụ; **giữ nguyên** tiêu chí "thắng ≥ 7/10 dataset trước baseline mạnh nhất của từng mức" và báo cáo cả kết quả theo tiêu chí G3 gốc. Mọi thay đổi phải ghi ở RESEARCH_PLAN §9 **trước** khi đóng băng, kèm lý do là tính toán lực kiểm định ở đây | **Đã quyết 26/10/2026:** thầy chọn (B). Khi viết lại phát hiện ra (B) chính là phát biểu H2 đã đóng băng ở G3; bản nháp G5 đã viết chặt hơn G3. Đối thủ chính gồm cả TFM ngây thơ (đúng câu hỏi RQ2) → Holm trên 4 phép so sánh. Ghi ở giao thức Mục 6, RESEARCH_PLAN §9, CLAIM #6 |
| 2 | **Lớn** | Rò rỉ do cấu trúc nhóm hoặc chuỗi (MiceProtein, wall-robot-navigation; BirdSong ở RQ5) có thể có lợi cho phương pháp dựa trên ngữ cảnh hơn cho baseline MLP | DATASET_CARDS C, D | Đã thêm phân tích độ nhạy S1 đăng ký trước (giao thức Mục 1.5). Phương án mạnh hơn nếu người kiểm duyệt muốn: chia theo nhóm cho MiceProtein nếu OpenML còn giữ mã con chuột *[kiểm ở G6]* | Đã xử lý (S1); thầy chấp nhận 26/10/2026 |
| 3 | Nhỏ | balance-scale và car sinh từ quy tắc đã biết; tiêu chí đa lớp không có điều khoản "dữ liệu thật" (khác tiêu chí nhị phân). kr-vs-kp và tic-tac-toe có nhãn tất định từ luật chơi | Giao thức Mục 1.1, 1.5 | Giữ theo tiêu chí đã viết; balance-scale và car vào S1. Người kiểm duyệt quyết có thêm kr-vs-kp, tic-tac-toe vào một phân tích độ nhạy tương tự cho nhóm nhị phân hay không | Thầy duyệt theo đề xuất của người tạo (26/10/2026): giữ theo tiêu chí, balance-scale và car vào S1, **không** thêm phân tích độ nhạy cho kr-vs-kp và tic-tac-toe |
| 4 | **Lớn** | PU-SAR là bối cảnh chính nhưng **chưa có baseline chuyên cho SAR** (SAR-EM) trong bất kỳ lượt thí điểm nào | Giao thức Mục 4 | Cài hoặc lấy code SAR-EM đầu G6 và tái hiện số công bố trước khi bắt đầu G7; nếu SAR-EM mạnh hơn rõ trên tập phát triển thì ghi nhận trước khi chạy tập đánh giá | Việc G6 |
| 5 | Lớn | Nhiều baseline còn "xác minh ở G6" (Easy-LLP, VPU, In-Context PU, VIME, PRODEN, ILL, RC/CC). Baseline ở thí điểm chạy mặc định, chưa tune; cách biệt PLL so với PRODEN (0,011–0,064) có thể hẹp lại khi PRODEN được tune | Giao thức Mục 4, F5 | G6: code chính thức hoặc cài lại + tái hiện; không gian tune của từng baseline ghi trước trong `docs/tuning_spaces.md`. Label spreading phải tune cả kernel và số láng giềng (bản kNN mặc định để tới 29% hàng test không nhận được nhãn lan truyền ở tỉ lệ nhãn 5%, blood-transfusion) | Việc G6 |
| 6 | Nhỏ | RQ5-PLL dùng TabICL v2, các RQ khác dùng TabPFN v2 | Giao thức Mục 1.4, 4 | Ghi rõ trong bài; ablation TFM (Mục 7) chạy cả hai TFM trên các bối cảnh chính | Đã ghi |
| 7 | Nhỏ | F7 (xác minh bộ PLL thật) chưa có kết quả | Mục 12 | Chạy trước khi ký; quy tắc dự phòng đã viết (bỏ bộ không đạt; không còn bộ nào thì RQ5 chỉ còn LLP-Bench) | **Xong 26/10/2026:** cả 3 bộ đạt |
| 8 | Nhỏ | Chưa có md5/SHA-256 của dữ liệu OpenML | DATASET_CARDS A1 | G6 ghi vào `docs/dataset_hashes.csv` | Việc G6 |
| 9 | Nhỏ | Loại trùng làm n của analcatdata_dmft giảm từ 797 xuống 556 (vẫn ≥ 500); dataset này nhãn rất nhiễu | DATASET_CARDS C | Giữ; báo cáo kèm chú thích | Đã ghi |
| 10 | Nhỏ | Mục G4-2 (dự đoán quy nạp hay chuyển nạp) chưa đóng; giao thức chọn quy nạp cho thực nghiệm chính, chuyển nạp là ablation | Giao thức Mục 4, 7 | Người kiểm duyệt xác nhận lựa chọn | Thầy duyệt theo đề xuất (26/10/2026): quy nạp cho thực nghiệm chính, chuyển nạp là ablation |

## 3b. Phát hiện của người kiểm duyệt
| # | Mức | Mô tả | Vị trí | Đề xuất | Trạng thái |
|---|---|---|---|---|---|
| — | — | Thầy duyệt toàn bộ biên bản, không thêm phát hiện (26/10/2026) | — | — | — |

## 4. Quyết định
☑ **Qua** (ngày đóng băng 26/10/2026 đã ghi vào giao thức)  ☐ **Sửa** (giữ ở G5)  ☐ **Dừng**

Điều kiện để qua: (1) thầy quyết Phát hiện 1 — xong; (2) F7 — xong; (3) người kiểm duyệt duyệt hai checklist ở Mục 2 — xong. Chuyển sang G6: Phát hiện 4, 5, 8 và mọi mục *[kiểm ở G6]* của DATASET_CARDS.

Chữ ký người kiểm duyệt: thầy — duyệt qua tin nhắn ngày 26/10/2026 ("đã duyệt GATE_REVIEW_G5")
