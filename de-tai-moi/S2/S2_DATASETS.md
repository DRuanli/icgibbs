# S2 — tập đánh giá mới: kết quả đếm, rà tay, danh sách đề xuất (01/11/2026)

**Trạng thái: thầy đã duyệt ngày 01/11/2026** (danh sách như đề xuất; S1 theo đề xuất ở Mục 4.2). Đóng băng cùng S2_PLAN. Mọi quyết định ở đây chỉ dựa trên metadata và mô tả của OpenML. Chưa tải bảng dữ liệu nào, chưa sinh nhãn yếu, chưa chạy phương pháp nào trên các dataset này.

- **Nguồn đếm:** `census/` (do `G4/code/s2_dataset_census.py` tạo, thầy chạy ngày 01/11).
- **Rà tay:** `s2_review.csv`.
- **Chọn:** `s2_select.py`, cho ra `s2_selected.csv`. Script tất định; chạy lại cho đúng danh sách dưới đây.

## 1. Kết quả đếm
| Nguồn | Số dataset |
|---|---|
| OpenML100 (14), AMLB (269, 270, 271), Grinsztajn (334, 337), TabZilla (379), TabArena-v0.1 (457) | 262 |
| Sau khi bỏ CC18 và 33 dataset của NC1 (theo ID) | 194 |
| Bậc 0 (đúng tiêu chí NC1), trước rà tay | nhị phân 25; đa lớp 13 |

## 2. Rà tay: các dataset bị loại và lý do
| ID | Tên | Nhóm | Lý do loại |
|---|---|---|---|
| 333, 334, 335 | monks-problems-1/2/3 | nhị phân | bài toán nhân tạo (Thrun 1991) |
| 1479 | hill-valley | nhị phân | dữ liệu sinh nhân tạo |
| 40981 | Australian | nhị phân | **cùng nguồn** với credit-approval (ID 29) của tập đánh giá NC1 (dữ liệu duyệt tín dụng của Quinlan, đã đổi mã thuộc tính) |
| 41144 | madeline | nhị phân | nhân tạo: "MADELINE is an artificial dataset that is very non-linear" (Guyon et al. 2019, *Analysis of the AutoML Challenge Series*) |
| 60 | waveform-5000 | đa lớp | sinh từ bộ sinh sóng, nhân tạo |
| 377 | synthetic_control | đa lớp | chuỗi thời gian tổng hợp |
| 40496 | LED-display-domain-7digit | đa lớp | sinh từ bộ sinh LED có nhiễu |
| 1459 | artificial-characters | đa lớp (dự phòng) | nhân tạo |

Cơ sở để coi là "dữ liệu thật":
- **TabArena:** tiêu chí tuyển chọn của TabArena yêu cầu dữ liệu "stems from a real random distribution, and is not generated" và IID.
- **Các bộ UCI và StatLib:** đọc mô tả nguồn.
- **Bộ của thử thách AutoML:** jasmine, philippine, sylvine là dữ liệu thật có thêm feature nhiễu hoán vị, theo cùng bài của Guyon et al.

## 3. Danh sách đề xuất (bậc 0 cho cả hai nhóm)

**Nhị phân: 19 dataset.** Lấy hết vì ≤ 20. Dùng cho LLP túi 8.

| # | ID | Tên | n | d | Lớp nhỏ nhất | S1 | Ghi chú |
|---|---|---|---|---|---|---|---|
| 1 | 24 | mushroom | 8.124 | 22 | 48% | | gần như tách được hoàn toàn |
| 2 | 312 | scene | 2.407 | 299 | 18% | | |
| 3 | 451 | irish | 500 | 5 | 44% | | n đúng ngưỡng; có thể trượt sau loại trùng |
| 4 | 470 | profb | 672 | 9 | 33% | | |
| 5 | 934 | socmob | 1.156 | 5 | 22% | | mỗi dòng là một ô của bảng tần số |
| 6 | 40536 | SpeedDating | 8.378 | 120 | 16% | ✓ | một người ở nhiều dòng |
| 7 | 41143 | jasmine | 2.984 | 144 | 50% | | |
| 8 | 41145 | philippine | 5.832 | 308 | 50% | | |
| 9 | 41146 | sylvine | 5.124 | 20 | 50% | | |
| 10 | 41156 | ada | 4.147 | 48 | 25% | | |
| 11 | 44130 | eye_movements | 7.608 | 20 | 50% | ✓ | cùng người đọc, cùng câu |
| 12 | 45026 | heloc | 10.000 | 22 | 50% | | |
| 13 | 45039 | compas-two-years | 4.966 | 11 | 50% | | |
| 14 | 46911 | Bank_Customer_Churn | 10.000 | 10 | 20% | | |
| 15 | 46927 | Fitness_Club | 1.500 | 6 | 30% | | |
| 16 | 46930 | hazelnut-spread-contaminant-detection | 2.400 | 30 | 50% | | |
| 17 | 46938 | Is-this-a-good-customer | 1.723 | 13 | 11% | | |
| 18 | 46940 | Marketing_Campaign | 2.240 | 25 | 15% | | |
| 19 | 46969 | NATICUSdroid | 7.491 | 86 | 35% | | |

**Đa lớp: 10 dataset, vừa đủ ngưỡng.** Dùng cho PLL q = 0,3 và 0,6.

| # | ID | Tên | n | d | Số lớp | S1 | Ghi chú |
|---|---|---|---|---|---|---|---|
| 1 | 181 | yeast | 1.484 | 8 | **10** | | bản UCI đã dùng ở G6 để tái hiện baseline (không chạy Gibbs) |
| 2 | 375 | JapaneseVowels | 9.961 | 14 | **9** | ✓ | khung của cùng phát âm, 9 người nói |
| 3 | 1466 | cardiotocography | 2.126 | 35 | **10** | | |
| 4 | 23380 | cjs | 2.796 | 34 | 6 | ✓ | đo lặp trên cùng cây; khoảng 70% ô thiếu |
| 5 | 40498 | wine-quality-white | 4.898 | 11 | 7 | | |
| 6 | 46906 | anneal | 898 | 38 | 5 | | |
| 7 | 46941 | maternal_health_risk | 1.014 | 6 | 3 | | nhiều dòng trùng, có thể trượt n ≥ 500 |
| 8 | 46960 | students_dropout_and_academic_success | 4.424 | 36 | 3 | | |
| 9 | 46963 | website_phishing | 1.353 | 9 | 3 | | |
| 10 | 46980 | MIC | 1.699 | 111 | **8** | | có giá trị thiếu |

- **Bốn dataset có ≥ 8 lớp** (yeast, JapaneseVowels, cardiotocography, MIC) nằm đúng vùng mà TabPFN sụp ở NC1. Vì vậy S2 kiểm thật sự xem TabICL có giữ được ở vùng này không.
- **Dự phòng đa lớp,** dùng theo thứ tự nếu sau loại trùng còn < 10 (S2_PLAN §3.4):
  1. gas-drift (1476; bậc 1, lấy mẫu con 10.000 hàng; S1);
  2. soybean (42; bậc 2, 19 lớp);
  3. micro-mass (1515);
  4. dilbert (41163);
  5. fabert (41164);
  6. hiva_agnostic (46933).

  Mục 3, 4, 5, 6 thuộc bậc 3.

## 4. Thầy cần quyết trước khi đóng băng
1. **Duyệt danh sách:** 19 dataset nhị phân, 10 dataset đa lớp, danh sách dự phòng, và các quyết định loại ở Mục 2.
2. **Cách xử lý S1 khi còn ít dataset** (S2_PLAN §1, đề xuất).
   - Bỏ 2 dataset đa lớp gắn cờ S1 thì còn 8. Với n = 8, Wilcoxon + Holm chỉ đạt khi thắng mọi dataset.
   - Em đề xuất: với nhóm này, S1 kiểm **chiều** của hiệu ứng (chênh trung bình không đổi dấu) và **độ rộng** (thắng ≥ 60% dataset), thay vì kiểm ý nghĩa thống kê.
   - Nếu thầy muốn giữ nguyên quy tắc S1 của NC1 thì vẫn được, nhưng khẳng định PLL có thể trượt chỉ vì cỡ mẫu.

## 5. Danh sách cuối, sau lượt kiểm dữ liệu S2D (02/11/2026)
Áp S2_PLAN §3.4, quy tắc ghi trước, bằng `s2_select.py --hashes raw/S2D/s2_dataset_hashes.csv`, ra `s2_final.csv`:
- **Bỏ** irish (451): n = 470 sau loại trùng (30 dòng trùng). **Bỏ** maternal_health_risk (46941): n = 452 sau loại trùng (562 dòng trùng). Cả hai trượt n ≥ 500.
- **Nhị phân: 18 dataset**, không cần bổ sung.
- **Đa lớp: 9 + 1 = 10 dataset.** Bổ sung dự phòng 1 là **gas-drift** (1476; 6 lớp; S1, vì dữ liệu trôi theo thời gian). Bộ này có n = 13.910, nên lấy mẫu con 10.000 hàng, phân tầng, seed 0, sau loại trùng và trước khi chia.
- Độ nhạy S1:
  - nhị phân bỏ SpeedDating, eye_movements → còn 16;
  - đa lớp bỏ JapaneseVowels, cjs, gas-drift → còn 7, nên dùng quy tắc S1 theo chiều và độ rộng của hiệu ứng (§1).
- Vẫn còn **4 dataset ≥ 8 lớp**: yeast, JapaneseVowels, cardiotocography, MIC.
