# Giai đoạn 0 của V2: kết quả (05/11/2026)

- **Dự đoán ghi trước:** [PHASE0_PREDICTIONS](PHASE0_PREDICTIONS.md), commit `1a8859f`. Code chạy là commit `b939041`. Script chấm `V2/analysis/v0_report.py` được commit cùng code, trước khi có số, và không sửa sau khi xem kết quả. Đầu ra: [`results/v0_report.txt`](results/v0_report.txt), [`results/v0_summary.json`](results/v0_summary.json).
- **Phân tích viết sau khi xem số (thăm dò):** `V2/analysis/v0_explore.py`, đầu ra [`results/v0_explore.txt`](results/v0_explore.txt). Mọi số ở §4 và §5 là thăm dò.
- **Tính chất:** toàn bộ giai đoạn 0 là **thăm dò**, vì dữ liệu Study 1 đã được xem từ trước. R4 chạy trên Study 2 nhưng không thay kết quả xác nhận đã đăng ký.

## 0. Tóm tắt

| | Dự đoán | Kết quả |
|---|---|---|
| **P1** | Khởi động từ CC tune cứu được chuỗi TabPFN ở PLL q = 0,6 | **ĐÚNG.** r_final 0,953–0,989 ở cả 9/9 lần chạy; accuracy hơn CC tune trên cả 3 dataset (+0,8 đến +1,9 điểm) |
| **P2** | semeion vẫn sụp (r_final < 0,5) | **SAI.** Chuỗi trôi xuống, nhưng dừng ở r ≈ 0,76–0,81 chứ không sụp |
| **P3** | TabPFN, LLP: gain âm thì Gibbs làm kém điểm xuất phát | **ĐÚNG** trên cả 2 dataset |
| **P4** | WPLL chọn đúng ở ≥ 80% ô | **SAI.** 19/24 ô, tức 79,2%. PLL đúng 16/16; LLP đúng 3/8 |
| **Cổng V0** | Theo quy tắc ghi trước: P1 đúng, P4 sai | **Qua một phần:** giữ nhiều điểm xuất phát, tìm tiêu chí chọn khác trên tập phát triển trước Study 3 |

**Ba phát hiện chính:**
1. **Kiểu sụp của Study 1 sửa được.** Trên bốn dataset từ 8 lớp trở lên với TabPFN v2, v2 nâng accuracy trung bình từ 0,399 lên 0,938, gần bằng CC tune (0,947). v2 thắng CC tune trên 3/4 dataset; semeion thua 8 điểm.
2. **R4: kết quả PLL q = 0,6 của Study 2 nhạy với nhiễu Monte Carlo.** Thay chuỗi đã đăng ký bằng một chuỗi khác seed thì H4′ ở q = 0,6 không còn đạt (p_Holm = 0,074). Trung bình hai chuỗi thì mọi phép thử lại đạt, và Δ lớn hơn. LLP và PLL q = 0,3 bền.
3. **Ánh xạ phản ứng dự đoán điểm cuối của chuỗi rất tốt với TabICL**, sai số tuyệt đối trung vị 0,018. Với TabPFN, dự đoán sai lớn ở các ô có hai lưu vực.

## 1. Kiểm toàn vẹn

| Kiểm | Kết quả |
|---|---|
| Mã băm code | 72/72 tệp của cả V0A và V0B khớp commit `b939041` |
| Bộ test trên Kaggle | 126 qua, 2 bỏ qua, ở cả hai lượt |
| Số dòng ok so với dự kiến | V0A 318/318; V0B 312/312; **0 lỗi** |
| Bảng, lần chia, nhãn yếu | V0 so với Study 1: 18/18 ô khớp. R4 so với Study 2: 114/114 ô khớp |
| Tính tất định | Chuỗi `warm` của V0 tái tạo **đúng từng số** `gibbs_warm` của Study 1 với cả hai TFM (44 ô, chênh tối đa 0). Chuỗi a của R4 tái tạo `gibbs_warm` của Study 2 (chênh tối đa 5·10⁻⁵, do làm tròn) |
| Phiên bản thư viện | Giống các lượt trước, trừ torch của V0A là bản `+cu130` thay vì `+cu128`. Kết quả tái tạo đúng nên không ảnh hưởng |
| Thời gian | V0A 4 giờ 6 phút; V0B 4 giờ 46 phút (2 × T4) |

## 2. Chấm các dự đoán (theo đúng PHASE0_PREDICTIONS)

### P1: ĐÚNG

TabPFN v2, PLL q = 0,6, khởi động từ CC tune:

| Dataset | r_final (3 seed) | Gibbs (base) | CC tune | Study 1 (warm) |
|---|---|---|---|---|
| mfeat-factors | 0,956 / 0,953 / 0,953 | **0,968** | 0,951 | 0,100 |
| optdigits | 0,963 / 0,968 / 0,954 | **0,980** | 0,973 | 0,726 |
| MiceProtein | 0,967 / 0,971 / 0,989 | **0,969** | 0,951 | 0,478 |

(a) r_final ≥ 0,80 ở 9/9 lần chạy: đạt. (b) Gibbs ≥ CC tune − 0,02 trên từng dataset: đạt, và thực tế Gibbs **cao hơn** CC tune ở cả ba.

### P2: SAI

- **Gain đo được** với TabPFN, semeion, q = 0,6: −0,100 ở ρ = 0,85; −0,095 ở ρ = 0,90; −0,093 ở ρ = 0,95. Cả ba đều âm, nên theo quy tắc ghi trước dự đoán là "sụp".
- **Thực tế:** chuỗi bắt đầu ở r = 0,91, trôi xuống và dừng ở r_final 0,763 / 0,807 / 0,779. Không có seed nào dưới 0,5.
- **Đọc kết quả:** chiều trôi đúng, nhưng độ lớn sai.
  - Accuracy test của chuỗi là 0,835, thấp hơn CC tune 0,915. Nghĩa là Gibbs làm kém điểm xuất phát, đúng như gain âm báo trước.
  - Nhưng chuỗi dừng ở một điểm bất động mà ánh xạ gieo lỗi không thấy được (xem §4, E2).

### P3: ĐÚNG

| Dataset | r_start | Gain gần r_start | Dự đoán | Gibbs (base) | DLLP-MLP tune | Kết quả |
|---|---|---|---|---|---|---|
| tic-tac-toe | 0,749 | −0,107 | Gibbs < start − 0,01 | 0,703 | 0,874 | đúng |
| cylinder-bands | 0,619 | −0,022 | Gibbs < start − 0,01 | 0,611 | 0,692 | đúng |

Với TabPFN, ánh xạ LLP có một điểm hút thấp duy nhất: 0,61 ở tic-tac-toe, 0,57 ở cylinder-bands. Từ điểm xuất phát nào chuỗi cũng trôi về đó. Với TabICL, tic-tac-toe không có điểm hút thấp, và chuỗi đạt AUC 0,99.

### P4: SAI (79,2%, ngưỡng 80%)

- Có 24 ô đủ điều kiện, tức chênh chỉ số test giữa các điểm xuất phát > 0,01. WPLL chọn đúng 19 ô.
- **Tách theo bối cảnh** (thăm dò, E1):
  - PLL: đúng **16/16**.
  - LLP: đúng **3/8**.
- Ở cả 5 ô chọn sai, các điểm xuất phát đều kết thúc gần như cùng một chỗ: trung vị khoảng cách r_final chỉ 0,026. Ở các ô chọn đúng, khoảng cách này là 0,304.
- **Đọc kết quả:** khi các chuỗi rơi vào cùng một lưu vực, chênh chỉ số test giữa chúng chủ yếu là nhiễu Monte Carlo. Không tiêu chí nào chọn được nhiễu. WPLL chọn đúng khi giữa các chuỗi có khác biệt thật.

### Cổng V0

Theo quy tắc ghi trước, P1 đúng và P4 sai cho kết quả **"qua một phần"**:
- giữ nhiều điểm xuất phát;
- tìm tiêu chí gộp hoặc chọn khác trên tập phát triển, trước khi thiết kế Study 3.

## 3. R4: chuỗi thứ hai trên mọi ô của Study 2 (thăm dò)

114/114 ô ok. Chuỗi b cùng điểm khởi động với chuỗi a, chỉ khác seed lấy mẫu.

**(a) Các phép thử của Study 2 với từng bộ ước lượng** (thăm dò, E4). Mỗi ô ghi Δ trung bình, rồi p_Holm.

| Phép thử | Chuỗi a (đã đăng ký) | Chuỗi b | Trung bình a và b |
|---|---|---|---|
| H2′, LLP so DLLP tuyến tính | +0,052; p < 0,001; đạt | +0,046; p = 0,002; đạt | +0,052; p < 0,001; đạt |
| H4′, PLL q = 0,3 so CC tune | +0,012; p = 0,016 | +0,013; p = 0,008 | +0,013; p = 0,008 |
| H4′, PLL q = 0,6 so CC tune | +0,012; p = 0,016; **đạt** | +0,011; **p = 0,074; không đạt** | +0,013; p = 0,008; đạt |

- **Vì sao H4′ q = 0,6 không đạt với chuỗi b:** số thắng/hoà/thua vẫn là 8/1/1, nhưng đổi chỗ. cjs từ thắng thành thua: chuỗi b đạt 0,973, CC tune 0,991, do seed 2 chỉ đạt 0,969 so với 1,000 của chuỗi a. yeast thì từ thua thành thắng. Lần thua mới lớn hơn, nên Wilcoxon mất ý nghĩa.

**(b) Độ lớn của nhiễu Monte Carlo**

| Bối cảnh | Trung vị \|a − b\| | p90 | Tối đa | MCSE của trung bình theo dataset (trung vị) |
|---|---|---|---|---|
| LLP túi 8 | 0,003 | 0,039 | 0,166 (profb, seed 2) | 0,002 |
| PLL q = 0,3 | 0,000 | 0,005 | 0,009 | 0,001 |
| PLL q = 0,6 | 0,005 | 0,015 | 0,031 (cjs, seed 2) | 0,002 |

- 9/114 ô có |a − b| > 0,03. Tám ô là LLP trên dataset nhỏ hoặc nhiễu (profb ×3, Is-this-a-good-customer ×2, scene, compas, Marketing); một ô là cjs ở PLL q = 0,6.
- **Đổi dấu chênh so với đối thủ chính** giữa chuỗi a và chuỗi b: LLP 3/18 dataset; PLL q = 0,3 1/10; PLL q = 0,6 2/10.

**(c) R̂ trên Study 2:** tỉ lệ nhãn có R̂ > 1,1, trung vị qua các ô:

| Bối cảnh | Trung vị | p90 |
|---|---|---|
| LLP | 7,3% | 23,5% |
| PLL q = 0,3 | 3,2% | 8,8% |
| PLL q = 0,6 | 10,0% | 45,9% |

**(d) Chuỗi đã ổn định ở vòng quét 30** (E5): độ trôi của r giữa vòng 20 và vòng 30 có trung vị 0,002–0,006, tối đa 0,047. Tỉ lệ đổi nhãn ở 5 vòng cuối là 13% với LLP và 4–5% với PLL. Đây là lấy mẫu từ bất định thật, không phải chuỗi còn trôi.

**Hệ quả cho bài:**
- **Không được viết** rằng kết quả PLL q = 0,6 của Study 2 bền với nhiễu Monte Carlo. Phải báo cáo trung thực bảng (a) ở §6.6 và ở Limitations.
- Đồng thời, bảng (a) là bằng chứng cho hướng v2: **gộp nhiều chuỗi** làm kết luận bền hơn và Δ lớn hơn.
- Điều này trả lời trực tiếp ý 13 của reviewer ngoài 01, giờ đã đo trên chính dữ liệu xác nhận.

## 4. Phân tích thăm dò

**E2. Ánh xạ phản ứng dự đoán điểm cuối của chuỗi.**
- **Cách làm:** lặp r ← Φ(r) 30 lần từ r_start, rồi so với r_final thật.

| TFM, bối cảnh | Ánh xạ gieo đều: trung vị sai số (p90) | Lỗi có cấu trúc: trung vị sai số (p90) |
|---|---|---|
| TabICL, PLL | **0,018** (0,065) | **0,018** (0,077) |
| TabICL, LLP | 0,025 (0,095) | — |
| TabPFN, PLL | 0,080 (0,629) | 0,221 (0,799) |
| TabPFN, LLP | 0,056 (0,092) | — |

- **TabICL:** ánh xạ dự đoán điểm cuối gần như chính xác, đạt ngưỡng W1 (≤ 0,1) với dư địa lớn.
- **TabPFN:** trung vị đạt ngưỡng, nhưng sai lớn ở các ô có hai lưu vực. Đó là semeion và các chuỗi `warm` của MiceProtein, optdigits, vốn xuất phát sát điểm bất động không ổn định.
- **Lỗi có cấu trúc không giúp TabPFN.** Giả thuyết F2 ("phải đo trên lỗi giống lỗi của chuỗi") đúng về nguyên tắc, nhưng cách gieo lỗi theo khởi động ấm chưa giống lỗi chuỗi tự sinh ra.
- **Hướng tiếp cho W1** (đặt ra sau khi xem số, chưa kiểm):
  - Lỗi của chuỗi là lỗi mà chính TFM "tin". Vì vậy có thể đo Φ dọc theo vết thật của chuỗi.
  - Ở vùng hai lưu vực, coi kết quả là **xác suất thoát** thay vì một điểm cuối duy nhất.

**E3. Tác động của v2 với TabPFN v2, PLL q = 0,6** (bốn dataset Study 1 từng sụp):

| | v1 (Study 1) | v2 (chọn theo WPLL) | CC tune |
|---|---|---|---|
| Trung bình 4 dataset | 0,399 | **0,938** | 0,947 |

v2 thắng CC tune trên mfeat-factors (+1,7 điểm), optdigits (+0,8) và MiceProtein (+1,9); thua trên semeion (−8,0).

**v2 với TabICL v2** (cùng các ô; không có dự đoán riêng):
- MiceProtein tăng **+4,7 điểm** (0,937 lên 0,985). Đây là chỗ TabICL yếu nhất ở Study 1.
- Các dataset PLL khác chênh không quá ±0,1 điểm.
- LLP: cylinder-bands giảm 2,2 điểm, do chọn sai (E1); tic-tac-toe tăng 0,3 điểm.

**E6. Chọn một chuỗi so với lấy trung bình mọi chuỗi** (không có tham số nào chỉnh trên dữ liệu này):

| | warm (v1) | Chọn theo WPLL | Trung bình mọi chuỗi | Tốt nhất (biết sau) |
|---|---|---|---|---|
| TabICL, LLP | 0,854 | 0,844 | **0,856** | 0,867 |
| TabICL, PLL | 0,966 | 0,978 | **0,978** | 0,978 |
| TabPFN, LLP | 0,682 | 0,674 | **0,693** | 0,708 |
| TabPFN, PLL | 0,399 | **0,938** | 0,935 | 0,938 |

- Lấy trung bình sửa được chỗ yếu của cách chọn ở LLP mà gần như không mất gì ở PLL.
- Lý do: chuỗi đã sụp cho dự đoán ít tự tin, nên trong trung bình xác suất nó gần như không làm đổi argmax.
- Kết quả này gợi ý dùng **stacking** các chuỗi. Đây là phương pháp đã có: Yao, Vehtari và Gelman (JMLR 2022) gộp các lần chạy MCMC không trộn được thành hỗn hợp có trọng số, chọn trọng số bằng sai số dự đoán kiểm chéo. Ở bài toán của mình, mục tiêu tự nhiên là WPLL. Cách này bao trọn cả hai trường hợp: chọn một chuỗi (trọng số one-hot) và lấy trung bình (trọng số đều).

## 5. Những gì giai đoạn 0 chưa giải được

1. **Gibbs làm kém điểm xuất phát khi gain ở đó âm.** Đã thấy ở semeion với TabPFN và LLP với TabPFN.
   - Cần một cơ chế bảo vệ không dùng nhãn: đưa chính learner khởi động vào làm một ứng viên khi stacking.
   - Muốn vậy phải có WPLL của learner đó, tính ngoài fold.
2. **Dự đoán định lượng cho TFM có hai lưu vực** (W1, xem E2).
3. **LLP với TabPFN:** từ mọi điểm xuất phát, chuỗi đều trôi về điểm hút thấp. Với TFM như vậy, v2 chỉ an toàn nếu có cơ chế bảo vệ ở mục 1.

## 6. Đề xuất cho giai đoạn 1

Đây là đề xuất của em, chờ thầy duyệt. Mọi thử nghiệm làm trên tập phát triển và được ghi trước.

1. **Bộ gộp chuỗi thay cho cách chọn argmax WPLL:**
   - (i) trung bình đều (không tham số);
   - (ii) stacking theo WPLL (Yao et al. 2022);
   - (iii) như (ii) nhưng có thêm learner khởi động làm ứng viên.
   
   So ba cách trên **các dataset phát triển chưa dùng ở giai đoạn 0** (9 dataset phát triển của Study 1 và các dataset còn lại của Study 1 và 2), với cả hai TFM. Chọn một cách trước Study 3 theo quy tắc ghi trước.
2. **Code:** lưu WPLL theo từng vòng quét và WPLL ngoài fold của learner khởi động; lưu dự đoán train ngoài khối để tính stacking.
3. **Bài báo:** đưa R4 (§3) vào §6.6 và Limitations, đồng thời bỏ con số "3 dataset nhỏ nhất". Em đề xuất làm việc này ngay, vì nó liên quan đến tính trung thực của kết quả đã đăng ký.
4. **W1:** đo Φ dọc vết chuỗi; mô hình xác suất thoát ở vùng hai lưu vực; viết mệnh đề cho trường hợp TFM có một điểm hút (đã khớp TabICL tới 0,02).
5. **Giữ nguyên lịch:** cổng V1 ngày 20/12.

## Nguồn

- Yao, Y., Vehtari, A., Gelman, A. (2022). Stacking for non-mixing Bayesian computations: the curse and blessing of multimodal posteriors. *JMLR* 23(79):1–45. https://jmlr.org/papers/v23/20-1426.html. Cần đưa vào REFERENCE_CHECK trước khi trích trong bài.
