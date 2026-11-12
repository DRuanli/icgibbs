# Study 3 (bản gọn): kết quả xác nhận (12/11/2026)

- **Protocol:** [STUDY3_PLAN](STUDY3_PLAN.md), ghi trước ở commit `c9009e8`, trước code, trước script chấm và trước khi tải dữ liệu.
- **Chấm:** `S3/s3_report.py`. Đầu ra ở `S3/results/s3_report.txt` và `s3_summary.json`.
- **Lượt chạy:** 3 phiên Kaggle 2 × T4, ngày 10–12/11.
  - Code khớp `7cc9679` (v14, phiên 1) và `2e04ac4` (v15, phiên 2–3), 85/85 tệp.
  - Bộ test: 161 qua, 2 bỏ qua.
  - Mọi dòng `ok`.
  - Cả 180 tệp npz của phiên 2 không đổi một byte nào ở phiên 3.
- **Độ phủ:** đủ 71/71 cặp (TFM, dataset), 213/213 ô. Cả 20 dataset đều tải được và đạt tiêu chí sau khi loại trùng. Không dataset nào bị loại.

## 0. Kết luận

| | Nội dung ghi trước | Kết quả | Kết luận |
|---|---|---|---|
| **H1** | v2 (A1) hơn v1 | δ₁ trung bình **+0,058** (khoảng tin cậy 95%: +0,016 đến +0,114), trung vị +0,003; thắng 13, hoà 6, thua 1; p (Holm) = **0,0004** | **ĐÚNG** |
| **H2** | đa-TFM hơn một TFM | δ₂ trung bình **+0,046** (+0,019 đến +0,079), trung vị +0,016; thắng 18, hoà 2, thua 0; p (Holm) = **0,000002** | **ĐÚNG** |
| **H3** | H8′: AUC(c3 → T2) ≥ 0,80 | AUC **0,766** (khoảng tin cậy 0,59–0,92), 66 ô dương tính trong 213 ô | **SAI** |
| **C4** | tỉ lệ thụt lùi của A1 ≤ 10% | 10/71 = **14,1%** | **KHÔNG ĐẠT** |
| **C5** | đa-TFM so với TFM tốt nhất | chênh trung bình **+0,00002**; thua quá 0,02 ở 1/20 dataset | **ĐẠT** |

- **Độ nhạy S1** (bỏ 5 dataset có cấu trúc nhóm hoặc chuỗi): H1 vẫn đúng (p = 0,0034), H2 vẫn đúng (p = 0,0001).

## 1. H1: v2 hơn v1, nhưng lợi tập trung ở PLL

| TFM | Số dataset | G = A1 − v1 (trung bình) | Thụt lùi (so với max(v1, L) − 0,02) |
|---|---|---|---|
| TabICL v2 | 20 | +0,024 | 1 |
| TabPFN-3 | 20 | **+0,112** | 4 |
| TabDPT | 20 | +0,037 | 3 |
| TabPFN v2 | 11 | +0,013 | 2 |

- **Theo bối cảnh:** δ₁ trung bình là **+0,136** với PLL (8 dataset) và **+0,007** với LLP (12 dataset).
- **Lợi lớn nhất ở nơi chuỗi v1 sụp.** Ví dụ rõ nhất là TabPFN-3 trên PLL: v1 đạt 0,371, A1 đạt 0,650. Với TabICL trên LLP, lợi chỉ +0,002.
- **Đọc đúng mức:** kiểm định dấu hạng có ý nghĩa vì v2 hiếm khi kém v1 (thua 1/20 dataset). Trung vị +0,003 cho thấy ở một dataset điển hình, lợi là nhỏ.

## 2. H2 và C5: gộp nhiều TFM là kết quả mạnh nhất

- Bản gộp X **không thua TFM trung bình ở dataset nào** (thắng 18, hoà 2), và ngang TFM tốt nhất của từng dataset (chênh +0,00002).
- Dataset duy nhất mà X thua TFM tốt nhất quá 0,02 là micro-mass (1515): 0,645 so với 0,671.
- Lợi lớn nhất so với trung bình các TFM:
  - micro-mass: +0,24;
  - cnae-9: +0,20;
  - vowel: +0,17.
- **Cơ chế** (như ở V1C §4, thăm dò): chuỗi sụp cho dự đoán gần đều, nên khi lấy trung bình, TFM không sụp tự chiếm trọng số.

## 3. H3 sai: tín hiệu chẩn đoán không bền giữa các TFM

| TFM | TabPFN-3 | TabICL v2 | TabPFN v2 | TabDPT | Gộp |
|---|---|---|---|---|---|
| AUC(c3 → T2) | 0,93 | 0,85 | 0,64 | 0,59 | **0,766** |

- Trên tập phát triển, AUC là 0,85–0,86 (thăm dò, và P18). Ở Study 3 nó tụt xuống 0,766, dưới ngưỡng ghi trước 0,80.
- Tín hiệu hoạt động tốt với TabPFN-3 và TabICL, nhưng gần như ngẫu nhiên với TabDPT. V1D đã cho dấu hiệu này (0,74 với TabDPT).
- **Kết luận cho bài:** không tuyên bố có phép chẩn đoán dùng chung cho mọi TFM. Báo cáo H8′ như một kết quả âm tính đã ghi trước, kèm số theo từng TFM.

## 4. C4 không đạt: A1 của một TFM vẫn thụt lùi ở 14% số cặp

Mười cặp thụt lùi:

| TFM | Dataset | v1 | A1 | L |
|---|---|---|---|---|
| TabPFN-3 | 1468 cnae-9 | 0,141 | 0,377 | 0,841 |
| TabPFN-3 | 1515 micro-mass | 0,047 | 0,221 | 0,477 |
| TabDPT | 1515 micro-mass | 0,279 | 0,320 | 0,477 |
| TabDPT | 1467 climate | 0,417 | 0,528 | 0,650 |
| TabICL v2 | 46956 seismic-bumps | 0,554 | 0,491 | 0,719 |
| TabPFN-3 | 46956 seismic-bumps | 0,755 | 0,731 | 0,719 |
| TabDPT | 46937 coupon | 0,666 | 0,682 | 0,707 |
| TabPFN v2 | 46937 coupon | 0,659 | 0,676 | 0,707 |
| TabPFN-3 | 4538 Gesture | 0,470 | 0,470 | 0,496 |
| TabPFN v2 | 1487 ozone | 0,875 | 0,868 | 0,890 |

- **Tách theo mốc:** thụt lùi so riêng với v1 chỉ có **2/71** cặp. So riêng với learner L có 9/71.
- Vậy thất bại của C4 gần như hoàn toàn là "A1 của một TFM chưa bắt kịp learner đã tune", không phải "A1 làm hỏng v1".
- **Hai kiểu:**
  - chuỗi sụp quá sâu, nên A1 kéo lên vẫn chưa đủ (cnae-9, micro-mass);
  - dataset mà learner L vốn hơn mọi chuỗi (coupon, seismic-bumps, climate).
- **Trong bài:** phải ghi rõ rằng A1 của một TFM đơn lẻ **không** đảm bảo không kém learner đã tune.

## 5. Phân tích thăm dò (không ghi trước; viết sau khi xem số)

- **Bản gộp X so với learner L:**
  - X hơn L ở **20/20 dataset**, trung bình +0,064, trung vị +0,030.
  - X không kém L quá 0,02 ở dataset nào.
  - Theo bối cảnh: LLP +0,045, PLL +0,092.
- **A1 của một TFM so với L** (báo cáo kèm đã ghi trước): +0,018, thắng 14, thua 6, p = 0,14, tức không có ý nghĩa thống kê.
- **v1 so với L:** với LLP hơn +0,018; với PLL **kém −0,128**. Con số này lặp lại thất bại F1 của Study 1 trên dataset mới.
- **Lưu ý:** phép so X với L không nằm trong protocol. Nếu đưa vào bài, phải ghi là thăm dò.

## 6. Ý nghĩa cho bài

1. **Tuyên bố được, có xác nhận:**
   - v2 (nhiều điểm xuất phát) hơn v1;
   - gộp nhiều TFM hơn việc dùng một TFM và ngang TFM tốt nhất.
2. **Phải báo cáo trung thực là âm tính:** H3 (chẩn đoán không cần nhãn) và C4 (thụt lùi của A1 đơn TFM).
3. **Khuyến nghị thực hành rút ra:** dùng bản gộp nhiều TFM làm cấu hình mặc định. Nó sửa được cả các ô mà A1 đơn TFM còn thụt lùi. Phần so với L mới chỉ là thăm dò.
4. **Giới hạn:**
   - 20 dataset (8 đa lớp);
   - Study 3 không so với các baseline ngoài, trừ learner L;
   - chi phí gấp 3 lần v1 cho A1, và khoảng 12 lần cho bản gộp 4 TFM.
