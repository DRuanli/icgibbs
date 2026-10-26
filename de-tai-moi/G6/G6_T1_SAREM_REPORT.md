# G6 · T1 — SAR-EM: tái hiện và so sánh trên tập phát triển (26/10/2026)

Kết quả Kaggle lượt `g6_code_v1`: `G4/pilot/g6_sarem_reproduction.csv`, `g6_sarem_dev.csv`, `g6_pusar01_dev.csv` (kèm log và dự đoán). Bảng gộp PU-SAR: `G4/pilot/g6_pusar_dev_merged.csv`, gồm cả các dòng PU-SAR mức 0,3 của lượt v2. Mọi số ở đây chỉ trên **tập phát triển** hoặc dữ liệu của bài gốc; chưa chạm tập đánh giá.

## 1. Tính đúng của dữ liệu
- Đủ dòng: SAR-EM 72/72 (6 dataset × 2 mức × 3 seed × 2 bản); PU-SAR mức 0,1 144/144; tái hiện 2 dataset × 25 thí nghiệm × 4 phương pháp.
- Nhãn yếu giống hệt nhau giữa các lượt: chữ ký (số dương có nhãn, η₀) trùng ở 36/36 ô (dataset × mức × seed); n_train, n_test trùng. Vậy SAR-EM, Gibbs và các baseline cũ được so trên **cùng** nhãn yếu.

## 2. Tái hiện số công bố của SAR-EM (code chính thức, 25 thí nghiệm mỗi dataset)

| Dataset | Phương pháp | ROC-AUC tái hiện | công bố | MSE của f tái hiện | công bố |
|---|---|---|---|---|---|
| Mushroom | SAR-EM | 1,000 ± 0,000 | 1,000 ✓ | 0,098 ± 0,009 | 0,103 ✓ |
| Mushroom | Naive | 1,000 | 1,000 ✓ | 0,189 ± 0,003 | 0,173 ✗ |
| Mushroom | Supervised | 1,000 | 1,000 ✓ | 0,000 | 0,000 ✓ |
| Breast Cancer | SAR-EM | 0,994 ± 0,004 | 0,983 ✗ (cao hơn) | 0,096 ± 0,008 | 0,058 ✗ |
| Breast Cancer | Naive | 0,994 ± 0,004 | 0,992 ✓ | 0,155 ± 0,014 | 0,127 ✗ |
| Breast Cancer | Supervised | 0,996 ± 0,002 | 0,994 ✓ | 0,025 ± 0,007 | 0,028 ✓ |

- **Bản bọc của dự án trùng code chính thức tuyệt đối** trên cả 50 thí nghiệm (chênh AUC = 0). Phần cài đặt mà G6 phải kiểm vì vậy đạt.
- Mushroom đạt tiêu chí. Breast Cancer lệch ở SAR-EM (AUC cao hơn công bố, MSE cao hơn). **Naive (cũng là code gốc) lệch cùng chiều về MSE ở cả hai dataset**, nên nguyên nhân nhiều khả năng nằm ở thiết lập dữ liệu và gán nhãn, không ở SAR-EM. Hai giả thuyết:
  1. Tập thuộc tính của bộ phân loại: bài không nói rõ; notebook gốc ghi `0-<d>`, cận trên tính cả thuộc tính propensity đầu tiên.
  2. Cách gán cột trong Hình 1. Đây là dot plot; bản trích chữ chỉ cho thấy số đầu tiên nằm cạnh nhãn "Naive", và đó là căn cứ để em đọc thứ tự (Naive, SAR-EM, KM2, TIcE, Sup.).
- **Xếp loại tạm thời:** (a) có nguyên nhân khả dĩ, đang kiểm. Script đã có `--cls_mode orig|orig_plus1|all` để kiểm giả thuyết 1 ở lượt Kaggle tới. Cần thầy nhìn Hình 1 của bài để xác nhận giả thuyết 2.

## 3. PU-SAR trên tập phát triển — có thêm SAR-EM (ROC AUC, trung bình 3 seed)

| mức | dataset | naive | nnPU | uPU | SAR-EM | SAR-EM tune | Gibbs ấm | hard-EM ấm | Gibbs, propensity đã biết | oracle |
|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0,1 | banknote | 0,985 | 0,953 | 0,825 | 0,987 | 0,877 | 0,974 | 0,983 | 0,997 | 1,000 |
| 0,1 | blood-transfusion | 0,419 | 0,383 | 0,404 | 0,337 | 0,343 | 0,441 | 0,419 | 0,613 | 0,768 |
| 0,1 | credit-g | 0,632 | 0,647 | 0,639 | 0,626 | 0,587 | 0,624 | 0,634 | 0,664 | 0,798 |
| 0,1 | diabetes | 0,686 | 0,662 | 0,648 | 0,688 | 0,701 | 0,698 | 0,687 | 0,696 | 0,843 |
| 0,1 | ilpd | 0,553 | 0,611 | 0,600 | 0,653 | 0,654 | 0,573 | 0,562 | 0,734 | 0,742 |
| 0,1 | wdbc | 0,932 | 0,943 | 0,743 | 0,870 | 0,862 | 0,865 | 0,896 | 0,570 | 0,992 |
| 0,3 | banknote | 0,997 | 0,979 | 0,898 | 0,996 | 0,965 | 0,996 | 0,993 | 0,999 | 1,000 |
| 0,3 | blood-transfusion | 0,629 | 0,546 | 0,548 | 0,482 | 0,595 | 0,636 | 0,629 | 0,751 | 0,768 |
| 0,3 | credit-g | 0,652 | 0,703 | 0,664 | 0,689 | 0,680 | 0,644 | 0,650 | 0,688 | 0,798 |
| 0,3 | diabetes | 0,700 | 0,684 | 0,660 | 0,746 | 0,719 | 0,722 | 0,721 | 0,782 | 0,843 |
| 0,3 | ilpd | 0,640 | 0,630 | 0,614 | 0,677 | 0,689 | 0,703 | 0,639 | 0,731 | 0,742 |
| 0,3 | wdbc | 0,954 | 0,972 | 0,884 | 0,933 | 0,959 | 0,962 | 0,915 | 0,912 | 0,992 |
| 0,1 | **trung bình** | 0,701 | 0,700 | 0,643 | 0,694 | 0,671 | 0,696 | 0,697 | 0,712 | 0,857 |
| 0,3 | **trung bình** | 0,762 | 0,752 | 0,711 | 0,754 | 0,768 | 0,777 | 0,758 | 0,811 | 0,857 |

Log loss trung bình:

| mức | naive | nnPU | uPU | SAR-EM | SAR-EM tune | Gibbs ấm | hard-EM ấm | Gibbs, propensity đã biết | oracle |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0,1 | 0,981 | 0,987 | 3,041 | 0,962 | 1,048 | 1,237 | 1,488 | 0,866 | 0,344 |
| 0,3 | 0,893 | 1,175 | 2,357 | 0,655 | 0,601 | 0,777 | 0,986 | 0,518 | 0,344 |

**Theo tiêu chí đã dùng cho RQ4** (Gibbs ấm ≥ cả baseline peer-review tốt nhất lẫn naive, tính trên từng ô dataset × mức):

| Tập baseline so sánh | mức 0,1 | mức 0,3 | Tổng |
|---|---|---|---|
| nnPU, uPU (như lượt v2) | 2/6 | 3/6 | 5/12 |
| nnPU, uPU, SAR-EM (mặc định và tune) | 1/6 | 2/6 | **3/12 (25%)** |
| *Tham khảo:* Gibbs với propensity đã biết, so với cùng các baseline | 4/6 | 4/6 | 8/12 |

**Đọc kết quả**
1. **SAR-EM là đối thủ thật.** Trung bình AUC ngang nnPU (0,724 so với 0,726 qua hai mức) và log loss tốt hơn Gibbs rõ ở mức 0,3 (0,60–0,66 so với 0,78). Phát hiện 4 của GATE_REVIEW_G5 đã thành sự thật: khi có baseline chuyên cho SAR, lợi thế của Gibbs ở PU-SAR trên tập phát triển không còn.
2. **Ở mức 0,1, mọi phương pháp gần như không hơn naive.** Ở blood-transfusion, AUC của mọi phương pháp học từ nhãn yếu dưới 0,5: chọn mẫu theo z đảo ngược thứ hạng. Đây là chế độ ít thông tin, giống LLP túi 32 và PU-SCAR c = 0,1.
3. **Nút thắt là việc học propensity (η), không phải bước Gibbs.** Khi biết propensity, Gibbs đạt 8/12 ô và AUC trung bình 0,811 ở mức 0,3 (so với 0,768 của baseline tốt nhất). Lưu ý: bản này dùng η thật, nhiều thông tin hơn SAR-EM; muốn so công bằng thì phải so với SAR-E (bộ phân loại với propensity đã biết) của cùng gói.
4. RQ3 vẫn giữ: Gibbs có log loss tốt hơn hard-EM ở 12/12 ô PU-SAR. Gộp mọi thí điểm: 63/72 ô = 87,5%.
5. Bản tune của SAR-EM (chọn theo loss trên nhãn yếu) **không** tốt hơn bản mặc định về AUC ở mức 0,1 (0,671 so với 0,694). Loss trên nhãn yếu không phải lúc nào cũng cùng chiều với AUC.

## 4. Hệ quả và việc cần quyết

> **Đã quyết ngày 26/10/2026** (thầy giao Claude chọn): cả hai đề xuất dưới đây được áp dụng — PU-SAR thành bối cảnh phụ; ứng viên đối thủ chính gồm cả bản mặc định lẫn bản tune. Ghi ở giao thức Mục 11 (deviation 2 và 3).

- **Phạm vi PU-SAR (thầy quyết).** Theo cùng ngưỡng 60% đã dùng để phân vai RQ4, PU-SAR (3/12) phải thành bối cảnh phụ. Khi đó khẳng định chính còn: LLP túi ≤ 16, PLL, RQ3 và lý thuyết. "Thành công tối thiểu" của G3 (H2 cho PU *hoặc* LLP) vẫn khả thi qua LLP. Giao thức đã đóng băng nên thay đổi này phải ghi ở Mục 11, với căn cứ chỉ từ tập phát triển. Kết quả "propensity đã biết" có thể giữ làm phân tích, cho thấy bước Gibbs giúp được khi cơ chế quan sát đã biết.
- **Làm rõ quy tắc chọn đối thủ chính (đề xuất).** Vì bản tune có thể kém bản mặc định, đề xuất: ở bước chọn đối thủ chính trên tập phát triển, mỗi baseline được xét cả **bản mặc định lẫn bản tune**, và chọn bản tốt hơn trên tập phát triển. Cách này khắt khe hơn với phương pháp của mình và vẫn quyết trước khi chạy tập đánh giá. Ghi ở giao thức Mục 11 nếu thầy đồng ý.
- **Rủi ro tương tự với LLP và PLL.** Ở thí điểm, LLP mới so với DLLP, PLL mới so với PRODEN chưa tune. Việc tiếp theo của G6 (T3, T4) là đưa các baseline peer-review còn lại (LLP-VAT, EasyLLP, GLWS, ILL, PRODEN có tune) chạy trên tập phát triển trước G7, để biết lợi thế còn lại thật sự ở đâu trước khi chạm tập đánh giá.
