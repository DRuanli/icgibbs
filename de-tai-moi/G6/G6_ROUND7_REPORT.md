# G6 lượt 7 — tổng duyệt ablation, cỡ dữ liệu, TabICL và cách chạy 2 GPU; phát hiện ở baseline GLWS/ILL cho PLL (27/10/2026)

Mọi số trong file này chỉ từ **tập phát triển** (hoặc dữ liệu tổng hợp); chưa phương pháp nào chạy trên tập đánh giá. Kết quả ablation ở đây là **thăm dò**: dùng để kiểm code và để biết trước điều gì có thể xảy ra, không để đổi cấu hình đã đóng băng. Dữ liệu ở `G4/pilot/g6r7_*.csv` (kèm dự đoán), log gộp ở `g6r7_logs.txt`.

## 1. Đầy đủ, môi trường, cách chạy 2 GPU
- Kaggle, 2 × Tesla T4, 08:22–10:35 UTC (2 giờ 13 phút). **509/509 dòng `ok`**, không lỗi, không hết bộ nhớ. Code chạy trùng với kho.
- Số dòng đúng như dự kiến theo quy tắc của từng ablation, ví dụ: khởi tạo lạnh 36 (LLP 6 dataset × 3 seed + PLL 3 × 2 mức × 3), K 36, số vòng quét 36, chuyển nạp 18, prior η 72, cỡ dữ liệu 105 (TFM) + 21 (đối thủ chính), TabICL 180, TabICL 16 lớp 5.
- **Hai luồng song song, mỗi luồng một GPU, chạy tốt.** Hai luồng xong cách nhau 1 phút (131 và 133 phút). Thời gian mỗi chuỗi bằng khoảng 0,6–1,4 lần thời gian của chuỗi tương ứng ở lượt 5 (một luồng) — so thô vì chuỗi lạnh ở đây không gồm bước khởi động ấm — nên thông lượng tăng khoảng 1,6–2 lần.
- **TabICL v2 (đã sửa theo F6) chạy đúng:** checkpoint `tabicl-classifier-v2-20260212.ckpt` tải một lần rồi dùng lại; tabicl 2.2.0. Nhanh hơn TabPFN v2 khoảng **3 lần** (thời gian bằng 0,26–0,37 lần).

## 2. Kết quả thăm dò của các ablation (so ghép cặp với `gibbs_warm` của lượt 5, cùng seed)
| Ablation | LLP túi 8 (AUC) | PLL q = 0,3 (accuracy) | Đọc |
|---|---|---|---|
| Khởi tạo lạnh − ấm | −0,008 (18 ô; kém/hơn 7/8) | +0,000 (q = 0,6: +0,001) | không khác có hệ thống; blood −0,039 |
| K = 5 − K = 10 | −0,004 | −0,004 | không nhạy với K |
| K = 20 − K = 10 | −0,016 (blood −0,051) | −0,005 | không nhạy với K, trừ blood |
| 15 − 30 vòng quét | **−0,031** (blood −0,049; ilpd −0,045) | +0,000 | 15 vòng là quá ít ở LLP |
| 60 − 30 vòng quét | +0,007 (blood +0,010; ilpd +0,010) | −0,004 | thêm vòng quét giúp chút ít ở LLP |
| Chuyển nạp − quy nạp | **−0,033** (ilpd −0,158; một ô 0,36 so với 0,76) | — | chuyển nạp kém ổn định |

Cả 6 dòng đều không có ý nghĩa thống kê ở mức 0,05 (Wilcoxon, chỉ để tham khảo). Các quan sát:
- **Số vòng quét:** LLP nhạy với số vòng quét đúng ở hai dataset mà R̂ đã báo trộn chậm (blood, ilpd). 30 vòng chưa phải điểm bão hoà, nhưng tăng lên 60 chỉ thêm +0,007 AUC. Cấu hình đã đóng băng (30 vòng) giữ nguyên; ablation trên tập đánh giá sẽ đo lại.
- **Chuyển nạp:** ở ilpd seed 0, chuỗi chuyển nạp cho AUC 0,36, tức trôi về cách gán nhãn ngược. Điểm test không có ràng buộc nào (ℓ ≡ 1), nên khi TFM yếu, nhãn của chúng có thể tự củng cố theo hướng sai. Điều này ủng hộ lựa chọn quy nạp làm mặc định (METHOD_SPEC D2).
- **Prior η của PU-SAR:** N(0, 4²) hơn N(0, 1) +0,010 AUC (p = 0,03; 36 ô). Mặc định N(0, 2²) nằm giữa; lượt này không chạy mặc định (PU-SAR chưa chạy trên bộ chạy G7). Chú ý: trên bộ chạy G7, Gibbs PU-SAR với hai prior này đạt 0,66–0,72 AUC, thấp hơn số thí điểm cũ (0,70–0,78) — cùng chiều với việc LLP yếu đi khi xử lý dữ liệu đúng giao thức. PU-SAR là bối cảnh phụ.

## 3. TabICL v2 thay TabPFN v2 (ablation "TFM", hai bối cảnh chính)
| | TFM ngây thơ | Gibbs ấm | hard-EM | oracle | Đối thủ chính | Gibbs ≥ đối thủ (ô) |
|---|---|---|---|---|---|---|
| LLP túi 8, TabICL | 0,669 | **0,815** | 0,805 | 0,860 | 0,811 (DLLP tuyến tính) | 5/6 |
| LLP túi 8, TabPFN | 0,690 | 0,814 | 0,803 | 0,855 | | 5/6 |
| PLL, TabICL | 0,672 | **0,803** | 0,788 | 0,819 | 0,749 (GLWS/ILL tune*) | 6/6 |
| PLL, TabPFN | 0,600 | 0,785 | 0,748 | 0,805 | | 6/6 |

\* trước khi sửa GLWS/ILL (Mục 5). Kết luận định tính không đổi khi đổi TFM. Ở PLL, TabICL còn tốt hơn TabPFN (vehicle +0,05).

**Nhiều lớp (TabICL, dữ liệu tổng hợp 16 lớp, 1.400 mẫu train):** Gibbs 0,72 so với TFM ngây thơ 0,13, oracle 0,79 — đường chạy cho PLL thật > 10 lớp (RQ5) hoạt động.

## 4. Cỡ dữ liệu (n_train 500, 1.000 so với toàn bộ)
- **PLL:** Gibbs hơn GLWS/ILL tune ở mọi cỡ: segment 0,970 so với 0,938 (500), 0,975 so với 0,951 (1.000), 0,981 so với 0,962 (1.460); vehicle 0,831 so với 0,812 (500); cmc 0,518 so với 0,495 (500). Lợi thế ở segment lớn hơn khi ít dữ liệu.
- **LLP:** ở 500 mẫu, Gibbs ngang DLLP tuyến tính (credit-g 0,592 so với 0,591; diabetes 0,802 so với 0,800); hard-EM lại cao hơn ở credit-g (0,641).
- Tập phát triển chỉ có dataset nhỏ nên chỉ đo được cỡ 500 và 1.000; tập đánh giá sẽ có đủ 4 cỡ trên 3 dataset lớn nhất mỗi nhóm.

## 5. Phát hiện khi chuẩn bị T3: GLWS/ILL cho PLL thiếu một số hạng của bản chính thức
Khi đối chiếu config chính thức để chọn cấu hình tái hiện, em thấy **mọi config PLL** của kho GLWS/ILL (20 tệp `config/partial_label/classic_cv/*.yaml`: MNIST, Fashion-MNIST, CIFAR-10/100, STL-10, ImageNet-100) đều bật `average_entropy_loss: True`: số hạng 0,1 × CE(phân phối đều ‖ dự đoán trung bình của batch) trong `imp_partial_label.py`. Bản cài của dự án (`PLLNet("glws")`) **thiếu số hạng này**. Đây là đối thủ chính của H4.
- **Đã sửa:** thêm số hạng đúng công thức của kho; mặc định bật như config chính thức; không gian tune thêm bật/tắt (như đã làm cho ILL nhãn nhiễu và SSL, deviation 9). 3 test mới, gồm test khớp từng phép tính với công thức của kho.
- **Hệ quả:** `glwspll_def` và `glwspll_tun` phải chạy lại trên tập phát triển, rồi áp lại quy tắc chọn đối thủ chính của PLL (G6 lượt 8). Đối thủ chính của PLL tạm về trạng thái "chờ lượt 8". Ghi ở giao thức §11 (deviation 13).
- Các thuật toán khác của kho đã đối chiếu lại: LLP (`imp_proportion_label.py`) và PU (`imp_pos_ulb.py`) không có số hạng tương tự; nhãn nhiễu và SSL đã có từ lượt 4.

## 6. T3 — tái hiện số công bố: PRODEN, RC, CC trên UCI (sẵn sàng ở lượt 8)
Bài PRODEN chỉ có số MNIST trong hình, nhưng có **bảng số cho 5 bộ UCI với mô hình tuyến tính**; kho chính thức ghi sẵn lr, weight decay, cỡ batch cho từng bộ (`parameter_settings.txt`). Bài RC/CC cũng có bảng UCI với mô hình tuyến tính. Cả hai đúng dạng dữ liệu bảng của đề tài, rẻ (chỉ CPU), và không trùng dataset nào của tập phát triển hay tập đánh giá. Script `g6_reproduce_pll_uci.py`:
- **PRODEN** (Bảng 1, binomial q = 0,1 và 0,7): Yeast, Texture, Dermatology, Synthetic Control; sinh nhãn bằng hàm `partialize` chính thức; 5 lần chia 90/10.
- **RC, CC** (Bảng 3, tập ứng viên đều): Texture, Yeast, Dermatology, Har; lr và weight decay chọn trong lưới 6 × 6 của bài; 10 lần chia 90/10.
- 20Newsgroups (bản 300 chiều của văn liệu PLL): em chưa tìm được bản này trên OpenML nên bỏ.
- Tiêu chí như G6_PLAN §3. Chỗ bài không nói rõ (PRODEN: "5-fold" nhưng bảng số mẫu là 90/10; RC/CC: chọn siêu tham số một lần hay mỗi lần chia) em chọn và ghi ở đầu script.

**Còn lại của T3, cần quyết sau:**
- **GLWS/ILL:** mọi thí nghiệm của bài dùng ảnh, tăng cường mạnh và LeNet/WRN. Em đề xuất kiểm tương đương từng phép tính với code chính thức thay cho tái hiện số.
- **DLLP, EasyLLP:** bài EasyLLP chỉ có số trong hình. Phải đọc số từ hình, với sai lệch cho phép ghi trước.
- **LLP-VAT:** cần CIFAR với ResNet, tốn GPU.

## 7. Ước tính compute cho G7 (cập nhật)
TabICL nhanh hơn dự kiến, nên ablation TFM chỉ còn khoảng 4,5 giờ; tổng GPU khoảng **72 giờ**. Với 2 GPU song song, cần khoảng **40–45 giờ phiên**. Baseline vẫn khoảng 100 giờ CPU.
