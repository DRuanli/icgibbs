# G6 lượt 2 và 3 — baseline peer-review trên tập phát triển, chọn đối thủ chính, kiểm lại phạm vi (26/10/2026)

Việc T3–T4 của [G6_PLAN](G6_PLAN.md). Mọi số trong file này chỉ từ **tập phát triển**; chưa phương pháp nào chạy trên tập đánh giá. Dữ liệu: `G4/pilot/g6r2_{llp,pll,pu}.csv` (kèm `_preds.jsonl`, `_log.txt`), `g6r2_sarem_repro_*.csv`; script chọn: `G4/code/g6_select_comparators.py` (viết trước khi có kết quả, commit `3a900c1`); kết quả chọn: `G4/pilot/g6_comparators.csv`.

## 1. Tính đầy đủ và đúng của dữ liệu
- Chạy trên Kaggle CPU từ 08:11 đến 10:13 UTC (khoảng 2 giờ). **684/684 dòng `ok`** (PLL 144, PU 180, LLP 360) — đúng bằng số ô × phương pháp dự kiến. Không có dòng lỗi, dòng trùng, giá trị NaN, hay dự đoán hằng (AUC = 0,5).
- **Cùng dữ liệu với các lượt chạy Gibbs/naive trước:** `n_train`, `n_test` khớp ở mọi ô; nhãn test giống hệt ở 36/36 ô LLP (so file dự đoán); số dương có nhãn (PU) khớp 54/54 ô; số ứng viên trung bình (PLL) khớp 18/18 ô. So sánh giữa các lượt vì vậy là so trên cùng lần chia và cùng nhãn yếu.
- Môi trường: Python 3.13.15, numpy 2.1.3, torch 2.11.0+cpu; PRODEN chính thức ở commit `637b531`.
- Kiểm chéo cài đặt: DLLP tuyến tính bản mới (`dllplin_def`) khớp bản cũ dùng làm khởi động ấm (`dllp_linear`) tới 1e-4 AUC trung bình.

## 2. Đối thủ chính theo quy tắc ghi trước (giao thức Mục 6 và Mục 11)
Ứng viên = mọi baseline peer-review của bối cảnh, mỗi baseline cả bản mặc định (`_def`) lẫn bản tune 20 cấu hình (`_tun`). Đối thủ chính = ứng viên có metric chính trung bình tốt nhất (2 mức chính × 6 dataset × 3 seed; PLL: 3 dataset).

| Bối cảnh (metric) | Đối thủ chính | Trung bình | Các ứng viên còn lại (trung bình) |
|---|---|---|---|
| LLP (AUC), gộp túi 8 và 16 như kế hoạch ban đầu | DLLP tuyến tính, tune (`dllplin_tun`) | 0,8191 | dllplin_def 0,8182 · glws_def 0,8134 · dllp_def 0,8125 · llpvat_tun 0,8106 · llpvat_def 0,8063 · glws_tun 0,8062 · easyllp_tun 0,8029 · dllp_tun 0,8023 · easyllp_def 0,7750 |
| **LLP (AUC), túi 8 — mức chính sau quyết định ở Mục 6** | **DLLP tuyến tính, mặc định** (`dllplin_def`) | 0,8254 | dllplin_tun 0,8248 · glws_def 0,8136 · glws_tun 0,8122 · dllp_def 0,8122 · llpvat_tun 0,8114 · easyllp_tun 0,8099 · llpvat_def 0,8082 · dllp_tun 0,8078 · easyllp_def 0,7712 |
| PLL (accuracy) | **GLWS/ILL, tune** (`glwspll_tun`) — *tạm thời, xem Mục 5* | 0,7451 | cc_tun 0,7422 · rc_tun 0,7411 · glwspll_def 0,7402 · proden_tun 0,7161 · cc_def 0,6727 · rc_def 0,6724 · proden_def 0,5730 |
| PU-SAR (AUC; bối cảnh phụ) | nnPU mặc định (`nnpu`) | 0,7260 | nnpu_tun 0,7253 · sar_em 0,7238 · sar_em_tuned 0,7193 · upu_tun 0,6815 · upu 0,6771 |
| PU-SCAR (log loss khớp tiên nghiệm, thấp hơn là tốt; bối cảnh phụ) | nnPU mặc định (`nnpu`) | 0,7130 | upu_tun 0,7954 · nnpu_tun 0,8948 · upu 0,9786 |

Ghi chú:
- Tune trên loss của nhãn yếu không phải lúc nào cũng giúp: ở LLP bản tune xấp xỉ bản mặc định (GLWS tune còn kém hơn); ở PU-SCAR, nnPU tune tệ hơn về log loss vì được chọn theo rủi ro PU, không theo log loss. Quy tắc xét cả hai bản (deviation 3) vì vậy có tác dụng thật.
- PU-SCAR, PU-SAR: VPU và In-Context PU (baseline phụ) chưa cài; nếu cài xong trước G7 thì chạy lại bước chọn cho hai bối cảnh phụ này.
- RQ5 (H5) xếp hạng mọi phương pháp, không cần đối thủ chính.

## 3. Kiểm lại phạm vi khẳng định
Quy tắc (viết trong `g6_select_comparators.py` trước khi có kết quả; cùng ngưỡng 60% đã dùng để phân vai RQ4 và PU-SAR): Gibbs ấm phải đạt hoặc vượt **cả** ứng viên tốt nhất của từng ô (lấy max qua mọi ứng viên) **lẫn** TFM ngây thơ ở ≥ 60% ô (dataset × mức).

| Bối cảnh | Gibbs ấm | TFM ngây thơ | Đối thủ chính | Ô đạt (quy tắc chặt) | Ô đạt so với riêng đối thủ chính và naive |
|---|---|---|---|---|---|
| **LLP** (túi 8, 16) | 0,8158 | 0,6315 | 0,8191 | **7/12 (58%) — dưới ngưỡng một ô** | 9/12 |
| ↳ túi 8 | 0,8386 | 0,7037 | 0,8248 | 5/6 (83%) | 5/6 |
| ↳ túi 16 | 0,7931 | 0,5594 | 0,8134 | 2/6 (33%) | 4/6 |
| **PLL** (q = 0,3; 0,6) | 0,7750 | 0,6115 | 0,7451 | **6/6 (100%)** | 6/6 |
| PU-SAR (phụ) | 0,7366 | 0,7317 | 0,7260 | 3/12 | 5/12 |
| PU-SCAR (phụ; log loss) | 0,4343 | 0,4182 | 0,7130 | 3/12 | 3/12 |

**Tiêu chí dừng của RESEARCH_PLAN §5 (mục 2) không bị kích hoạt:** tính trung bình qua hai mức, Gibbs đạt cả ứng viên tốt nhất lẫn naive ở 4/6 dataset LLP (banknote, diabetes, ilpd, wdbc); tiêu chí chỉ kích hoạt khi thất bại ở cả PU lẫn LLP.

### 3.1 LLP theo từng ô (AUC, trung bình 3 seed)

| Túi | Dataset | Gibbs ấm | naive | DLLP tuyến tính tune | Ứng viên tốt nhất của ô | Gibbs − tốt nhất | Oracle |
|---|---|---|---|---|---|---|---|
| 8 | banknote | 1,0000 | 0,9815 | 0,9996 | 1,0000 (dllp_def) | 0,0000 | 1,0000 |
| 8 | blood-transfusion | 0,7562 | 0,6426 | 0,7705 | 0,7705 (dllplin_tun) | −0,0144 | 0,7680 |
| 8 | credit-g | 0,7530 | 0,5836 | 0,7380 | 0,7497 (glws_tun) | +0,0034 | 0,7977 |
| 8 | diabetes | 0,8139 | 0,5606 | 0,8107 | 0,8107 (dllplin_tun) | +0,0032 | 0,8435 |
| 8 | ilpd | 0,7179 | 0,6482 | 0,6411 | 0,6723 (easyllp_def) | +0,0456 | 0,7419 |
| 8 | wdbc | 0,9906 | 0,8057 | 0,9892 | 0,9893 (glws_tun) | +0,0012 | 0,9915 |
| 16 | banknote | 1,0000 | 0,6100 | 0,9994 | 1,0000 (dllp_def) | 0,0000 | 1,0000 |
| 16 | blood-transfusion | **0,6054** | 0,5080 | 0,7670 | 0,7670 (dllplin_tun) | **−0,1615** | 0,7680 |
| 16 | credit-g | **0,6522** | 0,5007 | 0,7012 | 0,7337 (dllp_def) | **−0,0815** | 0,7977 |
| 16 | diabetes | 0,7973 | 0,5272 | 0,7910 | 0,7997 (glws_def) | −0,0023 | 0,8435 |
| 16 | ilpd | 0,7156 | 0,6355 | 0,6356 | 0,6967 (easyllp_tun) | +0,0189 | 0,7419 |
| 16 | wdbc | 0,9880 | 0,5749 | 0,9865 | 0,9888 (glws_def) | −0,0008 | 0,9915 |

- Ở túi 8, Gibbs vượt đối thủ chính ở 5/6 dataset (trung bình +0,014 AUC) và vượt naive ở 6/6 (+0,135).
- Ở túi 16, Gibbs hỏng nặng ở blood-transfusion và credit-g, và **tệ hơn chính điểm khởi động của nó** (khởi động ấm là DLLP tuyến tính: 0,727 và 0,714). Chẩn đoán chuỗi (cột `extra` của `g4_pilot_v2.csv`): ở các ô này, độ chính xác của trạng thái chuỗi trên nhãn train giảm từ khoảng 0,75 (khởi động) xuống 0,64–0,69, với khoảng 30% nhãn còn đổi ở vòng quét cuối. Chuỗi trôi về những cách gán nhãn tệ hơn. Đây là hai dataset tín hiệu yếu (oracle 0,77 và 0,80), nơi ràng buộc túi 16 quá lỏng để giữ chuỗi. Không thấy dấu hiệu lỗi cài đặt.
- So với chính điểm khởi động (DLLP tuyến tính, cùng cài đặt): Gibbs +0,013 AUC ở túi 8, −0,018 ở túi 16.
- Log loss: Gibbs 0,390, DLLP tuyến tính tune 0,477, naive 0,614 (trung bình hai mức) — Gibbs hiệu chỉnh tốt hơn kể cả ở túi 16.

### 3.2 PLL theo từng ô (accuracy)

| q | Dataset | Gibbs ấm | naive | hard-EM ấm | GLWS/ILL tune | Ứng viên tốt nhất | Oracle |
|---|---|---|---|---|---|---|---|
| 0,3 | cmc | 0,5460 | 0,5302 | 0,5249 | 0,5136 | 0,5158 (rc_tun) | 0,5475 |
| 0,3 | segment | 0,9788 | 0,9452 | 0,9803 | 0,9654 | 0,9711 (glwspll_def) | 0,9894 |
| 0,3 | vehicle | 0,8268 | 0,7493 | 0,8241 | 0,7979 | 0,8136 (cc_tun) | 0,8530 |
| 0,6 | cmc | 0,5294 | 0,4578 | 0,4977 | 0,4947 | 0,4947 (glwspll_tun) | 0,5475 |
| 0,6 | segment | 0,9697 | 0,5296 | 0,9153 | 0,9533 | 0,9533 (glwspll_tun) | 0,9894 |
| 0,6 | vehicle | 0,7992 | 0,4567 | 0,7480 | 0,7454 | 0,7664 (glwspll_def) | 0,8530 |

Gibbs vượt đối thủ chính ở 6/6 ô (+0,014 đến +0,054) và vượt ứng viên tốt nhất của từng ô ở 6/6. Khẳng định chính cho PLL (H4) giữ nguyên.

## 4. Tái hiện SAR-EM trên Breast Cancer: kiểm giả thuyết 1
Giả thuyết 1 ([G6_T1_SAREM_REPORT](G6_T1_SAREM_REPORT.md) §2): chênh lệch do tập thuộc tính của bộ phân loại. Kết quả 25 thí nghiệm mỗi chế độ (ROC-AUC; MSE của f):

| Chế độ thuộc tính | SAR-EM AUC (công bố 0,983) | SAR-EM MSE f (công bố 0,058) | Naive AUC (công bố 0,992) | Naive MSE f (công bố 0,127) |
|---|---|---|---|---|
| `orig` (lượt 1) | 0,994 ± 0,004 | 0,096 ± 0,008 | 0,994 ± 0,004 | 0,155 ± 0,014 |
| `orig_plus1` | 0,994 ± 0,003 | 0,098 ± 0,009 | 0,990 ± 0,005 | 0,160 ± 0,014 |
| `all` | 0,988 ± 0,007 ✓ | 0,097 ± 0,008 | 0,980 ± 0,009 ✗ | 0,158 ± 0,013 |

- **Giả thuyết 1 bị bác:** không chế độ nào tái hiện được đồng thời cả bốn số. MSE của f gần như không đổi theo tập thuộc tính, và lệch cùng chiều ở cả SAR-EM lẫn naive (naive chỉ là hồi quy logistic trên s, không dùng thuật toán SAR-EM).
- Bản bọc của dự án vẫn trùng code chính thức tuyệt đối (chênh AUC = 0 ở cả 50 thí nghiệm mới).
- **Xếp loại: (b) — chưa rõ nguyên nhân, nhưng đã khoanh vùng ở phần thiết lập dữ liệu và đánh giá dùng chung cho mọi phương pháp, không ở thuật toán SAR-EM.** Mushroom đạt. SAR-EM vẫn được dùng (PU-SAR là bối cảnh phụ); bài ghi rõ chênh lệch. Giả thuyết 2 (cách đọc Hình 1) không còn cần để quyết định, nhưng nếu thầy xem được hình thì em ghi thêm.

## 5. PRODEN: bản dùng loss chính thức yếu hơn bản cũ
- PRODEN bản cũ của dự án (Adam, ẩn (64, 64)) đạt 0,740 accuracy trung bình trên tập phát triển; bản mới dùng `partial_loss` chính thức với SGD momentum 0,9 như kho tác giả chỉ đạt 0,573 (mặc định) và 0,716 (tune). Nguyên nhân: SGD với lr 1e-3, 200 epoch là chưa đủ trên dữ liệu bảng nhỏ (khoảng 1.000–1.600 bước), và không gian tune lr ≤ 1e-2 không bù hết.
- Không ảnh hưởng kết luận PLL (Gibbs 0,775 vượt cả hai bản), nhưng baseline phải ở dạng mạnh nhất hợp lý. **Đề xuất (khắt khe hơn với phương pháp của mình):** thêm bộ tối ưu ∈ {SGD momentum 0,9 (mặc định, như kho chính thức), Adam} vào không gian tune của PRODEN; chạy lại `proden_tun` trên tập phát triển (Kaggle CPU, khoảng 15 phút) và chạy lại bước chọn đối thủ chính cho PLL trước G7. Đối thủ chính của PLL là **tạm thời** tới khi xong việc này.

## 6. Hệ quả
- **PLL:** khẳng định chính giữ nguyên, bằng chứng trên tập phát triển mạnh hơn trước (6/6 ô với đối thủ đã tune).
- **LLP:** quy tắc 60% gộp hai mức cho 7/12 ô — thiếu một ô. Bằng chứng tách rõ theo mức: túi 8 đạt 5/6, túi 16 đạt 2/6. **Quyết định (thầy, 26/10/2026): khẳng định chính chỉ còn túi 8**; túi 16 thành phụ (giao thức §11, deviation 7). Đối thủ chính chọn lại theo quy tắc trên mức chính mới: DLLP tuyến tính mặc định (0,8254); Gibbs 0,8386, đạt 5/6 ô ở túi 8.
- **PU-SCAR, PU-SAR:** vẫn là bối cảnh phụ; kết quả lượt 2 không đổi điều đó.
- **Khớp với PROOFS §7:** chỗ để Gibbs giúp tăng theo cỡ túi, nhưng ở túi 16 chuỗi không giữ được trạng thái tốt trên dataset tín hiệu yếu — đúng loại rủi ro "trộn chậm" đã ghi ở METHOD_SPEC (rủi ro 3).

## 7. Lượt 3 (Kaggle CPU, 11:34–11:46 UTC)
- **Dữ liệu:** 162/162 dòng `ok` (PRODEN tune 18; label spreading tune, self-training tune, forward correction mặc định và tune: 36 mỗi loại). Cùng lần chia và nhãn yếu với các lượt Gibbs/naive: kích thước train/test, số ứng viên trung bình (PLL), số mẫu có nhãn (SSL), tỉ lệ nhiễu thực tế (nhãn nhiễu) khớp ở mọi ô; nhãn test giống hệt ở 90/90 ô có file dự đoán.
- **PRODEN tune có Adam:** 0,7378 (trước: 0,7161 khi chỉ có SGD; bản Adam cũ: 0,7399). Bản tune chọn Adam ở 16/18 ô. Cải thiện rõ nhất ở vehicle (+0,037 ở q = 0,3; +0,051 ở q = 0,6). Đối thủ chính của PLL không đổi: GLWS/ILL tune (0,7451) — **chốt**. Gibbs 0,7750, đạt 6/6 ô.
- **Nhãn nhiễu (phụ):** đối thủ chính theo quy tắc là CL + TFM (0,8465); forward correction tune 0,8097, mặc định đã ghi 0,8012, bản thí điểm 0,7749. Gibbs 0,8319, TFM ngây thơ 0,8418. Kiểm phạm vi: 4/12 ô.
- **SSL (phụ):** đối thủ chính là self-training tune (0,8185); self-training mặc định 0,8066; label spreading 0,7705 (tune 0,7704; trung bình 10 điểm test mỗi ô phải gán tiên nghiệm vì không nhận được nhãn lan truyền). Gibbs 0,8090, TFM ngây thơ 0,8148. Kiểm phạm vi: 2/12 ô (trước lượt 3, so với baseline mặc định: 3/12).
- Ở cả hai bối cảnh phụ, Gibbs thua TFM ngây thơ về AUC trung bình. Điều này khớp loại 1 của PROOFS §7: bộ ngây thơ dùng chính quan sát, nên không còn thông tin để Gibbs lấy lại. Dự đoán thăm dò D3 (chênh lệch AUC trung bình ≤ +0,02) vẫn đúng trên tập phát triển.
- **Còn thiếu (bối cảnh phụ):** GLWS cho PU; VPU, In-Context PU; ILL cho nhãn nhiễu và SSL; VIME. Đối thủ chính của các bối cảnh phụ là tạm thời tới khi có các baseline này.
