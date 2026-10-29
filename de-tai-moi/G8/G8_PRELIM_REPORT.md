# G8 — phân tích xác nhận sơ bộ (29/10/2026)

**Trạng thái dữ liệu.**
- Phần chính (TFM): có 2.165/2.190 ô. Thiếu 25 ô vì OpenML trả lỗi 503 ở lượt L3 (PLL q = 0,6, seed 1, 5 dataset).
- Baseline: đủ 3.492/3.492 ô.
- Chưa có L4–L6 (R̂, RQ5, ablation, TabICL). Các lượt này không thuộc H2, H3, H4.
- H2-LLP và H3 đã đủ dữ liệu. H4-PLL thiếu 5 ô; Mục 3 chỉ ra kết luận không đổi dù 5 ô này ra sao.
- G7_PLAN quy định không xem metric tổng hợp trước khi đủ dữ liệu. Phân tích này làm sớm theo yêu cầu của thầy (29/10). Từ kết quả này, **không đổi code, giao thức hay lượt chạy nào**; các lượt L4–L6 và lần chạy lại L3 giữ nguyên lệnh đã định.

**Cách làm.**
- Script: `G8/analysis/g8_confirmatory.py`. Script viết theo đúng kế hoạch thống kê đã đóng băng (giao thức Mục 6, deviation 1–13) và chỉ đọc `G7/raw/`.
- Kết quả: `G8/results/` (json, bảng theo dataset, từng ô H3, so sánh với từng baseline).
- Phân tích lỗi ở Mục 3.2 là **thăm dò, làm sau khi xem số liệu**: `g8_pll_collapse.py`, ra `pll_chain_diag.csv`.
- Chạy lại toàn bộ: `python g8_confirmatory.py && python g8_pll_collapse.py`.

## 1. Tóm tắt

| Khẳng định | Kết quả | Đạt? |
|---|---|---|
| **H2-LLP túi 8** (AUC) | Gibbs 0,855, DLLP tuyến tính 0,831: **+0,024**, CI 95% [−0,013; +0,057], thắng/thua 8/2, p = 0,23 (Holm 0,23). So với TFM ngây thơ: +0,148, Holm p = 0,004. So với baseline mạnh nhất trên tập đánh giá (LLP-VAT tune): −0,006, thắng 6/10 | **Không** (không có ý nghĩa so với DLLP tuyến tính; 6/10 < 7/10) |
| **H4-PLL** (accuracy) | q = 0,3: **+0,022** so với CC tune, CI [+0,005; +0,044], 10/4, p = 0,042, Holm 0,084. q = 0,6: **−0,133**, 7/7, p = 0,63. Hơn TFM ngây thơ có ý nghĩa ở cả hai mức (Holm ≤ 0,002). Quy tắc 60% đạt hay không tuỳ cách đọc (Mục 3.1) | **Không** (theo cách đọc yếu nhất; Mục 3.1) |
| **H3** (log loss, Gibbs so với hard-EM) | 128 ô; Gibbs tốt hơn ở **86,7%**; P(Gibbs tốt hơn) = 1,00, P(ROPE) = 0,00; Spearman ρ = 0,36 (p = 3·10⁻⁵) | **Có** |
| Định lý 1 + RQ1 | xong ở G4 | Có |
| Thăm dò D1, D2, D3 | cả ba đúng như dự đoán ghi trước (Mục 5) | — |

**Tiêu chí thành công** (RESEARCH_PLAN §5):
- Mức tối thiểu cần Định lý 1, RQ1, (H2-LLP *hoặc* H4-PLL) và H3. Trên dữ liệu sơ bộ, mức này **chưa đạt**: cả H2-LLP lẫn H4-PLL đều không đạt.
- Điều kiện chuyển sang phương án B đã ghi trước ("chỉ chuyển phương án B khi cả H2-LLP lẫn H4-PLL đều không đạt") hiện **đã thoả**.
- Quyết định chính thức thuộc cổng G8, sau khi đủ dữ liệu (Mục 6).

## 2. H2-LLP túi 8

Bảng dưới là AUC, trung bình 3 seed (`G8/results/table_llp8.csv`).

| dataset | TFM ngây thơ | Gibbs | hard-EM | DLLP tuyến tính (đối thủ chính) | LLP-VAT tune (mạnh nhất) | oracle |
|---|---|---|---|---|---|---|
| credit-approval | 0,608 | **0,941** | 0,927 | 0,897 | 0,892 | 0,952 |
| cylinder-bands | 0,591 | 0,658 | 0,693 | 0,686 | 0,691 | 0,867 |
| dresses-sales | 0,539 | 0,553 | 0,527 | 0,529 | 0,565 | 0,575 |
| kr-vs-kp | 0,852 | 0,999 | 0,998 | 0,990 | 0,998 | 0,999 |
| pc4 | 0,736 | **0,917** | 0,904 | 0,829 | 0,854 | 0,944 |
| phoneme | 0,822 | 0,900 | 0,912 | 0,806 | 0,903 | 0,965 |
| qsar-biodeg | 0,722 | 0,890 | 0,890 | 0,877 | 0,882 | 0,937 |
| sick | 0,789 | **0,997** | 0,983 | 0,916 | 0,932 | 0,998 |
| spambase | 0,840 | 0,986 | 0,987 | 0,966 | 0,964 | 0,991 |
| tic-tac-toe | 0,573 | 0,706 | 0,721 | 0,812 | **0,926** | 1,000 |
| **trung bình** | 0,707 | **0,855** | 0,854 | 0,831 | 0,861 | 0,923 |

- **So với kỳ vọng ghi trước.** Giao thức §11 (27/10) ghi kỳ vọng +0,003 AUC so với DLLP tuyến tính và "rủi ro cao không đạt". Trên tập đánh giá, chênh trung bình lớn hơn kỳ vọng (+0,024) nhưng dao động mạnh giữa các dataset. Riêng tic-tac-toe kéo xuống −0,106: dataset này có nhãn sinh từ một quy tắc logic, oracle đạt 1,000, và baseline MLP học quy tắc này tốt hơn hẳn.
- **Phân tích phụ, so với từng baseline LLP** (Holm trên 10 phép so sánh, `per_baseline.csv`):
  - Không phép so sánh nào có ý nghĩa sau Holm.
  - Gibbs thắng ≥ 7/10 dataset trước 8/10 baseline. Ngoại lệ là DLLP tune và LLP-VAT tune: Gibbs thắng 6/10 trước mỗi baseline này.
- **Kết luận được phép viết:** Gibbs ngang các baseline LLP mạnh nhất và hơn hẳn TFM ngây thơ. Không được viết là Gibbs vượt các baseline LLP.

## 3. H4-PLL

### 3.1 Kiểm định và quy tắc 60%
| | q = 0,3 | q = 0,6 |
|---|---|---|
| Gibbs − CC tune (đối thủ chính) | +0,022 [+0,005; +0,044], 10/4, p 0,042, **Holm 0,084** | −0,133 [−0,301; +0,002], 7/7, p 0,63, **Holm 0,63** |
| Gibbs − TFM ngây thơ | +0,099, 13/1, Holm 0,001 | +0,186, 12/2, Holm 0,002 |
| S1 (bỏ MiceProtein, wall-robot, balance-scale, car; 10 dataset): Gibbs − CC tune | +0,026, 8/2, Holm 0,055 | −0,148, 6/4, Holm 0,77 |

**Quy tắc 60% có nhiều cách đọc.** Giao thức Mục 6 ghi tiêu chí: "đạt hoặc vượt baseline mạnh nhất trên ≥ 60% dataset mỗi loại". Câu này không nói rõ hai điều: tính theo từng mức hay gộp hai mức; "baseline mạnh nhất" là một baseline chung hay baseline tốt nhất của từng dataset. Em tính mọi cách đọc (`rule60` trong json):

| Cách đọc | q = 0,3 | q = 0,6 | gộp hai mức | Đạt? |
|---|---|---|---|---|
| so với baseline có trung bình cao nhất ở từng mức (CC tune; GLWS/ILL tune) | 10/14 | 7/14 | 17/28 = 60,7% | gộp: đạt (dư đúng 1 ô); từng mức: không |
| so với baseline tốt nhất của từng dataset và TFM ngây thơ — cách đếm của quy tắc phân vai RQ4 ở thí điểm G4 và G6 lượt 2 | 10/14 | 6/14 | 16/28 = 57,1% | không |
| S1, so với CC tune | 8/10 | 6/10 | 70% | đạt |
| S1, so với baseline tốt nhất của từng dataset | 8/10 | 5/10 | 65% | gộp: đạt; từng mức: không |

**Kết luận H4: không đạt.** Lý do:
1. Giao thức §11 và RESEARCH_PLAN §9 (27/10) ghi rằng PLL thành khẳng định chính "với cùng cấu trúc kiểm định [như H2] (hai đối thủ chính, Holm)". Khẳng định "vượt" cần phép so sánh với đối thủ chính có ý nghĩa sau Holm. Phép so sánh này không có ý nghĩa ở cả hai mức, cả với 14 dataset lẫn với S1.
2. Riêng quy tắc 60% chỉ đạt ở một số cách đọc. Có cách đạt sát ngưỡng, dư đúng 1 ô. Cách đếm mà dự án đã dùng ở các lần trước (từng ô, baseline tốt nhất, kèm điều kiện hơn TFM ngây thơ) thì không đạt.
3. Nguyên tắc ghi trước của S1 là "kết luận khác nhau thì dùng kết luận yếu hơn". Em áp nguyên tắc này cho các cách đọc của quy tắc 60%: chọn cách đọc yếu nhất, không chọn cách đọc có lợi sau khi đã xem số.
4. **25 ô còn thiếu không đổi được kết luận.** Phép thử ở q = 0,6 có p = 0,63, do bốn dataset thua rất đậm (Mục 3.2). Cả năm dataset thiếu seed 1 đều không sụp và đều đang có 2/3 seed.

### 3.2 Phân tích lỗi (thăm dò, làm sau khi xem số liệu): chuỗi sụp ở q = 0,6 khi có ≥ 8 lớp

Bảng dưới là accuracy ở q = 0,6, trung bình theo seed (`table_pll_0.6.csv`; từng seed ở `pll_chain_diag.csv`).

| dataset | số lớp | n train | TFM ngây thơ | Gibbs | hard-EM | CC tune | oracle |
|---|---|---|---|---|---|---|---|
| mfeat-factors | 10 | 1.395 | 0,110 | **0,100** | 0,248 | 0,952 | 0,979 |
| semeion | 10 | 1.115 | 0,111 | **0,293** | 0,102 | 0,910 | 0,948 |
| MiceProtein | 8 | 756 | 0,172 | **0,478** | 0,222 | 0,956 | 1,000 |
| optdigits | 10 | 3.934 | 0,214 | **0,726** (0,961 / 0,294 / 0,923) | 0,383 | 0,964 | 0,989 |
| eucalyptus | 5 | 515 | 0,244 | 0,647 | 0,357 | 0,508 | 0,742 |
| steel-plates-fault | 7 | 1.358 | 0,395 | 0,735 | 0,604 | 0,676 | 0,823 |
| wall-robot-navigation | 4 | 3.819 | 0,891 | 0,983 | 0,980 | 0,873 | 0,993 |

**Phạm vi của lỗi.**
- Có 10 ô (dataset, mức, seed) mà Gibbs kém CC tune từ 0,2 accuracy trở lên. Ngưỡng 0,2 này em chọn sau khi xem số.
- Cả 10 ô đều ở q = 0,6 và đều nằm trên đúng bốn dataset có ≥ 8 lớp: MiceProtein, mfeat-factors, semeion (3/3 seed), optdigits (1/3 seed).
- Không có ô nào như vậy ở q = 0,3, cũng không có ô nào ở các dataset ≤ 7 lớp.
- Ở các dataset còn lại, Gibbs ở q = 0,6 thường thắng đậm, ví dụ eucalyptus +0,14 và wall-robot +0,11 so với CC tune.

**Cơ chế, đọc từ chẩn đoán chuỗi.**
- **Điểm khởi động gần như không mang thông tin.** Tập ứng viên trung bình có 5,2–6,4 nhãn. TFM ngây thơ, cũng là điểm khởi động ấm, ở mức đoán ngẫu nhiên (0,10–0,23).
- **Chuỗi ở lại trạng thái không thông tin.**
  - Sau vòng quét đầu, chỉ 15–25% nhãn gán đúng, tức gần 1/|S|.
  - Khi ngữ cảnh mang nhãn gần như ngẫu nhiên, TFM dự đoán gần như đều. Chuỗi vì thế rút nhãn gần như đều trong tập ứng viên.
  - Trong 5 vòng quét cuối, 49–76% nhãn đổi mỗi vòng ở 9/10 ô sụp (trung vị 0,55). Các chuỗi còn lại ở q = 0,6 có trung vị 0,09.
  - Đây là một trạng thái tự duy trì: nhãn ngẫu nhiên làm TFM dự đoán đều, và dự đoán đều lại sinh nhãn ngẫu nhiên.
- **optdigits có hai lưu vực.**
  - Ba seed xuất phát từ cùng một trạng thái không thông tin (sau vòng quét đầu 16–18% nhãn đúng).
  - Seed 0 và seed 2 leo lên 0,95 và 0,94 nhãn đúng. Seed 1 kẹt ở 0,27.
  - optdigits là dataset lớn nhất trong bốn dataset (3.934 mẫu train). Ba dataset nhỏ hơn (756–1.395 mẫu) không thoát được ở seed nào.
  - Em đọc đây là vấn đề tín hiệu trên mẫu. Nhãn thật luôn nằm trong tập ứng viên, còn mỗi nhãn sai chỉ có mặt với xác suất 0,6. Tín hiệu này yếu, và TFM chỉ khuếch đại được nó khi mỗi lớp có đủ mẫu.
  - CC tối ưu trực tiếp likelihood trên tập ứng viên bằng gradient nên dùng được tín hiệu đó (0,90–0,98).
- **hard-EM cũng sụp** (0,10–0,38). Vậy đây là giới hạn của việc gán nhãn bằng TFM nói chung, không riêng gì cách lấy mẫu Gibbs.

**Lỗi này đã có cảnh báo trước.**
- G6 lượt 6 chạy dữ liệu tổng hợp cỡ semeion. Gibbs đạt 0,11, ngang đoán ngẫu nhiên. Báo cáo lượt đó ghi: "Ghi vào danh sách cần xem khi phân tích lỗi ở G8".
- R̂ ở G6 lượt 5 cho thấy chuỗi trộn chậm ở PLL q = 0,6.
- Tập phát triển của PLL (vehicle 4 lớp, segment 7, cmc 3) **không có dataset nào ≥ 8 lớp**. Vì vậy thí điểm (6/6 ô đạt) không thể lộ ra kiểu lỗi này. Đây là điểm mù của thiết kế tập phát triển, bài cần ghi ở phần giới hạn.
- Liên hệ lý thuyết: Định lý 2 chỉ chặn sai lệch khi sai số của TFM nhỏ và chuỗi trộn được. Ở đây cả hai điều kiện đều không thoả, nên định lý không bảo đảm gì; kết quả này không mâu thuẫn với định lý.

**Không sửa trên tập đánh giá.** Có thể nghĩ tới các cách sửa như khởi động từ một baseline PLL hoặc tôi luyện (annealing). Mọi cách sửa thử trên các dataset này chỉ là kết quả thăm dò, không cứu được H4.

## 4. H3 — Gibbs so với hard-EM (log loss)

| Bối cảnh | số ô | Gibbs tốt hơn | chênh log loss trung bình | Spearman (mức yếu, chênh) |
|---|---|---|---|---|
| LLP túi 8, 16 | 20 | 95% | 0,30 | 0,31 |
| PLL q 0,3, 0,6 | 28 | 82% | 0,81 | 0,64 |
| PU-SAR (Gibbs biết propensity; cùng thông tin với hard-EM) | 20 | 85% | 0,35 | 0,07 |
| PU-SCAR (log loss khớp tiên nghiệm) | 20 | 75% | 0,12 | 0,28 |
| SSL | 20 | 100% | 0,36 | 0,38 |
| Nhãn nhiễu | 20 | 85% | 0,32 | 0,54 |
| **Tất cả** | **128** | **86,7%** | 0,40 | **0,36** (p = 3·10⁻⁵) |

- Bayesian signed-rank với ROPE ±0,005 cho P(Gibbs tốt hơn) = 1,00 và P(ROPE) = 0,00. Như vậy H3 đạt cả ba điều kiện: ≥ 70% ô, P(ROPE) ≤ 0,9, ρ > 0.
- Ở PU-SAR, nếu so hard-EM với `gibbs_warm` (Gibbs tự học propensity) thì Gibbs tốt hơn ở 100% ô.
- Chênh log loss lớn chủ yếu vì hard-EM quá tự tin. Đây đúng là điều H3 dự đoán: lấy mẫu posterior giữ được độ bất định, còn ước lượng điểm thì không.

## 5. Bối cảnh phụ và thăm dò

| Bối cảnh (metric) | Gibbs − đối thủ chính | Gibbs − TFM ngây thơ |
|---|---|---|
| LLP túi 16 (AUC) | +0,040 so với DLLP tuyến tính, 7/3, Holm 0,23 | +0,206, Holm 0,008 |
| LLP túi 32, kiểm tra áp lực | +0,048, 6/4, Holm 0,38 | +0,223, Holm 0,004 |
| PU-SAR c 0,1 / 0,3 (AUC) | +0,034 / +0,032 so với nnPU, Holm 0,97 / 0,11 | +0,007 / +0,017, không có ý nghĩa |
| PU-SCAR c 0,1 / 0,3 (log loss khớp tiên nghiệm) | tốt hơn vPU tune 0,118 / 0,179, Holm 0,11 | kém hơn 0,033 / 0,028 |
| SSL 5% / 20% (AUC) | −0,002 / −0,009 so với self-training tune | **−0,041 / −0,016**, Holm 0,008 (kém có ý nghĩa) |
| Nhãn nhiễu r 0,2 / 0,4 (AUC) | −0,006 / **−0,048** so với CL + TFM (Holm 0,02 ở r 0,4) | −0,013 / −0,054, Holm ≤ 0,02 (kém có ý nghĩa) |

**Thăm dò** (deviation 6, ghi trước):
- **D1 đúng.** Ở PLL, lợi thế so với TFM ngây thơ lớn hơn ở q = 0,6 (+0,186) so với q = 0,3 (+0,099); p một phía 0,045.
- **D2 đúng.** Ở LLP, lợi thế ở túi 16 (+0,206) lớn hơn ở túi 8 (+0,148); p 0,010.
- **D3 đúng.** Ở PU-SCAR, nhiễu và SSL, Gibbs không có lợi ích AUC nào so với TFM ngây thơ (từ −0,054 tới −0,013).
- Cả ba khớp Mệnh đề 5–6: Gibbs giúp khi bộ ngây thơ bỏ đi thông tin ở dạng tập hoặc tổng. Khi cơ chế sinh nhãn yếu là hàm affine của p(y|x), Gibbs không giúp.

## 6. Ý nghĩa và đề xuất (quyết định ở cổng G8)

**Trên dữ liệu sơ bộ, điều kiện chuyển phương án B đã ghi trước đã thoả.**
- 25 ô còn thiếu không đổi được kết luận (Mục 3.1, ý 4).
- L4–L6 chỉ phục vụ phân tích mô tả và H5, không ảnh hưởng H2 hay H4.

**Em đề xuất làm đúng quy tắc đã ghi trước: theo phương án B** (RESEARCH_PLAN §8). Đó là bài phân tích và benchmark "TFM học từ nhãn yếu tốt đến đâu", nhắm NeurIPS Datasets & Benchmarks 2027 hoặc TMLR, giữ nguyên dữ liệu G7. Lý do:
1. Thiết kế G7 khớp gần hết với phương án B: TFM với ba cách xử lý (ngây thơ, hard-EM, Gibbs), 5 loại nhãn yếu, 24 dataset, cùng baseline chuyên biệt đã tune.
2. Kết quả có nội dung rõ cho một bài phân tích:
   - (i) Khi nào lấy mẫu posterior bằng TFM giúp: LLP, PLL, với lợi ích lớn và có ý nghĩa so với TFM ngây thơ; D1, D2 đúng như lý thuyết dự đoán.
   - (ii) Khi nào không giúp: PU-SCAR, nhiễu, SSL (D3, Mệnh đề 5–6).
   - (iii) Lấy mẫu so với ước lượng điểm: H3 rất mạnh.
   - (iv) Một kiểu lỗi được chẩn đoán rõ: nhiều lớp kèm độ mơ hồ cao (Mục 3.2).
   - (v) Không cần huấn luyện mà vẫn ngang baseline chuyên biệt đã tune: ngang ở LLP túi 8; ở PLL q = 0,3 hơn CC tune trên 10/14 dataset, nhưng không có ý nghĩa sau Holm.
3. Định lý 1, Định lý 2 và các Mệnh đề 5–6 vẫn là khung giải thích của bài.

**Phương án em không đề xuất:** giữ bài phương pháp cho NeurIPS bằng cách đọc lại tiêu chí hoặc chọn cách đọc quy tắc 60% có lợi.
- Làm vậy là đổi tiêu chí sau khi đã thấy kết quả, đúng điều mà việc ghi trước nhằm ngăn.
- Muốn có một khẳng định xác nhận mới về phương pháp (ví dụ phương pháp đã sửa lỗi ở Mục 3.2), cần giao thức mới trên một tập đánh giá chưa dùng, tức quay lại G5. Với ngân sách GPU hiện tại thì không kịp NeurIPS 2027.

## 7. Việc tiếp
- **Thầy:**
  - Chạy lại lượt L3, thêm output L3 lần trước làm dữ liệu. Lượt này chỉ chạy 25 ô còn thiếu, khoảng 30–60 phút GPU.
  - Gửi L4–L6 khi xong.
- **Em, khi đủ dữ liệu:**
  - Chạy lại hai script.
  - Thêm R̂, RQ5 và H5 (thứ hạng trên PLL thật), các ablation, đường cong theo cỡ dữ liệu, TabICL.
  - Điền ma trận khẳng định–bằng chứng; lập biên bản cổng G8.
- **Thầy quyết ở cổng G8:** phương án B hay cách khác. Nếu chọn cách khác, cần ghi deviation và gắn nhãn rõ trong bài.
