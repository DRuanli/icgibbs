# G6 lượt 4 và 5 — baseline phụ, mã băm dữ liệu, tổng duyệt bộ chạy G7 (27/10/2026)

Việc T4, T5, T6 của [G6_PLAN](G6_PLAN.md) và phần (a) của việc mở G4-5 ([PROOFS §6](../G4/PROOFS.md)). Mọi số trong file này chỉ từ **tập phát triển**; chưa phương pháp nào chạy trên tập đánh giá (file mã băm có đọc dữ liệu tập đánh giá nhưng không chạy phương pháp nào).

Dữ liệu (thư mục `G4/pilot/`): lượt 4 — `g6r4_sslnz.csv`, `g6r4_pu.csv` (kèm `_preds.jsonl`, `_log.txt`), `g6r4_dataset_hashes.csv`; lượt 5 — `g6r5_tfm.csv`, `g6r5_base.csv`, `g6r5_rhat.csv` (kèm dự đoán và log). Bản thô từng mảnh ở `g6r4_raw/`, `g6r5_raw/` (không đưa lên git).

## 0. Tóm tắt
1. **Lượt 4 đủ, sạch** (504/504 dòng `ok`). Baseline còn thiếu của bối cảnh phụ (GLWS cho PU, VPU, ILL, VIME) **không thay đối thủ chính nào trừ PU-SCAR** (nnPU → VPU tune). Mọi đối thủ chính đã chốt ([docs/primary_comparators.md](../docs/primary_comparators.md)).
2. **T5 xong:** mã băm của 33 dataset ở [docs/dataset_hashes.csv](../docs/dataset_hashes.csv); 33/33 khớp n, d, số lớp, số hàng trùng với G5. Hai phiên Kaggle độc lập cho cùng mã băm bảng và lần chia.
3. **Lượt 5 (tổng duyệt bộ chạy G7):** 216/216 dòng TFM, 27/27 dòng R̂, 323/324 dòng baseline `ok`. Khi dữ liệu không đổi, bộ chạy tái lập thí điểm: baseline tất định khớp tới 4 chữ số, PLL khớp hoàn toàn. Gibbs ở LLP **không tái lập từng bit trên GPU** (lệch tới 0,017 AUC ở một ô).
4. **Phát hiện quan trọng nhất — LLP túi 8 yếu đi dưới xử lý đúng giao thức.** Với loại trùng, tiền xử lý fit trên train và không cắt train (R1–R3), lợi thế của Gibbs so với đối thủ chính (DLLP tuyến tính) giảm từ +0,013 xuống **+0,003 AUC**; quy tắc phạm vi 60% (dạng chặt) cho **3/6 ô** (thí điểm 5/6). PLL không đổi: 6/6 ô, +0,036 accuracy. Cần thầy quyết (Mục 6).
5. **G4-5:** chuỗi có sửa MH (chính xác) không tốt hơn chuỗi xấp xỉ (LLP −0,006 AUC; PLL −0,007 accuracy), tỉ lệ chấp nhận 0,82 (LLP).
6. **R̂:** ở mức nhãn từng mẫu, hai chuỗi chưa trộn đủ trong 30 vòng quét ở LLP (blood, ilpd) và PLL q = 0,6 (vehicle, segment): 8–45% mẫu có R̂ > 1,1, so với ≤ 4,5% khi trộn hoàn hảo.
7. **Lỗi của em về thời gian:** phần TFM chạy 10 giờ thay vì 2,5–3 giờ em ước tính, vì cờ `--mh-check` chạy cả PLL và chuỗi có sửa MH chậm tới 26 lần chuỗi thường (9,2 trong 10 giờ là chuỗi MH). Đã sửa (cờ trần nay chỉ chạy LLP). Ước tính compute cho G7 cũng đã cập nhật (Mục 5).

## 1. Lượt 4 — baseline của bối cảnh phụ
Kaggle CPU, 12:16–16:19 UTC 26/10/2026 (mã băm 1 phút, SSL + nhiễu 27 phút, PU 3 giờ 35 phút). **504/504 dòng `ok`**: SSL và nhiễu 216 (ILL, VIME, mỗi cái bản mặc định và bản tune; 36 ô mỗi phương pháp), PU 288 (GLWS cho PU, VPU; PU-SAR và PU-SCAR). Không có dòng trùng khoá. **Cùng dữ liệu với các lượt trước:** `n_train`, `n_test` khớp ở 144/144 ô. Code chạy trùng với kho ở mọi file liên quan (khác duy nhất: `g7/data.py`, `g7/registry.py` của kho có thêm phần PLL thật viết sau, không dùng ở lượt này).

Chọn lại đối thủ chính theo đúng quy tắc (`g6_select_comparators.py --round2 "../pilot/g6r2_[lp][lu]*.csv" --override "../pilot/g6r[34]_[ps][lsu]*.csv"`):

| Bối cảnh phụ (metric) | Đối thủ chính | Trung bình | Baseline mới của lượt 4 (trung bình) | Gibbs ấm | TFM ngây thơ | Ô đạt (chặt) |
|---|---|---|---|---|---|---|
| Nhãn nhiễu (AUC) | `cl_tfm` (không đổi) | 0,8465 | illnz_tun 0,8163 · illnz_def 0,7745 | 0,8319 | 0,8418 | 4/12 |
| SSL (AUC) | `st_tun` (không đổi) | 0,8185 | illssl_tun 0,8105 · illssl_def 0,7706 · vime_tun 0,7434 · vime_def 0,7413 | 0,8090 | 0,8148 | 2/12 |
| PU-SAR (AUC) | `nnpu` (không đổi) | 0,7260 | glwspu_def 0,7206 · glwspu_tun 0,7011 · vpu_tun 0,6889 · vpu_def 0,6381 | 0,7366 | 0,7317 | 2/12 |
| PU-SCAR (log loss khớp tiên nghiệm, thấp là tốt) | **`vpu_tun`** (trước: `nnpu` 0,7130) | 0,5933 | vpu_def 0,7454 · glwspu_def 0,7524 · glwspu_tun 0,9554 | 0,4343 | 0,4182 | 3/12 |

Ghi chú:
- Cả bốn bối cảnh vẫn không đạt ngưỡng 60%, đúng như vai trò phụ đã định; không đổi phạm vi.
- Bản tune không phải lúc nào cũng tốt hơn: GLWS cho PU tune kém bản mặc định ở cả PU-SAR và PU-SCAR (tune chọn theo rủi ro PU trên validation, không theo AUC hay log loss). Quy tắc xét cả hai bản (deviation 3) xử lý đúng trường hợp này.
- VIME thấp ở bảng ít feature (banknote, d = 4: AUC 0,93 trong khi mọi phương pháp khác đạt 1,00). Cài đặt theo đúng kiến trúc của kho chính thức: encoder một lớp ẩn cỡ d với ReLU, nên với d nhỏ biểu diễn mất thông tin. Tune (p_m, α, β, lớp ẩn) không đổi kết quả. Nhiều khả năng là giới hạn của kiến trúc gốc (chưa đối chiếu bằng cách chạy code TF1 chính thức); không ảnh hưởng đối thủ chính.
- In-Context PU (PU-SCAR) vẫn chưa tái hiện được (không có code truy cập được; docs/baseline_reproduction.md).

## 2. T5 — mã băm dữ liệu
`g6_dataset_hashes.py` (dùng đúng `icgibbs/g7/data.py` của bộ chạy) ghi 33 dataset vào [docs/dataset_hashes.csv](../docs/dataset_hashes.csv): `md5_checksum` của OpenML, n gốc, số hàng thiếu target (0 ở mọi dataset), số hàng trùng đã loại, n, d, số lớp, tỉ lệ lớp nhỏ nhất sau loại trùng, SHA-256 của bảng (X, y) sau loại trùng và mã hoá, SHA-256 của chỉ số train/test của 3 lần chia.
- **Đối chiếu với `g5_dataset_check.csv` (G5):** 33/33 khớp n gốc, d, số lớp, số hàng trùng (vd. spambase 391, first-order-theorem-proving 482, analcatdata_dmft 241). Tỉ lệ lớp nhỏ nhất khác ở 11 dataset vì G5 đo trước loại trùng (lớn nhất: blood-transfusion 23,8% → 27,9%). Đã ghi vào [DATASET_CARDS](../G5/DATASET_CARDS.md) A1.
- **Tái lập giữa hai phiên Kaggle:** mã băm bảng và mã băm lần chia ghi trên từng dòng kết quả của lượt 5 khớp file mã băm của lượt 4 ở cả 27 cặp (dataset, seed) — cùng dữ liệu, cùng lần chia, độc lập với phiên chạy.
- Hàng cùng X khác y vẫn còn sau loại trùng (G5 đã ghi): analcatdata_dmft 329, first-order-theorem-proving 209, cmc 119, blood-transfusion 40. Không xử lý thêm (giao thức chỉ loại trùng hoàn toàn).

## 3. Lượt 5 — tổng duyệt bộ chạy G7 trên tập phát triển
Kaggle GPU T4: phần TFM 12:36–22:42 UTC (10 giờ 6 phút), phần R̂ 53 phút; phần baseline chạy song song trên CPU (3 mảnh, tổng 5,0 giờ CPU). Phạm vi: LLP túi 8 (6 dataset nhị phân) và PLL q ∈ {0,3; 0,6} (vehicle, segment, cmc), 3 seed, đúng các mức chính.

### 3.1 Đầy đủ
TFM 216/216 `ok` (gồm `gibbs_warm_mh` cho cả LLP và PLL), R̂ 27/27, baseline 323/324. Dòng lỗi duy nhất: `proden_tun`, segment, q = 0,3, seed 2 — cấu hình tốt nhất trên 80% train cho xác suất NaN khi fit lại trên toàn bộ train. Đã thêm quy tắc dự phòng (Mục 5; giao thức §11).

### 3.2 So với thí điểm
Bộ chạy khác thí điểm có chủ đích ở R1–R3. Loại trùng làm đổi dữ liệu ở 5/9 dataset: banknote (loại 24 hàng; n_train 943 so với 960), blood-transfusion (215; 373 so với 523), ilpd (13; 399/408), cmc (48; 997/1031), segment (224; 1460/1617). Credit-g không có hàng trùng nhưng có feature phân loại: mã hoá cho TFM đổi (mã theo thứ tự chữ cái, `encode_tfm`) và tiền xử lý của baseline fit trên train (R2). Ba dataset còn lại (diabetes, wdbc, vehicle) có cùng dữ liệu với thí điểm:
- Phần lớn baseline khớp tới 1e-4 (mọi bản mặc định của LLP trên diabetes và wdbc; PRODEN, RC, CC mặc định trên vehicle). Vài bản lệch tới 0,02 (dllp_tun ở diabetes; glwspll_def, cc_tun ở vehicle 0,008), nhiều khả năng do phép tính số thực không tất định giữa hai máy (thứ tự cộng đa luồng) được khuếch đại qua huấn luyện.
- **PLL (vehicle): mọi phương pháp TFM khớp hoàn toàn** (gibbs, SEM, hard-EM, naive, oracle).
- **LLP: Gibbs không tái lập từng bit.** Diabetes seed 0: 0,8387 so với 0,8561; hai seed còn lại lệch ≤ 0,001; wdbc ≤ 0,001. Một sai khác số nhỏ của GPU làm một lần lấy mẫu khác đi, sau đó hai quỹ đạo tách hẳn. Đây là biến thiên Monte Carlo thật của phương pháp, cỡ 0,01–0,02 AUC ở một ô.
- TFM ngây thơ của LLP lệch rất nhiều (tới 0,37 AUC ở wdbc seed 1) vì bộ chạy dùng dãy ngẫu nhiên riêng cho nhãn ngây thơ (thí điểm dùng chung dãy). Trung bình gần như nhau (0,690 so với 0,704): phương pháp ngây thơ của LLP có phương sai lớn theo lần rút nhãn.

### 3.3 Kiểm lại phạm vi trên dữ liệu của bộ chạy
Cùng quy tắc của `g6_select_comparators.py`, áp lên dữ liệu đúng giao thức:

**LLP túi 8 (AUC)**

| Dataset | Gibbs ấm | Gibbs có sửa MH | TFM ngây thơ | dllplin_def (đối thủ chính) | Ứng viên tốt nhất của ô | Gibbs − đối thủ | Gibbs − tốt nhất | Thí điểm: Gibbs − đối thủ |
|---|---|---|---|---|---|---|---|---|
| banknote | 1,0000 | 1,0000 | 0,9601 | 0,9994 | 1,0000 (dllp_def) | +0,001 | 0,000 | +0,001 |
| blood-transfusion | 0,6586 | 0,6588 | 0,5275 | 0,7026 | 0,7183 (dllplin_tun) | **−0,044** | −0,060 | −0,012 |
| credit-g | 0,7338 | 0,7267 | 0,5911 | 0,7323 | 0,7485 (glws_tun) | +0,002 | −0,015 | +0,021 |
| diabetes | 0,8075 | 0,7920 | 0,7043 | 0,8066 | 0,8107 (dllplin_tun) | +0,001 | −0,003 | +0,007 |
| ilpd | 0,6923 | 0,6767 | 0,5665 | 0,6341 | 0,6622 (llpvat_tun) | **+0,058** | +0,030 | +0,061 |
| wdbc | 0,9904 | 0,9899 | 0,7916 | 0,9892 | 0,9895 (llpvat_tun) | +0,001 | +0,001 | +0,001 |
| **Trung bình** | **0,8138** | 0,8074 | 0,6902 | 0,8107 | — | **+0,003** | — | +0,013 |

- Quy tắc chặt (≥ ứng viên tốt nhất của ô và ≥ naive): **3/6 ô (50%)**, dưới ngưỡng 60%. So với riêng đối thủ chính và naive: 5/6. Trên dữ liệu thí điểm cả hai đều là 5/6.
- Theo quy tắc chọn, ứng viên tốt nhất trên dữ liệu bộ chạy là `dllplin_tun` (0,8136), sát `dllplin_def`; đối thủ chính đã chốt không đổi.
- Lợi thế co lại do hai nguồn: (i) blood-transfusion (mất 29% hàng khi loại trùng): Gibbs giảm 0,098, đối thủ giảm 0,066; (ii) credit-g (đổi mã hoá feature phân loại cho TFM): Gibbs giảm 0,019, đối thủ không đổi. Ô diabetes trượt quy tắc chặt chỉ 0,003 — nhỏ hơn biến thiên Monte Carlo ở Mục 3.2.
- Ở 4/6 dataset Gibbs gần như đứng yên tại điểm khởi động ấm (chênh ≤ 0,002); chỉ ilpd (+0,058) và blood (−0,044) đi xa.

**PLL (accuracy)** — vững:

| Mức | Gibbs ấm | Gibbs có sửa MH | glwspll_tun (đối thủ chính) | Gibbs − đối thủ | Ô đạt (chặt) |
|---|---|---|---|---|---|
| q = 0,3 | 0,7926 | 0,7919 | 0,7618 | +0,031 | 3/3 |
| q = 0,6 | 0,7775 | 0,7635 | 0,7369 | +0,041 | 3/3 |
| Trung bình (thí điểm) | 0,7850 (0,7750) | | 0,7493 (0,7451) | +0,036 (+0,030) | 6/6 (6/6) |

RQ3 (Gibbs so với hard-EM, cùng khởi động): LLP 0,8138 so với 0,8032; PLL q = 0,6 0,7775 so với 0,7048 — cùng chiều với thí điểm.

### 3.4 Chẩn đoán R̂ (giao thức Mục 5)
Hai chuỗi Gibbs ấm cùng điểm khởi động, khác seed, 30 vòng quét, 20 lần rút sau burn-in mỗi chuỗi; split-R̂ trên chỉ báo nhãn từng mẫu train; 3 dataset nhỏ nhất mỗi nhóm (quy tắc ghi trước).

**Hiệu chuẩn (mới):** với 20 lần rút độc lập mỗi chuỗi (trộn hoàn hảo), mô phỏng Bernoulli p ∈ [0,02; 0,5] cho tỉ lệ mẫu có R̂ > 1,1 là **0,4–4,5%** và phân vị 90 là 1,01–1,06. Ngưỡng 1,1 quen thuộc vì vậy không phải "luôn đạt" ở cỡ chuỗi này, nhưng giá trị vượt xa mức trên là dấu hiệu thật.

| Bối cảnh | Dataset (n_train) | Mẫu có phương sai | R̂ trung vị | Phân vị 90 | Tỉ lệ R̂ > 1,1 | R̂ toàn cục |
|---|---|---|---|---|---|---|
| LLP túi 8 | blood-transfusion (373) | 339 | 1,005 | 1,09 | 8,3% | 1,25 |
| | ilpd (399) | 348 | 1,020 | 1,16 | **20,6%** | 1,62 |
| | wdbc (398) | 28 | 1,003 | 1,08 | 4,3% | 1,06 |
| PLL q = 0,3 | cmc (997) | 476 | 0,993 | 1,07 | 5,8% | 1,06 |
| | segment (1460) | 57 | 1,005 | 1,09 | 10,8% | 1,07 |
| | vehicle (592) | 120 | 1,003 | 1,07 | 5,0% | 1,05 |
| PLL q = 0,6 | cmc (997) | 790 | 1,001 | 1,08 | 7,8% | 1,37 |
| | segment (1460) | 168 | 1,026 | **1,64** | **34,2%** | 1,66 |
| | vehicle (592) | 255 | 1,017 | 1,26 | **25,2%** | 1,21 |

(trung bình 3 seed; R̂ toàn cục là R̂ của tỉ lệ mẫu đang mang nhãn phổ biến nhất của chúng.)

Đọc: ở mức nhãn, chuỗi trộn đủ ở wdbc và PLL q = 0,3, **chưa đủ** ở LLP (ilpd, blood) và PLL q = 0,6 (vehicle, segment). Hai kiểu khác nhau: ở LLP (blood, ilpd) chuỗi đổi 27–29% nhãn mỗi vòng sau burn-in, tức không kẹt, nhưng một nhóm mẫu trộn chậm; ở PLL q = 0,6 segment chỉ 1,7% nhãn đổi mỗi vòng, tức nhóm mẫu mơ hồ gần như đứng yên ở hai trạng thái khác nhau trong hai chuỗi. Hệ quả:
- [METHOD_SPEC §6c](../G4/METHOD_SPEC.md) đã thu hẹp: không nói "chuỗi hội tụ trong 30 vòng quét", chỉ nói dự đoán ổn định theo khởi tạo; R̂ báo cáo đầy đủ.
- **Không đổi cấu hình** (30 vòng quét, D7 đã đóng băng): ablation số vòng quét (giao thức Mục 7) sẽ đo trên tập đánh giá xem trộn thêm có đổi metric không.
- Bộ chạy nay ghi thêm metric test của từng chuỗi trong phần R̂ (Mục 5), để tách "nhãn trộn chậm" khỏi "dự đoán khác nhau". Lượt 5 chưa có số này.

## 4. G4-5: bag-block xấp xỉ so với có sửa MH
Cùng điểm khởi động, cùng seed, cùng 30 vòng quét; chênh = (có sửa MH) − (xấp xỉ).

| Bối cảnh | Chênh trung bình | Khoảng | MH kém / hơn (ô) | Tỉ lệ chấp nhận MH | Thời gian chuỗi MH so với chuỗi thường |
|---|---|---|---|---|---|
| LLP túi 8 (AUC) | −0,0064 | −0,045 … +0,018 | 10 / 5 (18 ô; Wilcoxon thăm dò p = 0,11) | 0,82 (0,59–0,99) | 1,2–13 lần (credit-g 1.980 s so với 148 s; banknote, wdbc gần như không chậm) |
| PLL q = 0,3 (accuracy) | −0,0007 | −0,012 … +0,009 | | 0,68 | 9–26 lần (cmc 1.370 s so với 52 s) |
| PLL q = 0,6 (accuracy) | −0,0139 | −0,039 … +0,016 | (PLL gộp: 10 / 5; p = 0,03) | 0,58 | |

- **Sơ đồ xấp xỉ không làm hại dự đoán** trên tập phát triển; nếu có thì bản chính xác hơi kém ở PLL q = 0,6.
- Không tách được hai hiệu ứng: ở cùng số vòng quét, chuỗi MH vừa nhắm đúng π vừa trộn chậm hơn (bị từ chối 18–42% đề xuất). Chặn lý thuyết cho bag-block xấp xỉ vẫn còn mở; PROOFS G4-5 đã cập nhật phần (a).

## 5. Sửa sau lượt 5
1. **Quy tắc dự phòng khi fit lại cấu hình tune hỏng** (`icgibbs/tuning.py: refit_best`, áp cho 9 chỗ fit lại của mọi baseline có tune): nếu mô hình fit lại cho xác suất không hữu hạn trên chính X train, dùng cấu hình xếp hạng kế tiếp, tối đa 5; ghi thứ hạng. Chỉ nhìn X train. Khi không có lỗi, kết quả trùng hoàn toàn với trước. 4 test mới (`tests/test_tuning_refit.py`). Ghi ở giao thức §11 (deviation 10).
2. **`--mh-check` nhận danh sách bối cảnh; cờ trần = chỉ LLP** (trước đây chạy mọi bối cảnh — nguyên nhân 10 giờ). PLL đã đo một lần ở lượt 5, đủ cho G4-5.
3. **Phần R̂ ghi thêm metric test của từng chuỗi** (`metric_chain_a`, `metric_chain_b`, `metric_absdiff`); chuỗi a cùng seed với `gibbs_warm` của phần TFM, nên so hai số này còn đo luôn độ không tất định của GPU.
4. Sửa docstring `rhat.py`: R̂ toàn cục dùng tỉ lệ mẫu mang nhãn phổ biến nhất (đúng như code đã chạy), kèm số hiệu chuẩn.

**Ước tính compute cho G7 (cập nhật giao thức Mục 9, chỉ là ước tính):** từ thời gian đo được, ngoại suy theo n_train và d: phần TFM chính (Gibbs + hard-EM + naive + oracle) khoảng **42 giờ T4** (giao thức ước 18 giờ); toàn bộ phần GPU khoảng 60–65 giờ (2–2,5 tuần quota). Baseline (CPU) khoảng **140 giờ CPU** (giao thức ước 25 giờ): phải chia mảnh và chạy song song nhiều phiên CPU. Ngoại suy vượt xa n_train lớn nhất đã đo (1.460) nên bất định lớn; lượt 6 đo lại ở cỡ lớn bằng dữ liệu tổng hợp (Mục 6).

## 6. Hệ quả và việc cần thầy quyết
**Quyết định cần thầy — phạm vi của LLP túi 8.** Quy tắc 60% đã dùng ở lượt 2 (trên dữ liệu thí điểm) để giữ LLP túi 8 làm khẳng định chính. Áp cùng quy tắc lên dữ liệu đúng giao thức thì chỉ còn 3/6. Hai lựa chọn:
- **(A) Giữ LLP túi 8 là khẳng định chính, không đổi giao thức** — em đề xuất. Lý do: quy tắc đã được áp đúng điểm đã định (sau khi mọi baseline được tune, G6 lượt 2) và thầy đã chốt (deviation 7). Áp lại quy tắc mỗi khi pipeline đổi là thêm một lần chọn theo kết quả. Ba ô trượt quy tắc chặt: blood (−0,060 so với ứng viên tốt nhất; dataset mất 29% hàng khi loại trùng), credit-g (−0,015 so với GLWS tune; đổi mã hoá feature phân loại), diabetes (−0,003, nhỏ hơn nhiễu Monte Carlo); chỉ cần một ô đổi chiều là đạt 4/6. H2-LLP là kiểm định xác nhận hai phía trên tập đánh giá: nếu Gibbs chỉ ngang DLLP tuyến tính thì H2 không đạt và bài báo cáo đúng như vậy. Kèm theo: ghi trước (giao thức §11, dạng ghi chú) rằng trên tập phát triển đã xử lý đúng, hiệu ứng kỳ vọng so với DLLP tuyến tính chỉ khoảng +0,003 AUC, nên H2-LLP có rủi ro cao không đạt; phần có cơ sở vững của LLP trên tập phát triển là so với TFM ngây thơ (+0,12 AUC).
- **(B) Áp lại quy tắc, chuyển LLP túi 8 thành bối cảnh phụ.** Khẳng định chính còn PLL (H4), RQ3 và lý thuyết. Nhất quán với cách đã xử lý túi 16, PU-SAR, SSL, nhãn nhiễu, nhưng bài mất trụ cột LLP và đây là lần thu hẹp thứ năm dựa trên cùng 6 dataset phát triển.

Việc còn lại của G6 (không phụ thuộc quyết định trên):
- **Lượt 6 (GPU khoảng 2 giờ, CPU song song; đã sẵn sàng: `g6_code_v6.zip`, `run_g6_round6.sh`, hướng dẫn ở G4/README):** tập phát triển không có dataset nào lớn hơn segment, và chạm tập đánh giá trước G7 là không được, nên lượt này chạy bộ chạy G7 trên **dữ liệu tổng hợp cùng cỡ** với 4 dataset lớn nhất của tập đánh giá (`--source synthetic --synthetic_shape`: 5395 × 5, 4210 × 57, 6430 × 36 với 6 lớp, 1593 × 256 với 10 lớp), mỗi bối cảnh 1 mức, 1 seed. Mục đích: thay ngoại suy ở Mục 5 bằng số đo, bắt lỗi hết bộ nhớ GPU, ghi phiên bản thư viện (`pip freeze`) để chốt cho G7.
- T3: tái hiện số công bố của baseline (GLWS, LLP-VAT, PRODEN/RC/CC trên MNIST-family, EasyLLP).
- Bộ chạy G7 bước 4e: phần LLP-Bench và phần ablation (chốt các điểm ở G7_RUNNER_DESIGN §4).
- G4-1, G4-6 (trích dẫn), T8 (chạy từ clone sạch).
