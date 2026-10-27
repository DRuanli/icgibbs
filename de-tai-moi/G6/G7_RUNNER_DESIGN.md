# Thiết kế bộ chạy G7 (G6, việc T6) — bản nháp 26/10/2026

Mục tiêu: một bộ chạy duy nhất cho mọi thực nghiệm trên **tập đánh giá** ở G7, đúng giao thức đã đóng băng ([G5](../G5/EXPERIMENT_PROTOCOL.md) và các deviation ở Mục 11). Các script thí điểm (`kaggle_g4_pilot.py`, `kaggle_g4_rq4_pilot.py`) chỉ dùng cho tập phát triển và **không** dùng lại nguyên trạng, vì chúng chưa đạt một số yêu cầu dưới đây.

## 1. Những điểm script thí điểm chưa đạt (phải sửa trong bộ chạy G7)

| # | Yêu cầu của giao thức | Script thí điểm | Bộ chạy G7 |
|---|---|---|---|
| R1 | Loại hàng trùng **trước khi chia**, báo cáo số hàng bị loại (Mục 2, L1.4) | không loại trùng | loại hàng trùng hoàn toàn (X, y) như cách đếm trong DATASET_CARDS (ví dụ spambase 4.601 → 4.210); ghi số hàng bị loại vào mỗi dòng kết quả |
| R2 | Điền thiếu, chuẩn hoá, one-hot **fit trên train của từng lần chia** (Mục 2, L1.2) | trung vị và one-hot tính trên toàn bộ dataset | `Preprocessor` fit trên train; one-hot bỏ mức chưa gặp; TFM nhận giá trị thiếu dạng NaN (tiền xử lý nội bộ của TFM) |
| R3 | Train tới 7.000 hàng; Gibbs dùng ngữ cảnh ≤ 2.000 (lấy mẫu con mỗi lần gọi, D4) (Mục 3) | `max_train = 2000` cắt bớt train cho mọi phương pháp | không cắt train; chỉ giới hạn ngữ cảnh của TFM bên trong sampler |
| R4 | Metric phụ: PR-AUC (PU), ECE; tính toán: GPU, bộ nhớ đỉnh (Mục 5) | thiếu PR-AUC, ECE, bộ nhớ | đủ mọi metric; `torch.cuda.max_memory_allocated` khi có GPU |
| R5 | Chẩn đoán chuỗi: tỉ lệ nhãn đổi mỗi vòng; $\hat R$ từ 2 chuỗi trên 3 dataset (Mục 5) | chỉ tỉ lệ đổi ở vòng cuối | lưu cả dãy tỉ lệ đổi; phần chạy riêng 2 chuỗi cho $\hat R$ |
| R6 | Nhãn sạch của train chỉ dùng để sinh nhãn yếu và chẩn đoán (Mục 8) | đúng, nhưng chỉ dựa vào quy ước | hàm của phương pháp chỉ nhận `WeakData` (không có trường nhãn sạch); oracle chạy ở nhánh riêng; test kiểm điều này |
| R7 | Có thể kiểm lại từng ô | không có mã băm | mỗi dòng ghi mã băm SHA-256 của dữ liệu (đối chiếu DATASET_CARDS), của chỉ số train/test, của nhãn yếu, commit git và cấu hình |

## 2. Kiến trúc

```
code/
  run_g7.py                      CLI: --part, --shard i/n, --out, --resume; một dòng CSV cho mỗi (dataset, bối cảnh, mức, seed, phương pháp)
  icgibbs/g7/
    registry.py                  danh sách dataset của tập đánh giá (ID OpenML, phiên bản, SHA-256 theo DATASET_CARDS), bối cảnh, mức
    data.py                      tải, kiểm mã băm, loại trùng (R1); Preprocessor fit trên train (R2)
    splits.py                    3 lần chia phân tầng 70/30, seed 0–2; PLL thật: 10 lần chia 90/10 không phân tầng (Mục 1.4)
    weakgen.py                   sinh nhãn yếu theo Mục 3.1 — cùng thuật toán và cùng cách dùng seed như thí điểm (test so khớp)
    methods.py                   bảng phương pháp: Gibbs ấm (+ SEM từ cùng chuỗi), hard-EM ấm, TFM ngây thơ, oracle, mọi baseline (_def, _tun)
    metrics.py                   metric chính và phụ (R4); log loss khớp tiên nghiệm
    io.py                        ghi CSV chạy tiếp được, dự đoán (jsonl), mã băm (R7)
  tests/test_g7_*.py             rò rỉ (R6, L1.1), tiền xử lý (R2), loại trùng (R1), khớp nhãn yếu với thí điểm, chạy tiếp
```

**Các phần chạy** (tách để phần không cần GPU chạy trên notebook CPU):

| Phần | Nội dung | Phần cứng |
|---|---|---|
| `main-tfm` | Gibbs ấm, SEM, hard-EM, TFM ngây thơ, oracle — LLP (8, 16, 32), PU-SAR, PU-SCAR, SSL, nhãn nhiễu; 10 dataset × 3 seed | GPU T4 |
| `pll-tfm` | như trên cho PLL tổng hợp, 14 dataset | GPU T4 |
| `baselines` | mọi baseline peer-review, bản mặc định và bản tune, mọi bối cảnh | CPU |
| `rq5` | PLL thật (TabICL v2, 10 lần chia 90/10) và LLP-Bench | GPU |
| `ablation` | Mục 7 của giao thức | GPU |
| `rhat` | 2 chuỗi trên 3 dataset (R5) | GPU |

## 3. Tiến độ và việc tiếp theo
| # | Việc | Trạng thái |
|---|---|---|
| 1 | `registry.py`, `data.py`, `splits.py` và test (R1, R2, R7) — `tests/test_g7_data.py`, 8 test | Xong 26/10/2026 |
| 2 | `weakgen.py` — cùng nhãn yếu như thí điểm ở mọi bối cảnh (`tests/test_g7_weakgen.py`, 10 test; kể cả kiểm R6: `WeakData` không có trường nhãn sạch). Khác thí điểm có chủ đích: PU-SAR lấy z từ cột đầu với giá trị thiếu = 0 sau `nan_to_num` (thí điểm điền trung vị toàn bộ trước) | Xong 26/10/2026 |
| 3 | T5: `g6_dataset_hashes.py` — mã băm (X_tfm, y) và 3 lần chia của mọi dataset, dùng đúng `data.py` | Code xong; chạy ở G6 lượt 4 (cần Internet tới OpenML) |
| 4a | `metrics.py` (metric chính và phụ: PR-AUC, ECE, khớp tiên nghiệm; R4), `io.py` (CSV chạy tiếp được, dự đoán, mã băm nhãn yếu và cấu hình, phiên bản code; R7) — 5 test | Xong 26/10/2026 |
| 4b | `methods.py` (đúng cấu hình đã chốt và tên phương pháp như thí điểm; mỗi phương pháp có khung bắt lỗi riêng), `run_g7.py` (hai phần `tfm`/`baselines`, chia mảnh, chạy tiếp, mặc định tập phát triển; tập đánh giá cần `--split eval --eval-ok`) — 3 test (chặn tập đánh giá; chạy đầu-cuối với TFM giả; chạy tiếp không thêm dòng) và chạy thử đầu-cuối mọi bối cảnh, mọi loại phương pháp trên dữ liệu tổng hợp | Xong 26/10/2026 |
| 4c | Phần `rhat` (R5): 2 chuỗi Gibbs ấm cùng điểm khởi động, khác seed; split-R̂ (Vehtari et al. 2021) trên chỉ báo nhãn từng mẫu sau burn-in — trung vị, phân vị 90, tỉ lệ R̂ > 1,1, R̂ toàn cục (`icgibbs/g7/rhat.py`, 3 test). **Quy tắc ghi trước:** 3 dataset nhỏ nhất theo n của mỗi nhóm — tập đánh giá: dresses-sales, cylinder-bands, credit-approval; balance-scale, eucalyptus, analcatdata_dmft | Xong 26/10/2026 |
| 4d | Bối cảnh `pll_real` (RQ5, PLL thật): đọc .npz của việc F7, đối chiếu SHA-256 tệp gốc với DATASET_CARDS, 10 lần chia 90/10 không phân tầng, số lớp Q lấy từ tập ứng viên, nhãn ngây thơ như thí điểm many-class; chỉ chạy với `--split eval --eval-ok` (1 test với dữ liệu giả). Cờ `--mh-check` cho PROOFS G4-5 | Xong 26/10/2026 |
| 4e | Phần `ablation` (giao thức Mục 7; định rõ ở Mục 4, deviation 11) và ablation cỡ dữ liệu (`--n_train_max`); 5 test (`tests/test_g7_ablation.py`). LLP-Bench: đề xuất ở Mục 4b, chờ kiểm quyền truy cập dữ liệu Criteo | Ablation xong 27/10/2026; LLP-Bench chưa |
| 5 | Chạy tổng duyệt toàn bộ trên **tập phát triển** (phần `tfm` trên GPU, phần `baselines` trên CPU), so với kết quả thí điểm để thấy sai khác do R1–R3 | **Xong 27/10/2026** (G6 lượt 5, [G6_ROUND45_REPORT](G6_ROUND45_REPORT.md) §3): 216/216 TFM, 27/27 R̂, 323/324 baseline; mã băm khớp T5 ở 27/27 cặp; tái lập thí điểm khi dữ liệu không đổi (PLL từng bit; LLP Gibbs không tái lập từng bit trên GPU) |
| 5a | Sửa sau lượt 5: `refit_best` (fit lại cấu hình tune kế tiếp khi xác suất không hữu hạn; giao thức §11 deviation 10); `--mh-check` nhận danh sách bối cảnh, cờ trần = LLP; phần `rhat` ghi metric test của từng chuỗi (`metric_chain_a/b`, chuỗi a cùng seed với `gibbs_warm`); `--synthetic_shape` | Xong 27/10/2026 (4 test mới) |
| 5b | Đo thời gian và bộ nhớ ở cỡ tập đánh giá bằng dữ liệu tổng hợp (`run_g6_round6.sh`) | **Xong 27/10/2026** ([G6_ROUND6_REPORT](G6_ROUND6_REPORT.md)): 92/92 dòng ok, không hết bộ nhớ (đỉnh 1,7 GB) |
| 5c | Sửa backend TabICL của `run_g7.py` cho đúng cấu hình F6 (giữ checkpoint trong bộ nhớ, `support_many_classes=True`, 2 bộ ước lượng khi dự đoán cuối) — bản trước nạp lại checkpoint mỗi lần fit và thiếu tham số; chưa từng chạy nên chưa ảnh hưởng kết quả nào | Xong 27/10/2026 |
| 5d | Tổng duyệt ablation, cỡ dữ liệu, TabICL trên tập phát triển, dùng 2 GPU song song (`run_g6_round7.sh`) | Chờ G6 lượt 7 |
| 6 | Người kiểm duyệt chạy từ clone sạch (T8) | Chưa |

## 3b. Bài học từ lượt 5 cho G7
- **Tái lập:** cùng dữ liệu và cùng seed, Gibbs ở LLP vẫn có thể lệch tới 0,017 AUC giữa hai lần chạy GPU (một sai khác số nhỏ đổi một lần lấy mẫu, sau đó hai quỹ đạo tách). Bài phải nói kết quả tái lập được về phân phối (qua seed), không từng bit; lưu dự đoán (`_preds.jsonl`) của mọi ô để kiểm lại phân tích mà không cần chạy lại.
- **Thời gian:** chuỗi có sửa MH chậm 1,2–26 lần chuỗi thường; chỉ bật ở nơi cần (`--mh-check llp`, ablation trên 3 dataset nhỏ). Ước tính compute G7 ở giao thức Mục 9 (ghi chú 27/10/2026), sẽ thay bằng số đo của lượt 6.
- **Baseline:** 1/324 ô lỗi vì fit lại phân kỳ → quy tắc dự phòng. Ghi phiên bản thư viện (`pip freeze`) ở mọi lượt G7.

## 4. Ablation (giao thức Mục 7): các điểm đã định rõ (27/10/2026; giao thức §11, deviation 11)
Thầy giao em chọn; ghi trước khi chạy bất kỳ ablation nào trên tập đánh giá. Nguyên tắc: dùng lại đúng quy tắc chọn dataset đã ghi cho R̂ (3 dataset nhỏ nhất theo n mỗi nhóm) để không có thêm lựa chọn tự do; mỗi biến thể là **một chuỗi Gibbs cùng seed với `gibbs_warm`** của phần chính, chỉ khác đúng một thành phần, nên so ghép cặp được (đã kiểm: biến thể không đổi gì cho đúng từng chữ số kết quả của `gibbs_warm`).

| Dòng của Mục 7 | Bối cảnh, mức | Dataset (tập đánh giá) | Biến thể (tên phương pháp) | Ghi chú |
|---|---|---|---|---|
| RQ3 (Gibbs / SEM / hard-EM) | mọi ô của phần chính | mọi dataset | `gibbs_warm`, `sem_warm`, `hard_em_warm` | đã có trong phần `tfm`, không chạy thêm |
| Khởi tạo ấm / lạnh | LLP túi 8; PLL q = 0,3 và 0,6 | mọi dataset | `gibbs_cold` | lạnh = y_i ~ ℓ_i chuẩn hoá (LLP: hoán vị ngẫu nhiên khớp số đếm của túi). Cột "Dải" của Mục 7 để trống như dòng RQ3, nên hiểu là mọi ô của hai bối cảnh chính |
| Số khối K ∈ {5, 10, 20} | LLP túi 8; PLL q = 0,3 | 3 nhỏ nhất mỗi nhóm: dresses-sales, cylinder-bands, credit-approval; analcatdata_dmft, balance-scale, eucalyptus | `gibbs_warm_K5`, `gibbs_warm_K20` | K = 10 lấy từ phần chính |
| Số vòng quét ∈ {15, 30, 60} | như trên | như trên | `gibbs_warm_S15`, `gibbs_warm_S60` | burn-in = S/3 (5, 10, 20); dự đoán dùng 10 mẫu cách đều sau burn-in |
| Quy nạp / chuyển nạp (G4-2) | LLP túi 8 | mọi dataset | `gibbs_warm_trans` | X_test vào chuỗi như điểm tự do (ℓ ≡ 1), khởi động bằng cùng DLLP tuyến tính; dự đoán = trung bình Rao–Blackwell của điều kiện trong chuỗi (TFM trong chuỗi, `n_estimators = 1`). Khác bộ dự đoán quy nạp cả ở số `n_estimators` — ghi rõ khi báo cáo |
| Khối có sửa MH (bật / tắt) | LLP túi 8; PLL q = 0,3 | 3 nhỏ nhất mỗi nhóm | `gibbs_warm_mh` | chậm 1,2–26 lần chuỗi thường (G6 lượt 5) |
| Propensity: học / đã biết; prior η | PU-SAR, cả hai mức | mọi dataset | `gibbs_warm_prior1`, `gibbs_warm_prior4` (mặc định N(0, 2²) ở phần chính); "đã biết" = `gibbs_known_warm` của phần chính | bối cảnh phụ |
| TFM: TabPFN v2 / TabICL v2 | LLP túi 8; PLL q = 0,3 và 0,6 | mọi dataset | mọi phương pháp TFM của phần chính với `--backend tabicl_v2` | TabICL đúng cấu hình F6: `support_many_classes=True`, 1 bộ ước lượng trong chuỗi, 2 khi dự đoán cuối |
| Cỡ dữ liệu n_train ∈ {500, 1.000, 2.000, 4.000} | LLP túi 8 (3 nhị phân lớn nhất: phoneme, spambase, sick); PLL q = 0,3 (3 đa lớp lớn nhất: satimage, first-order-theorem-proving, optdigits) | như cột trước | mọi phương pháp TFM + đối thủ chính (`dllplin_def`, `glwspll_tun`) | `--n_train_max`: mẫu con lồng nhau, phân tầng, cố định theo seed; test không đổi. Cỡ ≥ toàn bộ train thì dùng kết quả phần chính. Thay đề xuất cũ (chỉ PLL): LLP là khẳng định chính nên cũng cần đường cong theo cỡ dữ liệu |

Dòng lệnh: `run_g7.py --part ablation --ablation {init, blocks, sweeps, trans, mh, prior}` (quy tắc dataset và mức tự áp theo `--split`); TabICL: `--part tfm --backend tabicl_v2 --settings llp,pll --levels "llp=8;pll=0.3,0.6"`; cỡ dữ liệu: `--part tfm|baselines --n_train_max M`.

Ước tính GPU trên tập đánh giá (mô hình thời gian sau lượt 6, sai số khoảng ±40%): khởi tạo lạnh 9 giờ, K và số vòng quét 1–2 giờ (dataset nhỏ), chuyển nạp 2 giờ, sửa MH 3 giờ, prior η 6 giờ, TabICL 6 giờ (giả định TabICL nhanh hơn TabPFN khoảng 2,5 lần như ở thí điểm many-class; lượt 7 đo lại), cỡ dữ liệu 5 giờ: **tổng khoảng 32 giờ**.

## 4b. LLP-Bench (RQ5, kiểm tra áp lực): đề xuất, chưa chốt
- **Dữ liệu:** bài LLP-Bench (Brahmbhatt et al., NeurIPS 2023 D&B) không công bố code hay dữ liệu đã chia túi; các túi được dựng lại từ dữ liệu Criteo (khoảng 45,8 triệu hàng; 13 feature số, 26 feature phân loại) theo mô tả trong bài. Nguồn: dữ liệu cuộc thi Criteo Display Advertising trên Kaggle, hoặc bản `reczoo/Criteo_x4` trên Hugging Face (cùng 45.840.617 hàng; tiền xử lý không hoàn toàn như AutoInt). **Việc cần thầy:** kiểm trên Kaggle xem thêm được dữ liệu Criteo vào notebook không, và điều khoản sử dụng của nguồn đó.
- **Cỡ:** TFM chỉ nhận ngữ cảnh ≤ 2.000, còn mỗi bộ của LLP-Bench có hàng triệu mẫu. Đề xuất: lấy mẫu con 5.000 mẫu train và 2.000 mẫu test cho mỗi cấu hình. Dùng 4 bộ túi ngẫu nhiên (cỡ 64, 128, 256, 512) và một số bộ túi theo feature (chọn bằng quy tắc ghi trước, ví dụ 4 bộ có cỡ túi trung bình gần trung vị). 5 lần chia như bài. So Gibbs ấm với DLLP tuyến tính và TFM ngây thơ.
- **Dự đoán ghi trước** (giao thức §1.4): phương pháp yếu ở túi cỡ ≥ 64.
