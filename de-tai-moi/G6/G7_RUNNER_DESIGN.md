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
| 4d | Các phần còn lại: `rq5` (PLL thật, TabICL v2, 10 lần chia 90/10; LLP-Bench), `ablation` (giao thức Mục 7) | Chưa |
| 5 | Chạy tổng duyệt toàn bộ trên **tập phát triển** (phần `tfm` trên GPU, phần `baselines` trên CPU), so với kết quả thí điểm để thấy sai khác do R1–R3 | Chưa — G6 lượt 5 |
| 6 | Người kiểm duyệt chạy từ clone sạch (T8) | Chưa |
