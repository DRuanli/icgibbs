# G6 lượt 6 — đo thời gian và bộ nhớ ở cỡ của tập đánh giá; phần ablation của bộ chạy G7 (27/10/2026)

Lượt 6 chạy bộ chạy G7 trên **dữ liệu tổng hợp** cùng cỡ với 4 dataset lớn nhất của tập đánh giá. Không tải, không chạm tập đánh giá. Metric trên dữ liệu tổng hợp không dùng cho kết luận nào. Dữ liệu ở `G4/pilot/`: `g6r6_tfm.csv`, `g6r6_base.csv` (kèm dự đoán, log), `g6r6_pip_freeze.txt`, `g6r6_gpu.txt`; bản thô ở `g6r6_raw/` (không đưa lên git).

## 1. Đầy đủ và môi trường
- Kaggle, 03:13–05:03 UTC 27/10/2026 (1 giờ 50 phút). Phiên có **2 GPU Tesla T4** (15 GB mỗi GPU); bộ chạy dùng 1 GPU.
- **92/92 dòng `ok`**: TFM 42 (4 cỡ dữ liệu, mọi bối cảnh nhị phân ở cỡ spambase), baseline 50. Code chạy trùng với kho ở mọi file liên quan.
- **Bộ nhớ:** không ô nào hết bộ nhớ. Đỉnh 1,7 GB (cỡ semeion, d = 257), 1,3 GB ở các cỡ khác. Còn dư rất nhiều.
- **Phiên bản thư viện** (`pip freeze`): tabpfn 9.2.0, torch 2.11.0+cu128, numpy 2.1.3, pandas 2.3.3, scikit-learn 1.6.1, scipy 1.16.3, cleanlab 2.9.0, sarpu ở commit 6e4fc3d. Ghi vào `G4/code/requirements-g7.txt` (việc T8).

## 2. Thời gian đo được
**Phần TFM.** Một ô gồm Gibbs (SEM lấy từ cùng chuỗi), hard-EM, TFM ngây thơ, oracle:

| Cỡ mô phỏng | Bối cảnh | n_train × d | Giây mỗi ô (Gibbs / hard-EM) | Ngoại suy cũ |
|---|---|---|---|---|
| spambase | LLP túi 8 | 2947 × 58 | 1.066 (710 / 320) | 890 |
| spambase | PU-SAR (kèm Gibbs biết propensity 603 s) | | 1.541 | |
| spambase | PU-SCAR / SSL / nhiễu | | 812 / 766 / 706 | |
| phoneme | LLP túi 8 | 3776 × 6 | 184 (109 / 68) | 344 |
| satimage | PLL q = 0,3, 6 lớp | 4501 × 37 | 576 (322 / 230) | 1.006 |
| semeion | PLL q = 0,3, 10 lớp | 1115 × 257 | 1.011 (757 / 235) | 849 |

Ngoại suy cũ đánh giá thấp ảnh hưởng của số feature d và đánh giá cao ảnh hưởng của n. Ước lượng lại trên 17 điểm đo (lượt 5 và 6): log giây = −1,58 + 0,70 log n + 0,66 log d (sai số dư khoảng ±40%).

**Baseline (CPU, một luồng), cỡ spambase:** LLP 2.919 s mỗi ô, PU-SAR 1.964, PU-SCAR 1.512, nhiễu 501, SSL 285. Cỡ satimage, PLL: 952 s. Chậm nhất: `llpvat_tun` 928 s, `glws_tun` 770 s, `glwspu_tun` 657–695 s, `sar_em_tuned` 411 s. Thời gian tăng chậm hơn tuyến tính theo n (mũ 0,44–1,07 tuỳ bối cảnh), nên ngoại suy tuyến tính trước đây (140 giờ) quá cao.

## 3. Ước tính compute cho G7 (thay ước tính ở G6_ROUND45_REPORT Mục 5; giao thức Mục 9, ghi chú)
| Khối việc | GPU T4 | CPU một luồng |
|---|---|---|
| Phần chính: 330 ô nhị phân + 84 ô PLL | khoảng 35 giờ (23 + 12) | — |
| Baseline mọi bối cảnh | — | khoảng 100 giờ (LLP 41, PU-SAR 20, PU-SCAR 19, PLL 13, nhiễu 5, SSL 3) |
| Ablation Mục 7 (định rõ ở deviation 11) | khoảng 32 giờ | ít (đối thủ chính ở các cỡ dữ liệu) |
| RQ5 (PLL thật, LLP-Bench) và R̂ | khoảng 6 giờ | ít |
| **Tổng** | **khoảng 73 giờ** | **khoảng 100 giờ** |

- Phiên Kaggle có 2 GPU: chạy hai luồng song song (mỗi luồng một GPU qua `CUDA_VISIBLE_DEVICES`) giảm thời gian thực khoảng một nửa. Lượt 7 thử cách này. Quota GPU tính theo giờ phiên hay theo GPU: thầy xem giúp trên trang quota của Kaggle; với cách thứ nhất, 73 giờ GPU chỉ tốn khoảng 37 giờ quota.
- Baseline chạy trên CPU (4 nhân mỗi phiên): 4 tiến trình song song cho khoảng 25 giờ thời gian thực, tức khoảng 3 phiên CPU, không tốn quota GPU.

## 4. Quan sát trên dữ liệu tổng hợp (không dùng cho kết luận)
- **Cỡ semeion (10 lớp, 256 feature, 1.115 mẫu train):** bài tổng hợp rất khó (oracle chỉ 0,38). Gibbs 0,11, ngang đoán ngẫu nhiên, thấp hơn cả TFM ngây thơ (0,13). Khi chính TFM yếu, chuỗi có thể trôi về cách gán vô nghĩa — cùng chiều với Định lý 2 (sai số của TFM bị khuếch đại). Ghi vào danh sách cần xem khi phân tích lỗi ở G8.
- **PU-SAR:** Gibbs 0,83 AUC so với ngây thơ 0,75; biết propensity thì 0,90 — cùng chiều tập phát triển.

## 5. Việc đã làm sau lượt 6
1. **`requirements-g7.txt`**: chốt phiên bản cho G7.
2. **Phần `ablation` của bộ chạy** (giao thức Mục 7; các điểm chưa rõ đã định ở [G7_RUNNER_DESIGN §4](G7_RUNNER_DESIGN.md), giao thức §11 deviation 11): khởi tạo lạnh, số khối K, số vòng quét, chuyển nạp, khối có sửa MH, prior η; cỡ dữ liệu qua `--n_train_max`. Mỗi biến thể là một chuỗi cùng seed với `gibbs_warm`, khác đúng một thành phần. Em đã kiểm: biến thể không đổi gì cho đúng từng chữ số kết quả của `gibbs_warm`. 5 test mới.
3. **Sửa backend TabICL của `run_g7.py` (lỗi em để sót):** bản cũ tạo `TabICLClassifier` trơn — nạp lại checkpoint từ đĩa ở mỗi lần fit (một chuỗi Gibbs gọi fit hàng trăm lần), thiếu `support_many_classes=True` và dùng số bộ ước lượng mặc định khi dự đoán cuối. Cấu hình đã chốt ở F6 là giữ checkpoint trong bộ nhớ, `support_many_classes=True`, 1 bộ ước lượng trong chuỗi và 2 khi dự đoán cuối, như `kaggle_g4_manyclass_pilot.py`. Backend này chưa từng chạy trong bộ chạy G7 nên chưa ảnh hưởng kết quả nào; lượt 7 chạy thử thật.
4. **LLP-Bench:** bài không công bố code hay dữ liệu đã chia túi; phải dựng lại từ dữ liệu Criteo. Đề xuất cách làm ở G7_RUNNER_DESIGN §4b; cần thầy kiểm quyền truy cập dữ liệu Criteo trên Kaggle.
5. **Lượt 7 (sẵn sàng: `g6_code_v7.zip`, `run_g6_round7.sh`):** tổng duyệt ablation, cỡ dữ liệu và TabICL trên tập phát triển, dùng cả 2 GPU song song; khoảng 2,5 giờ.
