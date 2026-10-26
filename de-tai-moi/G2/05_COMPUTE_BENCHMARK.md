# G2 · Đo compute trên Kaggle (việc mở #2)

**Vì sao cần:** ước tính compute ở TOPIC_CARD_A dựa trên giả định khoảng 1 giây mỗi lần forward của TFM, chưa đo. Môi trường của Claude không tải được trọng số TabPFN (cần token license) hay TabICL (Hugging Face bị chặn), và không có GPU. Vì vậy phép đo phải chạy trên Kaggle (hoặc Colab) của thầy.

## Các bước (khoảng 45 phút, phần lớn là chờ)
1. **Lấy token TabPFN:** vào https://ux.priorlabs.ai, đăng nhập, chấp nhận license ở tab *License*, tạo API key.
2. **Tạo notebook Kaggle:** *Settings → Accelerator → GPU T4 x1* (hoặc P100); *Internet → On*. Thêm token: *Add-ons → Secrets*, tên `TABPFN_TOKEN`.
3. **Tải lên** `kaggle_g2_benchmark.py` (trong thư mục `compute/`), rồi chạy các ô:
   ```python
   !pip install -q tabpfn tabicl
   from kaggle_secrets import UserSecretsClient; import os
   os.environ["TABPFN_TOKEN"] = UserSecretsClient().get_secret("TABPFN_TOKEN")
   !python kaggle_g2_benchmark.py --backends tabpfn_v2,tabicl_v2 --out g2_benchmark.csv
   ```
   Nếu chưa có token: chạy `--backends tabicl_v2` trước. TabICL dùng trọng số mở (BSD-3), không cần đăng ký.
4. **Tuỳ chọn:** chạy thêm trên MacBook M1 với `--backends tabicl_v2 --out g2_benchmark_m1.csv`, để biết máy cá nhân gánh được phần nào (phát triển, thí điểm nhỏ).
5. **Gửi lại** file `g2_benchmark.csv`. Claude sẽ chạy `budget_calculator.py` và cập nhật C6 trong TOPIC_CARD_A.

## Những gì script đo
| Cột | Ý nghĩa |
|---|---|
| `block_median_s` | Thời gian một bước Gibbs theo khối: ngữ cảnh 90% hàng, truy vấn 10% hàng (trung vị 3 lần, sau 1 lần làm nóng) |
| `final_median_s` | Dự đoán 500 hàng test với ngữ cảnh đầy đủ |
| `n_estimators` | `1` (một thành viên, nhanh nhất) và `default` (mặc định của gói, ensemble) |

Lưới đo: n ∈ {500, 1.000, 2.000, 5.000} × số feature ∈ {10, 50} × số lớp ∈ {2, 5}. CSV được ghi sau mỗi dòng, nên phiên bị ngắt vẫn giữ phần đã đo.

## Công thức ngân sách (`budget_calculator.py`)
Mỗi lần chạy (một dataset × một loại nhãn yếu × một mức × một seed):
- forward Gibbs = số chuỗi × (burn-in + số mẫu) × K khối, mặc định 2 × (20 + 30) × 10 = 1.000;
- forward dự đoán cuối = số chuỗi × số mẫu = 60.

Giao thức mặc định: 12 dataset × 5 loại × 3 mức × 5 seed = 900 lần chạy, cộng baseline, nhân 2 dự phòng.

> Lưu ý: với mặc định này, nếu mỗi forward thật sự mất khoảng 1 giây thì mỗi lần chạy mất khoảng 18 phút và toàn bộ cần khoảng 265 giờ GPU cho phương pháp, nhiều hơn mức 120 giờ ghi ở TOPIC_CARD_A. Mức 120 giờ ở TOPIC_CARD_A tính 500 forward mỗi lần chạy. Con số nào đúng sẽ do số đo quyết định. Nếu vượt quota, các nút điều chỉnh là: số chuỗi, số vòng quét, số seed ở thí điểm, và chạy `n_estimators=1` trong Gibbs, chỉ dùng ensemble cho dự đoán cuối.

## Kết quả đo (thầy chạy trên Kaggle Tesla T4, ngày 24/10/2026)
File: [compute/g2_benchmark.csv](compute/g2_benchmark.csv). 64/64 cấu hình chạy được (`tabpfn 9.2.0`, `tabicl 2.2.0`, `torch 2.11.0+cu128`).

**Thời gian một bước Gibbs theo khối (giây, trung vị, 2 lớp):**

| Backend | n_estimators | Số feature | n = 500 | 1.000 | 2.000 | 5.000 |
|---|---|---|---|---|---|---|
| TabPFN v2 | 1 | 10 | 0,14 | 0,16 | 0,26 | 0,72 |
| TabPFN v2 | 1 | 50 | 0,38 | 0,59 | 1,05 | 3,25 |
| TabPFN v2 | mặc định | 10 | 0,34 | 0,41 | 0,85 | 2,71 |
| TabPFN v2 | mặc định | 50 | 0,94 | 1,54 | 3,46 | 11,76 |
| TabICL v2 | 1 | 10 | 0,50 | 0,54 | 0,52 | 0,63 |
| TabICL v2 | 1 | 50 | 0,50 | 0,57 | 0,57 | 0,77 |
| TabICL v2 | mặc định | 10 | 0,71 | 0,97 | 0,91 | 1,73 |
| TabICL v2 | mặc định | 50 | 1,15 | 1,68 | 1,68 | 3,43 |

**Nhận xét:**
1. TabPFN v2 với 1 thành viên nhanh nhất khi n ≤ 2.000. Chi phí tăng mạnh theo n và theo số feature: n = 5.000 với 50 feature chậm hơn n = 500 với 10 feature khoảng 23 lần.
2. TabICL gần như phẳng, khoảng 0,5 giây ở mọi kích thước. Đây nhiều khả năng là **chi phí cố định khi khởi tạo** (script tạo estimator mới ở mỗi lần gọi). Khi cài đặt thật (G6), dùng lại cùng một estimator thì số này có thể giảm; **cần đo lại**.
3. Với TabPFN, ensemble mặc định đắt hơn 1 thành viên khoảng 2,4–3,8 lần, nên chỉ dùng cho dự đoán cuối, không dùng trong vòng Gibbs.

## Ngân sách (`budget_calculator.py`, Gibbs dùng 1 thành viên, dự đoán cuối dùng ensemble mặc định)

| Kịch bản | Số lần chạy | GPU-giờ (phương pháp) | Tổng (kèm baseline ước tính, ×2 dự phòng) | Tuần (30 giờ/tuần) |
|---|---|---|---|---|
| **Giao thức đầy đủ như TOPIC_CARD_A** (12 dataset × 5 loại × 3 mức × 5 seed; 2 chuỗi × 50 vòng quét) | 900 | TabPFN 276 · TabICL 178 | 703 · 505 | **23,4 · 16,8 — không vừa** |
| 3 seed | 540 | 166 · 107 | 482 · 363 | 16,1 · 12,1 |
| 3 seed, 30 vòng quét | 540 | 102 · 65 | 353 · 280 | 11,8 · 9,3 |
| **3 seed, 30 vòng quét, ngữ cảnh Gibbs ≤ 2.000 hàng** | 540 | **50 · 57** | **250 · 264** | **8,3 · 8,8** |
| Thí điểm (PU + LLP, 10 dataset, 2 mức, 3 seed) | 120 | 34 · 23 | 98 · 77 | **3,3 · 2,6** |

**Kết luận về C6:**
- Giao thức đầy đủ như ghi ở TOPIC_CARD_A **không vừa** quota miễn phí trước NeurIPS 2027.
- Giao thức rút gọn (3 seed, 30 vòng quét, ngữ cảnh Gibbs tối đa 2.000 hàng) **vừa**, khoảng 8–9 tuần, và thí điểm khoảng 3 tuần. Kết luận: **C6 đạt với giao thức rút gọn**.
- Phần baseline (75 giờ, nhân 2) chỉ là ước tính và đang chiếm phần lớn ngân sách. Nhiều baseline cho dữ liệu bảng (SAR-EM, PU ExtraTrees, IPAL, Confident Learning) chạy trên CPU, nên số thật có thể thấp hơn nhiều; **đo ở G6**.
- Giới hạn ngữ cảnh 2.000 hàng làm thay đổi phân phối mục tiêu của Gibbs. Đây là quyết định phương pháp, phải chốt ở G5 và có ablation.

## Kiểm tra đã làm
Script đã chạy thử ở chế độ `--dry-run` (LogisticRegression thay TFM) để kiểm tra luồng code và định dạng CSV. Cả hai lớp `TabPFNClassifier.create_default_for_version(ModelVersion.V2, n_estimators=1, …)` và `TabICLClassifier(n_estimators=1)` đều khởi tạo được với gói `tabpfn 9.2.0` và `tabicl 2.2.0`. Chưa chạy với trọng số thật.
