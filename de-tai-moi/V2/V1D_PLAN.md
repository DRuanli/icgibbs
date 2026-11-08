# Quyết định sau V1C và lượt V1D (ghi trước, 08/11/2026)

- **Thầy duyệt toàn bộ đề xuất ngày 08/11/2026** ("duyệt toàn bộ và thực hiện tiếp"). Căn cứ: [V1C_REPORT](V1C_REPORT.md) §6–§8.
- **Thứ tự:** tài liệu này được commit trước code của lượt V1D (`s3_dataset_census.py` và lượt V1D trong `run_s2_lot.sh`). Chưa có số nào của V1D.

## 1. Quyết định

| # | Quyết định | Căn cứ |
|---|---|---|
| Q1 | **v2 = A1:** 3 điểm xuất phát (warm, base, cold), mỗi chuỗi 30 vòng quét, gộp đều. Không có bản rẻ hơn | P12 sai |
| Q2 | **v2-đa-TFM:** gộp đều A1 của 4 TFM (TabPFN v2, TabICL v2, TabPFN-3, TabDPT). Vào Study 3 như một biến thể chính của v2, cùng H7 | P13 đúng |
| Q3 | **Cơ chế nhãn yếu của Study 3:** LLP túi ngẫu nhiên 8 và 32; PLL iid q = 0,6; PLL phụ thuộc mẫu với q chọn ở V1D (§2) | V1C phần G |
| Q4 | **Bỏ túi gom theo k-means.** Không thêm túi theo feature phân loại, vì cần chọn feature riêng cho từng dataset, ngoài phạm vi | V1C phần G: dễ hơn túi iid |
| Q5 | **H8 viết lại (H8′).** Bỏ yêu cầu "đo trước khi chạy chuỗi"; đó trở thành giới hạn của bài. Tín hiệu mới c3 = TV trung bình giữa Q̄_warm và Q̄_base (trên train, không cần nhãn, không tốn thêm lượt gọi TFM). H8′: AUC(c3 → T2) ≥ 0,8 ở Study 3, với T2 = v1 thua max(A1, L) quá 0,02 | P15 sai. Thăm dò `h8_explore.py` trên 276 ô cũ: AUC 0,85. Thăm dò, chưa phải bằng chứng |
| Q6 | **Nguồn dataset của Study 3:** như S2, thêm OpenML-CC18. Loại mọi dataset đã dùng và 123 dataset huấn luyện của TabDPT (`G4/code/s3_exclude_ids.csv`, 176 ID) | Nguồn của S2 sau khi loại chỉ còn ≤ 13 nhị phân và ≤ 7 đa lớp, dưới mục tiêu của Study 3 |

## 2. Lượt V1D (Kaggle 2 × T4, khoảng 2–3 giờ)

**V1D-1: đếm dataset** (`s3_dataset_census.py`, CPU). Chỉ đọc metadata OpenML. Số liệu dùng để viết protocol Study 3 trước cổng V2.

**V1D-2: hiệu chỉnh q cho PLL phụ thuộc mẫu.**
- **Cấu hình:** `--weak_variant inst`, q ∈ {0,2; 0,3; 0,4}, trên tập phát triển PLL (54, 36, 23), seed 0–2, A1 với 3 điểm xuất phát, cấu hình V1.
- **TFM:** TabICL v2 (GPU 1), TabPFN-3 (GPU 0).
- **Quy tắc chọn q, ghi trước, theo TabICL v2:**
  - Chọn q **lớn nhất** trong {0,2; 0,3; 0,4} sao cho trung bình theo dataset của (A1_inst(q) − A1 iid q 0,6 của V1) **≥ −0,10**.
  - Nếu không mức nào đạt, chọn q = 0,2.
- **Lý do:** cần mức khó nhưng còn học được. Ở q = 0,6, dataset 36 rơi từ 0,97 xuống 0,52.
- **TabPFN-3:** chỉ báo cáo.
- **Chấm:** tại đây, bằng tay theo đúng quy tắc trên. Bảng tính bằng `V2/analysis/v1c_report.py` (phần G đọc thư mục npz).
