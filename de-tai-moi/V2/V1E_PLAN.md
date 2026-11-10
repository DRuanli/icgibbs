# Lượt V1E (ghi trước, 10/11/2026; trước code và trước mọi số V1E)

- **Căn cứ:** [V1D_REPORT](V1D_REPORT.md) §4–§5.
- **Duyệt:** thầy đã duyệt hướng chung ngày 08/11 ("duyệt toàn bộ và thực hiện tiếp"), kèm yêu cầu dùng GPU cho nhiều việc trong một lần chạy.
- **Tính chất:** thăm dò, trên tập phát triển và dataset cũ; không chạm dữ liệu Study 3.

## 1. Nội dung

**E1. Hoàn tất V1D-2 và V1D-4 sau khi sửa bug D-V1D-1.**
- Chạy lại ô lỗi 36 / q 0,4 / seed 1 với TabICL. Các ô đã xong được bỏ qua.
- Chọn q* theo đúng quy tắc V1D_PLAN §2, không đổi gì.
- Chạy PLL phụ thuộc mẫu ở q* với TabPFN-3, TabDPT và TabPFN v2, trên dataset 54, 36, 23, seed 0–2.
- Chỉ báo cáo, như V1D-4.

**E2. Đếm dataset Study 3 mở rộng** (`s3_dataset_census.py --all_openml`).
- Như V1D, thêm mọi dataset đang hoạt động trên OpenML có 500 ≤ n ≤ 20.000 và 2 ≤ C ≤ 20.
- Áp cùng danh sách loại trừ và cùng các bậc nới tiêu chí.
- Gắn cờ `artificial_flag` theo mẫu tên (BNG, fri_c, synthetic, …) để rà tay.

**E3. Bằng chứng 3 TFM trên các ô V1 chỉ có TabICL** (dữ liệu mới).
- **Ô:** các ô TabICL của V1 mà TabPFN v2 không có: S2-PLL (10 dataset), S2-LLP (18), LLP 31, 37, 23381, 38. Tổng 32 dataset × 3 seed = 96 ô.
- **TFM chạy mới:** TabPFN-3 và TabDPT, cấu hình V1, 3 điểm xuất phát, có bộ nhớ đệm L.
- **Lịch:** CPU tính trước L cho S2-LLP từ đầu lượt.
- **P19 (v2-đa-TFM, mẫu mới):**
  - **Gộp:** X3 = trung bình đều dự đoán A1 của TabICL, TabPFN-3 và TabDPT.
  - **Điều kiện:** trên 32 dataset, trung bình (X3 − tiên tri trong 3 TFM) **≥ −0,01**; X3 thua quá 0,02 ở **≤ 4/32** dataset; và X3 ≥ A1 của từng TFM, tính trung bình theo dataset.
  - **Dự đoán:** đúng.
- **P20 (A1 với TFM mới, mẫu mới):** với TabPFN-3 và với TabDPT, trên 32 dataset:
  - **Lợi so với v1:** G(A1 − v1) **> 0**;
  - **Thụt lùi:** số thụt lùi **≤ 3**.
  - **Dự đoán:** đúng cho TabPFN-3. Với TabDPT em không chắc, vì ở V1D G chỉ +0,0002.
- **P21 (H8′, mẫu mới):** trên 192 ô mới (2 TFM × 96), AUC(c3 → T2) **≥ 0,80**. Nếu T2 có dưới 5 ô dương tính thì báo "không đủ ô". Dự đoán: đúng.

## 2. Thứ tự và chi phí (Kaggle 2 × T4, khoảng 1–2 phiên)

- **CPU:** đếm dataset mở rộng; rồi tính trước L cho E3, ưu tiên S2-LLP.
- **GPU 1:** ô lỗi (TabICL) → q* → E1 với TabDPT → E1 với TabPFN v2 → E3 với TabDPT → mảnh 2/3 của E3 S2-LLP với TabPFN-3.
- **GPU 0:** E1 với TabPFN-3 (chờ q*) → E3 với TabPFN-3 (S2-PLL, các LLP còn thiếu, mảnh 0/3 và 1/3 của S2-LLP).
- **Chấm:** `V2/analysis/v1e_report.py`, viết và commit trước khi có số.
