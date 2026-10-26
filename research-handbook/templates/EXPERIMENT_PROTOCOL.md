# Giao thức thực nghiệm, đóng băng (G5)

> Commit **trước** khi chạy thực nghiệm chính. Người kiểm duyệt ký ở Mục 10.

Đề tài: ____________  Ngày đóng băng: ____________  Commit: ____________

## 1. Dataset (≥10 độc lập)
Tiêu chí chọn (viết trước): ____________

| # | Tên | Nguồn / ID | Nhóm nguồn (cho tính độc lập) | n | d | Đặc điểm (lớp, thiếu dữ liệu, kiểu feature) | Giấy phép | Dataset card |
|---|---|---|---|---|---|---|---|---|
| 1 | | | | | | | | |

Số nguồn độc lập: ____

## 2. Tiền xử lý
| Bước | Fit trên | Tham số |
|---|---|---|
| | train của từng fold | |

## 3. Chia dữ liệu
- Phương pháp (stratified / group / temporal / split chuẩn):
- k = ____, số lần lặp = ____, seed: ____
- Chọn siêu tham số: nested CV / validation tách từ train. Chi tiết:

## 4. Phương pháp so sánh
| Phương pháp | Loại (đề xuất / peer-reviewed / đơn giản / SOTA preprint / ablation) | Venue, năm | Code | Không gian tune | Ngân sách tune |
|---|---|---|---|---|---|
| | | | | | |

Quy tắc công bằng: ____________ (ví dụ: mỗi phương pháp 50 cấu hình random search, cùng vòng trong 5-fold)

## 5. Metric
- Metric chính (cho khẳng định chính):
- Metric phụ:
- Metric tính toán (thời gian, bộ nhớ, phần cứng):

## 6. Kế hoạch thống kê
| Khẳng định | Kiểm định | α | Hiệu chỉnh đa so sánh | Effect size | ROPE (nếu dùng) |
|---|---|---|---|---|---|
| | Wilcoxon signed-rank trên N dataset | 0,05 | Holm | chênh lệch trung bình + CI | |

## 7. Ablation và độ nhạy
| Thí nghiệm | Thành phần hoặc siêu tham số | Dải giá trị |
|---|---|---|
| | | |

## 8. Checklist rò rỉ (người kiểm duyệt đánh dấu)
- [ ] L1.1 có test riêng  - [ ] L1.2 tiền xử lý trong fold  - [ ] L1.3 chọn feature trong fold  - [ ] L1.4 đã loại trùng
- [ ] L2 không có feature bất hợp lệ  - [ ] L3.1 không rò rỉ thời gian  - [ ] L3.2 nhóm không nằm ở cả hai phía  - [ ] L3.3 test đại diện

## 9. Compute dự kiến
Phần cứng: ____  Ước lượng tổng giờ: ____ (× 2 dự phòng)

## 10. Ký duyệt
Người kiểm duyệt: ____  Ngày: ____  ☐ Đạt  ☐ Sửa

## 11. Deviations (ghi sau khi đóng băng)
| Ngày | Thay đổi | Lý do | Kết quả liên quan được gắn nhãn exploratory? |
|---|---|---|---|
| | | | |
