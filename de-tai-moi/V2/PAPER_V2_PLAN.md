# Bản thảo NeurIPS 2027 (bài v2): phân tích và dàn ý (12/11/2026)

- **Phạm vi:** viết lại `paper/` để gộp phương pháp v2, lý thuyết W1/R2 và Study 3 vào bài gốc.
- **Nguồn số:** mọi số mới sinh bằng `paper/scripts/make_v2.py` → `paper/numbers_v2.tex` và các bảng, hình mới.
- **Trạng thái:** bản nháp đầu biên dịch được, phần chính vừa đúng 9 trang với style tạm; đang qua các vòng phản biện.

## 1. Câu chuyện của bài

1. **Ý tưởng** (không đổi): dưới tính trao đổi được, xác suất có điều kiện của một nhãn ẩn chính là PPD mà TFM tính. Một forward pass của TFM đóng băng là một bước Gibbs.
2. **Ba nghiên cứu ghi trước, mỗi nghiên cứu sinh ra từ thất bại của nghiên cứu trước:**
   - Study 1 (TabPFN v2): chuỗi sụp ở PLL nhập nhằng cao.
   - Study 2 (TabICL v2, 28 dataset mới): một chuỗi hơn các baseline chuyên biệt mạnh nhất.
   - Study 3 (4 TFM, 20 dataset mới): nhiều điểm xuất phát và nhiều TFM hơn một chuỗi (H1, H2 đúng; C5 đạt); H3 và C4 không đạt.
3. **Lý thuyết nối hai nửa:** ánh xạ phản ứng quyết định chuỗi đi đâu (Định lý 3). Hai điểm hút thì điểm xuất phát quyết định. Đây là lý do có bản nhiều điểm xuất phát.

## 2. Phân tích thăm dò mới trên Study 3 (`S3/s3_explore.py`, viết sau khi chấm)

Các kết quả này **không ghi trước**. Em báo cả những kết quả bất lợi cho thiết kế v2:

| Kết quả | Số |
|---|---|
| Bản gộp nhiều TFM so với L | hơn ở 20/20 dataset, +0,064; 0/71 cặp thụt lùi theo thước đo C4 |
| Bản gộp chỉ gồm chuỗi v1 (không dùng L) so với L | hơn ở 18/20 dataset, +0,036 |
| Chuỗi xuất phát từ L, một mình, so với A1 | +0,001: **ngang** A1 |
| Chuỗi lạnh | không thêm gì: warm + cold chỉ hơn v1 +0,008 |
| v1 lấy trung bình với dự đoán của L (không chạy thêm chuỗi) so với A1 | +0,008, p = 0,26: **ngang** A1 |
| A1 lấy trung bình thêm với L so với A1 | +0,021, p = 0,005 |
| Bỏ TabICL khỏi bản gộp | mất 0,028; bỏ TFM khác mất ≤ 0,001 |
| Bản gộp so với riêng A1 của TabICL | +0,016, 10/5/5, p = 0,08: không có ý nghĩa |
| Ba chuỗi về cùng lưu vực | 81% số ô (PLL 61%); lợi A1 ở PLL: +0,328 khi khác lưu vực, +0,006 khi cùng |

**Hệ quả cho cách viết:**
- Không viết rằng cả ba điểm xuất phát đều cần. Bài viết thẳng: lợi đến từ điểm xuất phát learner.
- Không viết bản gộp hơn mọi TFM. Bài viết: bản gộp giúp khỏi phải biết trước TFM nào tốt.
- Phép so với L luôn ghi "exploratory" và không vào abstract.

## 3. Cấu trúc bài (phần chính 9 trang)

| Mục | Nội dung | Thay đổi so với bài gốc |
|---|---|---|
| 1 Introduction | ba nghiên cứu; hai tiêu chí không đạt nêu ngay ở đây | viết lại |
| 2 Related work | rút gọn | bảng so sánh chuyển xuống Phụ lục |
| 3 Setup | giữ | — |
| 4 Method | một chuỗi; thêm đoạn nhiều điểm xuất phát và nhiều TFM | mã giả chuyển xuống Phụ lục B |
| 5 Theory | Định lý 1, 2 (rút gọn); đoạn khối không sửa (Mệnh đề 5 mới); **Định lý 3 mới** | Mệnh đề Bayes và Mệnh đề "naive" chuyển phát biểu xuống Phụ lục A |
| 6 Experiments | thiết kế 3 nghiên cứu; Study 2; sụp và ánh xạ phản ứng (kèm phép kiểm P8); **Study 3**; nhãn thật, độ chính xác, chi phí | Bảng 2 mới; hình theo dataset của Study 2 và Study 3 ở Phụ lục |
| 7 Limitations, 8 Conclusion | viết lại | — |

**Phụ lục mới:** chứng minh Định lý 3 (Bổ đề một vòng quét), Mệnh đề mô-men dừng, Mệnh đề độ lệch dự đoán; Phụ lục F (Study 3: protocol, dataset, bảng theo dataset, theo TFM, các cặp thụt lùi, phân tích thăm dò); Phụ lục G (giai đoạn phát triển: bảng 13 dự đoán ghi trước, trong đó 5 sai).

## 4. Việc còn mở

- Thay `neurips_placeholder.sty` bằng style chính thức khi có; kiểm lại số trang.
- Hai `\todo` (đo rủi ro riêng tư của LLP; chính sách LLM của hội nghị) và một `\pending` (giấy phép) trong checklist.
- Kiểm nội dung từng chỗ trích trước khi nộp (REFERENCE_CHECK, việc của G10).
