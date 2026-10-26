# 10 · Checklist tổng hợp trước khi nộp (G10)

[← Về README](README.md)

Người kiểm duyệt độc lập điền checklist này. Mỗi dòng ghi **Có / Không / N/A** kèm vị trí trong bài (mục, bảng, trang). "Không" hay "N/A" chấp nhận được khi có lý do, đúng tinh thần của NeurIPS Checklist. Mục đánh dấu ★ là bắt buộc phải "Có".

## A. Tiêu chí riêng của nhóm
- [ ] ★ ≥10 bộ dữ liệu độc lập (khác nguồn thu thập), danh sách và lý do chọn có trong bài.
- [ ] ★ ≥2 baseline đã qua peer-review cho đúng bài toán, cộng baseline đơn giản và baseline mạnh nhất hiện tại.
- [ ] ★ Venue đích khớp phạm vi. Với tạp chí: đã kiểm tra chỉ mục trên Scopus/WoS và dấu hiệu săn mồi hay chiếm đoạt. Với hội nghị: đã kiểm tra hạng CORE, giới hạn trang, quy định ẩn danh và chính sách preprint.
- [ ] ★ Đóng góp chính không thuộc nhân quả, fairness, hay kiểm định như là chủ đề.

## B. Khẳng định và giới hạn (NeurIPS Q1–Q2)
- [ ] ★ Abstract và giới thiệu chỉ khẳng định điều có trong ma trận khẳng định–bằng chứng.
- [ ] ★ Có mục Limitations với giới hạn thật: giả định mạnh, phạm vi dataset, trường hợp phương pháp thua.
- [ ] Có mục mối đe doạ tính hợp lệ (nội tại, ngoại tại, cấu trúc, kết luận).

## C. Lý thuyết (NeurIPS Q3)
- [ ] ★ Mọi giả định nằm trong phát biểu định lý.
- [ ] ★ Chứng minh đầy đủ ở bài hoặc phụ lục; có phác thảo ở bài chính.
- [ ] Mỗi bổ đề đã qua kiểm chứng số; có người đọc chéo độc lập.

## D. Thực nghiệm (NeurIPS Q4, Q6, Q7, Q8; REFORMS)
- [ ] ★ Giao thức đóng băng trước khi chạy; các thay đổi có trong mục Deviations và được báo cáo là exploratory.
- [ ] ★ Cùng bộ split cho mọi phương pháp; mọi tiền xử lý fit trong từng fold train.
- [ ] ★ Checklist rò rỉ 8 loại (L1.1–L3.3) đạt ([06 §4](06_THIET_KE_THUC_NGHIEM.md)).
- [ ] ★ Baseline: code chính thức hoặc cài đặt đã kiểm chứng; đã tái hiện số liệu gốc; cùng ngân sách tune.
- [ ] ★ Siêu tham số và cách chọn được báo cáo (bài chính hoặc phụ lục).
- [ ] ★ Error bar hoặc khoảng tin cậy cho kết quả chính; nêu rõ chúng phản ánh nguồn biến thiên nào và là std hay SE.
- [ ] ★ Kiểm định thống kê đúng kế hoạch (Wilcoxon + Holm, hoặc Bayesian với ROPE); có effect size.
- [ ] ★ Báo cáo điểm thô theo từng dataset (phụ lục), không chỉ điểm trung bình hay chuẩn hoá.
- [ ] Ablation cho mỗi thành phần được khẳng định là có ích.
- [ ] Phân tích độ nhạy siêu tham số; phân tích khả năng mở rộng.
- [ ] Phân tích lỗi: dataset hay cấu hình mà phương pháp thua.
- [ ] Phần cứng, thời gian mỗi lần chạy, tổng compute (gồm cả các lần thử).
- [ ] Kết quả từ dữ liệu tổng hợp được dán nhãn và không làm bằng chứng chính.

## E. Tái lập (NeurIPS Q4–Q5; ACM Badging)
- [ ] ★ Một người không tham gia đã tái tạo ít nhất một bảng chính từ clone sạch.
- [ ] ★ README có lệnh chính xác cho từng kết quả; phiên bản thư viện được pin; seed được ghi.
- [ ] Code ẩn danh (nếu double-blind); có kế hoạch lưu trữ DOI khi được chấp nhận.

## F. Dữ liệu, giấy phép, đạo đức (NeurIPS Q9–Q15)
- [ ] ★ Mỗi dataset có nguồn, phiên bản, giấy phép, trích dẫn người tạo.
- [ ] ★ Dữ liệu cá nhân có cơ sở pháp lý (Luật 91/2025/QH15, Nghị định 356/2025/NĐ-CP), đã ẩn danh; có đồng ý của người đại diện với dữ liệu trẻ em.
- [ ] IRB hoặc hội đồng đạo đức nếu có người tham gia.
- [ ] Tác động xã hội tiêu cực tiềm năng (nếu liên quan) và biện pháp giảm thiểu.
- [ ] Asset mới (dataset, mô hình, code) có tài liệu, giấy phép, giới hạn.

## G. Trích dẫn và liêm chính
- [ ] ★ **Mọi** tài liệu tham khảo đã đối chiếu DOI, tiêu đề, tác giả, năm; nội dung được dẫn đúng với bài gốc.
- [ ] ★ Có công trình 12 tháng gần nhất, gồm arXiv và OpenReview; nhật ký tìm kiếm cập nhật trong vòng 30 ngày.
- [ ] ★ Đã chạy kiểm tra đạo văn, và tỉ lệ trùng lặp đã được giải thích.
- [ ] ★ Khai báo dùng AI đúng chính sách của nhà xuất bản đích.
- [ ] ★ Tác giả thỏa 4 tiêu chí ICMJE; có khai báo CRediT; mọi tác giả đã duyệt bản cuối.
- [ ] Khai báo tài trợ và xung đột lợi ích.

## H. Trình bày
- [ ] Ký hiệu nhất quán; có bảng ký hiệu nếu nhiều công thức.
- [ ] Hình đọc được khi in đen trắng; chú thích tự giải thích được.
- [ ] Đúng template và giới hạn độ dài; tài liệu tham khảo đúng định dạng của tạp chí.

---

**Kết luận của người kiểm duyệt:** ☐ Nộp được  ☐ Sửa lỗi Lớn rồi kiểm lại  ☐ Không nộp (có lỗi Nghiêm trọng)

Người kiểm duyệt: ____________  Ngày: ____________
