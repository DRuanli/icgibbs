# 04 · Tổng quan tài liệu và kiểm tra tính mới (G2)

[← Về README](README.md)

Có hai mức:
- **Tổng quan để định vị** cho mọi bài nghiên cứu: mục đích là định vị đóng góp và chứng minh tính mới.
- **Tổng quan có hệ thống (SLR)** khi chính bài tổng quan là sản phẩm: theo Kitchenham & Charters (2007) và báo cáo theo PRISMA 2020.

Cả hai mức đều dùng **tìm kiếm có nhật ký**, vì câu "to the best of our knowledge" chỉ có giá trị khi có nhật ký đi kèm.

## 1. Quy trình tìm kiếm có nhật ký (mức bắt buộc cho mọi bài)

1. **Định nghĩa phạm vi** bằng ba thành phần: *Bài toán* (loại task, dạng dữ liệu), *Cách tiếp cận* (họ phương pháp), *Bối cảnh* (giả định, ràng buộc).
2. **Tập hạt giống:** 5–10 bài chắc chắn liên quan, gồm survey gần nhất và các bài ở venue mạnh nhất.
3. **Từ khoá:** lấy từ tiêu đề và abstract của tập hạt giống. Lập bảng đồng nghĩa (ví dụ "multi-output" / "multi-target" / "multi-label" / "multivariate response").
4. **Tìm trên tối thiểu 4 nguồn**, ghi lại câu truy vấn, ngày, số kết quả, số bài được chọn:

   | Nguồn | Mạnh ở |
   |---|---|
   | Scopus, Web of Science | Tạp chí có chỉ mục, lọc theo năm và chủ đề |
   | IEEE Xplore, ACM Digital Library | Hội nghị và tạp chí của IEEE và ACM |
   | DBLP | Tra cứu đầy đủ theo tác giả và venue trong KHMT |
   | arXiv, OpenReview | Công trình mới nhất, chưa xuất bản; bài đang phản biện ở ICLR, TMLR |
   | Semantic Scholar, Google Scholar | Tìm theo trích dẫn, tìm toàn văn |
   | GitHub (issue, README) | Bằng chứng nhu cầu và công trình chưa thành bài báo |

5. **Snowballing** (Wohlin 2014): với mỗi bài được chọn, duyệt danh sách tài liệu tham khảo (*backward*) và các bài trích dẫn nó (*forward*). Lặp lại đến khi một vòng không thêm bài mới nào (bão hoà).
6. **Tiêu chí chọn và loại**, viết trước khi đọc. Ví dụ: chọn bài giải *đúng* bài toán; loại bài chỉ là ứng dụng không có phương pháp mới; loại bài không phải tiếng Anh nếu không đọc được.
7. **Ma trận tài liệu** `related_work_matrix.csv`: một hàng mỗi bài, các cột ở mục 3.
8. **Cập nhật:** chạy lại toàn bộ truy vấn mỗi tháng và một lần trong vòng 30 ngày trước khi nộp (G11).

## 2. Đọc bài báo: phương pháp ba lượt (Keshav 2007)

| Lượt | Thời gian | Mục tiêu | Ra quyết định |
|---|---|---|---|
| 1 | 5–10 phút | Tiêu đề, abstract, giới thiệu, đề mục, kết luận, lướt tài liệu tham khảo | Trả lời 5 câu: loại bài, bối cảnh, đúng hay không, đóng góp, viết rõ không. Có đọc tiếp không? |
| 2 | ~1 giờ | Đọc kỹ, xem hình và bảng, đánh dấu tài liệu cần đọc | Tóm tắt được nội dung chính cho người khác |
| 3 | Vài giờ | Tái tạo bài báo trong đầu: thử làm lại chính các bước với cùng giả định | Chỉ ra được giả định ẩn, lỗi, và chỗ còn mở |

Các bài gần nhất với đóng góp (≥3 bài) **bắt buộc** đọc tới lượt 3.

## 3. Ma trận tài liệu liên quan (cột bắt buộc)

| Cột | Nội dung |
|---|---|
| `id`, `citation`, `venue`, `year`, `peer_reviewed` | Thông tin cơ bản; `peer_reviewed` dùng cho tiêu chí C2 |
| `problem` | Bài toán chính xác mà bài giải |
| `assumptions` | Giả định chính |
| `method_family` | Họ phương pháp |
| `datasets` | Dataset dùng, số lượng |
| `baselines` | Họ so với ai |
| `metrics` | Metric |
| `code_url`, `license` | Để dùng làm baseline |
| `stated_limitations` | **Trích nguyên văn** phần limitation hay future work |
| `diff_from_ours` | Khác đóng góp dự kiến ở đâu, một câu |
| `read_pass` | 1, 2 hoặc 3 |

## 4. Viết khẳng định tính mới

Mẫu câu có giới hạn, kiểm chứng được:

> "Theo tìm kiếm của chúng tôi trên Scopus, IEEE Xplore, ACM DL, arXiv và OpenReview (truy vấn ở Phụ lục X, cập nhật ngày …), chưa có công trình nào [đóng góp cụ thể] trong [bối cảnh]. Gần nhất là A, vốn [khác ở …], và B, vốn [khác ở …]."

Không viết "first ever", "novel paradigm" hay "for the first time" khi không có nhật ký tìm kiếm đi kèm.

## 5. Khi bài báo chính là một tổng quan có hệ thống

- **Lập kế hoạch** (Kitchenham & Charters 2007): câu hỏi tổng quan, protocol (nguồn, truy vấn, tiêu chí chọn và loại, đánh giá chất lượng, trích xuất dữ liệu), cho người thứ hai rà soát protocol.
- **Thực hiện:** hai người sàng lọc độc lập (ít nhất trên một mẫu), báo cáo độ đồng thuận (ví dụ Cohen's κ), giải quyết bất đồng.
- **Báo cáo:** theo PRISMA 2020 (27 mục, kèm sơ đồ luồng gồm số bản ghi tìm thấy, loại trùng, sàng lọc, đọc toàn văn, được chọn). Nghiên cứu dạng *mapping* theo Petersen et al. (2015).
- **Công bố:** protocol, truy vấn, danh sách bài ở mỗi bước, và bảng trích xuất dữ liệu.

## 6. Lỗi thường gặp

1. Chỉ trích dẫn bài cũ: thiếu công trình 12 tháng gần nhất trên arXiv và OpenReview.
2. Related work liệt kê từng bài ("A làm X. B làm Y.") mà không định vị: thay bằng nhóm theo tiêu chí kèm bảng so sánh.
3. Bỏ qua bài cạnh tranh trực tiếp. Phản biện thường chính là tác giả của bài đó.
4. Trích dẫn sai nội dung (nói bài A chứng minh điều mà A không chứng minh). Đọc lại đoạn gốc trước khi dẫn.
5. Trích dẫn không có thật (thường do công cụ AI gợi ý). Đối chiếu DOI cho **mọi** mục tài liệu tham khảo ở G10.
