# 01 · Nguyên tắc và liêm chính học thuật

[← Về README](README.md)

## 1. Bốn nguyên tắc nền (Singapore Statement on Research Integrity, 2010)

1. **Trung thực** trong mọi khâu: thiết kế, thu thập, phân tích, báo cáo.
2. **Chịu trách nhiệm** về việc thực hiện nghiên cứu.
3. **Công bằng và tôn trọng** khi làm việc với người khác (đồng tác giả, phản biện, người tham gia).
4. **Quản lý tốt** nguồn lực và dữ liệu thay mặt cho người khác.

## 2. Vi phạm nghiêm trọng (cấm tuyệt đối)

| Vi phạm | Định nghĩa | Ví dụ trong KHMT/KHDL |
|---|---|---|
| Bịa đặt (fabrication) | Tạo ra dữ liệu hay kết quả không có thật | Điền số liệu baseline "ước lượng" thay vì chạy |
| Làm sai lệch (falsification) | Thao túng dữ liệu, thiết bị hay quy trình, bỏ bớt kết quả | Bỏ các dataset mà phương pháp thua; chạy lại seed tới khi thắng rồi chỉ báo cáo lần thắng |
| Đạo văn (plagiarism) | Dùng ý tưởng, câu chữ, code của người khác không ghi nguồn | Chép đoạn related work; dùng code GitHub không ghi nguồn hay trái giấy phép |

## 3. Thực hành đáng ngờ (Questionable Research Practices) cần chặn bằng quy trình

| Thực hành | Cách chặn trong sổ tay này |
|---|---|
| HARKing: đặt giả thuyết sau khi thấy kết quả (Kerr 1998) | Đăng ký trước nội bộ ở G3 ([RESEARCH_PLAN.md](templates/RESEARCH_PLAN.md)) |
| p-hacking, chọn lọc metric hay dataset | Đóng băng giao thức ở G5; báo cáo mọi metric và dataset đã đăng ký |
| Baseline yếu hoặc tune không công bằng | Quy tắc baseline ở [06 §3](06_THIET_KE_THUC_NGHIEM.md) |
| Chia nhỏ một kết quả thành nhiều bài (salami slicing) | Mỗi bài phải có một đóng góp độc lập, ghi trong TOPIC_CARD |
| Nộp song song, xuất bản trùng lặp | Một bản thảo chỉ ở một tạp chí tại một thời điểm; khai báo bài liên quan đang xét duyệt |
| Tự đạo văn | Dùng lại nội dung đã công bố phải trích dẫn và viết lại; ghi rõ khác biệt với bài trước |
| Trích dẫn ép (citation coercion), trích dẫn vòng | Chỉ trích dẫn vì liên quan về nội dung |

## 4. Tác giả và đóng góp

**Tiêu chí tác giả (ICMJE).** Một người là tác giả khi thỏa **cả bốn** điều kiện:
1. Đóng góp đáng kể vào ý tưởng, thiết kế, hoặc thu thập, phân tích, diễn giải dữ liệu.
2. Viết hoặc sửa bản thảo một cách có nội dung trí tuệ quan trọng.
3. Phê duyệt bản cuối cùng.
4. Đồng ý chịu trách nhiệm về mọi khía cạnh của công trình.

Ai không đủ bốn điều kiện thì ghi trong phần Acknowledgments.

**Khai báo vai trò bằng CRediT** (ANSI/NISO Z39.104-2022, 14 vai trò): Conceptualization, Data curation, Formal analysis, Funding acquisition, Investigation, Methodology, Project administration, Resources, Software, Supervision, Validation, Visualization, Writing – original draft, Writing – review & editing.

**Thoả thuận sớm.** Thứ tự tác giả và vai trò được thống nhất bằng văn bản ở G3, xem lại ở G9. Tranh chấp xử lý theo hướng dẫn của COPE.

## 5. Dùng công cụ AI (LLM, trợ lý lập trình) trong nghiên cứu

Tổng hợp chính sách của các nhà xuất bản lớn (tình trạng cuối 2025; kiểm tra lại trang chính sách của tạp chí đích trước khi nộp):

| Nhà xuất bản | AI làm tác giả? | Khai báo ở đâu | Ghi chú |
|---|---|---|---|
| Springer Nature | Không | Phần Methods hoặc vị trí phù hợp | Cấm ảnh do AI sinh, trừ ngoại lệ hẹp; sửa chính tả không cần khai báo |
| Elsevier | Không | Tuyên bố khai báo khi nộp: công cụ, mục đích, mức giám sát của con người | Kiểm tra ngữ pháp cơ bản được miễn |
| IEEE | Không nêu rõ | Acknowledgments: hệ thống, các phần bị ảnh hưởng, mức độ | Phản biện không được đưa bản thảo lên nền tảng AI công khai |
| ACM | Không | Acknowledgments: công cụ và cách dùng cho văn bản, bảng, hình, code, dữ liệu | |
| Taylor & Francis | Không | Methods hoặc Acknowledgments, kèm tên và phiên bản | Cấm dùng AI tạo hay sửa ảnh, hình, dữ liệu gốc |
| NeurIPS | — | Mục 16 của checklist, khi LLM là phần quan trọng hoặc không chuẩn của phương pháp | Dùng chỉ để sửa câu chữ thì không cần khai báo |

**Quy tắc nội bộ:**
1. AI không bao giờ là tác giả. Người nghiên cứu chịu trách nhiệm hoàn toàn với mọi dòng chữ, dòng code và con số.
2. **Mọi trích dẫn do AI gợi ý phải được mở ra và đối chiếu** (DOI, tiêu đề, tác giả, năm, và nội dung được dẫn có đúng là bài đó nói không). Trích dẫn bịa là lỗi bịa đặt.
3. Mọi đoạn code do AI viết phải qua test ở G6 như code tự viết.
4. Khi làm phản biện cho tạp chí: **không đưa bản thảo của người khác vào công cụ AI**.
5. Lưu lại cách dùng AI (công cụ, phiên bản, mục đích, phần bị ảnh hưởng) trong `docs/AI_USAGE.md` của dự án để khai báo chính xác lúc nộp.

## 6. Dữ liệu cá nhân và giấy phép dữ liệu

**Pháp luật Việt Nam:**
- Luật Bảo vệ dữ liệu cá nhân số **91/2025/QH15**, thông qua 26/6/2025, hiệu lực **01/01/2026**.
- Nghị định hướng dẫn **356/2025/NĐ-CP**, hiệu lực 01/01/2026, thay thế Nghị định 13/2023/NĐ-CP.
- Xử lý dữ liệu cá nhân của **trẻ em** cần sự đồng ý của người đại diện theo pháp luật. Điều này áp dụng trực tiếp cho dữ liệu học sinh phổ thông.
- Với các trường hợp ngoại lệ cho nghiên cứu khoa học hay thống kê, cần đọc nguyên văn luật và nghị định, hoặc hỏi bộ phận pháp chế của cơ quan. Sổ tay này không thay thế tư vấn pháp lý.

**Quy tắc nội bộ:**
1. Dữ liệu có thông tin định danh: cần thoả thuận sử dụng dữ liệu (DUA) hoặc văn bản cho phép của đơn vị sở hữu, cùng sự đồng ý theo luật. Ẩn danh hoá trước khi phân tích, và lưu bảng khoá định danh tách riêng, có kiểm soát truy cập.
2. Nghiên cứu có người tham gia (khảo sát, phỏng vấn, thí nghiệm người dùng): xin phép hội đồng đạo đức (IRB hoặc tương đương) nếu cơ quan hay tạp chí yêu cầu.
3. Mỗi dataset ghi rõ nguồn, phiên bản, **giấy phép** (CC BY, CC BY-NC, giấy phép riêng…) và điều khoản sử dụng trong [DATASET_CARD.md](templates/DATASET_CARD.md). Dataset có giấy phép phi thương mại vẫn dùng được cho nghiên cứu, nhưng phải ghi rõ.
4. Không công bố lại dữ liệu gốc khi giấy phép không cho phép. Thay vào đó công bố script tải và tiền xử lý.

## 7. Xung đột lợi ích và nguồn tài trợ

Khai báo mọi nguồn tài trợ và quan hệ có thể ảnh hưởng đến kết quả: công ty liên quan đến phương pháp, bằng sáng chế, hợp đồng tư vấn. Paper TabPFN (Nature 2025) là ví dụ khai báo đầy đủ: nêu rõ tác giả gắn với công ty Prior Labs và các đơn sáng chế liên quan.
