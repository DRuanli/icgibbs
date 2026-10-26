# 09 · Viết bài, chọn tạp chí, nộp bài và trả lời phản biện (G9–G12)

[← Về README](README.md)

## 1. Cấu trúc bài báo KHMT/KHDL

| Phần | Nội dung bắt buộc | Lỗi thường gặp |
|---|---|---|
| Tiêu đề | Nêu bài toán và đóng góp; tránh tính từ quảng cáo | "A novel framework for…" |
| Abstract | Bài toán → khoảng trống → phương pháp → kết quả có **con số** → ý nghĩa | Không có con số; khẳng định vượt bằng chứng |
| 1. Giới thiệu | Bối cảnh; khoảng trống (trích dẫn); câu hỏi; **danh sách đóng góp có trỏ tới mục và bảng**; cấu trúc bài | Đóng góp mơ hồ ("we study…") |
| 2. Công trình liên quan | Nhóm theo tiêu chí, có **bảng so sánh** với 3–6 công trình gần nhất; nêu rõ khác biệt | Liệt kê từng bài; thiếu bài cạnh tranh trực tiếp |
| 3. Kiến thức nền, phát biểu bài toán | Ký hiệu, định nghĩa hình thức | Ký hiệu thay đổi giữa các mục |
| 4. Phương pháp | Mô tả đủ để cài đặt lại; mã giả; độ phức tạp | Bỏ chi tiết quan trọng vào code |
| 5. Lý thuyết (nếu có) | Định lý, giả định, phác thảo chứng minh ([07](07_LY_THUYET_VA_CHUNG_MINH.md)) | Giả định ẩn |
| 6. Thực nghiệm | Thiết lập (dataset, baseline, metric, split, tune, phần cứng); kết quả **theo từng RQ**; ablation; độ nhạy; thống kê | Kết quả không gắn với câu hỏi; chỉ có trung bình |
| 7. Thảo luận | Giải thích *vì sao*, khi nào phương pháp thua | Lặp lại bảng bằng lời |
| 8. Giới hạn, mối đe doạ tính hợp lệ | Giới hạn thật (NeurIPS Q2); bốn loại tính hợp lệ ([06 §8](06_THIET_KE_THUC_NGHIEM.md)) | Giới hạn hình thức kiểu "cần thêm dữ liệu" |
| 9. Kết luận | Tóm tắt khẳng định đã được chứng minh; hướng mở có căn cứ | Khẳng định mới chưa có trong bài |
| Khai báo | Data & Code Availability; Funding; Competing interests; **CRediT**; khai báo dùng AI | Thiếu khai báo AI |
| Phụ lục | Chứng minh đầy đủ, siêu tham số, kết quả theo từng dataset, phân tích bổ sung | |

## 2. Quy tắc viết

1. **Ma trận khẳng định–bằng chứng** ([template](templates/CLAIM_EVIDENCE_MATRIX.md)): mỗi câu khẳng định trong abstract, giới thiệu và kết luận trỏ tới một bảng, hình hoặc định lý. Không có bằng chứng thì xoá câu đó.
2. **Tách quan sát với giải thích:** "Phương pháp đạt X (Bảng 3)" là quan sát. "Điều này *có thể* do Y" là giả thuyết, và cần thí nghiệm để kiểm tra (Lipton & Steinhardt 2019).
3. **Thuật ngữ nhất quán**: một khái niệm một tên. Định nghĩa thuật ngữ gợi ý ("hiểu", "suy luận", "chú ý") trước khi dùng.
4. **Hình và bảng sinh bằng script.** Error bar có định nghĩa trong chú thích (std hay SE, qua bao nhiêu lần chạy). Hình đọc được khi in đen trắng.
5. **Thứ tự viết khuyến nghị:** hình và bảng → kết quả → phương pháp → thực nghiệm → giới thiệu → kết luận → abstract → tiêu đề.
6. **Tiếng Anh:** câu ngắn, chủ động khi có thể. Kiểm tra ngữ pháp bằng công cụ được phép (không cần khai báo AI nếu chỉ sửa chính tả và ngữ pháp, theo đa số nhà xuất bản). Mọi nội dung do AI sinh vượt mức sửa câu chữ phải khai báo ([01 §5](01_NGUYEN_TAC_VA_LIEM_CHINH.md)).

## 3. Chọn tạp chí (G1 và G11)

### 3.1 Tiêu chí
| Tiêu chí | Cách kiểm tra |
|---|---|
| **Phạm vi khớp** | Đọc *Aims & Scope*; tìm ít nhất 3 bài cùng chủ đề đã đăng trong 5 năm. Baseline đăng ở đâu là gợi ý mạnh |
| Chỉ mục | Kiểm tra trực tiếp trên danh sách nguồn của Scopus hoặc Web of Science, không tin website tạp chí |
| Xếp hạng | SJR (Scimago) theo chuyên ngành; JCR nếu cơ quan yêu cầu; CORE với hội nghị |
| Thời gian phản biện, tỉ lệ nhận | Trang tạp chí; bài đăng gần đây (ngày nhận, ngày chấp nhận) |
| Phí và mô hình truy cập | APC (tạp chí OA), hoặc mô hình hybrid hay thuê bao (không phí) |
| Chính sách preprint | Đa số cho phép arXiv; kiểm tra với tạp chí double-blind |
| Yêu cầu trong nước (nếu có) | Danh mục tính điểm của cơ quan hay của Hội đồng Giáo sư nhà nước, nếu cần cho xét chức danh |

### 3.2 Tránh tạp chí săn mồi và tạp chí bị chiếm đoạt
- Dùng checklist **Think. Check. Submit.**
- **Tạp chí bị chiếm đoạt (hijacked):** website giả mạo tạp chí thật. Luôn truy cập tạp chí qua liên kết từ trang nguồn của Scopus hay Web of Science, không qua email mời viết bài. Tra Hijacked Journal Checker của Retraction Watch.
- Dấu hiệu cảnh báo: email mời nộp bài hàng loạt; hứa thời gian phản biện vài ngày; phí cao không minh bạch; ban biên tập không kiểm chứng được; danh mục chủ đề quá rộng.
- Kiểm tra tạp chí có trong danh sách bị Scopus ngừng chỉ mục (*discontinued titles*) hay không.

### 3.3 Khi venue đích là hội nghị hạng A/A* (NeurIPS, ICML, ICLR, KDD, AAAI…)
- **Hạn nộp cố định theo chu kỳ.** Kế hoạch G3 lùi ngược từ hạn nộp (abstract và full paper thường cách nhau khoảng 1 tuần). Kiểm tra ngày chính xác trên trang call for papers của năm đó.
- **Giới hạn trang nghiêm ngặt**; phụ lục và tài liệu bổ sung không bắt buộc reviewer đọc, nên khẳng định chính phải được chứng minh ngay trong bài chính.
- **Double-blind:** không để lộ danh tính trong bài, code, hay tên file. Chính sách preprint arXiv khác nhau giữa các hội nghị, kiểm tra trước.
- **Checklist bắt buộc:** NeurIPS desk reject bài thiếu Paper Checklist. ICML và ICLR có yêu cầu tương tự về reproducibility và tác động.
- **Rebuttal ngắn** (khoảng 1 tuần): chuẩn bị sẵn thí nghiệm bổ sung có khả năng bị hỏi (baseline mạnh hơn, thêm dataset, ablation) trước khi kết quả phản biện về.
- **Reciprocal reviewing:** nhiều hội nghị yêu cầu tác giả nhận phản biện; dự trù thời gian.
- **Chi phí tham dự:** thường cần ít nhất một tác giả đăng ký và trình bày, nhiều khi trực tiếp. Dự trù phí đăng ký, đi lại, visa ngay từ G0.
- **Xếp hạng:** tra hạng hội nghị trên CORE Conference Portal.

### 3.4 Chiến lược bậc thang
Chọn **1 tạp chí đích và 2 dự phòng** ngay từ G1 (tiêu chí C3), theo thứ tự tham vọng giảm dần. Khi bị từ chối, sửa theo ý kiến phản biện rồi chuyển sang tạp chí dự phòng, chứ không nộp lại nguyên bản.

## 4. Gói nộp bài (G11)

- [ ] Bản thảo theo đúng template và giới hạn độ dài của tạp chí.
- [ ] Cover letter: bài toán, đóng góp, **vì sao khớp với tạp chí** (trích bài đã đăng ở đó), khẳng định bài không được xét ở nơi khác, các bài liên quan của chính tác giả.
- [ ] Highlights hoặc graphical abstract nếu tạp chí yêu cầu (Elsevier thường yêu cầu highlights).
- [ ] Đề xuất phản biện (nếu được hỏi): không đồng tác giả trong 3–5 năm, không cùng cơ quan, có chuyên môn đúng.
- [ ] Data & Code Availability: link kho (ẩn danh nếu double-blind).
- [ ] Khai báo: tài trợ, xung đột lợi ích, CRediT, dùng AI, đạo đức hoặc IRB nếu có.
- [ ] Nhật ký tìm kiếm tính mới cập nhật trong vòng 30 ngày.
- [ ] Đã qua [10_CHECKLIST_TRUOC_KHI_NOP.md](10_CHECKLIST_TRUOC_KHI_NOP.md).

## 5. Trả lời phản biện (G12)

Dùng [RESPONSE_TO_REVIEWERS.md](templates/RESPONSE_TO_REVIEWERS.md). Quy tắc:
1. **Trả lời từng ý**, trích nguyên văn ý phản biện, đánh số (R1.1, R1.2…).
2. Mỗi ý có: *Trả lời* → *Thay đổi đã làm* → *Vị trí trong bản sửa* (trang, dòng, mục).
3. **Không bỏ qua ý nào**, kể cả ý không đồng ý.
4. Không đồng ý thì lập luận bằng bằng chứng (thí nghiệm mới, trích dẫn), với giọng lịch sự. Cảm ơn ý đúng.
5. Thí nghiệm bổ sung tuân theo cùng giao thức, ghi rõ là bổ sung sau phản biện.
6. Đánh dấu thay đổi trong bản thảo (màu hoặc track changes) theo yêu cầu của tạp chí.
7. Phản biện yêu cầu trích dẫn bài không liên quan: có thể từ chối lịch sự và báo biên tập viên (COPE coi việc ép trích dẫn là sai phạm).

## 6. Khi bị từ chối

1. Đợi 24 giờ trước khi đọc kỹ.
2. Phân loại lý do:
   - Ngoài phạm vi → đổi tạp chí.
   - Thiếu tính mới → quay G2.
   - Lỗi phương pháp → quay G5 hoặc G6.
   - Thiếu baseline hay dataset → quay G5.
   - Trình bày kém → sửa ở G9.
3. Desk reject (bị từ chối không qua phản biện) thường do phạm vi, chất lượng trình bày, hoặc thiếu đóng góp rõ ràng trong abstract và giới thiệu.
4. Ghi bài học vào `docs/submission_history.md`: tạp chí, ngày, quyết định, lý do, thay đổi.
