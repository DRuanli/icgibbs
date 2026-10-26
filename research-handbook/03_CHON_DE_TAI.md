# 03 · Chọn đề tài (G0–G1)

[← Về README](README.md)

Chọn đề tài là quyết định đắt nhất của cả dự án: mọi lỗi ở đây được trả giá ở G5–G12. Quy trình: **kiểm kê tài nguyên → sinh ứng viên → lọc bằng tiêu chí bắt buộc → chấm điểm → quét tính mới sơ bộ → chọn và viết tiêu chí dừng.**

## 1. Bắt đầu từ tài nguyên, không bắt đầu từ ý tưởng

Điền [RESOURCE_INVENTORY.md](templates/RESOURCE_INVENTORY.md) trước. Năm câu hỏi loại bỏ nhanh nhất:
1. Có GPU không, loại nào, bao nhiêu giờ mỗi tuần?
2. Có quyền truy cập dữ liệu riêng nào không, và đã có văn bản cho phép chưa?
3. Có bao nhiêu giờ mỗi tuần cho dự án, và hạn chót là khi nào?
4. Có ai đồng hành (hướng dẫn, đồng tác giả, chuyên gia lĩnh vực)?
5. Có ngân sách phí xuất bản (APC) không? Quyết định này giới hạn lựa chọn giữa tạp chí truy cập mở và tạp chí thuê bao. Nếu nhắm hội nghị: có ngân sách đăng ký, đi lại, visa không?

## 2. Nguồn sinh ứng viên

| Nguồn | Cách khai thác |
|---|---|
| Mục *Limitations* và *Future work* của bài mới ở venue mạnh (1–2 năm gần nhất) | Tác giả tự nêu khoảng trống; ghi trích dẫn nguyên văn vào TOPIC_CARD |
| Issue trên GitHub của thư viện phổ biến | Nhu cầu thật chưa được đáp ứng; ví dụ issue "multi-output chưa hỗ trợ" của TabPFN |
| Mâu thuẫn giữa các bài | Hai bài báo cáo kết quả ngược nhau trên cùng bài toán |
| Đóng góp sẵn có của chính mình | Mở rộng có nguyên tắc từ bài cũ, nhưng không chia nhỏ một kết quả (xem file 01 §3) |
| Chuyển phương pháp sang *loại bài toán* mới | Khác với "áp dụng X cho dataset Y": cần thay đổi phương pháp hay lý thuyết để giải bài toán mới |

## 3. Tiêu chí bắt buộc (lọc cứng)

Ứng viên trượt **bất kỳ** tiêu chí nào thì bị loại, không chấm điểm tiếp.

| Mã | Tiêu chí | Cách kiểm tra |
|---|---|---|
| **C1** | **Dữ liệu đủ: ≥10 bộ dữ liệu độc lập**, truy cập được ngay hoặc đã có văn bản cho phép | Liệt kê tên, nguồn, kích thước, giấy phép. *Độc lập* nghĩa là khác nguồn thu thập: hai phiên bản hay hai mẫu con của cùng một nguồn chỉ tính là 1 (ví dụ atp1d/atp7d, rf1/rf2 trong Mulan). Loại trùng lặp giữa các kho |
| **C2** | **≥2 baseline đã qua peer-review** cho *đúng* bài toán, có code hoặc mô tả đủ để cài đặt | Ghi venue, năm, link code. Preprint chưa phản biện có thể thêm vào nhưng không được tính vào con số 2 |
| **C3** | **Có venue đích rõ ràng** (tạp chí hoặc hội nghị) cùng 2 venue dự phòng | Đích phải đã đăng ít nhất 3 bài cùng chủ đề trong 5 năm gần nhất; các baseline ở C2 đăng ở đó là dấu hiệu tốt. Với hội nghị: ghi rõ chu kỳ (hạn nộp) nhắm tới và hạng CORE |
| **C4** | **Xác định mảng cần đi sâu**: data-centric AI, data science, deep learning, lý thuyết… và so với kỹ năng hiện có | Ghi rõ "cần học thêm X, khoảng N tuần". Nếu N vượt 30% thời gian dự án thì ghi là rủi ro |
| **C5** | **Không thuộc danh mục loại trừ:** suy luận nhân quả (causal inference), fairness, kiểm định hay thẩm định (statistical testing/verification) **như là chủ đề nghiên cứu** | Đề tài không được có đóng góp chính thuộc ba mảng này. *Lưu ý:* kiểm định thống kê vẫn **bắt buộc dùng làm công cụ** đánh giá kết quả ([06 §6](06_THIET_KE_THUC_NGHIEM.md)); C5 chỉ loại bỏ nó như là *chủ đề* |
| **C6** | **Khả thi với tài nguyên G0**: compute, thời gian, dữ liệu | Ước lượng compute cho toàn bộ giao thức, nhân 2 cho các lần chạy thử và sửa lỗi |
| **C7** | **Không có rào cản đạo đức hay pháp lý chưa giải quyết** | Dữ liệu cá nhân có cơ sở pháp lý ([01 §6](01_NGUYEN_TAC_VA_LIEM_CHINH.md)) |

## 4. Tám câu Heilmeier (DARPA)

Mỗi ứng viên phải trả lời bằng ngôn ngữ không chuyên môn:
1. Bạn đang cố làm gì? Nói rõ mục tiêu, tuyệt đối không dùng biệt ngữ.
2. Hiện nay người ta làm thế nào, và giới hạn của cách làm hiện tại là gì?
3. Điều gì mới trong cách tiếp cận của bạn, và vì sao bạn nghĩ nó sẽ thành công?
4. Ai quan tâm? Nếu thành công, nó tạo ra khác biệt gì?
5. Rủi ro là gì?
6. Tốn bao nhiêu?
7. Mất bao lâu?
8. "Bài kiểm tra" giữa kỳ và cuối kỳ để biết thành công là gì?

Câu 8 chính là tiêu chí dừng và các mốc ở G3.

## 5. Ma trận chấm điểm (chỉ với ứng viên đã qua C1–C7)

Chấm từ 1 đến 5, nhân trọng số. Trọng số là mặc định và có thể chỉnh, nhưng phải chỉnh **trước** khi chấm.

| Tiêu chí | Trọng số | 1 điểm | 5 điểm |
|---|---|---|---|
| Tính mới (sau quét sơ bộ) | 0,25 | Đã có bài gần như trùng | Không thấy bài nào trong chỗ giao giữa các hướng; có bằng chứng nhu cầu (issue, limitation của người khác) |
| Tầm quan trọng | 0,20 | Cải thiện nhỏ trên benchmark hẹp | Mở ra loại bài toán mới hoặc gỡ một giới hạn đã được công nhận |
| Khả năng có đóng góp lý thuyết | 0,15 | Chỉ có thực nghiệm | Có định lý mạnh, kiểm chứng được |
| Độ khớp với kỹ năng và bài trước | 0,15 | Hoàn toàn mới | Dùng trực tiếp kinh nghiệm sẵn có |
| Rủi ro bị đi trước (scoop) | 0,10 | Nhiều nhóm mạnh đang làm | Ít người làm, hoặc có lợi thế riêng (dữ liệu, kỹ thuật) |
| Có phương án dự phòng | 0,10 | Thất bại là mất trắng | Kết quả âm vẫn đăng được, hoặc có bài nhỏ hơn làm phương án B |
| Độ khớp với tạp chí đích | 0,05 | Lệch phạm vi | Khớp đúng chủ đề tạp chí đang đăng |

## 6. Phân loại đóng góp (Shaw 2003)

Ghi rõ trong TOPIC_CARD bài báo *trả lời loại câu hỏi nào*, *tạo ra loại kết quả nào*, và *kiểm chứng bằng gì*. Chi tiết ở [05 §1](05_LOAI_HINH_NGHIEN_CUU.md). Phản biện thường bác bài khi loại kiểm chứng không tương xứng với loại khẳng định, ví dụ khẳng định "tổng quát" nhưng chỉ có một ví dụ minh hoạ.

## 7. Tiêu chí dừng (viết trước, ở G1)

Ví dụ tiêu chí dừng tốt, đều đo được:
- "Nếu ở G2 tìm thấy công trình làm đúng đóng góp X trên cùng loại bài toán: dừng hoặc thu hẹp sang khác biệt Y."
- "Nếu đến tuần 8 phương pháp không vượt baseline mạnh nhất trên ít nhất 6/10 dataset theo metric chính: chuyển sang phương án B (bài phân tích thực nghiệm)."
- "Nếu không xin được quyền truy cập dữ liệu trước ngày D: thay dataset theo danh sách dự phòng."

## 8. Lỗi thường gặp khi chọn đề tài

1. Chọn phương pháp trước rồi mới đi tìm bài toán cho nó.
2. Đề tài "áp dụng mô hình M lên dữ liệu D" không có thay đổi phương pháp. Tạp chí phương pháp sẽ từ chối; tạp chí ứng dụng cần đóng góp về lĩnh vực.
3. Không kiểm tra C1, C2 trước, rồi đến G5 mới phát hiện thiếu dữ liệu hoặc không có baseline.
4. Đánh giá tính mới chỉ dựa trên cảm giác "chưa thấy ai làm" mà không có nhật ký tìm kiếm.
5. Đề tài quá rộng cho một bài báo. Một bài nên có 1 đóng góp chính và tối đa 2–3 đóng góp phụ.
