# 07 · Lý thuyết và chứng minh (G4)

[← Về README](README.md)

## 1. Chuẩn trình bày (NeurIPS Checklist Q3 và thực hành của JMLR)

1. **Bảng ký hiệu** ở đầu phần lý thuyết. Mỗi ký hiệu chỉ có một nghĩa trong toàn bài.
2. **Định nghĩa trước khi dùng**, đánh số định nghĩa.
3. **Mọi giả định nằm trong phát biểu định lý**, hoặc được tham chiếu bằng số (Giả định A1, A2). Không có giả định ẩn trong chứng minh.
4. **Chứng minh đầy đủ** trong bài hoặc phụ lục, cộng *phác thảo chứng minh* trong bài chính.
5. **Lập luận phi hình thức** trong bài chính phải có chứng minh hình thức tương ứng ở phụ lục.
6. **Phân biệt** định lý, bổ đề, mệnh đề, hệ quả, và *giả thuyết* (conjecture: điều chưa chứng minh được).

## 2. Mỗi định lý phải trả lời

| Câu hỏi | Vì sao |
|---|---|
| Giả định nào là cần thiết? Bỏ đi thì sao? | Cho một phản ví dụ khi bỏ từng giả định quan trọng |
| Chặn có chặt không? | Ví dụ đạt được chặn, hoặc chặn dưới, hoặc thực nghiệm cho thấy khoảng cách |
| Trường hợp riêng nào ra kết quả đã biết? | Kiểm tra tính nhất quán với tài liệu |
| Định lý nói gì về thực hành? | Liên kết với thiết kế thuật toán hoặc với thí nghiệm trong bài |

## 3. Kiểm chứng trước khi tin (bắt buộc ở G4)

1. **Kiểm chứng số từng bổ đề:** sinh ngẫu nhiên hàng nghìn trường hợp nhỏ, kiểm tra bất đẳng thức hay đẳng thức bằng code. Một trường hợp vi phạm là đủ bác bỏ.
2. **Săn phản ví dụ có chủ đích:** trường hợp biên (tập rỗng, một phần tử, giá trị bằng 0, phân phối suy biến), và tìm kiếm tối ưu hoá để *cực đại hoá* mức vi phạm.
3. **Kiểm tra thứ nguyên và đơn vị:** hai vế cùng đơn vị, cùng bậc theo n.
4. **Kiểm tra độ phức tạp bằng hai cách:** chứng minh bằng giải tích, *và* vẽ thời gian chạy theo n trên trục log–log để so độ dốc với bậc lý thuyết.
5. **Đọc chéo độc lập:** một người không viết chứng minh đọc lại từng dòng và đánh dấu những bước "hiển nhiên" chưa được giải thích.
6. **Tuỳ chọn, cho bổ đề then chốt:** hình thức hoá bằng trợ lý chứng minh (Lean, Isabelle, Coq).

## 4. Độ phức tạp

- Phát biểu theo **mọi** tham số liên quan (n, d, k, kích thước đầu ra), không chỉ n.
- Phân biệt trường hợp xấu nhất, trung bình (nêu rõ phân phối), và chi phí khấu hao.
- Thuật toán khai phá mẫu: độ phức tạp thường phụ thuộc vào **kích thước đầu ra** (output-sensitive). Nêu rõ điều này.
- Bộ nhớ: tách riêng bộ nhớ làm việc và bộ nhớ đầu ra.

## 5. Lỗi thường gặp

1. Chứng minh dùng một giả định không có trong phát biểu, ví dụ tính độc lập hay tính bị chặn.
2. Chặn trên là đúng nhưng quá lỏng để có ý nghĩa, và bài không thảo luận điều đó.
3. "Mathiness": định lý không liên quan đến phương pháp thật sự được dùng trong thực nghiệm (ví dụ chứng minh cho phiên bản lý tưởng hoá, nhưng thực nghiệm dùng một heuristic khác).
4. Đổi thứ tự giới hạn, kỳ vọng và tích phân mà không kiểm tra điều kiện.
5. Ký hiệu trùng: một chữ dùng cho hai đại lượng ở hai mục khác nhau.
