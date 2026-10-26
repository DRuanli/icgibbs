# Phản hồi kiểm duyệt độc lập phần toán (G6, việc T7) — 26/10/2026

**Cách kiểm duyệt (thầy giao Claude chọn):** một phiên riêng, không thấy quá trình soạn, đọc `G4/PROOFS.md` ở commit `c08e02b` theo quy trình phản biện toán (từng định nghĩa, giả định, bước chứng minh; săn phản ví dụ; kiểm số). Người kiểm duyệt không đọc bản tự rà soát trước khi viết xong nhận xét. Báo cáo gốc: [PROOFS_REVIEW_G6.md](PROOFS_REVIEW_G6.md).

**Kết quả:** 0 Critical, 4 Major, 12 Minor. Không định lý hay mệnh đề nào sai như đã phát biểu, trừ Hệ quả 3.1 khi có nhiều điểm test. Các vấn đề Major nằm ở phạm vi áp dụng, ở định vị đóng góp và ở văn xuôi nói quá.

**Cách xử lý:** mỗi phát hiện được kiểm lại độc lập bằng code riêng (`G4/code/experiments/g6_review_checks.py`, không dùng script của người kiểm duyệt) hoặc bằng đối chiếu nguồn, rồi mới sửa. Mọi chỗ sửa trong PROOFS được đánh dấu "sau kiểm duyệt" kèm mã phát hiện.

| # | Mức | Phát hiện (tóm tắt) | Kiểm lại độc lập | Đồng ý? | Đã làm |
|---|---|---|---|---|---|
| F1 | Major | Cấu hình thực nghiệm thật sự dùng (khối K = 10; bag-block cho LLP) không có đảm bảo khác rỗng; bag-block xấp xỉ chưa đo; chặn 4(b) tăng theo n | TV(π̃, π) của bag-block xấp xỉ trên bài chính xác LLP 3 × 4: 0–0,52 (người kiểm duyệt 0,16–0,64). Chặn O(n/K²) và κ_B ≈ 10^16 (túi 8) kiểm bằng lập luận | Có | Viết lại "Giới hạn của 4(b)"; nói rõ Định lý 1 và Mệnh đề 3 chỉ áp cho sơ đồ một mẫu, hoán đổi, khối có sửa MH; thêm việc mở **G4-5** với kế hoạch đo (tập phát triển: bag-block xấp xỉ so với có sửa MH ở LLP túi 8; tập đánh giá: ablation có sẵn ở giao thức Mục 7); METHOD_SPEC ghi chú |
| F2 | Major | Hệ quả 3.1 sai với chuyển nạp nhiều điểm test (đúng chế độ `fit_transductive`) | Liệt kê chính xác, hỗn hợp 3 thành phần, 30 prior: chênh tới 0,172 (trung vị 0,012) | Có | Thêm Mệnh đề 3′ cho f(X, W, X_*); viết lại Hệ quả 3.1 thành ba trường hợp (một điểm; nhiều điểm; quy nạp với prior tách được) |
| F3 | Major | Chặn chính của Định lý 2 là trường hợp yếu hơn của chặn nhiễu loạn kinh điển (π̃ − π = π̃EZ; τ₁(Z)) | Đồng nhất thức đúng tới 4e-15; τ₁(Z) ≤ τ(P) ở 200/200 chuỗi | Có | Thêm đoạn định vị: chặn chính không mới về dạng; đóng góp là Bước 0 (duy nhất khi không tương thích), các ước lượng ε cho nhân dựng từ TFM, Mệnh đề 4(b). Thêm trích dẫn Schweitzer 1968, Meyer 1980, Seneta 1988, Cho & Meyer 2001 (số trang cần kiểm, G4-6) |
| F4 | Major | Văn xuôi §7 dùng "nhất quán" quá điều đã chứng minh; "chỉ PU-SAR" mâu thuẫn với RQ5; SSL xếp loại 1 không khớp phần áp dụng | Đọc lại §7: đúng | Có | Viết lại: "mục tiêu quần thể của bộ học từng điểm xếp hạng đúng"; "chuỗi có phân phối dừng là posterior dưới prior của TFM" thay cho "Gibbs nhất quán"; giới hạn ở bối cảnh tổng hợp và thêm câu về RQ5; SSL kèm điều kiện prior tách được |
| F5 | Minor | "P_sw nói chung không khả nghịch" sai — dưới (A1) nó khả nghịch | Sai số cân bằng chi tiết ≤ 2e-16 (quét cố định: 1e-4–2e-2) | Có | Sửa §2(a) |
| F6 | Minor | Nhận xét 4.1: δ(P^t) = 1 chỉ với t < \|F\|/2; số "2–15 lần", "thêm 10 lần" lệch RQ1 | Bước đầu có δ < 1: 1, 2, 2, 3, 3 với \|F\| = 2..6; bảng E3 cho 15–44 lần | Có | Sửa Nhận xét 4.1 (τ ≥ ⌈\|F\|/2⌉; 2,2–12 lần; 15–44 lần) và RQ1_REPORT |
| F7 | Minor | Hệ quả 1.1 viết vòng; câu "suy từ softmax" không phủ ngữ cảnh ngoài supp(π) | Đọc lại: đúng | Có | Viết chứng minh quy nạp; (A2) thành giả định về prior |
| F8 | Minor | (A0) thiếu điều kiện cho ℓ của SSL, LLP; bộ sinh SSL của giao thức vi phạm nhẹ | Ví dụ 2/30 so với 14/270: đúng | Có | Thêm điều kiện vào (A0); ghi là sai lệch mô hình nhỏ đã biết của bối cảnh phụ SSL (không đổi bộ sinh đã đóng băng) |
| F9 | Minor | Định lý 1(c)(3) cho LLP và công thức Rao–Blackwell khi m ≥ 2 | Đọc lại: đúng | Có | Bổ sung hai câu |
| F10 | Minor | "chỉ đúng khi" và điều kiện tách được dùng không nhất quán; "O(·)" rỗng | Phản ví dụ W xác định y: đúng | Có | Thống nhất "điều kiện đủ: prior tách được"; thay O(·) bằng chặn TV |
| F11 | Minor | 4(b): nhân phân tích khác nhân cài đặt; Bổ đề 1 mở rộng chưa viết; tham số phiền nhiễu | Đọc lại: đúng | Có | Thêm ghi chú phạm vi ba điểm |
| F12 | Minor | Quan sát 4(c): "O(1/n)" không được dữ liệu ủng hộ | Độ dốc log–log −0,5 đến −0,8 | Có | Sửa câu |
| F13 | Minor | ε̄ định nghĩa trước Bước 0; trường hợp ∞·0 | Đọc lại: đúng | Có | Thêm câu ở Bước 0 |
| F14 | Minor | Nhận xét 4.2: "đúng bậc" là suy diễn chéo bối cảnh | Hai định nghĩa sai số khác nhau: đúng | Có | Thay bằng lập luận tự đủ (dấu bằng trên chuỗi 2 trạng thái); sửa RQ1_REPORT |
| F15 | Minor | Ký hiệu trùng (π, p, q, B, C, …) | — | Có | §7: ϖ cho tỉ lệ lớp, N cho biến nhị thức, ghi chú về q; C_erg cho hằng số Mitrophanov |
| F16 | Minor | G4-1 dễ đóng hơn; vấn đề thật là định danh yếu của η ở PU-SAR | Đọc lại: đúng | Có | Viết lại §2(c) và G4-1 |

**Trích dẫn:** G6-T7a đã đóng — Agarwal (2014), JMLR 15(49):1653–1674, Hệ quả 12 (đối chiếu trên bản arXiv 1207.0268). Các trích dẫn còn cần đối chiếu gom vào việc mở G4-6.

**Ảnh hưởng tới khẳng định của bài:**
1. Lý thuyết (Định lý 1, Mệnh đề 3) không phủ cấu hình thực nghiệm mặc định. Bài phải nói rõ điều này, và trình bày khẳng định chính LLP túi 8 như một kết quả thực nghiệm của một sơ đồ xấp xỉ, kèm đo sai lệch theo G4-5.
2. Định lý 2 trình bày như một bổ đề nhiễu loạn đã biết, áp vào nhân dựng từ TFM; không quảng bá là đóng góp lý thuyết mới về dạng chặn.
3. Mệnh đề 5–6 (PROOFS §7) đứng vững; chỉ phần văn xuôi được thu hẹp.
