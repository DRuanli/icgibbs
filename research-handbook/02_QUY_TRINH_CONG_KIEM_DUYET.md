# 02 · Quy trình 13 cổng kiểm duyệt

[← Về README](README.md)

Mỗi cổng có: **mục tiêu**, **việc làm**, **sản phẩm** (file được commit), **tiêu chí qua cổng**, và **lỗi thường gặp**. Cuối mỗi cổng, một người kiểm duyệt **không tham gia tạo sản phẩm** điền [GATE_REVIEW.md](templates/GATE_REVIEW.md) và quyết định **Qua / Sửa / Dừng**. Người kiểm duyệt có thể là đồng tác giả, người hướng dẫn, hoặc một agent AI được giao vai phản biện độc lập. Nếu là agent, người nghiên cứu vẫn phải tự xác nhận các phát hiện của nó.

Ba nguyên tắc chung:
- **Đóng băng trước khi chạy.** G3 (kế hoạch nghiên cứu) và G5 (giao thức thực nghiệm) được commit vào git *trước* khi chạy thực nghiệm chính.
- **Thay đổi giao thức sau G5 không bị cấm**, nhưng phải ghi vào mục *Deviations* của `EXPERIMENT_PROTOCOL.md`, kèm lý do và ngày. Kết quả từ phần thay đổi được báo cáo là *exploratory*.
- **Tiêu chí dừng được viết trước** (ở G1 và G3), không nghĩ ra sau khi đã tốn công.

---

## G0 · Kiểm kê tài nguyên và ràng buộc

- **Mục tiêu:** biết mình *có gì* trước khi chọn *làm gì*.
- **Việc làm:** liệt kê thời gian, nhân lực, compute, quyền truy cập dữ liệu, kỹ năng, code và bài sẵn có, ngân sách (APC, dịch vụ), hạn chót và yêu cầu của cơ quan.
- **Sản phẩm:** [RESOURCE_INVENTORY.md](templates/RESOURCE_INVENTORY.md)
- **Qua cổng khi:** mọi mục có giá trị cụ thể (con số, tên, ngày), không ghi "chưa rõ" ở compute, thời gian và dữ liệu.
- **Lỗi thường gặp:** chọn đề tài cần GPU lớn hoặc dữ liệu riêng mà không có; đánh giá quá cao thời gian rảnh.

## G1 · Chọn bài toán

- **Mục tiêu:** một bài toán *quan trọng, mới, khả thi với tài nguyên G0*, và thỏa tiêu chí bắt buộc.
- **Việc làm:**
  - Sinh 3–5 ứng viên.
  - Trả lời 8 câu Heilmeier cho từng ứng viên.
  - Chấm điểm theo ma trận ở [03_CHON_DE_TAI.md](03_CHON_DE_TAI.md).
  - Quét tính mới sơ bộ (30–60 phút mỗi ứng viên).
- **Sản phẩm:** một [TOPIC_CARD.md](templates/TOPIC_CARD.md) cho mỗi ứng viên; ứng viên được chọn có tiêu chí dừng.
- **Qua cổng khi:** ứng viên được chọn đạt mọi tiêu chí bắt buộc (C1–C7 ở file 03) và có điểm cao nhất; đã nêu được venue đích (tạp chí hoặc hội nghị, kèm chu kỳ nộp) cùng 2 venue dự phòng.
- **Lỗi thường gặp:** chọn theo phương pháp đang "hot" thay vì theo bài toán; dạng đề tài "áp dụng X cho Y" không có đóng góp phương pháp.

## G2 · Tổng quan tài liệu và kiểm tra tính mới

- **Mục tiêu:** biết chính xác ai đã làm gì, và đóng góp dự kiến khác ở đâu.
- **Việc làm:**
  - Tìm kiếm có nhật ký trên tối thiểu 4 nguồn.
  - Snowballing ngược và xuôi tới bão hoà.
  - Lập ma trận tài liệu liên quan.
  - Viết câu khẳng định tính mới có giới hạn.
  - Quy trình đầy đủ ở [04_TONG_QUAN_TAI_LIEU.md](04_TONG_QUAN_TAI_LIEU.md).
- **Sản phẩm:** [NOVELTY_SEARCH_LOG.md](templates/NOVELTY_SEARCH_LOG.md), `docs/related_work_matrix.csv`
- **Qua cổng khi:**
  - Có ít nhất 3 công trình gần nhất được đọc toàn văn, kèm câu "khác ở đâu" cho từng công trình.
  - Không tìm thấy công trình làm đúng đóng góp dự kiến.
  - Danh sách baseline tiềm năng đạt tiêu chí C2.
- **Dừng khi:** đã có công trình làm đúng đóng góp, và không còn khác biệt nào đủ cho một bài báo.
- **Lỗi thường gặp:** chỉ tìm Google Scholar; không tìm arXiv và OpenReview (công trình mới nhất thường ở đó); không ghi lại câu truy vấn.

## G3 · Câu hỏi nghiên cứu, giả thuyết, đóng góp

- **Mục tiêu:** chuyển đề tài thành các câu hỏi kiểm chứng được, cùng kết quả dự đoán.
- **Việc làm:**
  - Viết 2–4 câu hỏi nghiên cứu (RQ), mỗi câu ứng với một khẳng định trong bài.
  - Với mỗi RQ: giả thuyết, metric, kết quả dự đoán, và kết quả nào sẽ bác bỏ giả thuyết.
  - Phân loại đóng góp theo Shaw (2003) ([05 §1](05_LOAI_HINH_NGHIEN_CUU.md)).
  - Khởi tạo ma trận khẳng định–bằng chứng.
  - Thoả thuận vai trò và thứ tự tác giả.
- **Sản phẩm:** [RESEARCH_PLAN.md](templates/RESEARCH_PLAN.md), [CLAIM_EVIDENCE_MATRIX.md](templates/CLAIM_EVIDENCE_MATRIX.md) (cột bằng chứng còn trống)
- **Qua cổng khi:** mỗi khẳng định dự kiến trong abstract đã có một RQ, một thí nghiệm hoặc định lý dự kiến, và một tiêu chí bác bỏ.
- **Lỗi thường gặp:** RQ chung chung kiểu "phương pháp có tốt không?"; không có kết quả dự đoán, nên kết quả nào cũng "xác nhận" được.

## G4 · Phát triển phương pháp hoặc lý thuyết

- **Mục tiêu:** phương pháp và lý thuyết đúng *trước khi* tối ưu hiệu năng.
- **Việc làm:**
  - Viết đặc tả: input, output, giả định, độ phức tạp.
  - Chứng minh nháp.
  - Kiểm chứng định lý bằng số trên các ví dụ nhỏ.
  - Chạy thử trên một ví dụ đồ chơi có đáp án đã biết.
- **Sản phẩm:** `docs/method_spec.md`, `docs/proofs.md` (hoặc LaTeX), notebook kiểm chứng.
- **Qua cổng khi:**
  - Mỗi định lý có giả định đầy đủ và chứng minh hoàn chỉnh.
  - Mỗi bổ đề đã qua kiểm chứng số (xem [07](07_LY_THUYET_VA_CHUNG_MINH.md)).
  - Phương pháp cho đáp án đúng trên ví dụ đồ chơi.
- **Lỗi thường gặp:** "mathiness" (công thức trang trí không gánh lập luận); độ phức tạp chỉ đo bằng thời gian chạy chứ không chứng minh.

## G5 · Thiết kế và đóng băng giao thức thực nghiệm

- **Mục tiêu:** quyết định mọi thứ về đánh giá *trước khi* thấy kết quả.
- **Việc làm:** chốt
  - dataset (≥10 bộ độc lập, có [DATASET_CARD](templates/DATASET_CARD.md));
  - baseline (≥2 đã được peer-review, cộng baseline đơn giản);
  - metric, cách chia dữ liệu, ngân sách tune cho *mọi* phương pháp;
  - số seed, kế hoạch phân tích thống kê, ablation, phân tích độ nhạy;
  - phần cứng.

  Chi tiết ở [06](06_THIET_KE_THUC_NGHIEM.md).
- **Sản phẩm:** [EXPERIMENT_PROTOCOL.md](templates/EXPERIMENT_PROTOCOL.md) được commit, ghi ngày đóng băng.
- **Qua cổng khi:** người kiểm duyệt xác nhận checklist rò rỉ dữ liệu (8 loại) và checklist công bằng baseline đều đạt.
- **Lỗi thường gặp:** chọn dataset sau khi thử; tune phương pháp mình 200 lần nhưng chạy baseline mặc định.

## G6 · Cài đặt và kiểm chứng code

- **Mục tiêu:** code làm đúng điều bài báo nói.
- **Việc làm:**
  - Cấu trúc repo theo [08](08_TAI_LAP_VA_MA_NGUON.md).
  - Viết test theo thứ tự: shape/dtype → đáp án biết trước → kiểm chứng số (gradient check, so với thư viện chuẩn) → bất biến.
  - Tái hiện số liệu đã công bố của từng baseline trên dataset gốc của baseline đó.
- **Sản phẩm:** repo có test chạy xanh; `docs/baseline_reproduction.md` (số công bố vs số tái hiện, sai lệch, giải thích).
- **Qua cổng khi:**
  - Mọi test qua.
  - Mỗi baseline tái hiện trong khoảng sai lệch chấp nhận (ghi trước, ví dụ ±1 điểm %), hoặc chênh lệch được giải thích và xếp loại.
  - Người kiểm duyệt chạy được từ clone sạch.
- **Dừng hoặc quay G4 khi:** phương pháp sai trên ví dụ đồ chơi.

## G7 · Chạy thực nghiệm

- **Mục tiêu:** chạy đúng giao thức đã đóng băng và ghi lại mọi lần chạy.
- **Việc làm:**
  - Mỗi lần chạy lưu: commit hash, config, seed, môi trường, phần cứng, thời gian, kết quả thô (theo fold và seed).
  - Không xoá lần chạy thất bại. Chạy lại phải có lý do được ghi.
- **Sản phẩm:** [EXPERIMENT_LOG.md](templates/EXPERIMENT_LOG.md), `results/raw/` (không sửa tay).
- **Qua cổng khi:** mọi ô (dataset × phương pháp × fold × seed) trong giao thức đều có kết quả, hoặc có lý do ghi rõ (timeout, lỗi bộ nhớ, được báo cáo trong bài).

## G8 · Phân tích và đối chiếu khẳng định

- **Mục tiêu:** biết dữ liệu thật sự ủng hộ điều gì.
- **Việc làm:**
  - Tính bảng và hình bằng script từ `results/raw/`.
  - Phân tích thống kê đúng kế hoạch G5.
  - Chạy ablation và phân tích độ nhạy.
  - Phân tích lỗi: phương pháp thua ở đâu, vì sao.
  - Điền cột bằng chứng của ma trận khẳng định.
- **Sản phẩm:** `analysis/` (script tái tạo mọi bảng và hình), ma trận khẳng định–bằng chứng đã điền.
- **Qua cổng khi:** mọi khẳng định dự kiến được đánh dấu *ủng hộ / ủng hộ một phần / không ủng hộ*. Khẳng định không được ủng hộ phải bị **bỏ hoặc thu hẹp**.
- **Lỗi thường gặp:** chỉ báo cáo trung bình; giấu dataset thua; diễn giải p > 0,05 là "tương đương".

## G9 · Viết bản thảo

- **Mục tiêu:** bản thảo mà mỗi câu khẳng định trỏ được tới bằng chứng.
- **Việc làm:** theo cấu trúc và quy tắc ở [09](09_VIET_BAI_VA_NOP_BAI.md). Viết theo thứ tự: hình và bảng → kết quả → phương pháp → giới thiệu → abstract → tiêu đề.
- **Sản phẩm:** bản thảo LaTeX theo template của tạp chí đích; phần Data & Code Availability; khai báo AI, tài trợ, xung đột lợi ích, CRediT.
- **Qua cổng khi:** abstract và phần giới thiệu không khẳng định điều gì vượt ma trận khẳng định; phần Limitations có nội dung thật.

## G10 · Kiểm duyệt nội bộ độc lập

- **Mục tiêu:** tìm lỗi *trước* phản biện của tạp chí.
- **Việc làm:**
  - Ít nhất một người hoặc agent chưa tham gia viết đọc như reviewer khó tính.
  - Chạy [10_CHECKLIST_TRUOC_KHI_NOP.md](10_CHECKLIST_TRUOC_KHI_NOP.md).
  - Tái lập một bảng chính từ clone sạch.
  - Đối chiếu từng trích dẫn với bản gốc.
  - Kiểm tra đạo văn bằng công cụ của cơ quan.
- **Sản phẩm:** `GATE_REVIEW.md` của G10, danh sách lỗi phân loại Nghiêm trọng / Lớn / Nhỏ, kèm trạng thái sửa.
- **Qua cổng khi:** không còn lỗi Nghiêm trọng hay Lớn chưa xử lý.

## G11 · Chọn tạp chí và nộp bài

- **Mục tiêu:** nộp đúng nơi, đúng định dạng, đủ hồ sơ.
- **Việc làm:**
  - Kiểm tra lại độ phù hợp với phạm vi tạp chí và chỉ mục (Scopus/WoS).
  - Kiểm tra dấu hiệu tạp chí săn mồi hoặc bị chiếm đoạt (hijacked).
  - Chuẩn bị cover letter, highlights, đề xuất phản biện không có xung đột lợi ích.
  - Đưa artifact lên kho lưu trữ (ẩn danh nếu double-blind).
  - Chạy lại tìm kiếm tính mới. Nhật ký tìm kiếm phải có ngày không quá 30 ngày trước ngày nộp.
- **Sản phẩm:** gói nộp bài (`submission/`), bản cập nhật của `NOVELTY_SEARCH_LOG.md`.

## G12 · Phản biện, sửa bài và lưu trữ

- **Việc làm:**
  - Trả lời từng ý phản biện theo [RESPONSE_TO_REVIEWERS.md](templates/RESPONSE_TO_REVIEWERS.md).
  - Thí nghiệm bổ sung tuân theo cùng giao thức, ghi là bổ sung sau phản biện.
  - Khi được chấp nhận: lưu trữ artifact có DOI (Zenodo), cập nhật README với trích dẫn chính thức.
  - Khi bị từ chối: phân loại lý do, sửa, chọn tạp chí dự phòng, quay lại G11. Không nộp lại nguyên bản.
- **Sau xuất bản:** phát hiện lỗi thì liên hệ tạp chí để đính chính (erratum/corrigendum). Trả lời trung thực các yêu cầu tái lập.

---

## Phân bổ thời gian (quy tắc kinh nghiệm, không phải chuẩn đã công bố)

Với một bài tạp chí khoảng 6–9 tháng: G0–G3 chiếm 15–20%, G4–G6 30–35%, G7–G8 20–25%, G9–G11 15–20%. G12 nằm ngoài tính toán vì phụ thuộc tạp chí. Dự án mà G0–G3 chiếm dưới 10% thời gian thường phải quay lại G1 ở giữa chừng.
