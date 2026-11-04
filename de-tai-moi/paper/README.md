# Bản thảo bài In-Context Gibbs (G9)

Kế hoạch: [G9/G9_PLAN.md](../G9/G9_PLAN.md). Viết theo thứ tự của handbook 09 §2.5: hình và bảng → kết quả → phương pháp → thực nghiệm → giới thiệu → kết luận → abstract → tiêu đề.

- **Dựng:** `make` (cần Python có pandas, matplotlib; TeX Live có latexmk). `make data` tạo lại mọi con số, bảng, hình; `make pdf` chỉ biên dịch.
- **Số liệu:** không gõ tay. Mọi con số trong bài lấy từ `numbers.tex` (macro `\val{...}`) và `tables/*.tex`. Các tệp này do `scripts/make_tables.py` sinh từ `G8/results/` và `S2/results/`, là kết quả của các script phân tích đã đăng ký. Bảng phụ lục do `scripts/make_appendix.py` sinh (kể cả số RQ1 từ `G4/results/rq1`). Hình do `scripts/make_figures.py` sinh từ kết quả thô.
- **Template:** `neurips_placeholder.sty` chỉ mô phỏng khổ trang NeurIPS (5,5 × 9 inch, Times 10 pt). Thay bằng style chính thức NeurIPS 2027 khi có (G9_PLAN O5).
- **Trỏ khẳng định:** `\claim{6}` in số dòng của [CLAIM_EVIDENCE_MATRIX](../G3/CLAIM_EVIDENCE_MATRIX.md) ở bản nháp; `\finaltrue` trong `main.tex` để ẩn. `\todo{...}` là việc còn lại.
- **Tài liệu tham khảo:** `refs.bib` chỉ gồm 32 tài liệu được trích, đã đối chiếu ngày 04/11/2026 ([G9/REFERENCE_CHECK](../G9/REFERENCE_CHECK.md)). `refs_candidates.bib` là các tài liệu khác của G2, **chưa đối chiếu**; trích thêm thì phải đối chiếu trước.
- **Ẩn danh:** bài nộp double-blind. Không ghi tên, cơ quan, đường dẫn kho thật trong bài.
