# Bản thảo bài In-Context Gibbs (G9)

Kế hoạch: [G9/G9_PLAN.md](../G9/G9_PLAN.md). Viết theo thứ tự của handbook 09 §2.5: hình và bảng → kết quả → phương pháp → thực nghiệm → giới thiệu → kết luận → abstract → tiêu đề.

- **Đích hiện tại (14/11/2026): IEEE TPAMI.** `main.tex` (bài, mẫu IEEEtran compsoc hai cột) và `supplement.tex` (mọi phụ lục, một tệp riêng theo quy định của IEEE CS). Hai tệp tham chiếu chéo qua `xr-hyper`. Bản TMLR là commit `0b425e2`.
- **Dựng:** `make` (cần Python có pandas, matplotlib; TeX Live). `make data` tạo lại mọi con số, bảng, hình; `make pdf` chạy `scripts/build_pdf.sh` (biên dịch xen kẽ hai tệp); `make submission` tạo `_submission/source.zip`, `main.pdf`, `supplement.pdf` (dừng nếu còn `[TODO:`) và `prereg_record.zip`.
- **Số liệu:** không gõ tay. Mọi con số trong bài lấy từ `numbers.tex` (macro `\val{...}`) và `tables/*.tex`. Các tệp này do `scripts/make_tables.py` sinh từ `G8/results/` và `S2/results/`, là kết quả của các script phân tích đã đăng ký. Bảng phụ lục do `scripts/make_appendix.py` sinh (kể cả số RQ1 từ `G4/results/rq1`). Hình do `scripts/make_figures.py` sinh từ kết quả thô.
- **Template:** `IEEEtran.cls` V1.8b và `IEEEtranN.bst`, bản gốc không sửa (`style_ieee/SOURCE.txt`). Các style cũ: `style_tmlr/`, `neurips_placeholder.sty`.
- **Trỏ khẳng định:** `\claim{6}` in số dòng của [CLAIM_EVIDENCE_MATRIX](../G3/CLAIM_EVIDENCE_MATRIX.md) ở bản nháp; `\finaltrue` trong `main.tex` để ẩn. `\todo{...}` là việc còn lại.
- **Tài liệu tham khảo:** `refs.bib` chỉ gồm 32 tài liệu được trích, đã đối chiếu ngày 04/11/2026 ([G9/REFERENCE_CHECK](../G9/REFERENCE_CHECK.md)). `refs_candidates.bib` là các tài liệu khác của G2, **chưa đối chiếu**; trích thêm thì phải đối chiếu trước.
- **Ẩn danh:** TPAMI phản biện single-anonymous theo mặc định, nên bài ghi tên tác giả. Nếu xin double-anonymous thì chạy `python3 scripts/make_submission_source.py --anonymous --forbid <tên>` và bỏ tên khỏi `main.tex`.
