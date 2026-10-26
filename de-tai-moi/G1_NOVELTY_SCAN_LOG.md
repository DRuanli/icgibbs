# Nhật ký quét tính mới sơ bộ (G1), ngày 24/10/2026

> Đây là quét **sơ bộ** của G1 (vài truy vấn mỗi ứng viên, qua công cụ tìm kiếm web và trang arXiv/OpenReview/GitHub). Quét đầy đủ theo [04_TONG_QUAN_TAI_LIEU.md](../research-handbook/04_TONG_QUAN_TAI_LIEU.md) (Scopus, IEEE Xplore, ACM DL, DBLP, snowballing) thuộc G2, chỉ làm cho ứng viên được chọn.

| # | Ứng viên | Truy vấn (tóm tắt) | Công trình gần nhất tìm được | Kết luận sơ bộ |
|---|---|---|---|---|
| A | Học từ nhãn yếu bằng TFM, không huấn luyện lại | "learning from label proportions" + TabPFN / tabular foundation model / in-context; "weak supervision" + TabPFN + EM; "positive-unlabeled" + PFN; "partial label" + PFN; Gibbs / EM + prior-data fitted network | **In-Context PU Learning** (Liu et al., arXiv 2605.05591, 2026): pretrain một PFN riêng cho PU, prior chỉ mã hoá SCAR, ghi SAR là future work, không có xác minh hay quyết định. **GLWS** (Chen et al., ICML 2024) và **ILL** (Chen et al., NeurIPS 2024): khung EM thống nhất cho nhiều loại nhãn yếu, nhưng dùng mạng huấn luyện bằng gradient. **Noise Immunity** (arXiv 2604.04868): khảo sát thực nghiệm độ bền của TabPFN với nhãn lật đối xứng. **TICDA** (arXiv 2610.07996): phát hiện lỗi nhãn, không học từ nhãn yếu. Không thấy công trình LLP, PLL, hay khung nhãn yếu tổng quát nào dùng TFM | **Còn trống** ở chỗ giao: TFM có sẵn + suy luận Bayes trên nhãn ẩn + nhiều loại nhãn yếu |
| B | Dự đoán đa nhãn Bayes-optimal bằng phân phối đồng thời của TFM | "TabPFN multilabel classifier chain"; "multi-label in-context learning tabular" | **Coarse composition suffices** (arXiv 2608.30337, 2026): dùng TabPFN với BR, LP, PCC, CC, ECC cho đa nhãn trên **1** dataset (peptide). **Dembczyński et al. (ICML 2010)**: PCC với suy luận Bayes-optimal (cho bộ phân loại bất kỳ). Bài thép (arXiv 2603.22738): regression đa target, chạy từng target | **Đã bị lấp một phần**: TFM kết hợp classifier chain đã có; phần Bayes-optimal của PCC đã có từ 2010 |
| C | PFN đa đầu ra gốc (sửa kiến trúc + pretrain) | (từ quét ngày 24/10, lần trước) | GitHub issue #248, #367 của TabPFN: chưa hỗ trợ; không thấy bài | **Còn trống**, nhưng cần pretrain (xem C6) |
| D | Sinh dữ liệu tổng hợp hoặc tăng cường dữ liệu bằng TFM | "TabPFN data augmentation synthetic generation" | **TabPFGen** (arXiv 2406.05216), **TabMDA** (arXiv 2406.01805), chức năng sinh dữ liệu trong bài Nature (Fig. 6b) và trong gói TabPFN | **Đông** |
| E | Chọn và tinh chỉnh ngữ cảnh, định giá dữ liệu cho TFM | (từ quét ngày 24/10) | **TICDA** (10/2026), **VIP-COP** (arXiv 2605.12904), DemoShapley, các phương pháp ngữ cảnh truy hồi (retrieval) | **Rất đông, thay đổi nhanh** |

## Đã cân nhắc và loại sớm (trước khi chấm điểm)
- **Phát hiện "lát cắt dữ liệu lỗi" (slice discovery) bằng khai phá mẫu:** khớp kỹ năng khai phá mẫu, nhưng văn liệu chủ yếu đặt vấn đề theo hiệu năng nhóm con, dễ trượt sang fairness → rủi ro C5.
- **Label shift cho TFM:** đã có *Mitigating Label Shift in Tabular In-Context Learning via Test-Time Posterior Adjustment* (arXiv 2605.04363).
- **Bán giám sát cho TFM** như một đề tài riêng: đã có lý thuyết (*When and How Unlabeled Data Provably Improve In-Context Learning*, NeurIPS 2025). Gộp làm một trường hợp của ứng viên A (dữ liệu không nhãn là nhãn yếu nhất).

## Mốc nộp bài (để kiểm tra C3, đều là **ước tính**, phải xác minh trên trang chính thức)
Nguồn: bảng tổng hợp opencurious.com/ai-conference-deadlines (cập nhật 28/8/2026).

| Hội nghị | Hạn (ước tính) | Ghi chú |
|---|---|---|
| ICML 2027 | abstract 16/01/2027, bài 22/01/2027 | Quá gấp cho G2–G10 |
| KDD 2027 | khoảng 02/2027 | Quá gấp |
| **NeurIPS 2027** | abstract 14/05/2027, bài 21/05/2027 | **Đích đề xuất**, khoảng 6,5 tháng |
| ICLR 2028, ICML 2028 | chưa công bố | Dự phòng theo bậc thang |
