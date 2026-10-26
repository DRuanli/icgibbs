# 05 · Chuẩn riêng cho từng loại hình nghiên cứu

[← Về README](README.md)

Denning et al. (1989) mô tả KHMT gồm ba "paradigm": **lý thuyết** (định nghĩa, định lý, chứng minh), **trừu tượng hoá / mô hình hoá** (giả thuyết, mô hình, thực nghiệm, phân tích) và **thiết kế** (yêu cầu, đặc tả, cài đặt, kiểm thử). Hầu hết bài KHMT và KHDL kết hợp hai hoặc ba paradigm này. Mỗi phần dưới đây nêu điều phản biện *bắt buộc* phải thấy.

## 1. Phân loại đóng góp (Shaw 2003)

| Loại câu hỏi | Loại kết quả | Kiểm chứng tương xứng |
|---|---|---|
| Cách làm / phát triển X tốt hơn? | Quy trình, kỹ thuật, thuật toán | Phân tích (chứng minh), đánh giá thực nghiệm có hệ thống |
| Cách phân tích, đánh giá X? | Phương pháp phân tích, metric | Đánh giá thực nghiệm; so sánh với phương pháp đánh giá cũ |
| Thiết kế hay đánh giá một hệ thống cụ thể? | Giải pháp, prototype | Đánh giá trên tình huống thật; kinh nghiệm triển khai |
| Tổng quát hoá hay đặc trưng hoá X? | Mô hình định tính hay định lượng, mô hình giải tích | Chứng minh; thực nghiệm trên nhiều bối cảnh độc lập |
| Có khả thi không, có tồn tại không? | Bằng chứng khả thi | Ví dụ minh hoạ, phản ví dụ |

Shaw ghi nhận các loại kiểm chứng yếu dần: **phân tích → đánh giá → kinh nghiệm → ví dụ → thuyết phục → khẳng định suông**. Hai loại cuối không được chấp nhận làm bằng chứng chính.

## 2. Nghiên cứu lý thuyết

Xem chi tiết ở [07_LY_THUYET_VA_CHUNG_MINH.md](07_LY_THUYET_VA_CHUNG_MINH.md). Phản biện bắt buộc thấy:
- Mọi giả định nằm trong phát biểu định lý.
- Chứng minh đầy đủ (trong bài hoặc phụ lục), kèm phác thảo ở bài chính.
- Thảo luận độ chặt: có ví dụ đạt được chặn, hoặc có chặn dưới.
- Liên hệ với kết quả đã biết: chứng minh được kết quả cũ như trường hợp riêng.

## 3. Thuật toán và khai phá dữ liệu (thực nghiệm thuật toán)

Theo Johnson (2002), *A theoretician's guide to the experimental analysis of algorithms*, có mười nguyên tắc:
1. Làm thí nghiệm đáng đưa tin (newsworthy).
2. Gắn bài với tài liệu đã có.
3. Dùng bộ thử (testbed) hỗ trợ được kết luận tổng quát.
4. Thiết kế thí nghiệm hiệu quả.
5. Dùng cài đặt đủ hiệu quả.
6. Bảo đảm tái lập.
7. Bảo đảm so sánh được.
8. Báo cáo toàn bộ câu chuyện.
9. Rút ra kết luận có căn cứ và tìm lời giải thích.
10. Trình bày dữ liệu một cách giàu thông tin.

Bổ sung cho khai phá dữ liệu dạng mẫu (itemset, high-utility, uncertain pattern):
- **Tính đúng trước tốc độ:** chứng minh tính đúng và tính đầy đủ (complete và sound) của cắt tỉa. Kiểm tra bằng cách so tập kết quả với thuật toán chuẩn trên mọi dataset: **tập mẫu phải trùng khớp tuyệt đối**.
- **Cùng ngôn ngữ, cùng môi trường:** so thời gian chạy chỉ có nghĩa khi các cài đặt cùng ngôn ngữ, cùng framework (ví dụ cùng SPMF/Java), cùng máy, cùng tuỳ chọn biên dịch. Nếu không được thì báo cáo thêm metric không phụ thuộc cài đặt (số ứng viên sinh ra, số lần quét, số node duyệt).
- **Quét tham số:** chạy theo một dải ngưỡng (minsup, minutil, k), không chọn một điểm có lợi.
- **Khả năng mở rộng:** theo số giao dịch, số item, mật độ; thêm dữ liệu tổng hợp có điều khiển, *ngoài* dữ liệu thật.
- **Bộ nhớ đỉnh** và **thời gian**, báo cáo cả timeout.
- Chuẩn đánh giá benchmark nói chung: Bartz-Beielstein et al. (2020).

## 4. Phương pháp học máy và deep learning

Áp dụng toàn bộ [06_THIET_KE_THUC_NGHIEM.md](06_THIET_KE_THUC_NGHIEM.md) và 16 mục của NeurIPS Paper Checklist ([10](10_CHECKLIST_TRUOC_KHI_NOP.md)). Bốn điều phản biện ML hay bắt lỗi nhất (Lipton & Steinhardt 2019):
1. Không tách *giải thích* với *suy đoán*.
2. Không chỉ ra nguồn gốc của cải thiện, tức là không có ablation.
3. "Mathiness": toán để trang trí hoặc làm rối, không gánh lập luận.
4. Dùng sai ngôn ngữ: thuật ngữ gợi ý quá mức, dùng một từ với nhiều nghĩa.

## 5. Học máy ứng dụng và khoa học dữ liệu (ML-based science)

Khi đóng góp là *kết luận về một lĩnh vực* rút ra từ mô hình dự đoán (giáo dục, y tế, tài chính), áp dụng **REFORMS** (Kapoor et al., *Science Advances* 2024): 32 mục trong 8 module.

| Module | Câu hỏi chính |
|---|---|
| 1. Thiết kế nghiên cứu | Câu hỏi khoa học là gì; vì sao dự đoán trả lời được nó |
| 2. Tái lập tính toán | Có code, dữ liệu, môi trường không |
| 3. Chất lượng dữ liệu | Nguồn, đơn vị quan sát, đo lường, thiếu dữ liệu, độ lệch |
| 4. Tiền xử lý | Từng bước, fit trên phần nào của dữ liệu |
| 5. Mô hình hoá | Chọn mô hình, tune, so sánh với baseline đơn giản |
| 6. Rò rỉ dữ liệu | Kiểm tra theo 8 loại của Kapoor & Narayanan (2023) |
| 7. Metric và bất định | Metric phù hợp câu hỏi; khoảng tin cậy |
| 8. Tính tổng quát | Kết quả áp dụng cho quần thể nào, ngoài phạm vi nào |

Mẫu checklist chính thức: reforms.cs.princeton.edu.

Ứng dụng học máy trong an ninh mạng hay các lĩnh vực có đối thủ: thêm 10 lỗi của Arp et al. (USENIX Security 2022):
1. Sampling bias
2. Label inaccuracy
3. Data snooping
4. Spurious correlations
5. Biased parameter selection
6. Inappropriate baseline
7. Inappropriate performance measures
8. Base rate fallacy
9. Lab-only evaluation
10. Inappropriate threat model

## 6. Hệ thống, công cụ, artifact (design science)

Bảy nguyên tắc của Hevner et al. (2004):
1. Thiết kế là một artifact.
2. Bài toán có ý nghĩa thực tế.
3. Đánh giá thiết kế một cách chặt chẽ.
4. Đóng góp nghiên cứu rõ ràng.
5. Tính chặt chẽ của phương pháp.
6. Thiết kế là một quá trình tìm kiếm.
7. Truyền đạt cho cả người kỹ thuật lẫn người quản lý.

Quy trình sáu bước của Peffers et al. (2007):
1. Xác định vấn đề và động cơ.
2. Xác định mục tiêu của giải pháp.
3. Thiết kế và phát triển.
4. Minh hoạ.
5. Đánh giá.
6. Truyền đạt.

Artifact nên được nộp đánh giá theo ACM Artifact Review and Badging ([08 §6](08_TAI_LAP_VA_MA_NGUON.md)).

## 7. Bộ dữ liệu và benchmark

- **Datasheet** cho dataset (Gebru et al., *CACM* 2021): động cơ, thành phần, quy trình thu thập, tiền xử lý và gán nhãn, mục đích sử dụng, phân phối, bảo trì. Mẫu: [DATASET_CARD.md](templates/DATASET_CARD.md).
- **Model card** cho mô hình công bố (Mitchell et al. 2019).
- Ghi giấy phép của dữ liệu gốc và dữ liệu dẫn xuất; nêu rõ đồng ý của người tham gia và biện pháp bảo vệ (NeurIPS Q12–Q13).
- Benchmark phải có baseline chuẩn, split cố định công bố kèm, và lý do vì sao benchmark đo được điều nó tuyên bố đo.
- Nguyên tắc FAIR (Wilkinson et al. 2016): Findable, Accessible, Interoperable, Reusable. Dataset có DOI, metadata, định dạng mở.

## 8. Tổng quan có hệ thống

Xem [04 §5](04_TONG_QUAN_TAI_LIEU.md): Kitchenham & Charters (2007), PRISMA 2020, Wohlin (2014), Petersen et al. (2015).

## 9. Nghiên cứu thực nghiệm về thực hành (khảo sát, phỏng vấn, case study, khai phá repository)

Dùng **ACM SIGSOFT Empirical Standards** (Ralph et al. 2020). Mỗi chuẩn chia thuộc tính thành *essential*, *desirable* và *extraordinary*. Có các chuẩn riêng cho Experiment, Case Study, Systematic Review, Data Science, Engineering Research, Replication, Grounded Theory… Trang chuẩn có công cụ sinh checklist cho phản biện.
