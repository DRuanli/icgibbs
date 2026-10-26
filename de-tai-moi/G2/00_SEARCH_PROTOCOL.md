# G2 · Protocol tìm kiếm (viết trước khi chạy truy vấn)

Đề tài A: *In-Context Gibbs: học từ nhãn yếu bằng tabular foundation model, không huấn luyện lại*
Ngày viết protocol: 24/10/2026 · Người thực hiện: Claude, theo yêu cầu của thầy

## 1. Phạm vi
- **Bài toán:** phân loại trên dữ liệu bảng khi nhãn huấn luyện là *nhãn yếu*:
  - tỉ lệ nhãn theo nhóm (LLP);
  - dương và không nhãn (PU);
  - tập nhãn ứng viên (partial label);
  - nhãn nhiễu;
  - không nhãn (bán giám sát);
  - nhãn bổ sung (complementary).
- **Cách tiếp cận:** dùng mô hình học trong ngữ cảnh đã huấn luyện trước (TabPFN, TabICL, PFN nói chung; mở rộng sang LLM in-context learning) mà không cập nhật trọng số; hoặc suy luận trên nhãn ẩn (EM, Gibbs, lấy mẫu posterior, imputation).
- **Bối cảnh:** dữ liệu bảng cỡ nhỏ và vừa.

## 2. Khối từ khoá
| Khối | Từ khoá và đồng nghĩa |
|---|---|
| K1: mô hình | "tabular foundation model", TabPFN, TabICL, TabDPT, "prior-data fitted network", PFN, "in-context learning" + tabular |
| K2: nhãn yếu | "weak supervision", "weakly supervised", "label proportions", LLP, "aggregate labels", "positive-unlabeled", "PU learning", "partial label", "superset label", "candidate labels", "noisy labels", "label noise", "semi-supervised", "unlabeled data", "complementary label", "imprecise label" |
| K3: suy luận nhãn ẩn | Gibbs, "expectation maximization", EM, "latent labels", "posterior sampling", "pseudo-Gibbs", "chained equations", MICE, "dependency network", "multiple imputation" |

Truy vấn chính: **K1 × K2**. Truy vấn phụ: **K1 × K3** (để kiểm tra tính mới của phần phương pháp Gibbs) và **K2 × "in-context learning"** (văn liệu LLM).

## 3. Nguồn
| Nguồn | Truy cập từ môi trường này | Cách dùng |
|---|---|---|
| arXiv | Có, qua công cụ tìm web giới hạn domain `arxiv.org`, rồi đọc trang HTML | Nguồn chính cho công trình 2025–2026 |
| OpenReview | Có, qua tìm web giới hạn `openreview.net` | Bài ICLR, TMLR, NeurIPS đang hoặc đã phản biện |
| PMLR, NeurIPS Proceedings | Có, qua tìm web giới hạn `proceedings.mlr.press`, `proceedings.neurips.cc`, `papers.nips.cc` | Bài đã peer-review ở ICML, AISTATS, NeurIPS |
| IEEE Xplore, ACM DL | Một phần: chỉ tìm web giới hạn domain, không có công cụ tìm nâng cao | Bài KDD, TKDE, TNNLS |
| DBLP, Semantic Scholar | Một phần: API bị chặn, chỉ tìm web | Đối chiếu metadata |
| Scopus, Web of Science, Google Scholar ("Cited by") | **Không** (cần đăng nhập hoặc tài khoản cơ quan) | **Thầy chạy** các truy vấn soạn sẵn ở [03_QUERIES_FOR_THAY.md](03_QUERIES_FOR_THAY.md) |

Giới hạn này được ghi rõ để khẳng định tính mới ở G2 mang tính *có điều kiện*. Kết luận cuối cùng chỉ chắc chắn khi thầy chạy xong phần Scopus và Google Scholar.

## 4. Tiêu chí chọn và loại (viết trước khi đọc)
**Chọn** nếu thoả ít nhất một:
1. Dùng mô hình học trong ngữ cảnh hoặc PFN với **ít nhất một loại nhãn yếu** (K1 × K2).
2. Khung thống nhất cho **nhiều loại nhãn yếu** (bất kể mô hình), làm baseline hoặc đối chứng.
3. Phương pháp tiêu biểu đã peer-review cho **một loại nhãn yếu** trên dữ liệu bảng, làm ứng viên baseline.
4. Dùng Gibbs, EM hay imputation lặp **với xác suất có điều kiện do mạng học sẵn**, để kiểm tra tính mới của phần phương pháp.
5. Lý thuyết về in-context learning với nhãn không đầy đủ hoặc nhiễu.

**Loại** nếu:
- chỉ áp dụng TFM cho một bài toán ứng dụng có nhãn sạch;
- nhãn yếu trong thị giác hay NLP mà phương pháp phụ thuộc augmentation ảnh hay văn bản (vẫn ghi lại nếu là khung thống nhất);
- không phải tiếng Anh.

## 5. Quy trình
1. Chạy truy vấn K1 × K2 trên từng nguồn. Ghi ngày, nguồn, truy vấn nguyên văn, số kết quả, số bài chọn.
2. Snowballing **ngược** từ 3 bài hạt giống: GLWS (ICML 2024), ILL (NeurIPS 2024), In-Context PU (arXiv 2605.05591).
3. Snowballing **xuôi** (các bài trích dẫn 3 bài hạt giống): tìm web theo tên phương pháp. Danh sách "Cited by" đầy đủ trên Google Scholar do thầy chạy.
4. Lặp lại đến khi một vòng không thêm bài nào thoả tiêu chí 1 hoặc 4.
5. Đọc lượt 3 các bài gần nhất (ít nhất 3), viết câu "khác ở đâu".
6. Kết luận theo tiêu chí dừng của Mốc 1 trong TOPIC_CARD_A: dừng hoặc thu hẹp nếu đã có công trình dùng TFM có sẵn học từ ≥2 loại nhãn yếu bằng suy luận trên nhãn ẩn.
