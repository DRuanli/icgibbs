# G2 · Nhật ký truy vấn và snowballing (24/10/2026)

Protocol: [00_SEARCH_PROTOCOL.md](00_SEARCH_PROTOCOL.md). Công cụ: tìm kiếm web có giới hạn domain, rồi đọc trang kết quả. Cột "Chọn" ghi số bài thoả tiêu chí chọn; tên bài đã chọn có trong [related_work_matrix.csv](related_work_matrix.csv).

## 1. Truy vấn chính (K1 × K2)

| # | Nguồn (domain) | Truy vấn nguyên văn | Số kết quả xem | Chọn | Bài đáng chú ý |
|---|---|---|---|---|---|
| Q1 | arxiv.org | `TabPFN "label proportions"` | 10 | 3 | Không có bài TFM nào; chọn LLP-Bench, BP-LLP, Optimal LLP with general losses (2509.15145) làm baseline |
| Q2 | arxiv.org | `TabPFN positive-unlabeled PU learning in-context` | 10 | 2 | **In-Context PU (2605.05591)**; Focused PU (2605.14467), bài này ghi tích hợp TFM là future work |
| Q3 | arxiv.org | `TabPFN "partial label" OR "candidate labels" OR "complementary label" tabular foundation model` | 10 | 0 | Không có bài TFM nào |
| Q4 | arxiv.org | `tabular foundation model noisy labels label noise in-context learning robustness correction` | 10 | 4 | Noise Immunity (2604.04868); Adaptive Ensemble Selection for Noisy Labels on Tabular Data (2609.32976, không dùng TFM); In-Context Learning with Noisy Labels (2411.19581, LLM); Exploring the Robustness of ICL with Noisy Labels (2404.18191, LLM) |
| Q5 | arxiv.org | `tabular foundation model semi-supervised unlabeled data in-context TabPFN TabICL pseudo-labels` | 10 | 2 | When and How Unlabeled Data Provably Improve ICL (2506.15329); TabFM (2609.37959) làm backbone mới |
| Q6 | arxiv.org | `"tabular foundation model" "weak supervision" OR "weakly supervised" OR "weak labels"` | 10 | 1 | Universal Semantic Enhancement for Weakly Supervised Anomaly Detection on Tabular Data (2610.23765); **chưa đọc được toàn văn** (arXiv giới hạn tốc độ 2 lần), xem Mục 4 |
| Q7 | openreview.net | `TabPFN noisy labels OR positive-unlabeled OR label proportions OR partial label OR semi-supervised` | 10 | 2 | Count-based WSL (Shukla et al.); Weakly-Supervised Contrastive Learning for Imprecise Class Labels |
| Q8 | proceedings.mlr.press, proceedings.neurips.cc, papers.nips.cc | `prior-data fitted network weak supervision noisy labels unlabeled positive` | 10 | 1 | **A Unified Approach to Count-Based Weakly-Supervised Learning (NeurIPS 2023)**. Không có bài PFN nào |
| Q9 | ieeexplore.ieee.org, dl.acm.org | `tabular foundation model TabPFN label noise weak supervision positive unlabeled` | 10 | 0 | Chỉ có bài ứng dụng TabPFN với nhãn sạch |
| Q10 | dblp.org, semanticscholar.org | `"in-context" weak supervision OR "label proportions" OR "noisy labels" OR "positive-unlabeled" tabular` | 10 | 1 | Easy Learning from Label Proportions (baseline) |
| Q11 | web | `learning from label proportions tabular foundation model TabPFN in-context aggregate labels bags` | 9 | 0 | Không có |
| Q12 | web | `weak supervision tabular foundation model in-context learning expectation maximization TabPFN partial labels positive unlabeled label proportions training-free` | 9 | 2 | In-Context PU; GLWS |

## 2. Truy vấn phụ (K1 × K3: Gibbs / EM / imputation với xác suất có điều kiện do mạng học sẵn)

| # | Nguồn | Truy vấn | Xem | Chọn | Bài đáng chú ý |
|---|---|---|---|---|---|
| Q13 | web | `TabPFN multiple imputation chained equations Gibbs sampling conditionals missing values iterative imputation` | 9 | 2 | **Sepin (2026), Multiple Imputation of a Continuous Outcome … Using TabPFN** (*Stats*); Convergence of sequential regression MI (*JASA* 2015) |
| Q14 | web | `Gibbs sampling with learned neural conditionals pseudo-Gibbs incompatible conditionals transformer amortized latent variables` | 9 | 2 | Amortized Population Gibbs Samplers (1911.01382); bài về pseudo-Gibbs và conditional không tương thích (*AISM* 2019) |
| Q15 | arxiv.org, openreview.net | `TabPFN "Gibbs" sampling OR "data augmentation algorithm" latent variables in-context posterior` | 10 | 0 | Không có bài dùng TFM làm conditional trong Gibbs trên nhãn |
| Q16 | web | `TabImpute zero-shot missing data imputation …` | 9 | 1 | TabImpute (2510.02625): pretrain riêng cho imputation feature, không phải nhãn |

## 3. Văn liệu in-context learning của LLM (K2 × ICL)

| # | Truy vấn | Xem | Chọn | Ghi chú |
|---|---|---|---|---|
| Q17 | `large language model in-context learning "label proportions" OR "aggregate labels" OR "positive-unlabeled" demonstrations` | 9 | 0 | Chỉ có các bài về vai trò của nhãn trong demonstration; không có LLP hay PU |
| Q18 | `in-context learning from weak labels pretrained transformer amortized weak supervision prior-fitted 2026` | 10 | 2 | **In-Context Semi-Supervised Learning (2512.15934)**: pretrain transformer tổng hợp, không dùng TFM, không phải dữ liệu bảng; Transformers Can Learn Posterior Predictive Distributions In-Context (ICML 2026, lý thuyết) |

## 4. Snowballing

| Vòng | Bài hạt giống | Hướng | Kết quả |
|---|---|---|---|
| 1 | GLWS (ICML 2024) | Ngược | Liệt kê các phương pháp theo từng loại nhãn yếu (LLP: Quadrianto 2008, Scott & Zhang 2020, Zhang et al. 2022, LLP-VAT; PU: uPU, nnPU, VPU, DistPU; PLL: PRODEN, PiCO, LWS, RCR; Count Loss). **Không có bài nào dùng TFM.** GLWS chỉ thực nghiệm trên ảnh, ghi nhãn yếu có nhiễu là future work, giả định dự đoán độc lập có điều kiện (Assumption 3.1) |
| 1 | ILL (NeurIPS 2024) | Ngược | Baseline PLL, NLL, SSL, mixed (PiCO, PRODEN, DivideMix, SOP, FixMatch…). **Không có TFM, không có dữ liệu bảng.** EM tính tổng chính xác trên lớp, không lấy mẫu nhãn ẩn |
| 1 | In-Context PU (arXiv 2026) | Ngược | **Không trích dẫn công trình nào kết hợp TFM hay PFN với nhãn yếu.** Baseline: EMPU, VPU, uLSIF, (TED)^n |
| 1 | In-Context PU | Xuôi | Tìm theo tên "PUICL" / tiêu đề: chỉ ra trang của chính bài và các trang tổng hợp. Trang trích dẫn của Pith lỗi khi tải. Chưa thấy bài trích dẫn |
| 1 | GLWS, ILL | Xuôi | Tìm theo tên và từ khoá 2025–2026: không thấy bài tiếp nối nào dùng TFM. **Cần Google Scholar "Cited by" để hoàn tất** ([03](03_QUERIES_FOR_THAY.md)) |
| 2 | Các bài mới ở vòng 1 (Focused PU, Sepin 2026, In-Context SSL, Count-based WSL) | Ngược | Không thêm bài nào thoả tiêu chí 1 (TFM × nhãn yếu) hay tiêu chí 4 (Gibbs với TFM). **Bão hoà** với các nguồn truy cập được |

## 5. Không thực hiện được (cần thầy chạy)
- Scopus, Web of Science: tìm nâng cao theo trường TITLE-ABS-KEY.
- Google Scholar "Cited by" cho 3 bài hạt giống.
- Đọc toàn văn arXiv 2610.23765 (arXiv giới hạn tốc độ).

Truy vấn soạn sẵn: [03_QUERIES_FOR_THAY.md](03_QUERIES_FOR_THAY.md).
