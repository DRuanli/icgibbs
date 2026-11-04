# Đối chiếu tài liệu tham khảo (04/11/2026)

- **Phạm vi:** 32 tài liệu được trích trong bài (`paper/refs.bib`). Các tài liệu còn lại của ma trận G2 chưa được trích, nằm ở `paper/refs_candidates.bib` và **chưa đối chiếu**. Trích thêm tài liệu nào thì phải đối chiếu trước.
- **Cách làm:** với mỗi tài liệu, em đọc trang của nguồn chính thức hoặc nguồn thư mục đáng tin, rồi so tác giả (đủ, đúng thứ tự), tên bài, nơi đăng, năm, tập, trang, DOI. Nguồn là kỷ yếu PMLR/NeurIPS, trang arXiv, JMLR, mlanthology (BibTeX lấy từ kỷ yếu), RePEc/IDEAS, Scilit.
- **Việc G10 còn phải làm:**
  - đối chiếu **nội dung** từng chỗ trích, tức câu trong bài có đúng với điều bài gốc nói không;
  - cập nhật nơi đăng của các bản arXiv ngay trước khi nộp.

## Sai lệch đã sửa

| Khoá | Trước | Sau (theo nguồn) |
|---|---|---|
| `dllp` | ICDM 2017 | **ICDM Workshops 2017**, tr. 1017–1024, doi 10.1109/ICDMW.2017.144 (giao thức §11 cũng đã ghi là ICDMW) |
| `tabicl_v2` | bản arXiv 2602.11139 | **đã đăng ICML 2026** (ghi chú trên arXiv v2, 16/09/2026); tên đầy đủ "TabICLv2: A better, faster, scalable, and open tabular foundation model" |
| `ppd_icl_theory` | tác giả "Anonymous" | Kang, Lee, Cheng; ICML 2026, PMLR 306, tr. 55565–55596 |
| `noise_immunity` | tên rút gọn | thêm phụ đề "An Empirical Robustness Analysis of TabPFN's Attention Mechanisms" |
| `tabimpute` | tên của v1 | tên đã đổi ở v4 (17/02/2026): "TabImpute: Universal Zero-Shot Imputation for Tabular Data"; v3 từng mang tên khác. Khi nộp phải kiểm lại bản mới nhất |
| `count_wsl` | "Count-Based Weakly-Supervised" | bản kỷ yếu NeurIPS viết không có gạch nối: "Count-Based Weakly Supervised Learning" |
| `seneta1988` | ghi là sách, nơi đăng chưa rõ | bài báo, Adv. Appl. Probab. 20(1):228–230, doi 10.2307/1427277 |
| `tabpfn_nature` | thiếu tác giả | đủ 8 tác giả; Nature 637(8045):319–326 |
| `agarwal2014` | "Hệ quả 12" (mới đối chiếu trên arXiv) | đã đối chiếu bản JMLR 15(49):1653–1674: **Hệ quả 12 phát biểu cho bộ xếp hạng plug-in dựng từ một ước lượng xác suất lớp $\hat\eta\in[0,1]$**. Câu trong phụ lục A đã sửa theo đúng phạm vi đó; chứng minh của Mệnh đề 5(c) trong bài tự đủ |
| mọi mục khác | tên tác giả viết tắt hoặc thiếu | điền đủ họ tên, tập, trang từ nguồn |

## Danh sách đã đối chiếu

| Khoá | Nguồn đã đọc |
|---|---|
| inctx_pu | arXiv 2605.05591 |
| glws | mlanthology (PMLR 235) |
| ill | NeurIPS proceedings 2024 |
| count_wsl | mlanthology / NeurIPS 2023 |
| bp_llp | research.google (ICLR 2024) |
| easy_llp | mlanthology (NeurIPS 2023) |
| dllp | researchr (IEEE ICDMW 2017) |
| pico | mlanthology (ICLR 2022) |
| proden | PMLR 119 |
| feng2020provably | mlanthology (NeurIPS 2020) |
| tsai2020llpvat | PMLR 129 |
| hollmann2023tabpfn | mlanthology (ICLR 2023) |
| tabpfn_nature | PubMed, RePEc |
| qu2025tabicl | PMLR 267 |
| tabicl_v2 | arXiv 2602.11139 (v2) |
| muller2022pfn | mlanthology (ICLR 2022) |
| noise_immunity | alphaXiv, EmergentMind (arXiv 2604.04868) |
| ic_ssl | arXiv 2512.15934 |
| unlabeled_icl_theory | mlanthology (NeurIPS 2025) |
| ppd_icl_theory | PMLR 306, arXiv 2605.26713 |
| sepin_mi | MDPI Stats |
| tabimpute | arXiv 2510.02625 (v3, v4) |
| apgs | mlanthology (PMLR 119) |
| liu2014 | RePEc (Biometrika) |
| benavoli2017bayesian | JMLR |
| vehtari2021rhat | NASA ADS (Bayesian Analysis 16(2), doi). Trang cuối chưa xác nhận nên bỏ trường pages |
| schweitzer1968 | Scilit |
| meyer1980 | Scilit (SIAM) |
| seneta1988 | ACNP, Scilit, MaRDI |
| cho2001 | Scilit (Elsevier) |
| agarwal2014 | JMLR (trang và bản PDF) |
| grinsztajn2022tree | mlanthology (NeurIPS 2022) |
