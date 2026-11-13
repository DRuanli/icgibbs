# Đối chiếu tài liệu tham khảo (04/11/2026; bổ sung 12/11/2026)

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
| tabpfn3 | arXiv 2605.13986 (trang abs, v2 ngày 28/05/2026): "TabPFN-3: Technical Report", 41 tác giả, Grinsztajn … Hutter. Chưa có nơi đăng; kiểm lại trước khi nộp |
| tabdpt | kỷ yếu NeurIPS 2025 (papers.nips.cc), tập 38: Ma, Thomas, Hosseinzadeh, Labach, Cresswell, Golestan, Yu, Caterini, Volkovs |
| tabarena | kỷ yếu NeurIPS 2025, Datasets and Benchmarks Track, tập 38: Erickson, Purucker, Tschalzev, Holzmüller, Desai, Salinas, Hutter |
| bischl2021openml | kỷ yếu NeurIPS Datasets and Benchmarks 2021 (tập 1). Trang kỷ yếu lặp tên Bischl hai lần; em ghi 9 tác giả, mỗi người một lần |
| hoeffding1963 | trang thư mục UT Austin: JASA 58(301):13–30, 1963 |

## Bổ sung 12/11/2026 (bản v2 của bài)

- Thêm 5 tài liệu ở cuối bảng trên; tổng số tài liệu được trích là 37.
- `tabpfn3` là báo cáo kỹ thuật trên arXiv, chưa qua bình duyệt. Trong bài chỉ dùng để nêu nguồn của mô hình.
- TabPFN-3.5 (arXiv 2609.17895) **không** được trích trong bài: mô hình này chỉ xuất hiện ở phép chọn ghi trước giữa 3 và 3.5 trong phụ lục, mô tả bằng lời. Nếu sau này trích thì phải đối chiếu trước.
- Phần "TabDPT được huấn luyện trên dữ liệu thật" dựa vào tóm tắt của bài và danh sách 123 dataset trong repo huấn luyện (`V2/data/tabdpt_pretrain_ids.csv`, xem W4_TFM_REPORT).


## Bổ sung sau vòng phản biện (12/11/2026)

| Khoá | Nguồn đã đọc |
|---|---|
| heckerman2000 | trang JMLR (jmlr.org/papers/v1/heckerman00a.html): JMLR 1:49–75, 2000; 5 tác giả. **Nội dung:** tóm tắt không nhắc pseudo-Gibbs; phần "ordered pseudo-Gibbs sampler" nằm trong thân bài, em chưa đọc lại bản PDF → việc của G10 |
| chen2015gibbs | PubMed Central (PMC4572746): J. Stat. Comput. Simul. 85(16):3266–3275, doi 10.1080/00949655.2014.968159; trang ghi tên bài theo chính tả Mỹ ("Behavior"), đăng trực tuyến 10/2014, số báo năm 2015. Tóm tắt nói rõ: phân phối giới hạn phụ thuộc thứ tự quét khi các điều kiện không tương thích |

Tổng số tài liệu được trích: 39.

## Bổ sung ngày 13/11/2026 (bản chuyển sang tạp chí)

Mỗi dòng: đã mở trang arXiv (URL lấy từ kết quả tìm kiếm) và đối chiếu tên bài, tác giả, năm. Không trang nào ghi nơi đăng, nên cả sáu được trích dạng arXiv preprint.

| Khoá | Tên bài, tác giả, năm như trang ghi | Trang đã mở | Dùng để nói gì trong bài |
|---|---|---|---|
| `busafekete2025llp` | Nearly Optimal Sample Complexity for Learning with Label Proportions — Robert Busa-Fekete, Travis Dick, Claudio Gentile, Haim Kaplan, Tomer Koren, Uri Stemmer; nộp 08/05/2025 | arxiv.org/abs/2505.05355v1 | lý thuyết số mẫu của LLP theo cỡ túi |
| `icmil` | In-Context Multiple Instance Learning — Alexander Möllers, Marvin Sextro, Julius Hense, Gabriel Dernbach, Klaus-Robert Müller; 2026 (v2 08/09/2026) | arxiv.org/html/2606.06458 | huấn luyện trước một mô hình trong ngữ cảnh cho MIL trên dữ liệu tổng hợp |
| `zhu2024iterated` | Eliciting the Priors of Large Language Models using Iterated In-Context Learning — Jian-Qiao Zhu, Thomas L. Griffiths; 04/06/2024 | arxiv.org/abs/2406.01860v1 | lặp học trong ngữ cảnh như một chuỗi lấy mẫu từ prior của mô hình |
| `nagler2025martingale` | Uncertainty Quantification for Prior-Data Fitted Networks using Martingale Posteriors — Thomas Nagler, David Rügamer; 2025 (v3 01/2026) | arxiv.org/html/2505.11325v3 | rút nhãn từ dự đoán của PFN để lấy mẫu lại theo dự đoán (martingale posterior) |
| `ng2025tabmgp` | TabMGP: Martingale Posterior with TabPFN — Kenyon Ng, Edwin Fong, David T. Frazier, Jeremias Knoblauch, Susan Wei; 29/10/2025 | arxiv.org/abs/2510.25154v2 | martingale posterior dựng trên TabPFN (chỉ nói ở mức này; cơ chế chưa đọc) |
| `balakrishnan2014em` | Statistical guarantees for the EM algorithm: From population to sample-based analysis — Sivaraman Balakrishnan, Martin J. Wainwright, Bin Yu; 09/08/2014 | arxiv.org/abs/1408.2156v1 | vùng xuất phát mà từ đó EM hội tụ (lưu vực) |

Ghi chú: `unlabeled_icl_theory` (đã có) được dùng thêm cho câu "lặp TabPFN trên nhãn giả mềm của chính nó cho bán giám sát", theo lần đọc ngày 12/11/2026.


## Bổ sung ngày 13/11/2026, vòng 3 (theo phản biện TMLR F22, F7, F26)

Mỗi dòng: URL lấy từ kết quả WebSearch; đã mở trang và đối chiếu tên bài, tác giả, nơi đăng, năm, trang.

| Khoá | Thông tin như trang ghi | Trang đã mở | Dùng để nói gì trong bài |
|---|---|---|---|
| `johnson2013hogwild` | Analyzing Hogwild Parallel Gaussian Gibbs Sampling — Matthew J. Johnson, James Saunderson, Alan Willsky; NIPS 26, 2013 (trang không ghi số trang) | papers.nips.cc/paper/5043-… | Gibbs song song kiểu Hogwild (cập nhật song song với trạng thái cũ) và sai lệch của nó; chỉ nói ở mức "liên quan tới khối không hiệu chỉnh" |
| `desa2016asyncgibbs` | Ensuring Rapid Mixing and Low Bias for Asynchronous Gibbs Sampling — Christopher De Sa, Chris Re, Kunle Olukotun; ICML 2016, PMLR 48:1567–1576 | proceedings.mlr.press/v48/sa16.html | sai lệch và thời gian trộn của Gibbs bất đồng bộ |
| `celeux1985sem` | The SEM algorithm: a probabilistic teacher algorithm derived from the EM algorithm for the mixture problem — G. Celeux, J. Diebolt; Computational Statistics Quarterly 2:73–82, 1985 (theo danh mục tham khảo của arXiv 1310.5034; chưa mở bản gốc) | arxiv.org/pdf/1310.5034 | gốc của "stochastic EM" |
| `tanner1987da` | The Calculation of Posterior Distributions by Data Augmentation — Martin A. Tanner, Wing Hung Wong; JASA 82(398):528–540, 1987 | stat.cmu.edu/~brian/…/tanner-wong-1987-with-disc.pdf (trang đầu bản gốc) | data augmentation: lấy mẫu dữ liệu ẩn xen với tham số |
| `hullermeier2006plknn` | Learning from ambiguously labeled examples — Eyke Hüllermeier, Jürgen Beringer; Intelligent Data Analysis 10(5):419–439 (IOS Press; trang Paderborn ghi 419–440), 2006, doi 10.3233/IDA-2006-10503 | content.iospress.com/articles/intelligent-data-analysis/ida00259; ris.uni-paderborn.de/record/16177; bản nháp IDADRAFT.pdf §4.1 | PL-kNN: mỗi láng giềng bỏ phiếu cho mọi nhãn trong tập ứng viên của nó (bản nháp §4.1: thay I(λ = λ_x) bằng I(λ ∈ L_x)) |
| `zhang2015ipal` | Solving the Partial Label Learning Problem: An Instance-Based Approach — Min-Ling Zhang, Fei Yu; IJCAI 2015, tr. 4048–4054 | ijcai.org/Proceedings/15/Abstracts/568.html; mlanthology.org/ijcai/2015/zhang2015ijcai-solving | IPAL, baseline PLL dựa trên láng giềng, chưa chạy |
| `ruping2010invcal` | SVM Classifier Estimation from Group Probabilities — Stefan Rüping; ICML 2010, tr. 911–918 | mlanthology.org/icml/2010/ruping2010icml-svm | InvCal, baseline LLP phi nơ-ron, chưa chạy |
| `yu2013propsvm` | ∝SVM for Learning with Label Proportions — Felix Yu, Dong Liu, Sanjiv Kumar, Tony Jebara, Shih-Fu Chang; ICML 2013, PMLR 28:504–512 | mlanthology.org/icml/2013/yu2013icml-learning | ∝SVM, baseline LLP phi nơ-ron, chưa chạy |
| `balakrishnan2014em` (cập nhật) | bản đăng: The Annals of Statistics 45(1):77–120, 2017, doi 10.1214/16-aos1435 | escholarship.org/uc/item/7r74k08k | như trước; nay trích bản đăng tạp chí |

Định dạng: bốn bản arXiv (`inctx_pu`, `noise_immunity`, `ic_ssl`, `tabimpute`) chuyển sang kiểu `@article` để style không in "In arXiv preprint". `tabimpute` ghi rõ v4 (17/02/2026), vì tên bài và năm 2026 là của bản v4.
