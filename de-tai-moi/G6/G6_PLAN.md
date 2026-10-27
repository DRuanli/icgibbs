# G6 · Cài đặt và kiểm chứng code — kế hoạch (mở 26/10/2026)

Theo [02_QUY_TRINH_CONG_KIEM_DUYET §G6](../../research-handbook/02_QUY_TRINH_CONG_KIEM_DUYET.md) và giao thức đã đóng băng [G5/EXPERIMENT_PROTOCOL](../G5/EXPERIMENT_PROTOCOL.md). Lịch gốc: 20/12/2026 – 31/01/2027. Vì G5 xong sớm khoảng 7 tuần, G6 bắt đầu ngay; mục tiêu xong trước giữa tháng 12 để G7 có thể bắt đầu từ tháng 1.

**Qua cổng G6 khi:** mọi test qua; mỗi baseline tái hiện được số công bố trong sai lệch cho phép (ghi trước, Mục 3) hoặc chênh lệch được giải thích và xếp loại; người kiểm duyệt chạy được từ clone sạch.

## 1. Tình trạng code chính thức của baseline (việc F5, tra ngày 26/10/2026)

Cột "Kho" đã kiểm bằng cách clone thử và đọc README. Không kho nào được chạy trên dữ liệu của dự án.

| Baseline | Bối cảnh | Venue | Code chính thức | Giấy phép | Ghi chú |
|---|---|---|---|---|---|
| **SAR-EM** | PU-SAR (chính) | ECML-PKDD 2019 | `github.com/ML-KULeuven/SAR-PU` (nhóm của tác giả, KU Leuven; README dẫn đúng bài) | MIT | Python 3; có script sinh dữ liệu cho các bộ UCI trong bài → tái hiện được trên CPU. **Ưu tiên số 1** (GATE_REVIEW_G5 Phát hiện 4) |
| nnPU, uPU | PU | NeurIPS 2017; ICML 2015 | `github.com/kiryor/nnPUlearning` (tác giả đầu) | MIT | Chainer (cũ). Dùng làm tham chiếu để kiểm bản PyTorch của dự án |
| VPU | PU-SCAR (phụ) | NeurIPS 2020 | `github.com/HC-Feynman/vpu` (README ghi "official implementation") | MIT | PyTorch |
| In-Context PU (PUICL) | PU-SCAR (phụ) | preprint arXiv 2605.05591 | chỉ có liên kết ẩn danh `anonymous.4open.science/r/puicl-58B1` | ? | Transformer khoảng 1,23 triệu tham số, pretrain trên PU tổng hợp, giả định SCAR, không cần tiên nghiệm lớp. Thử tải ở G6; không được thì ghi là không tái hiện được |
| GLWS | LLP, PU, PLL | ICML 2024 | `github.com/Hhhhhhao/General-Framework-Weak-Supervision` | không có tệp LICENSE | PyTorch, thiết kế cho ảnh và văn bản → phải viết phần backbone MLP cho bảng |
| ILL | PLL, nhiễu, SSL | NeurIPS 2024 | cùng kho với GLWS (bài ILL dẫn kho này) | không có tệp LICENSE | Bài ILL không thử trên dữ liệu bảng |
| DLLP | LLP | **ICDM Workshops 2017** (đính chính, giao thức §11) | không có kho của tác giả; kho LLP-VAT có DLLP (`--consistency_type none`) | MIT (kho LLP-VAT) | Đối chiếu bản cài lại của dự án với bản trong kho LLP-VAT |
| LLP-VAT | LLP | **ACML 2020** (đính chính, giao thức §11) | `github.com/kevinorjohn/LLP-VAT` (tác giả đầu) | MIT | PyTorch, CIFAR |
| EasyLLP | LLP | NeurIPS 2023 | **chưa tìm thấy** (bài, Papers with Code và trang NeurIPS đều không dẫn code) | — | Cài lại và kiểm bằng tái hiện số công bố |
| PRODEN | PLL (chính) | ICML 2020 | `github.com/Lvcrezia77/PRODEN` (tác giả đầu) | không có tệp LICENSE | PyTorch |
| RC, CC | PLL thật (RQ5) | NeurIPS 2020 | **chưa tìm thấy** (bài không dẫn code) | — | Cài lại. Số công bố với mô hình tuyến tính: Lost 79,43 / 79,29; MSRCv2 46,56 / 47,22; BirdSong 71,94 / 72,22 (Feng et al. 2020, Bảng 4) |
| VIME | SSL (phụ) | NeurIPS 2020 | `github.com/jsyoon0823/VIME` (tác giả đầu) | không có tệp LICENSE | TensorFlow 1 → cân nhắc bản PyTorch đã kiểm |
| Forward correction | nhiễu (phụ) | CVPR 2017 | cài lại (T đã biết) | — | |
| Confident Learning | nhiễu (phụ) | JAIR 2021 | `cleanlab` | AGPL | |
| Label spreading, self-training | SSL (phụ) | NIPS 2003; ACL 1995 | scikit-learn | BSD | Tune cả kernel và số láng giềng |

Kho không có tệp LICENSE: chỉ dùng để chạy tái hiện và đối chiếu; không sao chép code của họ vào kho của dự án.

## 2. Việc của G6 theo thứ tự

| # | Việc | Ai, ở đâu | Sản phẩm |
|---|---|---|---|
| T1 | **SAR-EM:** cài từ kho chính thức, tái hiện số trong bài trên một bộ UCI của chính bài đó, bọc theo giao diện chung, chạy có tune trên 6 dataset phát triển (PU-SAR) | Claude (CPU tại đây) | `baselines/sar_em.py`, dòng trong `docs/baseline_reproduction.md` |
| T2 | Giao diện chung cho mọi baseline (fit với nhãn yếu, predict_proba) và không gian tune ghi trước | Claude | `docs/tuning_spaces.md`, test |
| T3 | Tái hiện số công bố của từng baseline trên dataset gốc của baseline đó. **Ngoại lệ:** nếu dataset gốc trùng tập đánh giá của dự án (Lost, MSRCv2, BirdSong cho RC/CC/PRODEN) thì tái hiện trên một bộ khác trong cùng bài (ví dụ MNIST hoặc Fashion-MNIST với mô hình tuyến tính), để không thấy số của baseline trên tập đánh giá trước G7 | Claude viết, thầy chạy phần GPU trên Kaggle | `docs/baseline_reproduction.md` |
| T4 | Chọn đối thủ chính của từng bối cảnh theo quy tắc ở giao thức Mục 6: baseline peer-review tốt nhất trên 6 dataset phát triển sau khi tune | Kaggle | `docs/primary_comparators.md` (trước G7) |
| T5 | Mã băm dữ liệu (md5 của OpenML, SHA-256 của bảng X, y), loại trùng, sinh lần chia và nhãn yếu theo seed cố định | Claude (CPU, cần Internet tới OpenML → chạy trên Kaggle) | `docs/dataset_hashes.csv`, `data/splits/` |
| T6 | Bộ chạy thực nghiệm chính cho G7: mỗi ô (dataset × bối cảnh × mức × seed × phương pháp) độc lập, chạy tiếp được khi bị ngắt, ghi đủ metric chính và phụ, thời gian, số lần gọi TFM | Claude | `run_g7.py` + test |
| T7 | Lý thuyết: chứng minh mệnh đề về xếp hạng (khi nào bộ ngây thơ đã nhất quán về xếp hạng); đóng G4-1, G4-3, G4-4 trong PROOFS; kiểm duyệt toán độc lập | Claude soạn; kiểm duyệt độc lập | `G4/PROOFS.md` (bổ sung) |
| T8 | Cấu trúc kho theo [08](../../research-handbook/08_TAI_LAP_VA_MA_NGUON.md); người kiểm duyệt chạy từ clone sạch | Claude + người kiểm duyệt | `README`, `environment.yml` hoặc `requirements.txt` có ghim phiên bản |

## 3. Sai lệch cho phép khi tái hiện (ghi trước khi chạy)
- Đạt: số tái hiện cách số công bố **≤ 1 điểm %** (accuracy hoặc AUC), hoặc nằm trong **1 độ lệch chuẩn** mà bài công bố — lấy ngưỡng rộng hơn.
- Không đạt: ghi chênh lệch và xếp loại — (a) có nguyên nhân rõ (khác phiên bản thư viện, khác tiền xử lý đã ghi trong bài); (b) chưa rõ nguyên nhân. Baseline loại (b) vẫn được dùng, nhưng bài phải ghi rõ.

## 4. Tiến độ
| Ngày | Việc | Trạng thái |
|---|---|---|
| 26/10/2026 | T1 SAR-EM: bọc code chính thức (`icgibbs/baselines_sar.py`), random search 20 cấu hình trên nhãn yếu (`icgibbs/tuning.py`), kiểm thử tổng hợp (`tests/test_baselines_g6.py`, 29/29 test qua), script tái hiện số công bố (`g6_reproduce_sarem.py`) | Code xong; chờ chạy Kaggle (`g6_code_v1.zip`) |
| 26/10/2026 | T2: không gian tune của mọi baseline ghi trước | Xong: [docs/tuning_spaces.md](../docs/tuning_spaces.md) |
| 26/10/2026 | Kho git riêng tư `github.com/DRuanli/icgibbs`; commit đóng băng `c4da1f4` | Xong, đã đẩy lên |
| 26/10/2026 | T1 chạy Kaggle: tái hiện SAR-EM (Mushroom đạt; Breast Cancer lệch, đang kiểm), SAR-EM trên tập phát triển, Gibbs ở PU-SAR mức 0,1 | Xong — [G6_T1_SAREM_REPORT](G6_T1_SAREM_REPORT.md). **Phát hiện:** khi có SAR-EM, Gibbs chỉ đạt 3/12 ô ở PU-SAR |
| 26/10/2026 | Quyết định (thầy giao Claude chọn): PU-SAR thành bối cảnh phụ; đối thủ chính xét cả bản mặc định lẫn bản tune; thêm RC, CC cho PLL tổng hợp; GLWS dùng E-step chính xác | Đã ghi ở giao thức Mục 11, RESEARCH_PLAN §9, CLAIM #6 |
| 26/10/2026 | T3: cài DLLP (MLP, tuyến tính), LLP-VAT, EasyLLP, GLWS cho LLP; PRODEN, GLWS/ILL, RC, CC cho PLL; nnPU, uPU có tune; SAR-E (`icgibbs/baselines_ext.py`, 16 test mới). Sửa lỗi ghép nhầm `else` trong `kaggle_g4_pilot.py` (không ảnh hưởng kết quả G6 lượt 1 vì lượt đó chạy với `--only`) | Code xong; chờ lượt 2 trên Kaggle (`g6_code_v2.zip`, `run_g6_round2.sh`) |
| 26/10/2026 | Smoke test lượt 2 (dry-run mọi phương pháp mới); sửa phần tóm tắt của `kaggle_g4_rq4_pilot.py` khi chỉ chạy baseline; commit `3a900c1` | Xong; `g6_code_v2.zip` đã gửi thầy |
| 26/10/2026 | T7 (phần 1): Mệnh đề 5 (mục tiêu ngây thơ affine ⇒ nhất quán về xếp hạng; chặn hối tiếc AUC), Mệnh đề 6 (bất đẳng thức xử lý dữ liệu cho thông tin Fisher), Quan sát 6(b) (LLP: hợp lý từng điểm ≈ 0,27/k hiệu quả). **Đính chính:** bộ ngây thơ ở PLL, LLP cũng nhất quán về xếp hạng. Thêm phân tích thăm dò D1–D3 (giao thức §11) | Xong — [PROOFS §7](../G4/PROOFS.md), `g6_t7_naive_theory.py`, 10 test. Còn: G4-1, G4-3, G4-4, G6-T7a, kiểm duyệt độc lập |
| 26/10/2026 | G6 lượt 2 trên Kaggle (CPU, 2 giờ): 684/684 dòng, cùng dữ liệu với Gibbs/naive. T4: đối thủ chính LLP — DLLP tuyến tính; PLL — GLWS/ILL tune (tạm thời); PU — nnPU. Kiểm phạm vi: PLL 6/6; LLP 7/12 (túi 8: 5/6, túi 16: 2/6). SAR-EM Breast Cancer: giả thuyết 1 bị bác, xếp loại (b) | Xong — [G6_ROUND2_REPORT](G6_ROUND2_REPORT.md), [docs/primary_comparators.md](../docs/primary_comparators.md) |
| 26/10/2026 | **Thầy quyết: LLP chính chỉ còn túi 8** (giao thức §11, deviation 7). PRODEN tune thêm Adam; thêm bản tune cho label spreading, self-training, forward correction (deviation 8) | Xong — lượt 3 chạy 12 phút trên Kaggle CPU |
| 26/10/2026 | G6 lượt 3: 162/162 dòng, cùng dữ liệu. **Đối thủ chính đã chốt cho hai bối cảnh chính:** LLP túi 8 — DLLP tuyến tính mặc định; PLL — GLWS/ILL tune (PRODEN có Adam 0,738 vẫn thấp hơn). Bối cảnh phụ (tạm thời): nhãn nhiễu — CL + TFM; SSL — self-training tune; PU — nnPU | Xong — [G6_ROUND2_REPORT §7](G6_ROUND2_REPORT.md), [docs/primary_comparators.md](../docs/primary_comparators.md) |
| 26/10/2026 | T7: kiểm duyệt độc lập PROOFS §0–5 và §7 (thầy giao Claude chọn cách: một phiên kiểm duyệt riêng, không thấy quá trình soạn, theo quy trình phản biện toán) | **Xong** — 0 Critical, 4 Major, 12 Minor; mọi phát hiện được kiểm lại độc lập rồi sửa ([phản hồi](G6_T7_REVIEW_RESPONSE.md), [báo cáo gốc](PROOFS_REVIEW_G6.md)). Hai hệ quả lớn: lý thuyết không phủ cấu hình mặc định (khối, bag-block) → việc mở G4-5 và đo ở lượt 5; Định lý 2 trình bày như bổ đề nhiễu loạn đã biết. G6-T7a (trích dẫn AUC) đã đóng |
| 26/10/2026 | T6: bộ chạy G7 — `icgibbs/g7/` (dữ liệu, chia, nhãn yếu, phương pháp, metric, ghi kết quả, R̂, PLL thật) và `run_g7.py`; 30 test; thiết kế [G7_RUNNER_DESIGN](G7_RUNNER_DESIGN.md). Còn: LLP-Bench, phần `ablation` | Phần chính xong; tổng duyệt ở lượt 5 |
| 26/10/2026 | Lượt 5 (GPU): tổng duyệt bộ chạy G7 trên tập phát triển (LLP túi 8, PLL) + G4-5 (sửa MH) + R̂ | Code xong; chờ thầy chạy (`g6_code_v5.zip`) |
| 26/10/2026 | Baseline còn thiếu của bối cảnh phụ: GLWS cho PU, VPU, ILL (nhãn nhiễu, SSL), VIME (`icgibbs/baselines_secondary.py`, 7 test). In-Context PU: không có code truy cập được → chưa tái hiện được | Code xong; chờ lượt 4 trên Kaggle (`g6_code_v4.zip`, `run_g6_round4.sh`, CPU khoảng 2,5–3,5 giờ) |
| 27/10/2026 | G6 lượt 4 (CPU, 4 giờ): 504/504 dòng. **Mọi đối thủ chính đã chốt** (PU-SCAR đổi sang VPU tune; các bối cảnh phụ khác không đổi). T5: mã băm 33 dataset, 33/33 khớp G5 | Xong — [G6_ROUND45_REPORT](G6_ROUND45_REPORT.md) §1–2, [docs/dataset_hashes.csv](../docs/dataset_hashes.csv) |
| 27/10/2026 | G6 lượt 5 (GPU 11 giờ — em ước sai, Mục 5 của báo cáo): bộ chạy G7 tái lập thí điểm khi dữ liệu không đổi. **LLP túi 8 dưới xử lý đúng giao thức: Gibbs − DLLP tuyến tính +0,003 AUC (thí điểm +0,013), quy tắc 60% chặt 3/6.** PLL vững (6/6, +0,036). G4-5: sửa MH không cải thiện. R̂: trộn chưa đủ ở mức nhãn trong 30 vòng quét (LLP, PLL q = 0,6) | Xong — [G6_ROUND45_REPORT](G6_ROUND45_REPORT.md) §3–4. **Thầy quyết (27/10): giữ LLP túi 8 là khẳng định chính**, ghi trước kỳ vọng hiệu ứng nhỏ (giao thức §11) |
| 27/10/2026 | Sửa sau lượt 5: quy tắc dự phòng khi fit lại baseline tune hỏng (deviation 10); `--mh-check` chỉ LLP; R̂ ghi metric từng chuỗi; `--synthetic_shape` để đo thời gian ở cỡ lớn. 97 test qua, 2 bỏ qua | Xong; lượt 6 (đo thời gian, dữ liệu tổng hợp) chờ thầy chạy (`g6_code_v6.zip`) |

## 5. Cần thầy làm hoặc xác nhận
- **Chạy G6 lượt 6 trên Kaggle (GPU khoảng 2 giờ, dữ liệu tổng hợp):** hướng dẫn ở [G4/README](../G4/README.md), mục "G6 lượt 6". Đo thời gian và bộ nhớ ở cỡ của tập đánh giá trước khi lên lịch G7.
- Không bắt buộc: nếu thầy mở được liên kết ẩn danh của In-Context PU (`anonymous.4open.science/r/puicl-58B1`) thì gửi em code và trọng số; nếu không, baseline này ghi là chưa tái hiện được.
- Không bắt buộc: Hình 1 của bài SAR-EM (giả thuyết 2); không còn ảnh hưởng tới quyết định nào.
- Đã xong: phạm vi LLP sau lượt 5 — giữ túi 8 là khẳng định chính (27/10); lượt 4 và lượt 5 trên Kaggle (27/10); kho git; phạm vi PU-SAR và quy tắc chọn đối thủ chính (26/10); phạm vi LLP — chỉ túi 8 (26/10); lượt 3 trên Kaggle (26/10); cách kiểm duyệt lý thuyết (thầy giao Claude chọn, 26/10).
