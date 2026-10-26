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
| 26/10/2026 | T1 chạy Kaggle: tái hiện SAR-EM (Mushroom đạt; Breast Cancer lệch, đang kiểm), SAR-EM trên tập phát triển, Gibbs ở PU-SAR mức 0,1 | Xong — [G6_T1_SAREM_REPORT](G6_T1_SAREM_REPORT.md). **Phát hiện:** khi có SAR-EM, Gibbs chỉ đạt 3/12 ô ở PU-SAR → chờ thầy quyết phạm vi PU-SAR |

## 5. Cần thầy quyết
- **Kho git:** xong — `github.com/DRuanli/icgibbs` (riêng tư), đẩy lên ngày 26/10/2026.
- **Phạm vi PU-SAR** và **quy tắc chọn đối thủ chính** (xét cả bản mặc định lẫn bản tune): xem [G6_T1_SAREM_REPORT](G6_T1_SAREM_REPORT.md) Mục 4.
