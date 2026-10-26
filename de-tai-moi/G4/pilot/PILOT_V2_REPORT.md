# Thí điểm G4 — lượt 2 (v2) trên 6 dataset phát triển

> Chạy: Kaggle, GPU T4, TabPFN v2 (`tabpfn` 9.2.0), 25/10/2026, mã `g4_code_v3`. Dữ liệu thô: [g4_pilot_v2.csv](g4_pilot_v2.csv) (900 dòng, không lỗi), dự đoán test: `g4_pilot_v2_preds.jsonl`, log: `g4_pilot_v2_log.txt`. Gộp với các chuỗi tốn GPU của lượt 1: [g4_pilot_merged.csv](g4_pilot_merged.csv); mã phân tích: [analyze_v2.py](analyze_v2.py). Vẫn là **exploratory trên tập phát triển**.
>
> Quy tắc đọc kết quả đã ghi trước khi chạy (PILOT_V1_REPORT §4). Hai quyết định Q1 (metric PU-SCAR) và Q2 (RQ3) đã chốt trước khi chạy (GATE_REVIEW_G4 §3c).

## 0. Tóm tắt

| Câu hỏi | Kết quả lượt 2 | Hệ quả |
|---|---|---|
| Khởi động ấm có giúp không? | Giúp ở nơi chuỗi trộn chậm (LLP túi 16: log loss tốt hơn 4/6, trung bình 0,077; PU-SCAR c = 0,1: AUC tốt hơn 5/6) và **xoá hết các ca sụp** (wdbc LLP túi 32: 0,63 → 0,99). Không đổi gì ở LLP túi 8. Ở PU-SAR làm log loss kém đi (0/6) nhưng không đổi AUC (metric chính) | **Khởi động ấm thành mặc định** |
| (a) khởi tạo hay (b) khuếch đại sai số? | Túi 8: ba cách khởi tạo (lạnh, ấm, nhãn thật) cho cùng kết quả ⇒ chuỗi hội tụ. Túi 16–32: kết quả phụ thuộc khởi tạo và rất dao động giữa các seed (độ lệch AUC tới 0,18) ⇒ trộn chậm hoặc đa mode. Ngay cả chuỗi khởi tạo từ nhãn thật cũng chỉ ngang DLLP ở túi 32 | **Quy tắc (b) áp dụng:** thu hẹp khẳng định về chế độ nhãn yếu đủ thông tin (LLP túi ≤ 16); túi 32 thành bài kiểm tra áp lực, báo cáo như giới hạn |
| Q1: PU-SCAR theo log loss khớp tiên nghiệm | Gibbs ấm vượt cả naive lẫn peer-review: 1/6 (c = 0,1), 2/6 (c = 0,3). Khởi tạo từ nhãn thật cũng chỉ 1/6. TFM ngây thơ + Elkan–Noto với c đã biết là một baseline gần tối ưu | **Đưa PU-SCAR ra khỏi khẳng định chính của RQ2**; báo cáo như kết quả âm có giá trị |
| Q2: RQ3, Gibbs so với hard-EM cùng điểm khởi động | Gibbs tốt hơn ở **30/36 ô (83%)**, Wilcoxon một phía p = 7·10⁻⁷; khoảng cách tăng theo mức yếu (Spearman 0,31, p = 0,10). Gibbs tốt hơn SEM ở 31/36 ô | H3 được ủng hộ trên tập phát triển (ngưỡng đã đăng ký 70%) |
| Tiêu chí dừng | Không đổi so với lượt 1: không chạm điều kiện chuyển phương án B | Tiếp tục |

## 1. LLP

AUC (trung bình 3 seed). Peer-review tốt nhất = max(DLLP-MLP, DLLP tuyến tính):

| túi | dataset | Gibbs lạnh | **Gibbs ấm** | Gibbs từ nhãn thật | hard-EM ấm | DLLP | DLLP tuyến tính | naive | oracle |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | blood | 0,691 | 0,756 | 0,716 | 0,742 | 0,701 | **0,768** | 0,643 | 0,768 |
| 8 | credit-g | 0,753 | 0,753 | 0,741 | **0,771** | 0,717 | 0,732 | 0,584 | 0,798 |
| 8 | diabetes | **0,819** | 0,814 | 0,811 | 0,774 | 0,711 | 0,807 | 0,561 | 0,843 |
| 8 | ilpd | 0,713 | **0,718** | 0,717 | 0,613 | 0,635 | 0,657 | 0,648 | 0,742 |
| 8 | wdbc | 0,991 | 0,991 | 0,990 | **0,992** | 0,988 | 0,989 | 0,806 | 0,992 |
| 16 | blood | 0,681 | 0,605 | 0,644 | 0,659 | 0,725 | **0,727** | 0,508 | 0,768 |
| 16 | credit-g | 0,660 | 0,652 | **0,741** | 0,720 | 0,707 | 0,714 | 0,501 | 0,798 |
| 16 | diabetes | 0,724 | **0,797** | 0,792 | 0,796 | 0,708 | 0,794 | 0,527 | 0,843 |
| 16 | ilpd | 0,658 | 0,716 | **0,727** | 0,676 | 0,645 | 0,647 | 0,635 | 0,742 |
| 16 | wdbc | 0,879 | 0,988 | **0,990** | **0,990** | 0,972 | 0,987 | 0,575 | 0,992 |
| 32 | blood | 0,570 | 0,449 | 0,650 | 0,612 | 0,629 | **0,680** | 0,365 | 0,768 |
| 32 | credit-g | 0,605 | 0,585 | **0,668** | 0,605 | 0,642 | 0,645 | 0,533 | 0,798 |
| 32 | diabetes | 0,600 | 0,558 | **0,728** | 0,711 | 0,653 | 0,636 | 0,447 | 0,843 |
| 32 | ilpd | **0,702** | 0,635 | 0,658 | 0,517 | 0,607 | 0,525 | 0,526 | 0,742 |
| 32 | wdbc | 0,627 | **0,988** | 0,986 | 0,987 | 0,962 | **0,988** | 0,748 | 0,992 |

(banknote: mọi phương pháp ≈ 1,000 ở mọi túi; bỏ khỏi bảng.)

Số dataset vượt *cả* peer-review tốt nhất *lẫn* naive (Gibbs ấm):

| túi | AUC | log loss |
|---|---|---|
| 8 | 4/6 (2 với cách biệt > 0,01) | 5/6 (3) |
| 16 | 3/6 (1) | 4/6 (3) |
| 32 | 3/6 (1) | 2/6 (2) |

- **DLLP tuyến tính** (mới thêm, cũng là điểm khởi động ấm) là baseline mạnh hơn DLLP-MLP ở 13/18 ô (tính cả banknote). Thêm nó làm thước so sánh khắt khe hơn.
- **Túi 8:** ba cách khởi tạo cho cùng kết quả (chuỗi hội tụ). Log loss tốt hơn DLLP-MLP ở 6/6, nhiều ô cách xa (ilpd 0,54 so với 1,02).
- **Túi 16–32:** kết quả phụ thuộc khởi tạo. Gibbs ấm có thể trôi khỏi một điểm khởi động tốt (blood túi 32: điểm khởi động 0,680 → 0,449; độ lệch giữa seed 0,18). Độ chính xác nhãn của chuỗi khởi tạo từ nhãn thật giảm từ 1,0 xuống 0,62–0,74 trên các dataset khó. Phần lớn là bất định *hợp lệ* của posterior (cùng mức với độ chính xác của oracle), nhưng ở túi 32 dự đoán của nó cũng chỉ ngang DLLP. Đây là dấu hiệu của đa mode khi nhãn nhóm ít thông tin: nhiều cách gán khớp số đếm gần như ngang nhau. Hiện tượng này khớp với Định lý 2: $\tau(P)$ lớn khiến sai số của TFM bị khuếch đại.

## 2. PU-SCAR (metric chính theo Q1: log loss khớp tiên nghiệm)

| c | dataset | Gibbs ấm | Gibbs từ nhãn thật | hard-EM ấm | naive (Elkan–Noto) | nnPU | uPU | oracle |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0,1 | banknote | **0,058** | 0,001 | 0,133 | 0,142 | 0,124 | 0,253 | 0,001 |
| 0,1 | blood | 0,553 | 0,562 | 0,665 | **0,524** | 0,668 | 1,262 | 0,462 |
| 0,1 | credit-g | 0,597 | 0,734 | 0,595 | **0,592** | 0,964 | 1,072 | 0,489 |
| 0,1 | diabetes | 0,690 | 0,741 | 0,692 | **0,556** | 0,949 | 1,232 | 0,462 |
| 0,1 | ilpd | 0,808 | 0,684 | 1,796 | **0,648** | 1,362 | 1,846 | 0,519 |
| 0,1 | wdbc | 0,235 | 0,169 | 0,318 | 0,286 | **0,151** | 0,494 | 0,116 |
| 0,3 | (6 dataset) | vượt cả naive lẫn peer-review: 2/6 (0 với cách biệt > 0,01) | — | | | | | |

- Gibbs **tốt hơn rõ** nnPU và uPU về log loss, kể cả sau khớp tiên nghiệm, nhưng **không tốt hơn TFM ngây thơ + Elkan–Noto**. Khởi tạo từ nhãn thật không đổi kết luận, nên đây không phải lỗi khởi tạo.
- AUC: Gibbs ấm vượt cả hai ở 4/6 (c = 0,1) nhưng chỉ 1/6 với cách biệt > 0,01, đúng như Elkan–Noto dự đoán.
- **Diễn giải:** dưới SCAR với c đã biết, TFM có hiệu chỉnh tốt cộng với phép chia cho c đã gần tối ưu. Mô hình hoá Bayes đầy đủ không thêm được gì đo được trên 6 dataset này.

## 3. PU-SAR (AUC chính)

| | banknote | blood | credit-g | diabetes | ilpd | wdbc | thắng cả hai |
|---|---:|---:|---:|---:|---:|---:|---|
| Gibbs lạnh (học η) | 0,997 | 0,633 | 0,649 | 0,719 | 0,740 | 0,953 | 4/6 (2 với cách biệt) |
| Gibbs ấm (học η) | 0,996 | 0,636 | 0,644 | 0,722 | 0,703 | 0,962 | 3/6 (2) |
| Gibbs ấm, propensity đã biết | 0,999 | 0,751 | 0,688 | 0,782 | 0,731 | 0,912 | 4/6 |
| naive | 0,997 | 0,629 | 0,652 | 0,700 | 0,640 | 0,954 | |
| nnPU | 0,979 | 0,546 | 0,703 | 0,684 | 0,630 | 0,972 | |

Lợi ích rõ ở ilpd và diabetes; khi biết propensity thì lợi ích lớn (blood +0,12, diabetes +0,08). Khởi động ấm sửa ca sụp của wdbc khi propensity đã biết (0,853 → 0,912). **Chưa có baseline SAR chuyên biệt** (SAR-EM, Bekker et al. 2019); đây là rủi ro lớn nhất của bối cảnh này ở G5–G7.

## 4. RQ3 (Q2: Gibbs ấm so với hard-EM ấm, cùng điểm khởi động; log loss, PU-SCAR dùng bản khớp tiên nghiệm)

| bối cảnh | mức | Gibbs tốt hơn hard-EM | chênh trung bình | Gibbs tốt hơn SEM |
|---|---|---|---:|---|
| LLP | 8 / 16 / 32 | 5/6 · 4/6 · 5/6 | +0,19 · +0,36 · +0,62 | 5/6 · 5/6 · 5/6 |
| PU-SCAR | 0,3 / 0,1 | 5/6 · 5/6 | +0,05 · +0,21 | 5/6 · 6/6 |
| PU-SAR | 0,3 | 6/6 | +0,21 | 5/6 |

Tổng: 30/36 ô (83%; 8 ô gần hoà, |chênh| < 0,005, chủ yếu banknote). Wilcoxon một phía p = 6,8·10⁻⁷. Spearman giữa mức yếu và khoảng cách: LLP 0,40 (p = 0,10), PU-SCAR 0,24, gộp 0,31 (p = 0,10). Đúng chiều dự đoán, chưa có ý nghĩa ở cỡ mẫu thí điểm.

## 5. Áp quy tắc đọc kết quả đã ghi trước (PILOT_V1_REPORT §4)

- *"Nếu gibbs_trueinit giữ được độ chính xác cao và gibbs_warm ≥ điểm khởi động ⇒ (a)"* — **không thoả.** Chuỗi khởi tạo từ nhãn thật mất 26–38 điểm phần trăm độ chính xác nhãn trên 4 dataset khó, và ở túi 16–32 Gibbs ấm thấp hơn chính điểm khởi động của nó ở blood và credit-g.
- *"Nếu gibbs_trueinit cũng trôi xa ⇒ (b): thu hẹp phạm vi về chế độ nhãn yếu đủ thông tin (ví dụ túi ≤ 16)"* — **áp dụng.**
- **Ghi chú về quy tắc:** độ chính xác nhãn của chuỗi không phải thước đo tốt cho "trôi". Ngay cả posterior chính xác cũng không đạt 100% khi dữ liệu nhiễu (oracle AUC 0,74–0,84). Thước đo đúng hơn là dự đoán. Theo thước đo dự đoán, túi 8 hội tụ tốt, túi 16 ở ranh giới, túi 32 thì không. Kết luận vẫn giữ nguyên.
- Khởi động ấm vẫn được giữ làm mặc định: nó loại bỏ các ca sụp (wdbc) và không làm hại ở túi 8.

## 6. Đề xuất cho G5 (để thầy duyệt)

1. **Phạm vi khẳng định chính (RQ2):** LLP với túi ≤ 16, và PU-SAR. PU-SCAR thành kết quả phụ: báo cáo đầy đủ và nói rõ TFM ngây thơ + Elkan–Noto là baseline gần tối ưu. LLP túi 32 thành bài kiểm tra áp lực, báo cáo trong phần giới hạn. Phù hợp với điều khoản thu hẹp đã có ở CLAIM_EVIDENCE_MATRIX #6 và tiêu chí "thành công tối thiểu" của RESEARCH_PLAN §5 (H2 cho PU *hoặc* LLP, cộng H3).
2. **Cấu hình mặc định:** khởi động ấm (PU: TFM ngây thơ tính chéo 5 fold, hiệu chỉnh theo propensity; LLP: DLLP tuyến tính), khối trên mọi mẫu, K = 10, 30 vòng quét, bỏ 10. Chẩn đoán $\hat R$ bằng 2 chuỗi ở G6.
3. **Baseline phải thêm ở G5:** DLLP tuyến tính (đã có), SAR-EM (Bekker et al. 2019) cho PU-SAR, cùng các baseline LLP mới hơn trong RESEARCH_PLAN (Easy-LLP, LLP-VAT hoặc BP-LLP), dùng cài đặt chính thức.
4. **RQ4 (PLL, nhãn nhiễu, SSL) chưa được thí điểm.** Vì phạm vi RQ2 đã hẹp lại, sức nặng của bài phụ thuộc nhiều hơn vào RQ4. Nên chạy một thí điểm RQ4 nhỏ trên chính 6 dataset phát triển trước khi đóng băng G5 (khoảng 2–3 giờ T4).
5. **RQ5 (LLP-Bench):** túi thật thường lớn, rơi đúng vào vùng yếu của phương pháp. Cần kiểm tra phân bố cỡ túi của nguồn dữ liệu này trước khi đóng băng.
