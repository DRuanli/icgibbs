# Thí điểm G4 — lượt 1 (v1) trên 6 dataset phát triển

> Chạy: Kaggle, GPU T4, TabPFN v2 (gói `tabpfn` 9.2.0), 24/10/2026. Dữ liệu thô: [g4_pilot_v1.csv](g4_pilot_v1.csv) (612 dòng, không lỗi), log: [g4_pilot_v1_log.txt](g4_pilot_v1_log.txt). Thí điểm là **exploratory trên tập phát triển**; không dùng dataset đánh giá; không phải kết quả chính của bài.

## 1. Tiêu chí dừng đã đăng ký (RESEARCH_PLAN §5, mục 2)

> "phương pháp không vượt *cả* baseline peer-review tốt nhất *lẫn* TFM ngây thơ trên ít nhất 4/6 dataset, ở cả PU và LLP → phương án B"

Kết quả theo AUC trung bình 3 seed (số dataset mà In-Context Gibbs vượt cả hai):

| Bối cảnh | Gộp các mức (cách script tính) | Theo từng mức | Theo từng mức, cách biệt > 0,01 |
|---|---|---|---|
| PU-SCAR | 2/6 | c = 0,1: 1/6 · c = 0,3: 3/6 | 1/6 · 2/6 |
| PU-SAR (1 mức) | **4/6** | 4/6 | 2/6 |
| LLP | 2/6 | túi 8: **4/6** · túi 32: 1/6 | 3/6 · 1/6 |

**Kết luận theo đúng câu chữ:** điều kiện chuyển phương án B **không bị chạm** (PU đạt 4/6 nhờ PU-SAR; LLP đạt 4/6 ở túi 8).

**Nhưng phải nói rõ ba điểm yếu của kết luận này:**
1. Cách gộp (gộp các mức hay theo từng mức; PU = PU-SCAR hay PU-SAR) **không được đăng ký trước**. Script gộp các mức và lấy PU tốt hơn trong hai loại.
2. PU-SAR được thêm vào thí điểm ở G4 (nó có trong RQ2 nhưng không có trong mô tả thí điểm ở G3).
3. Hai trong bốn lần thắng ở PU-SAR là hoà về thực chất (banknote +0,001; blood +0,004 so với naive). Nếu đòi cách biệt > 0,01 thì không bối cảnh nào đạt 4/6.

Đề xuất của em: ghi nhận là **"không chạm điều kiện dừng, nhưng sát ngưỡng"**. Không chuyển phương án B. Chưa qua G4 cho tới khi có kết quả lượt 2 (Mục 4).

## 2. Kết quả theo bối cảnh

### LLP
| AUC (log loss) | túi 8: Gibbs | DLLP | naive | túi 32: Gibbs | DLLP | naive |
|---|---|---|---|---|---|---|
| banknote | 1,000 (0,000) | 1,000 (0,009) | 0,982 (0,599) | 0,973 (0,204) | 1,000 (0,067) | 0,740 (0,681) |
| blood | 0,691 (0,529) | 0,701 (0,593) | 0,643 (0,544) | 0,570 (0,549) | 0,629 (0,544) | 0,365 (0,563) |
| credit-g | **0,753** (0,564) | 0,717 (0,764) | 0,584 (0,610) | 0,605 (0,605) | 0,642 (0,609) | 0,533 (0,610) |
| diabetes | **0,819** (0,504) | 0,711 (0,964) | 0,561 (0,647) | 0,600 (0,641) | 0,653 (0,653) | 0,447 (0,652) |
| ilpd | **0,713** (0,533) | 0,635 (1,016) | 0,648 (0,595) | **0,702** (0,582) | 0,607 (0,761) | 0,526 (0,600) |
| wdbc | 0,991 (0,117) | 0,988 (0,137) | 0,806 (0,639) | **0,627** (0,630) | 0,962 (0,304) | 0,748 (0,646) |

- **Túi 8: mạnh.** Vượt DLLP 4/6 (3/6 với cách biệt rõ: +0,04 đến +0,11), vượt naive 6/6; log loss tốt hơn DLLP ở 6/6, thường cách xa (diabetes 0,50 so với 0,96).
- **Túi 32: thất bại** ở 5/6 so với DLLP; wdbc sụp (0,63 ± 0,28 giữa các seed). Chẩn đoán: tỉ lệ nhãn đổi ở vòng quét cuối 0,35–0,41, tức chuỗi gần như đang xáo nhãn ngẫu nhiên, chưa hội tụ.

### PU-SCAR
- **AUC ≈ TFM ngây thơ** (chênh trung bình −0,025 đến +0,043, đều nằm trong độ lệch giữa các seed). **Đúng như lập luận Elkan–Noto** đã nêu ở GATE_REVIEW_G4 #2, nay có thêm bằng chứng trên dữ liệu thật.
- **Log loss của Gibbs kém naive (có hiệu chỉnh Elkan–Noto) ở c = 0,1** (ví dụ credit-g 0,73 so với 0,61; wdbc 0,51 so với 0,31). Nghi vấn: khởi tạo từ ℓ đặt khoảng 47% mẫu không nhãn là dương trong khi thật chỉ khoảng 28%, và 10 vòng bỏ đi chưa đủ để chuỗi về cân bằng.

### PU-SAR (E[e | y = 1] = 0,3, propensity dốc theo cột số đầu tiên)
- Gibbs học η: vượt cả hai ở 4/6, rõ ở ilpd (+0,10 so với naive) và diabetes (+0,02). η học được tản mát quanh giá trị thật — PU-SAR không định danh được, như đã biết.
- Gibbs với propensity **đã biết** tốt hơn rõ ở blood (0,752), credit (0,715), diabetes (0,782), nhưng **sụp ở wdbc** (0,853; log loss 1,6). Giải thích nghi vấn: với propensity đã biết, mẫu không nhãn có propensity cao được khởi tạo âm; TFM học lại chính các nhãn sai đó, nên chuỗi kẹt.

### RQ3 (tín hiệu sớm, AUC)
Gibbs > hard-EM: LLP 6/6, PU-SAR 5/6, PU-SCAR 2/6 (ở PU-SCAR hard-EM trùng naive vì argmax hiếm khi lật nhãn). Gibbs > SEM (chỉ mẫu cuối): 4–5/6 ở mọi bối cảnh.

### Chi phí
Chuỗi 30 vòng quét: 37–86 s mỗi lần chạy trên T4 (n_train 398–960), khớp ngân sách G2.

### Baseline
uPU có log loss rất tệ (điểm số không bị chặn). nnPU mạnh ở wdbc và banknote. Đây là các bản cài lại, chưa tune, nên chỉ có tính chỉ báo.

## 3. Chẩn đoán chung và liên hệ với lý thuyết

Các thất bại tập trung ở chế độ **nhãn yếu ít thông tin** (LLP túi 32, PU c = 0,1) và ở chỗ **khởi tạo xa posterior**. Hai cách giải thích, khác nhau về hệ quả:
- **(a) Khởi tạo và trộn chậm.** Từ nhãn ngẫu nhiên, TFM học trên nhãn gần ngẫu nhiên nên dự đoán gần phẳng; chuỗi ở lại vùng ngẫu nhiên. Sửa được bằng khởi động ấm, vì khởi tạo không đổi phân phối mục tiêu.
- **(b) Khuếch đại sai số.** Định lý 2: sai lệch $\le\tau(P)\,\bar\varepsilon$. Khi nhãn yếu ít thông tin thì chuỗi chính xác trộn chậm ($\tau$ lớn), nên sai số nhỏ của TFM cũng bị khuếch đại thành sai lệch lớn của phân phối dừng. Khởi động ấm **không** sửa được trường hợp này.

**Kiểm tra trên CPU** (`warm_start_check.py`, `drift_check.py`; TFM thay thế là mini-TabPFN của khóa học, dữ liệu breast_cancer = wdbc):
- LLP túi 32: khởi động ấm nâng AUC từ 0,64 ± 0,35 lên 0,84 ± 0,09, nhưng vẫn **thấp hơn chính điểm khởi động** (DLLP tuyến tính 0,977).
- Khởi tạo từ nhãn thật: chuỗi trôi về khoảng 55% (túi 32) và 65% (túi 8) nhãn đúng.
- PU: khởi động ấm không đổi gì.
- **Giới hạn của kiểm tra này:** mini-TabPFN chỉ được huấn luyện với ngữ cảnh ≤ 100 hàng và cho xác suất rất phẳng (log loss ≈ 0,6–0,65 dù AUC cao), nên là một $q$ kém hơn TabPFN nhiều. Kết quả **ủng hộ (b) với một $q$ yếu**, nhưng không cho biết TabPFN rơi vào (a) hay (b). Chỉ có GPU trả lời được.

## 4. Đề xuất lượt 2 (v2) — ghi trước khi chạy

**Mục đích:** phát triển phương pháp trên tập phát triển (đúng vai trò của tập này), **không** phải chạy lại tiêu chí dừng.

**Giữ nguyên:** 6 dataset, 3 seed, cách chia dữ liệu và sinh nhãn yếu (tất định theo seed, nên dùng lại được kết quả v1).

**Thêm:**
- `gibbs_warm` / `sem_warm`: chuỗi khởi động ấm. PU dùng TFM ngây thơ tính chéo 5 fold, hiệu chỉnh theo propensity; LLP dùng DLLP tuyến tính.
- `dllp_linear`: chính điểm khởi động của LLP, đồng thời là một biến thể DLLP để so sánh.
- `gibbs_known_warm` cho PU-SAR.
- Mức LLP túi 16, để thấy chỗ chuyển giữa 8 và 32.
- Với `--diag`: `gibbs_trueinit` (khởi tạo từ nhãn thật, chỉ để chẩn đoán) cho mọi mức LLP và PU-SCAR c = 0,1. Mọi chuỗi đều ghi độ chính xác của trạng thái so với nhãn thật qua các vòng quét.

**Quy tắc đọc kết quả (cố định trước):**
- Nếu `gibbs_trueinit` giữ được độ chính xác cao và `gibbs_warm` ≥ điểm khởi động của nó ⇒ (a). Khởi động ấm thành mặc định ở G5.
- Nếu `gibbs_trueinit` cũng trôi xa ⇒ (b). Thu hẹp phạm vi khẳng định về chế độ nhãn yếu đủ thông tin (ví dụ túi ≤ 16), ghi giới hạn này trong bài và trong RESEARCH_PLAN §9. Bản sửa MH (chính xác dưới A1) **không** giúp ở (b), vì (b) là sai số của chính $q$.
- Không đổi metric hay tiêu chí dừng dựa trên v2. Mọi lựa chọn rút ra từ v1 + v2 đều ghi trong EXPERIMENT_PROTOCOL ở G5, kèm ghi chú "chọn trên tập phát triển".

**Ước tính:** khoảng 2,5 giờ T4 khi chưa bật `--diag`, thêm khoảng 1,2 giờ nếu bật. Script tiếp tục từ CSV v1 (`--resume_from`).
