# Tái hiện số công bố của baseline (G6)

Theo [02 §G6](../../research-handbook/02_QUY_TRINH_CONG_KIEM_DUYET.md) và [G6_PLAN](../G6/G6_PLAN.md) Mục 3.

**Tiêu chí (ghi trước khi chạy, 26/10/2026):** đạt nếu số tái hiện cách số công bố ≤ 1 điểm % (accuracy hoặc AUC), hoặc nằm trong 1 độ lệch chuẩn mà bài công bố hay đo được qua các lần lặp — lấy ngưỡng rộng hơn. Không đạt thì xếp loại: (a) có nguyên nhân rõ; (b) chưa rõ nguyên nhân.

**Ngoại lệ về dữ liệu:** nếu dataset gốc của baseline trùng tập đánh giá của dự án (Lost, MSRCv2, BirdSong) thì tái hiện trên một dataset khác trong cùng bài.

## Bảng tổng hợp

| Baseline | Code | Dataset tái hiện | Metric | Công bố | Tái hiện | Kết luận |
|---|---|---|---|---|---|---|
| SAR-EM | chính thức (`sarpu`, commit 6e4fc3d8) | Breast Cancer, Mushroom (bài gốc, 4 thuộc tính propensity, 25 thí nghiệm) | ROC-AUC, MSE của f | Mushroom 1,000 / 0,103; Breast Cancer 0,983 / 0,058 | Mushroom 1,000 / 0,098; Breast Cancer 0,994 / 0,096 | Mushroom **đạt**; Breast Cancer **lệch**, xếp loại **(b)** (26/10/2026, sau G6 lượt 2): đã khoanh vùng ở phần thiết lập dữ liệu và đánh giá dùng chung, không ở thuật toán — Naive (code gốc) lệch cùng chiều; đổi tập thuộc tính không giải thích được. Bản bọc trùng code gốc tuyệt đối (100/100 thí nghiệm) |

## SAR-EM (Bekker, Robberechts & Davis, ECML-PKDD 2019)

- **Script:** `G4/code/g6_reproduce_sarem.py`. Dùng đúng code tiền xử lý trong notebook của kho gốc, đúng `generate_extended_data`, `label_data` (cơ chế `simple_0.2_0.8`) và `pu_learn_sar_em`.
- **Thích ứng để chạy được trên thư viện mới.** Không đổi hành vi thuật toán:
  1. `LogisticRegressionPU` của gói gọi `LogisticRegression(multi_class=...)`, tham số đã bị bỏ khỏi scikit-learn mới. Dự án truyền một lớp tương đương vào `pu_learn_sar_em`: cùng cách đánh trọng số propensity của `BasePU`, cùng `C = 1`, solver liblinear.
  2. `generate_attributes` gán mảng 1 phần tử vào một ô; numpy ≥ 2 báo lỗi. Dự án thay bằng hàm y hệt nhưng lấy phần tử `[0]`, nên cùng dãy số ngẫu nhiên.
  3. Notebook dùng `np.NaN` (đã bị bỏ ở numpy 2) và gán số vào cột chữ (pandas 3 không cho). Dự án đặt `np.NaN = np.nan` và tắt kiểu chuỗi mới của pandas.
- **Kiểm bản bọc của dự án** (`icgibbs.baselines_sar.SAREM`): trên cùng dữ liệu, bản bọc cho đúng cùng AUC với lời gọi trực tiếp code chính thức (chênh 0 trên dữ liệu thử). Script in lại chênh lệch này trên dữ liệu thật.
- **Kiểm trên dữ liệu tổng hợp** (`tests/test_baselines_g6.py`):
  - Khi thuộc tính propensity tách khỏi thuộc tính phân loại (thiết lập của bài): AUC của SAR-EM cách oracle ≤ 0,01 và MSE của propensity < 0,02.
  - Khi z vừa là feature vừa quyết định propensity (thiết lập PU-SAR của dự án): SAR-EM hơn bộ ngây thơ > 0,05 AUC nhưng không đạt oracle (0,82 so với 0,85). Đây là giới hạn định danh đã biết của mô hình SAR khi hai phần dùng chung thuộc tính.
- **Số công bố cần so** (Hình 1 của arXiv 1809.03207v4, 4 thuộc tính propensity):

  | Dataset | Phương pháp | ROC-AUC | MSE của f |
  |---|---|---|---|
  | Breast Cancer | SAR-EM | 0,983 | 0,058 |
  | Breast Cancer | Naive | 0,992 | 0,127 |
  | Breast Cancer | Supervised | 0,994 | 0,028 |
  | Mushroom | SAR-EM | 1,000 | 0,103 |
  | Mushroom | Naive | 1,000 | 0,173 |
  | Mushroom | Supervised | 1,000 | 0,000 |

  Các số này lấy từ bản trích văn bản của hình. Cách gán cột theo thứ tự (Naive, SAR-EM, KM2, TIcE, Sup.) là suy luận của em, vì Supervised phải tốt nhất. **Cần thầy xác nhận trên hình gốc** trước khi kết luận.
- **Kết quả (26/10/2026, Kaggle, `G4/pilot/g6_sarem_reproduction.csv`):** chi tiết ở [G6_T1_SAREM_REPORT](../G6/G6_T1_SAREM_REPORT.md) Mục 2.
  - Bản bọc của dự án cho đúng cùng kết quả với code chính thức (chênh AUC = 0 trên 50/50 thí nghiệm).
  - Mushroom đạt ở mọi metric của SAR-EM.
  - Breast Cancer: SAR-EM có AUC 0,994 so với 0,983 công bố (cao hơn), MSE 0,096 so với 0,058. Naive cũng lệch cùng chiều về MSE ở cả hai dataset.
  - Giả thuyết đang kiểm: (1) tập thuộc tính của bộ phân loại — chạy lại với `--cls_mode orig_plus1` và `all`; (2) cách gán cột trong Hình 1 — cần thầy xác nhận trên hình gốc.
- **Kết quả kiểm giả thuyết 1 (G6 lượt 2, `G4/pilot/g6r2_sarem_repro_*.csv`):** `orig_plus1`: SAR-EM 0,994 / 0,098, Naive 0,990 / 0,160; `all`: SAR-EM 0,988 / 0,097 (AUC đạt), Naive 0,980 / 0,158 (AUC lệch). Không chế độ nào tái hiện cả bốn số; MSE của f gần như không đổi theo tập thuộc tính. **Giả thuyết 1 bị bác.** Xếp loại cuối: **(b)**, chưa rõ nguyên nhân nhưng đã khoanh vùng ngoài thuật toán SAR-EM (Naive, chỉ là hồi quy logistic trên s, lệch cùng chiều). SAR-EM vẫn dùng được; bài ghi rõ chênh lệch. Chi tiết: [G6_ROUND2_REPORT](../G6/G6_ROUND2_REPORT.md) Mục 4.

## GLWS cho LLP — khác biệt giữa bài và code chính thức (phát hiện G6, 26/10/2026)
- Bài GLWS mô tả E-step là forward-backward trên NFA, tức hậu nghiệm của nhãn từng mẫu khi biết số đếm của túi.
- Dự án tính đúng đại lượng đó bằng quy hoạch động theo số đếm, và kiểm bằng liệt kê mọi cách gán nhãn (`tests/test_baselines_ext.py::test_glws_estep_is_exact_posterior`).
- So với hàm `create_proportion_graph` của kho chính thức (commit f577553):
  - Đại lượng log P(đếm = c | đếm ≤ c) **khớp số**.
  - **Nhãn giả không khớp**, và tổng theo túi không bằng c. Ví dụ túi 5 mẫu, c = 1: tổng chỉ 0,45; túi 4 mẫu, c = 2: tổng 1,59. Vậy hàm của kho không trả về hậu nghiệm chính xác.
- Quyết định: baseline GLWS của dự án dùng hậu nghiệm chính xác (bản đúng theo bài, và là bản mạnh hơn). Ghi ở giao thức Mục 11.
- Chưa tái hiện số công bố của GLWS: các thí nghiệm LLP của bài dùng CIFAR với ResNet, rất tốn GPU. Sẽ chọn cấu hình nhỏ nhất của bài để tái hiện ở lượt sau.

## LLP-VAT, DLLP
- Hàm `ProportionLoss` và `VATLoss` sao chép nguyên văn từ kho chính thức (MIT, `icgibbs/third_party/llp_vat_losses.py`; đã so `diff` với bản gốc).
- Proportion loss viết dạng vector, vì phép kiểm `simplex.check` của kho chỉ nhận vector một chiều ở torch mới. Test kiểm khớp số với cách gọi từng túi.
- Ramp-up của trọng số nhất quán dùng cùng công thức exp(−5(1−t)²) như kho (tệp ramps.py của kho có giấy phép CC BY-NC nên không sao chép).
- Chưa tái hiện số công bố (CIFAR).

## PRODEN
- Dùng đúng hàm `partial_loss` của kho chính thức, nạp từ kho đã ghim commit 637b531 (kho không có tệp giấy phép nên không sao chép).
- Dùng SGD momentum 0,9 như `main.py` của kho (cấu hình mặc định).
- **Phát hiện ở G6 lượt 2:** trên tập phát triển PLL, bản này chỉ đạt 0,573 accuracy (mặc định) và 0,716 (tune), thấp hơn bản Adam cũ của dự án (0,740) — SGD với lr ≤ 1e-2 chưa hội tụ trên dữ liệu bảng nhỏ. Từ lượt 3, không gian tune có thêm bộ tối ưu ∈ {SGD, Adam} (docs/tuning_spaces.md); mặc định vẫn là SGD như kho chính thức.
- Bản dự phòng cài lại cho kết quả trùng với hàm chính thức (test `test_proden_official_loss_matches_reimplementation`).
- Chưa tái hiện số công bố: bài dùng MNIST và các bộ tương tự, sẽ làm ở lượt sau.

## EasyLLP, RC, CC
- Không có code chính thức nên cài theo bài: EasyLLP theo Eq. 3; RC và CC theo Feng et al. 2020.
- Số công bố để tái hiện:
  - EasyLLP: accuracy trên MNIST chẵn/lẻ với hồi quy logistic, nhiều cỡ túi.
  - RC, CC: một trong các bộ chuẩn của bài (MNIST, Fashion-MNIST, Kuzushiji-MNIST); bảng cụ thể sẽ chọn khi làm. Không dùng Lost, MSRCv2, BirdSong vì đó là tập đánh giá RQ5.
- Lượt tái hiện này cần MNIST, sẽ làm sau lượt chọn đối thủ chính.

## Baseline phụ thêm ở G6 lượt 4 (26/10/2026) — `icgibbs/baselines_secondary.py`
- **GLWS cho PU** (kho GLWS, `pos_ulb/imp_pos_ulb.py`): CE trên mẫu dương có nhãn + CE với nhãn giả của batch không nhãn, coi batch là một túi có số dương = tiên nghiệm × cỡ batch. E-step dùng hậu nghiệm chính xác (forward–backward O(K²), `count_posterior`, khớp vét cạn và khớp DP autograd tới 1e-15) thay cho `create_proportion_graph`, cùng lý do như GLWS cho LLP (giao thức §11, deviation 5). Tập không nhãn gồm toàn bộ dữ liệu như `include_lb_to_ulb=True` của kho.
- **VPU** (kho chính thức HC-Feynman/vpu, MIT): loss biến phân và điều chuẩn MixUp viết lại đúng theo `vpu.py` (log Φ từ log-softmax, mục tiêu MixUp λΦ(x) + (1 − λ)·1, Adam betas (0,5; 0,99), lr giảm một nửa mỗi 20 epoch). Tập "x" gồm cả mẫu dương có nhãn như `x_loader` của kho.
- **ILL** (kho GLWS/ILL): nhãn nhiễu theo `noisy_label/imp_noisy_label.py` (noise_loss + em_loss + con_loss + cân bằng entropy; ma trận nhiễu khởi tạo đơn vị, chuẩn hoá hàng và cột); SSL theo `semisup/imp_semisup.py` (căn phân phối EMA 0,999, giữ top-3, cân bằng entropy). **Khác kho:** dữ liệu bảng không có phép tăng cường chuẩn nên góc nhìn "yếu" và "mạnh" là cùng một x; khi đó con_loss của ILL nhãn nhiễu trở thành một số hạng giống tối thiểu hoá entropy.
- **VIME** (Yoon et al., NeurIPS 2020): cài lại theo bài (kho TF1 không có tệp giấy phép). Tự giám sát: encoder một lớp ẩn cỡ d, ước lượng mặt nạ (BCE) và khôi phục feature (α·MSE), mặt nạ Bernoulli(p_m), giá trị thay thế lấy từ phân phối biên (hoán vị từng cột), RMSprop, 10 epoch. Bán giám sát: bộ dự đoán MLP trên encoder cố định, CE có nhãn + β·phương sai dự đoán qua K = 3 bản che, Adam, 1.000 vòng lặp, dừng sớm theo loss trên 10% phần có nhãn. *Ghi nhận ở G6 lượt 4:* với bảng ít feature, encoder cỡ d với ReLU (đúng kiến trúc của kho) làm mất thông tin — banknote (d = 4) chỉ đạt AUC 0,93 trong khi mọi phương pháp khác đạt 1,00; tune không đổi kết quả. Nhiều khả năng đây là giới hạn của kiến trúc gốc chứ không phải lỗi cài đặt, nhưng chưa đối chiếu bằng cách chạy code TF1 chính thức; bài sẽ nêu khi báo cáo SSL.
- **In-Context PU (PUICL, arXiv 2605.05591):** bài trên arXiv không dẫn kho công khai; liên kết ẩn danh `anonymous.4open.science/r/puicl-58B1` không truy cập được từ môi trường làm việc (26/10/2026). Theo giao thức: ghi là **chưa tái hiện được**; nếu thầy mở được liên kết và có trọng số pretrain thì bổ sung trước G7.
- Chưa tái hiện số công bố của các baseline phụ này trên dataset gốc (ảnh); việc đó thuộc T3 cùng các baseline chính.

