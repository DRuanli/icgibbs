# Đối thủ chính của từng bối cảnh (chọn trước G7)

Theo giao thức [G5 Mục 6](../G5/EXPERIMENT_PROTOCOL.md) và Mục 11 (deviation 3, 7, 8). **Quy tắc (ghi trước khi chạy tập đánh giá):** ứng viên là mọi baseline peer-review của bối cảnh, mỗi baseline xét cả bản mặc định lẫn bản tune 20 cấu hình. Đối thủ chính là ứng viên có metric chính trung bình tốt nhất trên tập phát triển (6 dataset; PLL: vehicle, segment, cmc), qua các **mức chính** của bối cảnh, 3 seed. Chọn bằng `G4/code/g6_select_comparators.py`; số liệu: `G4/pilot/g6_comparators.csv` (bản sau lượt 4), chi tiết ở [G6_ROUND2_REPORT](../G6/G6_ROUND2_REPORT.md) và [G6_ROUND45_REPORT](../G6/G6_ROUND45_REPORT.md).

Đối thủ thứ hai của mọi bối cảnh là **TFM ngây thơ** (cùng TFM, cùng ngữ cảnh, nhãn ngây thơ).

| Bối cảnh | Vai trò | Mức chính | Metric | Đối thủ chính | Trung bình trên tập phát triển | Trạng thái |
|---|---|---|---|---|---|---|
| LLP | Chính (H2) | túi 8 | ROC AUC | **DLLP tuyến tính, mặc định** (`dllplin_def`; Ardehaly & Culotta, ICDMW 2017) | 0,8254 (bản tune 0,8248) | **Chốt** 26/10/2026 |
| PLL tổng hợp | Chính (H4) | q = 0,3; 0,6 | accuracy | **GLWS/ILL, tune** (`glwspll_tun`) | 0,7451 (PRODEN tune có Adam: 0,7378) | **Chốt** 26/10/2026 sau lượt 3 |
| LLP túi 16 | Phụ | túi 16 | ROC AUC | cùng đối thủ của LLP chính (`dllplin_def`) | — | Chốt |
| LLP túi 32 | Kiểm tra áp lực | túi 32 | ROC AUC | cùng đối thủ của LLP chính | — | Chốt |
| PU-SAR | Phụ | E[e \| y=1] = 0,1; 0,3 | ROC AUC | nnPU, mặc định (`nnpu`) | 0,7260 (GLWS mặc định 0,7206; VPU tune 0,6889) | **Chốt** 27/10/2026 sau lượt 4 |
| PU-SCAR | Phụ | c = 0,1; 0,3 | log loss khớp tiên nghiệm (thấp là tốt) | **VPU, tune** (`vpu_tun`; Chen et al., NeurIPS 2020) | 0,5933 (nnPU mặc định 0,7130) | **Chốt** 27/10/2026 sau lượt 4 (In-Context PU: chưa tái hiện được) |
| Nhãn nhiễu phụ thuộc lớp | Phụ | r = 0,2; 0,4 | ROC AUC | Confident Learning + TFM (`cl_tfm`; Northcutt et al., JAIR 2021) | 0,8465 (ILL tune 0,8163; forward tune 0,8097) | **Chốt** 27/10/2026 sau lượt 4 |
| SSL | Phụ | 5%; 20% có nhãn | ROC AUC | Self-training, tune (`st_tun`) | 0,8185 (ILL tune 0,8105; label spreading 0,7705; VIME tune 0,7434) | **Chốt** 27/10/2026 sau lượt 4 |
| PLL thật (RQ5) | Kiểm chứng ngoại tại | — | accuracy | không cần (H5 xếp hạng mọi phương pháp) | — | — |

Ghi chú:
- **LLP:** đối thủ chính trùng với bộ khởi động ấm của In-Context Gibbs (DLLP tuyến tính, cùng cấu hình). Kiểm định H2-LLP vì vậy cũng trả lời câu hỏi: bước Gibbs có cải thiện được điểm xuất phát của nó không.
- **Ứng viên đã xét:** LLP: DLLP (MLP, tuyến tính), LLP-VAT, EasyLLP, GLWS. PLL: PRODEN, GLWS/ILL, RC, CC. PU-SAR: nnPU, uPU, SAR-EM, GLWS (PU), VPU. PU-SCAR: nnPU, uPU, GLWS (PU), VPU. Nhãn nhiễu: forward correction (cấu hình của thí điểm và cấu hình mặc định đã ghi), CL + TFM, ILL. SSL: label spreading, self-training, ILL, VIME. Mỗi baseline có bản mặc định và bản tune (CL + TFM không tune, giao thức Mục 4).
- **Mọi bối cảnh đã chốt** (27/10/2026, sau G6 lượt 4). Baseline duy nhất của giao thức Mục 4 không có trong ứng viên là In-Context PU (PU-SCAR): không có code truy cập được, ghi là chưa tái hiện được (docs/baseline_reproduction.md). Bối cảnh PU-SCAR là phụ.
- **PU-SCAR:** metric chính là log loss khớp tiên nghiệm, nên thứ tự ứng viên khác hẳn thứ tự theo AUC. Gibbs ấm trung bình 0,4343, tốt hơn đối thủ chính (0,5933) nhưng kém TFM ngây thơ (0,4182), như dự kiến cho nhãn yếu loại (1) (PROOFS §7).
- Mọi thay đổi sau ngày chốt phải ghi lý do ở cuối file và ở giao thức Mục 11, và phải xong trước khi chạy bất kỳ phương pháp nào trên tập đánh giá.

## Lịch sử
| Ngày | Thay đổi |
|---|---|
| 27/10/2026 | Lượt 4 (baseline còn thiếu của bối cảnh phụ): **chốt mọi bối cảnh phụ**. Nhãn nhiễu `cl_tfm`, SSL `st_tun`, PU-SAR `nnpu` không đổi; PU-SCAR đổi từ `nnpu` sang `vpu_tun`. Chi tiết: [G6_ROUND45_REPORT](../G6/G6_ROUND45_REPORT.md) |
| 26/10/2026 | Lượt 3: PRODEN tune có Adam (0,7378) không đổi đối thủ chính của PLL → chốt `glwspll_tun`. Nhãn nhiễu: `cl_tfm`; SSL: `st_tun` (tạm thời) |
| 26/10/2026 | Tạo file sau G6 lượt 2. LLP chốt `dllplin_def` (mức chính chỉ còn túi 8, deviation 7; khi gộp túi 8 và 16 quy tắc chọn `dllplin_tun`, 0,8191). PLL tạm `glwspll_tun`. PU-SAR, PU-SCAR: `nnpu` |
