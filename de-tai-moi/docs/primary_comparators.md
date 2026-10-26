# Đối thủ chính của từng bối cảnh (chọn trước G7)

Theo giao thức [G5 Mục 6](../G5/EXPERIMENT_PROTOCOL.md) và Mục 11 (deviation 3, 7, 8). **Quy tắc (ghi trước khi chạy tập đánh giá):** ứng viên là mọi baseline peer-review của bối cảnh, mỗi baseline xét cả bản mặc định lẫn bản tune 20 cấu hình. Đối thủ chính là ứng viên có metric chính trung bình tốt nhất trên tập phát triển (6 dataset; PLL: vehicle, segment, cmc), qua các **mức chính** của bối cảnh, 3 seed. Chọn bằng `G4/code/g6_select_comparators.py`; số liệu: `G4/pilot/g6_comparators.csv`, chi tiết ở [G6_ROUND2_REPORT](../G6/G6_ROUND2_REPORT.md).

Đối thủ thứ hai của mọi bối cảnh là **TFM ngây thơ** (cùng TFM, cùng ngữ cảnh, nhãn ngây thơ).

| Bối cảnh | Vai trò | Mức chính | Metric | Đối thủ chính | Trung bình trên tập phát triển | Trạng thái |
|---|---|---|---|---|---|---|
| LLP | Chính (H2) | túi 8 | ROC AUC | **DLLP tuyến tính, mặc định** (`dllplin_def`; Ardehaly & Culotta, ICDMW 2017) | 0,8254 (bản tune 0,8248) | **Chốt** 26/10/2026 |
| PLL tổng hợp | Chính (H4) | q = 0,3; 0,6 | accuracy | GLWS/ILL, tune (`glwspll_tun`) | 0,7451 | **Tạm thời** — chọn lại sau khi chạy lại PRODEN tune có Adam (lượt 3) |
| LLP túi 16 | Phụ | túi 16 | ROC AUC | cùng đối thủ của LLP chính (`dllplin_def`) | — | Chốt |
| LLP túi 32 | Kiểm tra áp lực | túi 32 | ROC AUC | cùng đối thủ của LLP chính | — | Chốt |
| PU-SAR | Phụ | E[e \| y=1] = 0,1; 0,3 | ROC AUC | nnPU, mặc định (`nnpu`) | 0,7260 | Chốt cho tới khi có VPU (nếu cài kịp trước G7) |
| PU-SCAR | Phụ | c = 0,1; 0,3 | log loss khớp tiên nghiệm | nnPU, mặc định (`nnpu`) | 0,7130 | như trên |
| Nhãn nhiễu phụ thuộc lớp | Phụ | r = 0,2; 0,4 | ROC AUC | — | — | Chờ lượt 3 (forward correction mặc định và tune; CL + TFM) |
| SSL | Phụ | 5%; 20% có nhãn | ROC AUC | — | — | Chờ lượt 3 (label spreading, self-training: mặc định và tune) |
| PLL thật (RQ5) | Kiểm chứng ngoại tại | — | accuracy | không cần (H5 xếp hạng mọi phương pháp) | — | — |

Ghi chú:
- **LLP:** đối thủ chính trùng với bộ khởi động ấm của In-Context Gibbs (DLLP tuyến tính, cùng cấu hình). Kiểm định H2-LLP vì vậy cũng trả lời câu hỏi: bước Gibbs có cải thiện được điểm xuất phát của nó không.
- **Ứng viên đã xét:** LLP: DLLP (MLP, tuyến tính), LLP-VAT, EasyLLP, GLWS. PLL: PRODEN, GLWS/ILL, RC, CC. PU-SAR: nnPU, uPU, SAR-EM. PU-SCAR: nnPU, uPU. Nhãn nhiễu: forward correction (cấu hình của thí điểm và cấu hình mặc định đã ghi), CL + TFM. SSL: label spreading, self-training. Mỗi baseline có bản mặc định và bản tune (CL + TFM không tune, giao thức Mục 4).
- **Chưa có trong ứng viên:** VPU, In-Context PU (PU, phụ); ILL cho nhãn nhiễu và SSL; VIME (SSL, phụ). Nếu cài xong trước G7 thì chạy lại bước chọn cho các bối cảnh phụ tương ứng và ghi ở đây.
- Mọi thay đổi sau ngày chốt phải ghi lý do ở cuối file và ở giao thức Mục 11, và phải xong trước khi chạy bất kỳ phương pháp nào trên tập đánh giá.

## Lịch sử
| Ngày | Thay đổi |
|---|---|
| 26/10/2026 | Tạo file sau G6 lượt 2. LLP chốt `dllplin_def` (mức chính chỉ còn túi 8, deviation 7; khi gộp túi 8 và 16 quy tắc chọn `dllplin_tun`, 0,8191). PLL tạm `glwspll_tun`. PU-SAR, PU-SCAR: `nnpu` |
