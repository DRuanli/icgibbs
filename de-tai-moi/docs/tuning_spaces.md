# Không gian tune của baseline (ghi trước — 26/10/2026)

Theo giao thức G5 (Mục 3, Mục 4 quy tắc 1) và GATE_REVIEW_G5 (B3, B4). Ghi **trước khi chạy bất kỳ baseline nào có tune**, trên cả tập phát triển lẫn tập đánh giá. Sửa sau ngày này phải ghi lý do ở cuối file và ở giao thức Mục 11.

**Quy tắc chung**
- Ngân sách: **20 cấu hình** random search mỗi (baseline × dataset × mức × seed). Cấu hình đầu tiên luôn là cấu hình mặc định của bài gốc, và nằm trong 20 cấu hình đó. Cài đặt: `icgibbs/tuning.py`.
- Validation: tách **20% train** theo seed của lần chia; phân tầng theo nhãn yếu khi có thể. Riêng LLP: tách theo **túi**, không tách theo mẫu.
- Điểm chọn: loss của chính baseline trên **nhãn yếu** của phần validation (bảng dưới). Không bao giờ dùng nhãn sạch.
- Sau khi chọn: fit lại trên toàn bộ train với cấu hình tốt nhất. Báo cáo cả bản mặc định (cấu hình đầu) lẫn bản tune.
- In-Context Gibbs không tune theo dataset (cấu hình chốt trên tập phát triển ở G4).

**Loss trên nhãn yếu dùng để chọn**

| Bối cảnh | Loss trên validation |
|---|---|
| PU-SAR | −log-likelihood của nhãn yếu theo mô hình SAR của chính phương pháp nếu có (SAR-EM); nếu không có: rủi ro PU không chệch (sigmoid loss) với tiên nghiệm lớp đã biết |
| PU-SCAR | rủi ro PU không chệch (sigmoid loss) với tiên nghiệm lớp đã biết |
| LLP | KL giữa tỉ lệ lớp thật của túi và trung bình xác suất dự đoán trong túi |
| PLL | −log Σ_{k ∈ tập ứng viên} p(k \| x) |
| Nhãn nhiễu | log loss đã hiệu chỉnh tiến (forward) với ma trận T đã biết trên nhãn nhiễu |
| SSL | log loss trên phần có nhãn của validation |

**Không gian tìm kiếm** (LogU = log-đều; {…} = chọn đều)

| Baseline | Mặc định (cấu hình đầu) | Không gian |
|---|---|---|
| SAR-EM (code chính thức, hồi quy logistic như bài gốc) | C_f = C_e = 1 | C_f, C_e ~ LogU(1e-3, 1e3); max_its = 500 cố định |
| nnPU, uPU (MLP) | lr 1e-3, wd 0, ẩn (100,), 200 epoch; nnPU β = 0, γ = 1 | lr ~ LogU(1e-4, 1e-2); wd ~ LogU(1e-6, 1e-2); ẩn ∈ {(64,), (128,), (128, 128)}; epoch ∈ {100, 200, 400} |
| VPU | theo kho chính thức | lr ~ LogU(1e-5, 1e-3); λ (mixup) ∈ {0,01; 0,03; 0,1; 0,3}; α ∈ {0,1; 0,3; 1}; ẩn như nnPU |
| DLLP (MLP và tuyến tính) | lr 1e-3, wd 0, 200 epoch | lr ~ LogU(1e-4, 1e-2); wd ~ LogU(1e-6, 1e-2); ẩn ∈ {(), (64,), (128, 128)}; epoch ∈ {100, 200, 400}; số túi mỗi batch ∈ {8, 32} |
| LLP-VAT | như DLLP + mặc định VAT của kho | như DLLP; ξ ∈ {1e-6, 1e-3}; ε ∈ {0,1; 0,5; 1; 2}; trọng số nhất quán ~ LogU(1e-2, 1) |
| EasyLLP | như DLLP | như DLLP |
| GLWS (LLP, PU, PLL) | theo kho chính thức, backbone MLP | lr ~ LogU(1e-4, 1e-2); wd ~ LogU(1e-6, 1e-2); ẩn ∈ {(64,), (128,), (128, 128)}; epoch ∈ {100, 200, 400} |
| ILL (PLL, nhiễu, SSL) | như GLWS | như GLWS |
| PRODEN | lr 1e-3, wd 1e-5, ẩn (128, 128), 200 epoch | lr ~ LogU(1e-4, 1e-2); wd ~ LogU(1e-6, 1e-2); ẩn ∈ {(), (64,), (128, 128)}; epoch ∈ {100, 200, 400} |
| RC, CC (PLL thật) | mô hình tuyến tính như bài gốc | như PRODEN |
| VIME | theo kho chính thức | p_m ∈ {0,1; 0,3; 0,5}; α ∈ {0,5; 1; 2}; β ∈ {0,1; 1; 10}; ẩn ∈ {(64,), (128,)} |
| Label spreading | kNN, k = 7, α = 0,2 | kernel ∈ {kNN, RBF}; k ∈ {3, 5, 7, 11, 15, 21}; γ ~ LogU(1e-3, 10); α ∈ {0,1; 0,2; 0,5; 0,8} |
| Self-training | hồi quy logistic C = 1, ngưỡng 0,75 | C ~ LogU(1e-3, 1e3); ngưỡng ∈ {0,6; 0,75; 0,9; 0,95} |
| Forward correction (MLP) | như nnPU | như nnPU |
| Confident Learning + TFM | mặc định (không tune, giao thức Mục 4) | — |

## Thay đổi sau ngày ghi
| Ngày | Thay đổi | Lý do |
|---|---|---|
| 26/10/2026 | GLWS (LLP) tune thêm số túi mỗi batch ∈ {8, 32} như các baseline LLP khác; RC và CC dùng cả cho PLL tổng hợp (giao thức Mục 11); GLWS cho PLL trùng với ILL trong kho chính thức nên tính là một baseline "GLWS/ILL" | Khi cài đặt ở G6, trước khi chạy bản tune nào |
| 26/10/2026 | Ứng viên đối thủ chính gồm cả bản mặc định lẫn bản tune (giao thức Mục 11) | Bản tune có thể kém bản mặc định |
