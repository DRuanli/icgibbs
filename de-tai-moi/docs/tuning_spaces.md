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
| 26/10/2026 | **PRODEN:** thêm bộ tối ưu ∈ {SGD momentum 0,9, Adam} vào không gian; cấu hình mặc định giữ SGD như kho chính thức | G6 lượt 2: SGD chưa hội tụ trên dữ liệu bảng nhỏ (0,716 tune so với 0,740 của bản Adam cũ); làm baseline mạnh hơn. Quyết trước khi chạy bản tune mới và trước tập đánh giá |
| 26/10/2026 | **Forward correction:** cấu hình mặc định đã ghi "như nnPU" được hiểu là lr 1e-3, wd 0, ẩn (100,), 200 epoch (`forward_def`); bản trong thí điểm (lr 1e-3, wd 1e-4, ẩn (64, 64), 200 epoch, tên `forward`) vẫn là một ứng viên đối thủ chính. **Label spreading:** với kernel kNN thì γ không dùng; cấu hình mặc định (kNN, k = 7, α = 0,2) trùng bản thí điểm. **SSL:** validation tách phân tầng theo nhãn quan sát (kể cả −1), loss chỉ tính trên phần có nhãn | Làm rõ khi cài ở G6 lượt 3, trước khi chạy |
| 26/10/2026 | **Baseline phụ còn thiếu (G6 lượt 4), trước khi chạy:** GLWS cho PU dùng không gian GLWS đã ghi (lr, wd, ẩn, epoch), mặc định MLP (100,), lr 1e-3, 200 epoch, batch dương 64, tỉ lệ không nhãn/dương 2 như kho. VPU: mặc định đúng `run.py` của kho (lr 3e-5, batch 500, 50 epoch × 30 vòng lặp, α = 0,3, λ = 0,03, mạng 6 lớp ẩn 300 như `model_vec.py`); tune theo không gian VPU đã ghi, các tham số còn lại giữ như mặc định. ILL nhãn nhiễu: thêm vào không gian "như GLWS" hai lựa chọn — ma trận nhiễu ∈ {học được (như kho, mặc định), T đã biết} và cân bằng entropy trung bình ∈ {có (như kho, mặc định), không}; ILL SSL: thêm cân bằng entropy ∈ {có, không}. VIME: mặc định như kho (p_m 0,3; α 2; β 1; K 3), bộ dự đoán ẩn (100, 100); tune theo không gian VIME đã ghi | Hai lựa chọn thêm cho ILL làm baseline mạnh hơn: cân bằng entropy về phân phối đều có hại khi lớp mất cân bằng (lớp nhỏ của các dataset nhị phân của dự án chiếm từ 6% (sick) tới 48%), và các phương pháp khác của bối cảnh nhãn nhiễu đều được biết T |
| 27/10/2026 | **Fit lại sau khi tune** (mọi baseline có bản tune): nếu mô hình fit lại trên toàn bộ train với cấu hình tốt nhất cho xác suất không hữu hạn trên chính X train, dùng cấu hình xếp hạng kế tiếp theo loss validation, tối đa 5 (`icgibbs/tuning.py: refit_best`; giao thức §11, deviation 10). Không gian tìm kiếm và ngân sách 20 cấu hình không đổi | G6 lượt 5: `proden_tun` cho NaN ở 1/324 ô (segment, q = 0,3, seed 2). Quyết trước tập đánh giá |
