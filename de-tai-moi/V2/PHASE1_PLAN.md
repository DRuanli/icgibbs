# Giai đoạn 1 của V2: chọn bộ gộp chuỗi (ghi trước, 05/11/2026)

- **Căn cứ:** [V0_REPORT §6](V0_REPORT.md). Cổng V0 qua một phần, theo quy tắc ghi trước: giữ nhiều điểm xuất phát, tìm tiêu chí gộp hoặc chọn khác trên tập phát triển. **Thầy duyệt ngày 05/11/2026**, gồm cả việc đưa R4 vào bài.
- **Tính chất:** phát triển phương pháp, tức **thăm dò**. Mọi dataset dùng ở đây đều đã được xem trong Study 1 hoặc Study 2, nên từ nay chúng là tập phát triển của Study 3. Phép xác nhận là Study 3.
- **Thời điểm ghi:** tài liệu này được commit **trước** khi viết code và trước mọi lượt chạy của giai đoạn 1. Script chấm `V2/analysis/v1_aggregate.py` sẽ được commit cùng code, trước khi có số. Mọi thay đổi về sau phải ghi thành deviation có ngày.

## 1. Ô thí nghiệm

Chỉ dùng các dataset **chưa dùng ở giai đoạn 0**. Sáu dataset của giai đoạn 0 bị loại: 12, 28, 40966, 1501, 50, 6332.

| Nhóm | TabPFN v2 | TabICL v2 |
|---|---|---|
| LLP túi 8, Study 1 (đánh giá) | 3, 29, 44, 1049, 1489, 1494, 23381, 38 (8 dataset) | như bên trái |
| LLP túi 8, Study 1 (phát triển) | 31, 37, 1464, 1462, 1480, 1510 (6 dataset) | như bên trái |
| LLP túi 8, Study 2 | — | 18 dataset của Study 2 |
| PLL q = 0,6, Study 1 (đánh giá) | 11, 46, 182, 188, 458, 469, 1475, 1497, 40975, 40982 (10 dataset) | như bên trái |
| PLL q = 0,6, Study 1 (phát triển) | 54, 36, 23 (3 dataset) | như bên trái |
| PLL q = 0,6, Study 2 | — | 10 dataset của Study 2 |

- **Seed:** 0, 1, 2. Lần chia và nhãn yếu giống hệt Study 1 và Study 2.
- **Vì sao TabPFN không chạy trên Study 2:** để giữ chi phí (xem §6). Việc chạy TabPFN trên Study 2 đã nằm trong W4 và Study 3.
- **Chuỗi:** như giai đoạn 0, mỗi ô có ba chuỗi `warm`, `base`, `cold`, cấu hình v1 (K = 10, T = 30, burn-in 10, S = 10, ngữ cảnh tối đa 2.000).
- **Learner khởi động L:** CC tune cho PLL, DLLP-MLP tune cho LLP, như giai đoạn 0. Phần mới là thêm **dự đoán ngoài fold** trên train: lấy cấu hình tune tốt nhất, fit lại trên 5 fold (PLL chia theo điểm, LLP chia theo túi; seed của fold là seed của ô), mỗi fold dự đoán phần bị giữ lại.

## 2. Đại lượng lưu cho mỗi ô

Chạy một lần trên GPU, lưu ra tệp npz. Mọi bộ gộp được tính lại trên CPU từ tệp này.

- **Với mỗi chuỗi k:**
  - Q̄ᵏ: dự đoán ngoài khối trên train, trung bình sau burn-in;
  - Pᵏ_test: dự đoán Rao–Blackwell trên test;
  - WPLL, r_start, r_final, vết monitor.
- **Với learner L:** P^L_oof (ngoài fold, trên train) và P^L_test (fit trên toàn train, chính là `base_start`).
- **Chung:** nhãn yếu (S hoặc túi và số đếm), nhãn thật của test. Nhãn thật của train chỉ dùng cho chẩn đoán.

## 3. Ba bộ gộp

Gọi u là một đơn vị quan sát yếu: một điểm tự do với PLL, một túi với LLP. Với ứng viên k, độ khớp của đơn vị u là:
- PLL: ℓ_uk = Σ_{c ∈ S_u} Q̄ᵏ_u(c);
- LLP: ℓ_uk = P(số đếm của túi u | các xác suất Q̄ᵏ của phần tử, độc lập), tính chính xác bằng quy hoạch động Poisson–binomial.

Với learner L, thay Q̄ᵏ bằng P^L_oof.

| Ký hiệu | Bộ gộp | Dự đoán test |
|---|---|---|
| A0 | Chọn theo argmax WPLL (v2 của giai đoạn 0, chỉ để đối chiếu) | P của chuỗi được chọn |
| **A1** | Trung bình đều ba chuỗi | (1/3) Σ_k Pᵏ_test |
| **A2** | Stacking ba chuỗi: w = argmax_{w ∈ Δ} Σ_u log Σ_k w_k ℓ_uk (Yao, Vehtari, Gelman 2022), giải bằng EM từ w đều, dừng khi thay đổi < 1e-10 hoặc sau 5.000 vòng | Σ_k w_k Pᵏ_test |
| **A3** | Stacking ba chuỗi cùng L (4 ứng viên), như A2 | Σ_k w_k Pᵏ_test + w_L P^L_test |

**Lưu ý đã biết trước khi chạy.** Q̄ᵏ_u là dự đoán khi u không nằm trong ngữ cảnh. Tuy vậy, nhãn đã lấy mẫu của u từng có mặt trong ngữ cảnh của các điểm khác ở những vòng quét trước, nên vẫn còn rò rỉ bậc hai. P^L_oof thì sạch. Vì vậy A3 có thể thiên về các chuỗi. Điều này sẽ được báo cáo cùng trọng số w_L.

## 4. Kết quả và quy tắc chọn

- **Chỉ số:** AUC cho LLP, accuracy cho PLL, trung bình 3 seed trên mỗi dataset.
- **v1** là chuỗi `warm`. **L** là `base_start`.
- **Bốn nhóm:** (TabPFN, LLP), (TabPFN, PLL), (TabICL, LLP), (TabICL, PLL). Nhóm TabICL gồm cả dataset Study 1 và Study 2.
- **Thụt lùi** của bộ gộp A trên dataset d: m_A(d) < max(m_v1(d), m_L(d)) − 0,02.
- **R(A):** tổng số thụt lùi trên cả bốn nhóm.
- **G(A):** trung bình qua bốn nhóm của [trung bình theo dataset của m_A − m_v1].

**Quy tắc chọn:**
1. Chọn A ∈ {A1, A2, A3} có R(A) nhỏ nhất.
2. Nếu hoà, chọn A có G(A) lớn nhất.
3. Nếu chênh G giữa các ứng viên còn hoà đều < 0,002, chọn bộ đơn giản hơn, theo thứ tự A1 < A2 < A3.

Bộ gộp được chọn là mặc định của v2 cho Study 3.

## 5. Dự đoán ghi trước

- **P5.** Với TabPFN (gộp hai nhóm LLP và PLL), số thụt lùi của A3 **nhỏ hơn hẳn** của cả A1 và A2.
- **P6.** Với TabICL, trong **từng** nhóm LLP và PLL, chênh G giữa mọi cặp trong {A1, A2, A3} < 0,005.
- **P7.** Bộ gộp được chọn có G ≥ −0,002 ở **mọi** nhóm, tức không kém v1 trung bình ở bất kỳ nhóm nào.

**Báo cáo kèm** (không phải dự đoán):
- trọng số trung bình theo ứng viên;
- tỉ lệ ô có w_L > 0,5;
- R và G của A0;
- bảng theo dataset;
- số thụt lùi so riêng với v1 và riêng với L.

## 6. Chi phí ước tính và các lượt

Ước tính từ thời gian thật của Study 1 và Study 2, chạy trên 2 × T4; sẽ đo lại khi chạy.
- **V1A, TabPFN v2:**
  - GPU 0: LLP (14 dataset, 3 seed, 3 chuỗi), khoảng 5 giờ chuỗi cộng thời gian fit L.
  - GPU 1: PLL (13 dataset), khoảng 7 giờ.
- **V1B, TabICL v2:** mỗi GPU chạy một nửa LLP (32 dataset) rồi một nửa PLL (23 dataset), khoảng 7–8 giờ mỗi GPU.
- Chạy tiếp được nếu hết 12 giờ. Hai lượt độc lập nhau.

## 7. Cổng V1-A (sau giai đoạn này)

- Chốt bộ gộp theo §4, và báo cáo P5–P7 đúng hay sai.
- Nếu bộ gộp được chọn vẫn còn **≥ 3 thụt lùi** trong một nhóm, ghi rõ giới hạn của v2 với nhóm đó. Theo V2_PLAN H7, giới hạn này đi vào giả thuyết của Study 3 dưới dạng ngưỡng thụt lùi đã biết trước.
- Không đổi gì khác của v2 trong giai đoạn này: K, T, các điểm xuất phát, TFM.
