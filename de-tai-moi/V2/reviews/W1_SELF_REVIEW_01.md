# Tự rà toán W1 lần 1 (skill dm-paper-math-review)

05/11/2026.
- **Đối tượng:** §2 của `V2/W1_R1A_REPORT.md`, bản đầu cùng ngày: Mệnh đề 4, Bổ đề 1–3, Hệ quả (c)–(d). Có liên hệ tới §0 và §3.
- **Người rà:** em. Đây là tự rà, chưa phải phản biện độc lập.
- **Đã sửa:** mọi mục dưới đây đều đã sửa trong report.

## Kết luận một dòng

**MAJOR ISSUES.** Giả định (H_δ) bản đầu so Φ tại r(y) của *cả* trạng thái và đòi dung sai đều trên *mọi* khối. Hai điều này làm δ có sàn không giảm theo n, và làm δ̂ đo trong mô phỏng không phải là δ của mệnh đề. Bản sửa đổi giả định, nên các con số so sánh ở §3 phải tính lại.

## Sổ lỗi

### [MAJOR-1] Φ tại r(y) thay vì tại tỉ lệ đúng của ngữ cảnh. Vị trí: §2.1, (H_δ)

- **Bản đầu viết:** |avg_{B∩F} c_i(y, B) − Φ(r(y))| ≤ δ, với mọi y và mọi B.
- **Vì sao sai:**
  - c_i(y, B) chỉ phụ thuộc y_{−B}, vì TFM chỉ thấy ngữ cảnh. Còn r(y) phụ thuộc cả y_B.
  - Giữ y_{−B} cố định và đổi y_B: vế trái không đổi, nhưng r(y) chạy trên một đoạn dài |B∩F|/n_F, có thể tới 1/(ρK).
  - Vậy (H) kéo theo dao động của Φ trên mọi đoạn dài 1/(ρK) phải ≤ 2δ, tức δ ≥ L/(2ρK) với Φ L-Lipschitz.
- **Bằng chứng:** ánh xạ 0,9 có L = 0,5, K = 10, ρ ≈ 1, cho δ ≥ 0,025 ở mọi n. Trong khi đó §3 báo δ̂ = 0,008 ở n = 3.000, và dùng con số này để nói "điều kiện γ > δ thoả cạnh u".
- **Ảnh hưởng:** bản tiệm cận "dải co về G_δ ± δ/K" và mọi so sánh với δ̂ ở §3.
- **Sửa:**
  - Viết (H) theo r_{−B}(y), tức tỉ lệ đúng trên F∖B, và thêm giả định Φ L-Lipschitz.
  - Bổ đề 1 có thêm số hạng ngữ cảnh. Theo cấu trúc, r_{−B} − R = S_B/(n_F − f), với S_B là tổng mẫu không hoàn lại có trung bình 0. Trên biến cố {|S_B| ≤ x_N}, có xác suất ≥ 1 − ε/N mỗi bước, số hạng này ≤ δ_ctx/K, với
    δ_ctx = L/(ρ^{3/2}(1 − 1/(ρK)))·√(ln(2N/ε)/(2K n_F)) → 0.
  - Chứng minh (a) và (b) dùng trội từng đường đi trên ∩_t G_t, cộng ε vào xác suất thất bại.

### [MAJOR-2] Dung sai đều trên mọi khối không khớp với không đồng nhất theo điểm. Vị trí: §2.1, §3

- **Bản đầu viết:** δ là chặn đều trên mọi khối. §3 lại đo δ̂ bằng *phân vị 99* của |trung bình h_i| trên khối ngẫu nhiên.
- **Vì sao sai:**
  - Với h_i cố định theo điểm, chặn đều trên mọi khối là max|h_i| ≈ 0,1, chứ không phải 0,008.
  - Mặt khác, phần không đồng nhất có trung bình 0 không cần nằm trong δ. Tổng của nó trên khối là một mẫu không hoàn lại, nên đi vào nhiễu ζ_2.
- **Ảnh hưởng:** cách đọc toàn bộ §3, gồm "dải khớp G_δ" và "n = 500 không thoả γ > δ".
- **Sửa:**
  - Viết c_i − h_i thay cho c_i trong (H). h_i cố định theo điểm, |h_i| ≤ h_max, và Σ_F h_i = 0 (không mất tổng quát, vì gộp được vào Φ).
  - Trong TFM giả: δ = 0, chỉ trừ ảnh hưởng của clip (≤ 0,001). Proxy nhiễu nhân với c_w = (1 + (1 + 2h_max + δ)²)/2.
  - §3 viết lại: δ′ = δ_ctx; dải và hiện tượng lẫn lưu vực ở n = 500 do **nhiễu** gây ra; δ̂_blk chỉ là thang nhiễu.

### [MAJOR-3] Phạm vi: LLP không được phủ. Vị trí: §1, §0

- **Bản đầu viết:** §1 nói "LLP thì mọi điểm đều tự do", ngụ ý mệnh đề phủ LLP.
- **Vì sao sai:** bước ζ_1 của Bổ đề 1 cần các lần rút trong khối độc lập. Với LLP, bộ lấy mẫu rút cả túi theo quy hoạch động có ràng buộc số đếm, nên các lần rút trong túi phụ thuộc nhau. Khối LLP cũng được chọn theo túi.
- **Sửa:**
  - Giới hạn phạm vi ở các bài rút độc lập: PLL, SSL, PU.
  - Thêm Nhận xét 5: hướng mở rộng cho LLP lấy túi làm đơn vị, nên proxy nhân thêm cỡ túi lớn nhất.

### [MINOR-1] Lọc không chứa B_t. Vị trí: §1, §2.2

- 𝓕_t sinh bởi Y_0, …, Y_t không cho biết B_t, vì một điểm rút lại đúng nhãn cũ thì không lộ ra. Do đó ζ_2 không thích nghi.
- **Sửa:** 𝓕_t = σ(Y_0, B_0, ξ_0, …, Y_t).

### [MINOR-2] (b) đòi R_0 < ℓ chặt. Vị trí: Hệ quả (c), khi chuỗi chạm biên từ trên

- **Sửa:** cho phép R_0 ∈ [ℓ, x̄]. Lần vượt đầu tiên bắt đầu tại x = R_0 ≤ x̄, và chứng minh không đổi.

### [MINOR-3] Bản tiệm cận nói quá. Vị trí: Hệ quả (c), đoạn "Bản tiệm cận"

- **Bản đầu viết:** dải → G_δ ± δ/K.
- **Vì sao chưa đúng:** ḡ = 0 chỉ khi 1/(ρK) < η. Với K cố định và η → 0, cửa sổ của ḡ chứa cả điểm phía bên kia r*.
- **Sửa:** dải → G_δ ± (ḡ + δ)/K. Ghi chú: dùng chặn bước nhảy xác suất cao, cỡ (max|g| + δ)/K + λ′, thì ḡ = O(1/K), và phần thêm vào là O(1/K²).

### [MINOR-4] Thời gian thoát ở (d) mới là phác thảo

- Câu "λ ≈ (r₂ − u − 2η)/2" chưa chọn λ, λ′ cụ thể. Hằng số chưa ghi rõ.
- **Sửa:** ghi rõ là phác thảo (mức L2), và thêm điều kiện r₂ − u > 2η + β.

### [MINOR-5] Hai câu nói quá

1. "Base learner có ích **đúng khi** r_base > u": nói quá so với (d), vì trong khoảng u ± 2η mệnh đề không nói gì.
2. "Đây là cơ chế của P1": nói quá, vì ở §4 chuỗi TabPFN khởi động ấm thoát một phần khỏi lưu vực dưới của ánh xạ gieo đều. Lưu vực đó không phải bẫy thật.

**Sửa:** viết thành "khớp về chiều" và trỏ tới §4.

### [MINOR-6] Bổ đề 3 chỉ phủ vùng gain đủ lớn

- Phác thảo chỉ cho R_{s+1} ≥ Φ(R_s) − … khi g(R_s) ≥ δ + Lε, và cần Φ không giảm. Nó không phủ pha "ở lại".
- **Sửa:** ghi rõ hai điều này.

### [MINOR-7] §3 nói "kiểm Mệnh đề 4", nhưng mô phỏng chạy lịch quét thật

- Mệnh đề 4 chứng minh cho lịch quét ngẫu nhiên. Mô phỏng chạy lịch quét thật, tức thứ mà Bổ đề 3 phác thảo.
- **Sửa:** ghi rõ.

## Phán quyết theo kết quả (sau khi sửa)

| Kết quả | Mức chứng minh | Phán quyết |
|---|---|---|
| Bổ đề 1 (bản sửa) | L3 | PROVEN dưới (H) sửa, Φ L-Lipschitz, n_F > m |
| Bổ đề 2 | L4 | PROVEN; kinh điển (Ville + Chernoff) |
| Mệnh đề 4(a) | L3 | PROVEN dưới (H⁻) sửa |
| Mệnh đề 4(b) | L3 | PROVEN dưới (H⁺) sửa |
| Hệ quả (c) | L3 | PROVEN; bản tiệm cận như MINOR-3 |
| Hệ quả (d), phần lưu vực | L3 | PROVEN |
| Hệ quả (d), thời gian thoát | L2 | INCOMPLETE: chưa chọn hằng số |
| Bổ đề 3 (lịch quét thật) | L2 | INCOMPLETE: phác thảo |
| Bản số hạng ngữ cảnh không có log khi Φ ∈ C² | L1 | INCOMPLETE: mới là nhận xét |

## Đã tấn công mà không phá được

1. **Biên.**
   - r_0 = 0 trong (a) với a = 0: κ = λ, vì R ≥ 0.
   - B ∩ F = ∅: bước không đổi gì, Bổ đề 1 vẫn đúng.
   - Φ_δ < 0: đã bỏ clamp, đoạn chứa 0 có độ dài ≤ w.
2. **Đoạn chứa giá trị a_i.** Đoạn chứa mọi a_i và 0 có độ dài ≤ 1 + 2h_max + δ trong mọi trường hợp dấu của Φ − δ ± h_max.
3. **Kỳ vọng.** E_B[S_B] = 0, vì Σ_F 1[đúng] = n_F R.
4. **Ghép proxy.** Hai proxy cộng được nhờ kỳ vọng lặp: ζ_2 đo được theo (𝓕_t, B_t), và ζ_1 có kỳ vọng 0 khi biết (𝓕_t, B_t).
5. **θ*.** θ* = 2(γ − δ′)/(Ks²) làm MGF của bước ≤ 1, kiểm lại đại số được 4ρn_F(γ − δ′)/c_w.
6. **Trội từng đường đi.** Trội trên ∩G_t kết hợp với siêu martingale của tổng trội. Không cần MGF có điều kiện trên biến cố G_t.

## Thứ tự sửa

1. MAJOR-1 và MAJOR-2: viết lại (H) và Bổ đề 1, rồi tính lại cột lý thuyết của §3. Đã xong: `w1_drift_check.py theory` dùng δ′ = δ_ctx.
2. MAJOR-3: ghi phạm vi.
3. Các mục MINOR.
4. **Còn mở:** Bổ đề 3 đầy đủ, thời gian thoát ở (d), bản số hạng ngữ cảnh không có log. Cần một lượt rà độc lập trước khi đưa vào bài.
