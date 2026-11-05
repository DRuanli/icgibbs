# Rà độc lập lần 2: §2.7, §2.8, §5.7 của `V2/W1_R1A_REPORT.md`

Người rà: agent độc lập, chưa thấy quá trình viết và không mở tệp nào khác trong `V2/reviews/`. Theo skill `dm-paper-math-review`: thang L0–L5, nhãn BLOCKING / MAJOR / MINOR / COSMETIC. Ngày 05/11/2026.

## Phán quyết một dòng

**BLOCKING (chỉ ở câu đề xuất đưa vào bài, sửa một dòng).** Câu "Đề xuất cho bài" ở §5.7 viết rằng "trong mô hình liên hợp", khối chỉ gồm điểm tự do cho dự đoán đúng tuyệt đối. Câu này **sai như đã viết**: trong chính mô hình liên hợp gộp đó, với PU, khối free lệch dự đoán tới 0,07 (E5c). Câu đó cũng nói Mệnh đề 5 "đủ để thay" Mệnh đề 3(b), trong khi 3(b) là phát biểu tổng quát. Các kết quả toán đã phát biểu (Bổ đề 3*, Mệnh đề 6, Mệnh đề 5(a)–(c)) **không sai**: em kiểm Mệnh đề 6 và Mệnh đề 5 chính xác tới sai số máy. Có 2 MAJOR, đều về hằng số và về lợi thế được quảng cáo, không về tính đúng của các bổ đề.

## Phạm vi

**Đã đọc kỹ:**
- §2.7: thiết lập, (H*), hằng số, Bổ đề 3*, hệ quả cho 4′, mô phỏng max_context, hằng số tinh chỉnh;
- §2.8: Mệnh đề 6, chứng minh, bảng so mô phỏng, giới hạn, ý nghĩa;
- §5.7: Mệnh đề 5(a)–(c), kiểm số, phạm vi, đề xuất cho bài;
- các dòng của §0 nhắc tới ba mục này.

**Đọc làm nền, không rà lại:** §1, §2.1–2.6, §5.1–5.6.

**Code đã đọc:**
- `sampler.py`: `_ctx`, `_sweep_instance`;
- `w1_drift_check.py`: `make_cond`, `refined_band`, `sweep_moments`, `sweep_stationary`, `second_order`, `context_cap`;
- `r1a_block_bias.py`: `lumped_all_exact`, `lumped_all_kernel`, `ssl_affine_means`, `e7f`;
- `rq1_exact.py`: `lumped`, `lumped_all`, `e5c`;
- `paper/sections/appendix_proofs.tex`, Prop. `prop:blocks`, để so phạm vi với "thay Mệnh đề 3(b)".

**Đã đối chiếu số trong bài với CSV:**
- `w1_context_cap_h002.csv`, `w1_partition_bound_refined.csv`, `w1_second_order_h002.csv`, `w1_theory_check_h002.csv`;
- `e7f_affine_check.csv`, `e7f_affine_closed_form.csv`, `rq1/e5c_block_composition.csv`.

**Script của người rà** (thư mục `/tmp/claude-0/-home-claude/643477cb-9e2c-5e3b-9dd1-1dba7a8f3cd0/scratchpad/review2/`):

| Script | Nội dung |
|---|---|
| `prop6_exact.py` | luật chính xác của S′ biết S cho lịch quét thật, bằng quy hoạch động theo khối (độc lập với đệ quy của tác giả); so với `sweep_moments` và `sweep_stationary` |
| `prop6_sampler_mc.py` | chạy **đúng** `InContextGibbs` với TFM giả, h = 0, ρ = 1, 40.000 vòng quét; thêm hai ô K ∤ n |
| `prop6_rho.py` | luật chính xác khi ρ < 1, có điểm cố định và khối trên mọi điểm |
| `prop5_check.py` | dạng đóng rút gọn của người rà; so với `ssl_affine_means` và với tiệm cận (C = 2, 3, 5; α = 0,5, 1, 2); đệ quy trung bình cho **lịch quét hệ thống**; Monte Carlo lịch quét hệ thống |
| `prop5_C3.py` | chuỗi gộp C = 3 chính xác (ma trận chuyển đầy đủ), kiểu all và free, nhân chính xác và nhân không sửa |
| `lemma3star_bad.py` | Σ_k \|B_k ∩ Bad_k\| dưới phân hoạch thật, với Bad tĩnh và Bad thích nghi |
| `refined_v.py` | tính lại dải Bernstein với chặn phương sai hợp lệ cho phía sàn |

**Không kiểm được:**
- phép khai triển sympy của §5.7: không có trong repo. Em thay bằng một rút gọn đại số chính xác của riêng mình, xem Phụ lục A;
- định lượng ảnh hưởng của h ≠ 0 lên Mệnh đề 6.

Bằng chứng số **không** thay chứng minh. Mỗi chỗ dùng số, em ghi rõ đã phủ những gì.

## Sổ lỗi

### [BLOCKING-1] Câu đề xuất thay Mệnh đề 3(b) sai như đã viết — §5.7, "Đề xuất cho bài"; §0 mục 5

**Khẳng định như đã viết.**
- "(b) … Nhưng ở mức dự đoán, **trong mô hình liên hợp**: khối chỉ gồm điểm tự do cho dự đoán đúng tuyệt đối, còn khối trên mọi điểm (mặc định) cho sai lệch α(1−λ)(1−Cp_c)/(λ(K−1)n) + O(n⁻²)."
- §0: "Phát biểu này đủ để thay Mệnh đề 3(b) đang rỗng."

**Vì sao sai.**
- Mệnh đề 5 chỉ được chứng minh cho **SSL**: điểm tự do không bị nghiêng, mô hình gộp không feature, lịch quét khối ngẫu nhiên. §5.7 tự ghi giới hạn này ở mục "Phạm vi" 1–2, nhưng câu đề xuất bỏ mất nó.
- Mô hình gộp Beta–Bernoulli với PU (ℓ = (1, 1−c)) **cũng là mô hình liên hợp**. Ở đó, khối free cho sai lệch dự đoán lớn và không giảm theo n.
- Ngoài ra, Prop. `prop:blocks`(b) trong `appendix_proofs.tex` là phát biểu cho mọi loại nhãn yếu, kể cả bag-block của LLP với κ_B. Mệnh đề 5 phủ một mô hình đồ chơi SSL, nên không "thay" được 3(b). Nó chỉ thay được phần SSL, hoặc thêm vào như một mệnh đề minh hoạ.

**Bằng chứng** (`G4/results/rq1/e5c_block_composition.csv`, kiểu free, PU, cùng mô hình liên hợp gộp; `pred_err_free`):

| n | K | pred_err_free |
|---|---|---|
| 50 | 10 | 0,021 |
| 1.000 | 10 | 0,031 |
| 1.000 | 5 | 0,070 |

Cùng tệp, SSL-asym cho `pred_err_free` ≈ 10⁻¹⁴, đúng như 5(b). Vậy câu "khối free đúng tuyệt đối trong mô hình liên hợp" sai, với phản ví dụ ngay trong dữ liệu của đề tài. Chính lập luận chọn mặc định `block_over="all"` (comment ở `sampler.py`, docstring `rq1_exact.lumped_all`, RQ1_REPORT §5) dựa trên việc khối free lệch khi có nhãn yếu nghiêng.

**Hệ quả.**
- Nếu câu này vào bài thành Prop 3(b), bài có một mệnh đề sai, trái với số liệu của chính bài.
- Câu §0 "đủ để thay Mệnh đề 3(b)" cũng phải sửa theo.

**Sửa tối thiểu.**
- Viết: "Trong mô hình liên hợp gộp **SSL** (điểm tự do không có nhãn yếu), với lịch quét khối ngẫu nhiên: (i) khối free cho trung bình dừng của dự đoán đúng tuyệt đối; (ii) khối all cho sai lệch … (dạng đóng ở Phụ lục A)."
- Giữ 3(b) tổng quát như cũ, hoặc chỉ thay câu "Why this is vacuous". Thêm Mệnh đề 5 như một mệnh đề riêng có phạm vi SSL.
- Nói rõ: với PU/PLL, đồng nhất thức dạng Poisson vẫn đúng nhưng không có dạng đóng, và khối free **lệch** (E5c).
- Tách "TV ≍ 1/K không giảm theo n" khỏi phần mệnh đề. Đó là số đo, xem MINOR-11.

### [MAJOR-1] Bản Bernstein/Freedman của hằng số tinh chỉnh chưa được chứng minh — §2.7, "Hằng số tinh chỉnh", cải tiến 2; `refined_band(method="bernstein")`

**Khẳng định như đã viết.**
- "Có hai cải tiến, **đều chặt về toán**."
- Ở cải tiến 2: v là chặn của c(1−c) và r(1−r) "trong dải", lặp tới điểm bất động, và "chỉ dùng cho dải bất biến (b′), vì phương sai cục bộ chỉ đúng trong dải".

**Vì sao chưa đủ.** Có bốn bước bị thiếu hoặc sai.
1. **Biến cố chỉ khống chế phía sàn.** Chứng minh sàn (Bổ đề 3(i)) chỉ cho r_k ≥ x − ν trên 𝓔; nó không cho chặn trên của r_k. Bản (b′) áp (i) và (ii) **riêng rẽ**, mỗi bản một biến cố, nên "trong dải" (hai phía) không phải là thông tin mà chứng minh sàn có.
   - Chặn phương sai từ một phía chỉ dùng được nếu c(1−c) đơn điệu trên vùng được khống chế. Ví dụ, toàn vùng c ≥ 1/2: đúng với ánh xạ 0,9.
   - Nếu không, phải lấy v = 1/4, hoặc làm quy nạp **đồng thời** cho sàn và trần.
2. **Freedman cần chặn biến thiên bậc hai khả đoán trên cả đường đi.** Chặn ⟨Z⟩ ≤ n_F·v chỉ đúng khi quy nạp còn đứng. Cần dừng martingale tại khối đầu tiên mà r_k rời vùng; khi đó ⟨Z^stop⟩ ≤ n_F·v luôn đúng. Rồi chỉ ra rằng trên giao các biến cố, việc dừng không xảy ra. Lập luận này chuẩn nhưng chưa có trong bài.
3. **(H) chỉ khống chế trung bình khối, không khống chế từng c_i.** Cần bước Jensen: c(1−c) lõm, nên Σ_{i∈B∩F} c_i(1−c_i) ≤ f·c̄(1−c̄). Rồi chặn c̄ qua (H) và h̄_B. Bước này chưa có.
4. **v trong code bỏ lề ngữ cảnh ν.** Trên biến cố sàn, ngữ cảnh chỉ thoả r_k ≥ x − ν, với x = lo là chân dải. Nhưng `refined_band` lấy max của c(1−c) với c = a + b·rr ± h và rr ∈ [lo, hi], tức bỏ đoạn [lo − ν, lo).

**Bằng chứng** (`refined_v.py`): tính lại với v hợp lệ cho phía sàn, tức max của c(1−c) trên c ∈ [a + b(lo − ν) − h, a + b + h] và của R(1−R) trên R ∈ [lo, 1].

| Ánh xạ | Nửa dải ở n_F = 3.000 | n_F để nửa dải ≤ 0,05 |
|---|---|---|
| 0,9 | 0,155, không đổi (v = 0,19, do R(1−R) tại lo quyết định) | 18.900 → 19.700 (+4%), trên lưới mịn hơn của người rà; CSV của tác giả ghi 19.126 vì lưới thô hơn |
| 0,6 | không đổi (v = 0,25) | không đổi |

Vậy **con số hầu như không đổi**: khuyết điểm nằm ở lập luận, không ở bảng. Trong trường hợp dùng ở đây, đánh giá một phía tình cờ đủ, vì c ≥ 1/2 với ánh xạ 0,9, còn ánh xạ 0,6 có v = 1/4 sẵn. Nhưng lý do trong bài ("phương sai cục bộ chỉ đúng trong dải") không phải lý do đúng.

**Hệ quả.**
- Hàng "+ Bernstein/Freedman" của bảng.
- Câu §0 "Hằng số tinh chỉnh giảm n_F cần thiết 2–3 lần": phần chặt (chia + Ville) chỉ cho 1,7–1,8 lần, còn "3 lần" dựa vào Bernstein.
- Kết luận "lỏng 8 lần" ở n_F = 3.000.

**Sửa tối thiểu.**
- Viết một Bổ đề 3-Bernstein với ba bước trên: dừng, Jensen, và v lấy trên [x − ν, 1], hoặc v = 1/4 khi vùng chứa 1/2.
- Sửa `refined_band` để v phủ lề ν.
- Ghi "đều chặt về toán" chỉ sau khi đã có ba bước đó.

### [MAJOR-2] Lợi thế định lượng của (H*) bị nói quá — §2.7, "Vì sao đây là giả định yếu hơn" và "Ví dụ"

**Khẳng định như đã viết.** "(H*) chỉ trả **D·η₀·H̃_K**". Ví dụ: 2% điểm lệch tới D = 1,1, K = 10, thì (H) cần δ ≥ 0,22, còn (H*) "chỉ trả 1,1 × 0,02 × 2,93 ≈ 0,064".

**Vì sao sai.** (H*) **yếu hơn về logic**: (H⁻_δ) kéo theo (H*) với Bad = ∅. Điều đó đúng. Nhưng phép tính giá bỏ hai số hạng mà chính Bổ đề 3* trả.
1. **Số hạng lệch D·ν_b** trong β_bad = D(η₀H̃_K + ν_b). Để K·e^{−2ρn_Fν_b²/K} nhỏ, ν_b phải cỡ √(K ln(K/q)/(2ρn_F)). Ở n_F = 3.000, với q ≈ 10⁻⁴: ν_b ≈ 0,14, nên D·ν_b ≈ 0,154 > D·η₀·H̃_K = 0,064.
2. **β_bad/a_m nằm trong ν′**, tức β_bad bị trả **hai lần**: một lần ngoài Φ̲, một lần bên trong qua Φ̲(x − ν′).

Với Φ tuyến tính độ dốc b, dải bị nới thêm β_bad(1 + b/a_m)/(1 − b) theo (H*), so với δ/(1 − b) theo (H).

**Bằng chứng** (b = 0,5, a_m = 0,9, D = 1,1, η₀ = 0,02, K = 10, p mỗi vòng = 0,05/120, chia 1/5 cho 𝓔_bad):

| n_F | β_bad | Giá trên dải, (H*) | Giá trên dải, (H) |
|---|---|---|---|
| 3.000 | 0,218 | 0,68 | 0,44 |
| 10.000 | 0,149 | 0,46 | 0,44 |
| 30.000 | 0,113 | 0,35 | 0,44 |
| ∞ | 0,064 | 0,20 | 0,44 |

- Ở cỡ dữ liệu của bài (n_F ≲ 7.000), (H*), **với hằng số như đã chứng minh**, **tệ hơn** (H).
- Điểm hoà vốn ở n_F ≈ 1,2·10⁴.
- Ở giới hạn, (H*) lợi khoảng 2,2 lần, không phải 0,22/0,064 ≈ 3,4 lần.

**Hệ quả.** Câu §0 "Bổ đề 3* cũng dùng một giả định yếu hơn" vẫn đúng, nhưng thông điệp ngầm "rẻ hơn" chỉ đúng tiệm cận. Không ảnh hưởng tính đúng của Bổ đề 3*.

**Sửa tối thiểu.**
- Ghi giá đầy đủ: D(η₀H̃_K + ν_b)(1 + L/a_m).
- Nói rõ lợi thế chỉ có khi n_F đủ lớn, hoặc thay Hoeffding cho |B_k ∩ Bad| bằng Bernstein/Chernoff siêu bội. Phương sai ở đây cỡ m·η₀H̃_K/K, nhỏ hơn nhiều so với biên độ 1 mà Hoeffding dùng, nên ν_b sẽ giảm hẳn.

### [MINOR-1] Lý do sai cho trường hợp ρ < 1 của Mệnh đề 6 — §2.8, "Giới hạn" 1

- **Khẳng định:** "Với ρ < 1, các điểm cố định chỉ thêm hằng số vào r_k, nên đệ quy vẫn đúng."
- **Vì sao sai:** r_k = r_{−B_k} chỉ tính trên F∖B_k, nên điểm cố định **không** vào r_k. Cái chúng làm là khiến f_k = |B_k ∩ F| ngẫu nhiên, và mẫu số n_F − f_k của r_k ngẫu nhiên. Khi đó c_k không còn affine theo Z, nên đệ quy như viết không áp dụng được.
- **Kết luận vẫn đúng**, nhưng vì lý do khác: điều kiện theo (f_1, …, f_K), độc lập với trạng thái, thì đệ quy affine–bậc hai đúng với hệ số phụ thuộc f. Trộn theo f thì E[R′|R] vẫn affine, và Var[R′|R] = E_f[bậc hai] + Var_f(affine) vẫn bậc hai.
- **Bằng chứng:** `prop6_rho.py` (n = 12, n_F = 10, K = 3 và n = 16, n_F = 12, K = 4) cho phần dư affine ≤ 3·10⁻¹⁶, phần dư bậc hai ≤ 2·10⁻¹⁶, và trung bình dừng = r* = 0,9 đúng tới 10 chữ số.
- **Sửa:** thay câu lý do bằng lập luận "điều kiện theo vector cỡ tự do của khối rồi trộn".

### [MINOR-2] "Dải 99% khớp tới 3 chữ số" nói quá — §2.8, "Đọc bảng"; §0

- **CSV `w1_second_order_h002.csv`:**

  | Ô | Gauss | Quan sát |
  |---|---|---|
  | (0,9; 500) | [0,851; 0,949] | [0,844; 0,944] |
  | (0,6; 500) | [0,511; 0,689] | [0,512; 0,686] |
  | (0,6; 3.000) | [0,564; 0,636] | [0,563; 0,636] |

  Ngoài đầu dưới của ô (0,9; 500) mà bài đã loại trừ, còn 3 đầu mút lệch 0,001–0,005.
- §0 bỏ luôn ngoại lệ.
- **Sửa:** "khớp trong ±0,005, và trong ±0,001 ở n = 3.000".

### [MINOR-3] So 4′ với Mệnh đề 6 là so hai đối tượng khác nhau — §2.8, "So với chặn của Mệnh đề 4′"; §2.7, "lỏng khoảng 8 lần"

- "Mệnh đề 6 cho **đúng** 2,576·σ = 0,020": σ thì đúng, nhưng 2,576σ là phân vị Gauss, tức xấp xỉ, như chính Giới hạn 3 nói.
- Dải của 4′ là dải **đều theo thời gian** trên H = 60 vòng quét, với xác suất thất bại tổng 0,05. Đối chứng công bằng là khoảng [min, max] quan sát trên 60 vòng, chẳng hạn late_min/late_max ở `w1_theory_check_h002.csv` cho [0,874; 0,924], tức nửa dải ≈ 0,025. Khi đó độ lỏng là khoảng 6 lần, không phải 8 lần.
- **Sửa:** ghi rõ hai đối tượng và lấy đúng đối chứng.

### [MINOR-4] Câu "Ý nghĩa cho bài" của Mệnh đề 6 mở rộng từ "affine" sang "gần affine" — §2.8

- "khi ánh xạ phản ứng **gần affine** quanh điểm hút, độ dao động … **có dạng đóng**". Mệnh đề 6 chỉ chính xác khi Φ affine tại **mọi** giá trị mà r_k đạt được. Chuỗi bất khả quy, nên tập này là cả lưới {j/(n−m)} ⊂ [0, 1].
- Với "gần affine", Giới hạn 2 tự nhận là mức L2, chưa chặn.
- Tương tự, câu "Φ affine trên vùng chuỗi đi qua" (Giới hạn 1) phải đọc là "trên mọi giá trị r_k có thể đạt, và Φ([0,1]) ⊆ [0,1]".
- **Sửa:** "khi Φ affine (không cắt) thì có dạng đóng; khi gần affine, dạng đóng là xấp xỉ bậc hai chưa có chặn sai số".

### [MINOR-5] Mệnh đề 5 chính xác cho lịch khối ngẫu nhiên, không phải lịch quét thật — §5.7 (c), §0, "Đề xuất cho bài" ("mặc định")

- Thiết lập ghi "khối ngẫu nhiên", và mọi ma trận kiểm (`lumped_all`, `lumped_all_kernel`) đều chọn một khối đều **mỗi bước**.
- Bộ lấy mẫu thật (`block_over="all"`, mặc định) dùng **hoán vị + array_split**. Chữ "(mặc định)" dễ được đọc là "đúng cho bộ lấy mẫu như chạy".
- **Bằng chứng** (`prop5_check.py`): đệ quy trung bình chính xác ở mức vòng quét cho lịch hệ thống. Kiểm tra hợp lệ của đệ quy:
  - với nhân chính xác, nó cho đúng trung bình hậu nghiệm ở cả 8 ô;
  - Monte Carlo 40.000 chuỗi khớp trong ≈ 1,6 SE.

  So sai lệch dự đoán của lịch hệ thống với lịch ngẫu nhiên:

  | Ô | Tỉ số hệ thống/ngẫu nhiên |
  |---|---|
  | n = 50, K = 5 | 1,010 |
  | n = 200, K = 10 | 1,003 |
  | n ≥ 1.000 | 1,000–1,001 |

  Vậy **số hạng chính đúng cho lịch thật**: đây là bằng chứng số, không phải chứng minh. Nhưng dạng đóng Ã_c/ã **không** chính xác cho lịch thật.
- **Sửa:** ghi rõ lịch. Nếu muốn phát biểu cho lịch thật, dùng (a) ở mức vòng quét: drift vòng quét vẫn affine theo k⁰, vì mọi bước tuyến tính theo (k, j) và E[j | cỡ khối, y⁰] = u·k⁰/U. Rồi chứng minh số hạng chính trùng nhau.

### [MINOR-6] Lẫn tiệm cận với dạng đóng, và hằng số của O(n⁻²) — §0 mục 5; §5.7 (c)

- §0 viết "sai lệch = α(1−λ)(1−Cp_c)/(λ(K−1)n) + O(n⁻²), một dạng đóng đã kiểm khớp mọi chữ số với ma trận chuyển". Cái khớp mọi chữ số là **Ã_c/ã**, và chỉ ở C = 2, α = 1, n ≤ 200. Công thức tiệm cận ở các n đó lệch đáng kể. Ở các ô C = 3 chính xác của người rà (n = 15–24), tiệm cận lớn gấp 2–4 lần giá trị thật.
- Phần dư đúng là O(n⁻²), nhưng hằng số tương đối là ≈ −(K + Cα + 2Cα/λ)/n:
  - −54/n khi λ = 0,1, K = 10;
  - −125/n khi λ = 0,05, K = 20, C = 5.

  Nghĩa là phần dư ≈ 5% ở n = 1.000, và trội lên khi n_lab = λn nhỏ. Bằng chứng: `prop5_check.py` (1), n·(chính xác − tiệm cận)/tiệm cận hội tụ về −54,2, −56,0, −125,1.
- **Sửa:** thay bằng dạng đóng chính xác gọn hơn ở Phụ lục A, và ghi hằng số của phần dư.

### [MINOR-7] Mô tả phép kiểm tiệm cận không tái lập được — §5.7 (c)

- "khai triển Taylor bậc hai … (sympy)" không có trong repo.
- "6 tổ hợp (λ, p, α, K)": `e7f` chỉ dùng α = 1 và C = 2.
- Kết quả vẫn đúng: người rà kiểm với C ∈ {2, 3, 5} và α ∈ {0,5; 1; 2}, cả qua rút gọn đại số (Phụ lục A) lẫn qua chuỗi C = 3 chính xác. Nhưng phần mô tả phép kiểm phải khớp với code.

### [MINOR-8] "Đoạn đầu chưa tiệm cận của cùng công thức" chỉ đúng cho nửa SSL — §5.7, "Kiểm số"

- Dải số mũ n^{−0,86…−0,93} của E5c là hai bối cảnh gộp lại. Hồi quy lại của người rà (n ≥ 200) cho:
  - SSL-asym: n^{−0,858}·K^{−1,154};
  - PU: n^{−0,935}·K^{−1,155}.
- Dạng đóng không phủ PU.
- **Sửa:** chỉ nói cho SSL-asym.

### [MINOR-9] Mô phỏng max_context: chọn lọc ô khớp, và ngoại suy ghi như quan sát — §2.7; §0

- "Dự đoán … khớp mô phỏng: ×1,32 với ×1,32, và ×1,12 với ×1,11" bỏ ô M_c = 1.000, nơi dự đoán ×1,04 mà quan sát ×1,09.
- Công thức dự đoán cũng bỏ hệ số hiệu chỉnh tổng thể hữu hạn 1 − M_c/(n − m). Ở M_c = 1.000, hệ số này là 0,63, và đẩy dự đoán **xuống** ×1,02.
- §0 viết "mô phỏng cho thấy ảnh hưởng thật rất nhỏ **ở cỡ dữ liệu của bài**". Mô phỏng chỉ có n = 3.000, M_c ≤ 1.000, Φ′ = 0,5; phần "cỡ dữ liệu của bài" (n ≤ 7.000, M_c = 2.000) là ngoại suy AR(1). §2.7 đã ghi đúng "chưa chứng minh", nhưng §0 thì không.
- Ngoài ra, 1/10 chuỗi đổi lưu vực khi M_c ≤ 300.
- **Sửa:** báo cả ba ô; ở §0 ghi "dự đoán theo xấp xỉ".

### [MINOR-10] "Giảm n_F cần thiết 2–3 lần" và "chia ν tối ưu" — §0; §2.7

- Phần đã chặt (chia + Ville) giảm 60.000 → 36.000 và 122.000 → 69.000, tức 1,7–1,8 lần. "3 lần" chỉ có ở Bernstein với ánh xạ 0,9, mà phần đó còn MAJOR-1.
- `refined_band` chia đều xác suất thất bại thành 4 phần (q = p/4) rồi giải từng mảnh. Đó không phải phép chia ν tối ưu, nên gọi là "chia theo xác suất" thì đúng hơn.

### [MINOR-11] Số đo đặt trong câu mệnh đề — §5.7, "Đề xuất cho bài"

"π̃ lệch π một lượng TV ≍ 1/K không giảm theo n (đo được)" nằm trong phần (b) của một mệnh đề. Ký hiệu ≍ gợi ý đã chứng minh, trong khi đây là hồi quy trên E5c/E7b (K^{−1,13}). Nên tách ra thành nhận xét thực nghiệm.

### [COSMETIC-1] 𝓖_{s,k−1} phải gồm cả C_1, …, C_{k−1} — §2.7, chứng minh Bổ đề 3*, đoạn mở đầu

Điều cần cho bước 1–2 thực ra yếu hơn: Bad_k chỉ cần đo được theo 𝓖_{s,k−1}, không nhất thiết "chỉ phụ thuộc y". Xem mục "Đã kiểm mà không phá được".

### [COSMETIC-2] Tiêu đề cột "α₁ quan sát (đã trừ chệch chuỗi ngắn)" đặt sai chỗ — §2.8, bảng

- Số quan sát là số thô; số trong ngoặc mới là dự đoán đã trừ chệch.
- Chuỗi dài L = 41, không phải 40.
- Công thức chệch (1 + 4α)/L là của ước lượng Yule–Walker. Code dùng hệ số tương quan Pearson theo từng chuỗi (`np.corrcoef`), nên chệch hơi khác. Ảnh hưởng ở chữ số thứ ba.

### [COSMETIC-3] Mệnh đề 6(b) nên ghi điều kiện mẫu số

Cần 1 − α₁² − q₂ > 0 và α₁ ≠ 1. Cả hai tự động đúng, vì Var R·(1 − α₁² − q₂) = q(r*) > 0 khi c_k ∈ (0, 1). Nên ghi một dòng.

### [COSMETIC-4] Mệnh đề 5(a): ghi rõ π̃ là phân phối dừng của P̃, và a ≠ 0 là đủ

Câu "Ý nghĩa" thiếu hệ số 1/(n + Cα): sai lệch dự đoán bằng π̃(d_c)/(a(n + Cα)).

### [COSMETIC-5] Câu cũ và số bỏ sót

- §2.7 dòng cuối, "Lập luận đó chưa viết", đã lỗi thời sau §2.8.
- Ở 5(b), dải "TV 0,003–0,011 (tv_free)" bỏ K = 5, nơi TV là 0,021–0,023. Dải đầy đủ là 0,003–0,023.
- Hàng Bernstein của ánh xạ 0,6 ở n_F = 3.000 (0,245) **kém** hàng chia + Ville (0,239). Nên lấy min của hai bản.

## Phán quyết theo kết quả

| Kết quả | Mức chứng minh | Phán quyết | Ghi chú |
|---|---|---|---|
| (H*) yếu hơn (H⁻_δ) về logic | L4 | PROVEN | Bad = ∅ |
| Bổ đề 3*, phía dưới | L4 | PROVEN | Đã kiểm: luật có điều kiện của B_k và C_k; kỳ vọng siêu bội và H̃_K; hai biến cố Hoeffding và ρ_D ≥ ρa_m; quy nạp với ν′; bước cuối. Chặn H̃_K bị **đạt** bởi Bad thích nghi (58,42 so với 58,58), nên không lỏng ở trường hợp xấu nhất. |
| Bổ đề 3*, phía trên ("đối xứng") | L2 | PROVEN UNDER ADDITIONAL ASSUMPTIONS | Đối xứng thật, nhưng chưa viết; Bad của phía trên có thể khác Bad của phía dưới |
| Hệ quả cho 4′ (ψ*, p*), bão hoà e^{c·M_c} của thời gian thoát chứng minh được | L3 | PROVEN | Thay thế trực tiếp; số hạng K·e^{−M_cρ²a_m²ν_C²/2} chặn 1/p* |
| Giá "chỉ D·η₀·H̃_K" của (H*) | — | FALSE như kế toán của Bổ đề 3* ở n_F ≲ 1,2·10⁴ | MAJOR-2 |
| Hằng số tinh chỉnh: chia + Ville | L3 | PROVEN | Ville cho tiền tố của martingale trong vòng: đúng, không cần thừa số K; các mũ khớp code |
| Hằng số tinh chỉnh: Bernstein/Freedman | L2 | INCOMPLETE | Thiếu dừng, Jensen, và v trên [x − ν, 1] (MAJOR-1). Bernstein cho mẫu không hoàn lại qua định lý trội lồi của Hoeffding (1963, Thm 4) là **đúng**. Số hầu như không đổi. |
| Dự đoán √(1 + n_FΦ′²/(KM_c)) cho max_context | L1 | INSUFFICIENT INFORMATION | Xấp xỉ AR(1); 2/3 ô khớp |
| Mệnh đề 6(a), mô-men có điều kiện affine/bậc hai, chỉ qua R | L5 | PROVEN | Quy hoạch động độc lập khớp `sweep_moments` tới ≤ 2·10⁻¹⁵ trên 20 cấu hình, gồm b < 0 |
| Mệnh đề 6(b): μ = r*, Var dừng, trễ 1 = α₁ | L5 | PROVEN | Ma trận chuyển đầy đủ, 20 cấu hình, khớp 10 chữ số. Bộ lấy mẫu thật (MC 40.000 vòng): sd 0,05458 ± 0,00022 so với 0,05459 |
| Mệnh đề 6, Giới hạn "ρ < 1 vẫn đúng" | L4 (người rà) | PROVEN, nhưng lý do trong bài sai | MINOR-1 |
| Bảng so mô phỏng của Mệnh đề 6 | — | Số khớp CSV | ±1,5% và 18–21% đúng; "3 chữ số" nói quá (MINOR-2) |
| Mệnh đề 5(a), đồng nhất thức | L4 | PROVEN | Cần π̃ dừng dưới P̃, a ≠ 0; không cần duy nhất |
| Mệnh đề 5(b), khối free đúng tuyệt đối (SSL) | L5 | PROVEN | Dirichlet–multinomial và tích categorical cùng biên, cùng ngữ cảnh n − u. Chuỗi C = 3 chính xác: trung bình trùng tới 10 chữ số. Người rà thấy kết luận cũng đúng cho lịch quét hệ thống (L2). |
| Mệnh đề 5(c), drift affine, A_c, a, Ã_c, ã, E[a\|l] = l·L_c/n_lab | L5 | PROVEN (lịch khối ngẫu nhiên) | Rút gọn độc lập (Phụ lục A); C = 2 khớp 2,5·10⁻¹¹; C = 3 khớp ma trận chuyển tới 10 chữ số |
| Mệnh đề 5(c), tiệm cận và dấu, O(n⁻²) | L4 | PROVEN | Dấu đúng (co về 1/C); hằng số phần dư lớn khi λ nhỏ (MINOR-6) |
| Mệnh đề 5(c) cho lịch quét thật ("mặc định") | L2 | PROVEN UNDER ADDITIONAL ASSUMPTIONS (lịch khối ngẫu nhiên) | Số hạng chính trùng theo số (MINOR-5) |
| Phạm vi: PLL/PU không affine; dạng Poisson π̃(f) − π(f) = π̃((P̃ − P)u_f) | L3 | PROVEN | Dấu và dạng đúng; E5c xác nhận d ≠ 0 với PU |
| Câu đề xuất thay Mệnh đề 3(b) | — | FALSE như đã viết | BLOCKING-1 |

## Bảng giả định

| Giả định | Nêu ở đâu | Dùng ở đâu | Cần không? | Thiếu thì hỏng gì |
|---|---|---|---|---|
| Bad(y) không phụ thuộc B, C | (H*) | Bổ đề 3*, bước 2 | Cần. Đủ cả bản yếu hơn: Bad đo được theo 𝓖_{s,k−1} | Nếu Bad được chọn theo B, mọi khối có thể chứa min(\|B\|, η₀n_F) điểm xấu, và quay về giá của (H) |
| C_k đều cỡ M_c, độc lập khi biết (𝓖, B_k) | §2.7, thiết lập | Bước 3 | Cần | Mất 𝓔_ctx. Code đúng như vậy (`rng.choice`, không hoàn lại) |
| n_F > m | Bổ đề 3* | a_m > 0 | Cần | Chia cho a_m |
| Phương sai cục bộ v | §2.7, cải tiến 2 | Bản Bernstein | Cần thêm dừng và Jensen | MAJOR-1 |
| h ≡ 0, δ = 0, ρ = 1, K \| n, Φ affine trên [0,1] với Φ([0,1]) ⊆ [0,1] | §2.8, thiết lập | Mệnh đề 6 | ρ = 1 không cần (MINOR-1); K \| n cần cho dạng đóng như viết. MC với K ∤ n cho sd đổi trơn theo n | Với h ≠ 0, R không còn Markov một mình; cần thêm (S, Σh_i o_i). Hệ đóng được về nguyên tắc (người rà, L1) |
| SSL, mô hình gộp, lịch khối ngẫu nhiên, K \| n | §5.7, thiết lập | Mệnh đề 5(b)(c) | SSL cần (E5c, PU); lịch chỉ cần cho tính chính xác | BLOCKING-1; MINOR-5 |

## Nhất quán giữa các mục

- §0 nói "dải 99% khớp tới 3 chữ số" và "mô phỏng cho thấy ảnh hưởng thật rất nhỏ ở cỡ dữ liệu của bài". Thân bài có ngoại lệ, hoặc tự ghi là dự đoán AR(1). Xem MINOR-2 và MINOR-9.
- §0 và §5.7: "đủ để thay Mệnh đề 3(b)", trong khi `prop:blocks`(b) là phát biểu cho mọi loại nhãn yếu, kể cả bag-block. Xem BLOCKING-1.
- §5.7 (b) khẳng định khối free đúng tuyệt đối cho SSL, đúng. Nhưng bài và code chọn mặc định `all` vì khối free lệch ở PU. Bài nên nói thẳng hệ quả: với SSL, chế độ `free` tốt hơn mặc định ở mức dự đoán; với PU thì ngược lại.
- §2.7 nói "lập luận bậc hai … chưa viết", nhưng §2.8 đã viết.
- §2.7 nói AR(1) khớp "6–22%", còn §2.8 nói Jacobi thấp hơn 18–21%. Hai số dùng hai đại lượng khác nhau (sd trong chuỗi theo thời gian với sd gộp) và hai tập ánh xạ khác nhau (có và không có ánh xạ hai điểm hút). Không mâu thuẫn, nhưng nên ghi rõ.

## Đã kiểm mà không phá được

- **Luật có điều kiện của B_k** (Bổ đề 3*, bước 1). Kiểm với `array_split(rng.permutation(n), K)`:
  - các lần rút ở khối j < k chỉ phụ thuộc (y, B_j, C_j) và nhiễu riêng;
  - B_k đều trên A_k khi biết 𝓖_{s,k−1};
  - |A_k| ≥ (K − k + 1)·m₋;
  - C_k được rút sau, bằng `rng.choice(…, replace=False)`.
- **Chặn H̃_K.** `lemma3star_bad.py`, 4.000 lần lặp mỗi ô, 3 cấu hình, gồm K ∤ n (n = 1.003) và ρ < 1:
  - Bad tĩnh cho tổng ≈ η₀n_F;
  - Bad thích nghi (luôn nằm trong các điểm chưa dùng) cho 58,42 so với chặn 58,58; 137,7 so với 139,0; 114,1 so với 114,2.

  Chặn đúng và chặt trước đối thủ thích nghi.
- **Bước 𝓔_ctx.** e_j nằm trong một đoạn dài 1 chứa 0, có tổng 0, và Σ_{C_k} e_j = |C_k∩F|(r_C − r_k). Hai lần Hoeffding không hoàn lại cho đúng các mũ M_cρ_D²/2 và M_cρ_D²ν_C²/2. ρ_D ≥ ρa_m vì n − |B_k| ≤ n.
- **Quy nạp với ν′ = ν + ν_C + β_bad/a_m.** Kiểm từng bất đẳng thức. Đoạn n_F − f_k ≥ a_m·n_F hợp lệ. Bước cuối dùng Σ_F h_i = 0 và Σ f_k = n_F.
- **Ville cho Z không cần thừa số K.** Đúng, theo bất đẳng thức cực đại cho siêu martingale mũ e^{−θZ_t − θ²t/8}.
- **Bernstein cho mẫu không hoàn lại** qua Hoeffding (1963, Thm 4). Đúng: exp lồi, và phương sai tổng thể ρR_s(1−R_s) đo được theo 𝓕_s.
- **Mệnh đề 6.** Ba phép kiểm độc lập:
  - quy hoạch động chính xác theo khối (hypergeom tuần tự cho O_k, nhị thức cho N_k), không dùng đệ quy của tác giả. Trên n ∈ {12, 20, 30, 40, 60}, K ∈ {3, …, 6}, (c₀, b) ∈ {(0,45; 0,5), (0,24; 0,6), (0,1; 0,8), (0,7; −0,5)}: E và Var khớp `sweep_moments` tới ≤ 2·10⁻¹⁵; phần dư affine/bậc hai ≤ 10⁻¹⁵; trung bình dừng = r*, sd dừng và tự tương quan trễ 1 khớp `sweep_stationary` tới 10 chữ số;
  - `InContextGibbs` thật, 40.000 vòng quét, n = 60 và 40: sd 0,05458 ± 0,00022 so với 0,05459, và 0,06702 ± 0,00027 so với 0,06676; trễ 1 là 0,355 so với 0,357;
  - trường hợp ρ < 1, quy hoạch động chính xác (MINOR-1).

  Câu hỏi "trạng thái có quan trọng ngoài R không" có câu trả lời **không**, khi h ≡ 0 và ρ = 1: phân hoạch độc lập với y, luật của O biết S không phụ thuộc điểm nào đúng, và N_k chỉ phụ thuộc r_k.
- **Mệnh đề 5.**
  - A_c, a, Ã_c, ã và E[l_c | l] = l·L_c/n_lab được dẫn lại độc lập.
  - Rút gọn (Phụ lục A) cho thấy Ã/ã là trung bình hậu nghiệm với α′ = αK(n−1)/((K−1)n).
  - C = 3 kiểm bằng ma trận chuyển đầy đủ (3 cấu hình, gồm α = 0,5), cho cả all lẫn free, cả P lẫn P̃.
  - Đồng nhất thức (a) cũng được xác nhận gián tiếp, vì nhân chính xác trong mọi đệ quy cho đúng trung bình hậu nghiệm.

## Thứ tự ưu tiên sửa

1. **BLOCKING-1:** viết lại câu đề xuất: thêm "SSL, nhãn yếu không nghiêng", ghi lịch khối ngẫu nhiên, bỏ "đủ để thay 3(b)" tổng quát, và tách TV ≍ 1/K ra làm số đo.
2. **MAJOR-2:** sửa phần "Vì sao đây là giả định yếu hơn". Ghi giá đầy đủ D(η₀H̃_K + ν_b)(1 + L/a_m) và nói lợi thế chỉ có tiệm cận, hoặc làm chặt |B_k ∩ Bad| bằng Bernstein siêu bội.
3. **MAJOR-1:** viết bản Bernstein/Freedman của Bổ đề 3 (dừng, Jensen, v trên [x − ν, 1]); sửa `refined_band`; sửa "2–3 lần".
4. **MINOR-5 và MINOR-6:** ghi lịch quét. Thay tiệm cận bằng dạng đóng chính xác α′ ở Phụ lục A, kèm hằng số phần dư.
5. **MINOR-1 đến MINOR-4:** sửa lý do ρ < 1; sửa "3 chữ số"; sửa phép so với 4′; sửa câu "gần affine".
6. Các MINOR và COSMETIC còn lại.

## Phụ lục A. Dạng đóng chính xác cho Mệnh đề 5(c) (rút gọn của người rà)

Dùng u = M − l, n − M − U + u = n_lab − l, và

  E[u(1 − l/n_lab)] = M(1−λ)(1 − 1/K)·n/(n−1),

tính từ phương sai siêu bội của l. Khi đó, với lịch khối ngẫu nhiên và khối all:

  E_π k_c = U(L_c + α)/(n_lab + Cα),  E_π̃ k_c = U(L_c + α′)/(n_lab + Cα′),  α′ = α·K(n−1)/((K−1)n).

Nghĩa là: **chuỗi không sửa có đúng trung bình hậu nghiệm của một prior mạnh hơn K/(K−1) lần.** Từ đó, với N = n_lab:

  π̃(f_c) − π(f_c) = U(α′ − α)(N − C·L_c) / ((n + Cα)(N + Cα′)(N + Cα)),  α′ − α = α(n − K)/((K−1)n).

Khai triển cho đúng α(1−λ)(1−Cp_c)/(λ(K−1)n)·[1 − (K + Cα + Cα/λ + Cα′/λ)/n + O(n⁻²)].

**Kiểm:**
- khớp `ssl_affine_means` tới 2,5·10⁻¹¹ (C = 2, 4 tổ hợp (λ, p), n tới 10⁵);
- khớp ma trận chuyển C = 3 tới 10 chữ số.

Khi p_c = 1/C, sai lệch bằng 0 **chính xác** với mọi n, không chỉ ở số hạng chính.

---

## Phản hồi của tác giả (em, 05/11/2026 tối)

Mọi mục đã sửa trong `V2/W1_R1A_REPORT.md`. Chỗ sửa trong bài được đánh nhãn [IR2-…].

| Mục | Cách sửa |
|---|---|
| BLOCKING-1 | Câu đề xuất viết lại: phạm vi "SSL, lịch khối ngẫu nhiên"; giữ 3(b) tổng quát, chỉ thay đoạn "Why this is vacuous"; Mệnh đề 5 thành mệnh đề riêng. TV ≍ 1/K tách thành nhận xét thực nghiệm. Nói rõ: với SSL, free tốt hơn; với PU, all tốt hơn. §0 sửa theo. |
| MAJOR-1 | Viết ba bước còn thiếu (dừng, Jensen, phương sai một phía trên [x − ν, 1]). `refined_band` sửa theo bước (iii): 19.100 → 19.800, khớp số của người rà. Đổi "2–3 lần" thành "1,7–1,8 lần (chặt), khoảng 3 lần với Bernstein". Ba bước mới chưa được rà lại. |
| MAJOR-2 | Ghi giá đầy đủ β_bad(1 + b/a_m)/(1 − b), kèm bảng của người rà và điểm hoà vốn n_F ≈ 1,2·10⁴. Ghi hướng cải thiện (Bernstein siêu bội), chưa làm. |
| MINOR-1 | Thay lý do cho ρ < 1 bằng lập luận "điều kiện theo (f_k) rồi trộn". |
| MINOR-2 | "±0,005, ±0,001 ở n = 3.000". |
| MINOR-3 | So 4′ với dải [min, max] trên 60 vòng quét (nửa dải ≈ 0,025): lỏng khoảng 6 lần. Ghi rõ 2,576σ là xấp xỉ. |
| MINOR-4 | Ghi "affine trên toàn [0, 1]" và sửa câu "gần affine". |
| MINOR-5 | Ghi lịch khối ngẫu nhiên. Ghi bằng chứng số của người rà cho lịch thật. |
| MINOR-6 | Thêm dạng đóng α′ (Phụ lục A, em đã kiểm lại tới 1,5·10⁻¹¹) và hằng số của phần dư. |
| MINOR-7 | Lưu `V2/analysis/r2_asymptotic.py` (sympy); `e7f` thêm `e7f_alpha_prime_check.csv` với α ∈ {0,5; 1; 3}. |
| MINOR-8 | Số mũ thực nghiệm chỉ nêu cho SSL-asym. |
| MINOR-9 | Bảng max_context báo đủ 3 ô, thêm cột có hệ số tổng thể hữu hạn. §0 ghi "dự đoán theo xấp xỉ AR(1)". |
| MINOR-10 | Đổi "chia ν tối ưu" thành "chia xác suất thất bại". |
| MINOR-11 | Gộp vào BLOCKING-1. |
| COSMETIC-1 | 𝓖 gồm C_j; ghi điều kiện yếu hơn (Bad đo được theo 𝓖). |
| COSMETIC-2 | Sửa tiêu đề cột, L = 41, ghi chú Yule–Walker. |
| COSMETIC-3 | Thêm dòng điều kiện mẫu số. |
| COSMETIC-4 | Thêm 1/(n + Cα) và "chỉ cần a ≠ 0". |
| COSMETIC-5 | Bỏ câu "chưa viết" ở §2.7. Dải TV sửa thành 0,003–0,023. Bernstein 0,6 ghi "lấy min". |
