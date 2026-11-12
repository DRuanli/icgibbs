# REVIEW_V2_MATH_01 — Phản biện toán học độc lập cho phần lý thuyết MỚI (NeurIPS 2027)

Ngày: 12/11/2026. Người phản biện: referee độc lập (không xem quá trình soạn bài).
Phạm vi đọc từng dòng: `paper/sections/theory.tex` (Assumption 1, Theorem 3 `thm:basins`, đoạn "Uncorrected blocks", đoạn "Read with the gain"),
`paper/sections/appendix_proofs.tex` (§"Uncorrected blocks: bias of the predictor" với `prop:predbias`; §"Dynamics of the sampler as run" với
`lem:sweep`, chứng minh `thm:basins`, Remarks, `prop:moments`), `abstract.tex`, `introduction.tex`, và các câu dùng định lý trong
`experiments.tex` §`sec:tfm`, `limitations.tex`, `conclusion.tex`. Đối chiếu mã: `G4/code/icgibbs/sampler.py`, `G4/code/icgibbs/weak.py`,
`G4/code/icgibbs/g7/methods.py`, `G4/code/s2x_response_map.py`, `G4/code/experiments/r1a_block_bias.py`, `G4/code/experiments/w1_drift_check.py`,
`paper/scripts/make_v2.py`. Các kết quả cũ (Lemma 1, Thm 1, Thm 2, Prop blocks, Prop naive) chỉ đọc lướt.

---

## 1. Kết luận trong 5 dòng

1. **Không có lỗi BLOCKING trong phát biểu và chứng minh**: `lem:sweep`, `thm:basins` (a)(b)(c), `prop:moments`, `prop:predbias` (a)(b)(c) đều ĐÚNG như đã viết trong phụ lục; tôi dựng lại toàn bộ đại số và kiểm số bằng ma trận chuyển chính xác (sai khác 1e-12 đến 1e-16).
2. **Vấn đề nằm ở cách bài viết DÙNG và MÔ TẢ các kết quả (4 MAJOR)**: câu "by Theorem 3(b), a chain that starts there cannot climb" (`experiments.tex`) là một suy diễn không được định lý chống đỡ — với chính các ánh xạ đo được và hằng số của định lý, p ≥ 1 trên cả bốn dataset, và (b) cần nửa trên (H⁺) mà chính mục đó nói là KHÔNG đúng với TabPFN v2.
3. Abstract/Introduction/Theory nói quá: "for the sampler as run with a real TFM", "the start decides", "exponential escape time", "Thirty sweeps cannot repair…" — định lý là có điều kiện (Assumption 1, chưa được kiểm đúng dạng đã phát biểu), chỉ cho nhãn yếu cấp mẫu, không subsample context (loại ~21/52 dataset của Study 1–2), và ở n_F ≤ 2 222 cần biên gain ≥ 0,17–0,35 trong khi gain đo được ≤ 0,11.
4. Câu tóm tắt `prop:predbias` trong thân bài SAI ở ba điểm: "stationary mean of the predictor equals the exact posterior mean under α′" (đúng cho E[k_c], không đúng cho predictor), "bias of order 1/(nK)" (chỉ đúng khi tỉ lệ có nhãn cố định; đúng ra là 1/((K−1)N)), và không nói rằng đây là quét khối ngẫu nhiên chứ không phải phân hoạch của Algorithm 1.
5. Còn 16 lỗi MINOR (định nghĩa response map không nhất quán, quy ước miền của Φ̲/Φ̄, (1−p)^H khi p > 1, phiên bản Azuma, câu "15 configurations… full transition matrices" không đúng với mã, v.v.) và 2 COSMETIC (va chạm ký hiệu). Tất cả sửa được bằng chỉnh câu chữ/giả thiết, không cần đổi định lý.

**Đếm: BLOCKING 0 · MAJOR 4 · MINOR 16 · COSMETIC 2.**

---

## 2. Bảng phát hiện

| ID | Mức | Vị trí | Sai ở đâu | Cách sửa cụ thể |
|---|---|---|---|---|
| M1 | MAJOR | `experiments.tex` §`sec:tfm`, dòng 69–73 ("by \cref{thm:basins}(b), a chain that starts there cannot climb") và dòng 78–79 | Suy diễn không hợp lệ: (i) (b) cần (H⁺_δ) nhưng dòng 78 nói với TabPFN v2 "only the lower half of the assumption holds"; (ii) với hằng số của định lý, p_min = 3,65–15 ≥ 1 trên 3 dataset áp dụng được, dataset thứ tư (optdigits, n_train = 3 934) nằm ngoài phạm vi vì context bị subsample; (iii) "cannot" là mệnh đề xác suất 1 − Hp | Đổi thành mệnh đề định tính: "a negative gain is the situation in which \cref{thm:basins}(b) gives a ceiling when the upper half of \cref{ass:H} holds and n_F is large; at our sizes the theorem's constants are not informative, so this is an interpretation, not a certificate". Xem §3.M1 |
| M2 | MAJOR | `abstract.tex` dòng 8–9; `introduction.tex` dòng 26–28 và mục Contributions (2); `theory.tex` dòng 90–95; `conclusion.tex` | Nói quá so với định lý: "for the sampler as run with a real TFM … governs where a chain ends", "the start decides", "exponential escape time", "leaving a basin takes at least 1/p sweeps", "Thirty sweeps cannot repair a start in the wrong basin". Định lý: có điều kiện theo Assumption 1; chỉ nhãn cấp mẫu; không subsample; cận dưới của KỲ VỌNG thời gian thoát; vô nghĩa (p ≥ 1/30) trừ khi biên ≥ 0,17 + δ ở n_F = 2 222 | Câu chữ đề xuất ở §5 |
| M3 | MAJOR | `theory.tex` dòng 45–48 (đoạn "Uncorrected blocks") | Ba sai lệch so với `prop:predbias`: (i) "stationary mean of the predictor equals the exact posterior mean under … α′" — sai, chỉ E_π̃[k_c] bằng trung bình hậu nghiệm dưới α′; (ii) "bias of order 1/(nK)" — chỉ khi λ = N/n cố định; N cố định thì bias KHÔNG về 0 theo n; (iii) không nói mệnh đề dùng khối ngẫu nhiên mới ở mỗi bước, không phải phân hoạch của Alg. 1 | Viết lại câu (§3.M3) |
| M4 | MAJOR | `theory.tex` Assumption 1 (`ass:H`); `experiments.tex` dòng 75–81 ("We checked \cref{ass:H} on … cells") | Giả thiết là sup trên MỌI trạng thái y và MỌI khối B: (i) không kiểm được bằng thực nghiệm và rất mạnh (khối chỉ có 1 điểm tự do ép từng c_i riêng lẻ); (ii) r_{−B}(y) không xác định khi F ⊆ B, cỡ khối không giới hạn; (iii) phép "kiểm" trong `sec:tfm` kiểm một thứ khác (điểm bất động của ánh xạ Jacobi dự đoán độ chính xác cuối), không phải bất đẳng thức của Assumption 1 | Thu hẹp giả thiết về các cặp (y, B) thật sự dùng trong chứng minh; đổi "We checked Assumption 1" thành "We checked the one-dimensional description that Assumption 1 formalises" (§3.M4) |
| m1 | MINOR | `introduction.tex` dòng 26–27; `experiments.tex` dòng 69 và chú thích Fig. `fig:mechanism`; so với `ass:H` | "Response map" được định nghĩa ba kiểu: intro/caption nói "expected label accuracy after ONE SWEEP as a function of the accuracy before it"; Assumption 1 là ánh xạ cấp KHỐI theo độ chính xác của context; mã `s2x_response_map.py` đo bản "Jacobi" (mọi khối dùng cùng z) = cấp khối. Ánh xạ cấp vòng quét khác Φ: theo `prop:moments`, hệ số góc là α₁ ≠ b (K = 10, b = 0,5: α₁ = 0,355; n = 12, K = 3: α₁ = 0,3646) | Định nghĩa một lần: "Φ(r): expected accuracy of a block redraw when its context has a fraction r of correct labels (every block conditioned on the same assignment)"; sửa intro và caption |
| m2 | MINOR | `theory.tex` dòng 90–93 | Đoạn "Read with the gain g" chỉ đúng khi Φ không giảm: điều kiện của định lý dùng Φ̲(x−ν) = inf_{r ≥ x−ν}Φ(r), không phải g cục bộ. "every start reaches the same band" cần bản đi xuống của (c) (chưa phát biểu) và cả hai nửa giả thiết. "starts above the crossing stays up" chỉ đúng khi ở trên điểm cắt một khoảng ≥ biên | Thêm "for non-decreasing Φ"; phát biểu (c′) đối xứng dưới (H⁺); "sufficiently far above/below the crossing" |
| m3 | MINOR | `appendix_proofs.tex` chứng minh `thm:basins`(a), dòng 265–266 | "≥ (1−p)^H ≥ 1−Hp" và "Σ_H (1−p)^H = 1/p" chỉ đúng khi p ≤ 1 (p = 3, H = 2 cho (1−p)^H = 4). Kết luận vẫn đúng (p ≥ 1 thì 1 − Hp ≤ 0 và E[τ−s] ≥ 1 ≥ 1/p) | Viết max{0, 1−p}^H; thêm "for every integer H ≥ 1"; ghi τ có thể = ∞ |
| m4 | MINOR | `theory.tex` dòng 71; `lem:sweep` | Φ̲(x−ν) khi x−ν < 0 và Φ̄(x+ν) khi x+ν > 1 chưa định nghĩa (Φ chỉ xác định trên [0,1]); Lemma (ii) thiếu "𝓕_s-measurable" | "Φ̲(x) = inf{Φ(r): r ∈ [0,1], r ≥ x}, inf ∅ = +∞" và tương tự cho Φ̄ |
| m5 | MINOR | `lem:sweep`, bước "Probabilities", dòng 254–255 | Hằng số exp(−2t²/n_F) cần bản Azuma–Hoeffding với khoảng DỰ BÁO ĐƯỢC độ dài 1 (ξ_i − c_i ∈ [−c_i, 1−c_i], c_i đo được theo quá khứ); Azuma cổ điển với |D_i| ≤ 1 chỉ cho exp(−t²/(2n_F)). Kết quả đúng nhưng trích dẫn chưa đủ | Thêm một dòng: "by Hoeffding's lemma applied conditionally to each increment (the martingale form of Hoeffding's inequality with predictable ranges, e.g. McDiarmid 1998)"; nêu rõ lọc 𝒢 = σ(𝓕_s, hoán vị, các lần rút trước) |
| m6 | MINOR | Remarks (i), dòng 275–277 | "uses the assumption only at states the chain visits" — thực ra là các trạng thái TRONG vòng quét (Y_{s,k−1}, B_k) với r_{−B_k} ≥ x−ν (không phải ≥ x), không chỉ trạng thái cuối vòng | Viết chính xác tập cặp (y, B) được dùng |
| m7 | MINOR | `prop:moments` phát biểu (dòng 292–294) và chứng minh (dòng 310); `theory.tex` dòng 96 | (i) "(that is, Φ(r)=c₀+br, h≡0, δ=0)" không tương đương: mệnh đề cần MỖI lần rút đúng với xác suất đúng bằng c₀+br_k (N_k nhị thức); (H₀) với h ≡ 0 chỉ ràng buộc trung bình khối, khi đó phần phương sai không còn đúng. (ii) "If R_s = r*" — r* không nhất thiết thuộc lưới {j/n}. (iii) Thân bài nói "have a closed form": chỉ E R = c₀/(1−b) là dạng đóng, phương sai qua đệ quy K bước; chỉ cho ρ = 1, K ∣ n | (i) bỏ "that is", viết "a special case of (H₀) in which …"; (ii) "substituting R_s = r* in the (affine) mean recursion"; (iii) "the stationary mean has a closed form and the variance an exact K-step recursion (all points free)" |
| m8 | MINOR | `prop:predbias`(c) | Cần K ∣ n và K ≥ 2 (không nêu). Dạng đúng tổng quát là α′ = α(n−1)/(n−M) với mọi cỡ khối M. Lưới E7b có ô n = 50, "K = 20" thực ra M = 2 (n//K): công thức theo K cho 19,3145, ma trận cho 19,3047 = công thức theo M | Phát biểu theo M: "α′ = α(n−1)/(n−M); for M = n/K this is αK(n−1)/((K−1)n)" |
| m9 | MINOR | "Reading", dòng 187–188 ("agrees with the stationary means computed from full transition matrices to within … in 15 configurations") | Không khớp với mã: 15 cấu hình (`e7f_alpha_prime_check.csv`, có n = 100 000 — không thể dựng ma trận) so dạng đóng với điểm bất động của drift affine (`ssl_affine_means`), tức chỉ kiểm phép rút gọn đại số. Phép so với ma trận chuyển là 9 cấu hình khác (`e7f_affine_check.csv`: α = 1, C = 2, n ≤ 200) | Sửa câu: "agrees with the fixed point of the numerically averaged drift in 15 configurations and with stationary means from full transition matrices in 9 (α = 1, C = 2, n ≤ 200)"; hoặc bổ sung kiểm ma trận cho α ≠ 1, C = 3 (tôi đã làm, §4) |
| m10 | MINOR | `thm:basins` phát biểu (thân bài) | Thiếu giả thiết đã có trong "Setting" của phụ lục: thừa số nhãn yếu cố định. Mã gọi `weak.update_params` sau mỗi vòng quét (PU học c, nhãn nhiễu học T, PU-SAR học η): khi đó c_i phụ thuộc tham số | Thêm "with fixed weak-label factors (no nuisance-parameter update)" vào phát biểu, hoặc yêu cầu (H_δ) đều theo tham số |
| m11 | MINOR | `theory.tex` dòng 93–94; Remarks (ii) | "exponential in n_F": số mũ là cρn_Fν² = cν²n_F²/n, với c ∝ a² = (1 − ⌈n/K⌉/n_F)². Chỉ là mũ theo n_F khi ρ > 1/K, K, biên, h_max, δ, Φ cố định | "exponential in n_F for a fixed fraction of free points and fixed margins" |
| m12 | MINOR | Remarks (iii), dòng 283–284 | Con số nửa bề rộng \val{th-band-bound-half} = 0,22 phụ thuộc lựa chọn không nêu: H = 60 vòng, tổng xác suất hỏng 5%, hai phía, chia đều hai số hạng của p (`w1_drift_check.py::partition_bound`). Tôi tái lập: ν = 0,148, λ = 0,0376, (bν+λ)/(1−b) = 0,223 | Nêu các tham số này trong câu |
| m13 | MINOR | `theory.tex` dòng 44–46 | "for it \eqref{eq:stability} is vacuous: … TV(π̃,π) does not shrink with n" là non sequitur (cận vượt 1 ≠ TV không giảm), và che mất việc TV thật rất nhỏ (0,0012–0,0248) | "the bound exceeds 1 in 22 of 24 solvable cases while the true TV is 0.001–0.025 and does not shrink with n" |
| m14 | MINOR (kết quả cũ, do nén) | `theory.tex` dòng 100–103 so với `prop:naive`(a) | Thân bài: "through a symmetric channel"; phụ lục: "equal off-diagonal entries in each column" (M_{y,k} = b_k với y ≠ k) — hai điều kiện khác nhau. "class-dependent noise" chỉ thoả điều kiện khi C = 2 (hoặc T có cấu trúc đó), và λ_k > 0 cần M_{kk} > b_k | Dùng đúng điều kiện của phụ lục; ghi "binary class-dependent noise with T₀₀ + T₁₁ > 1" |
| m15 | MINOR | `prop:predbias`(c), khai triển tiệm cận | "(1+O(1/n))" đúng nhưng hằng số lớn (≈ 2Cα/λ): n = 100, λ = 0,1, α = 1 thì số hạng dẫn đầu lệch 40% (tỉ số 0,603); n = 1 000: 0,948 | Ghi "(1 + O(Cα/N))" để thấy sai số tính theo số điểm có nhãn |
| m16 | MINOR | `sec:tfm` + Remarks (iv) | Ánh xạ đo trong `s2x_response_map.py` dùng `max_context=2000` (context bị subsample ở dataset lớn) — đúng với sampler, nhưng nằm ngoài phạm vi định lý; bài chỉ nói ngắn ở Remark (iv) | Nêu trong `sec:tfm` những dataset nào của Fig. `fig:mechanism` nằm ngoài phạm vi định lý (optdigits) |
| c1 | COSMETIC | toàn bài | Va chạm ký hiệu (bảng ở §6) — nặng nhất: N_k hai nghĩa trong cùng tiểu mục `app:dynamics`; x (mức độ chính xác) vs x_i (đặc trưng); δ, τ, ν, λ, a, c, p, b, q dùng lại | Đổi tên theo bảng §6 |
| c2 | COSMETIC | `prop:predbias`(b) | "u < U" thừa: (b) đúng cả khi u = U (a = 1) | Bỏ điều kiện hoặc ghi "u ≤ U" |

---

## 3. Giải thích chi tiết các phát hiện MAJOR

### M1 — "by Theorem 3(b), a chain that starts there cannot climb" không suy ra được từ định lý

**Câu trong bài** (`experiments.tex` dòng 69–72): "With TabPFN v2 it [the gain Φ(r) − r] is negative for intermediate r on all four datasets: by \cref{thm:basins}(b), a chain that starts there cannot climb."

**Vì sao hỏng — ba lý do độc lập.**

1. *Giả thiết bị chính bài viết phủ nhận.* (b) được chứng minh dưới (H⁺_δ). Bảy dòng sau (dòng 78–79) bài viết: "With TabPFN v2 the error is one-sided: chains do better than the map says, so only the lower half of the assumption holds, and the uniform map certifies that a chain will climb, not that it will stall." Tức là với đúng TFM và đúng ánh xạ (gieo lỗi đều) ấy, (H⁺) không đúng, nên (b) không áp dụng được. Hai câu mâu thuẫn nhau.

2. *Điều kiện định lượng không thoả với hằng số của định lý.* Tôi lấy chính dữ liệu ánh xạ đã đo (`S2/raw/S2X/s2x_tabpfn.csv`, PLL q = 0,6, trung bình 3 seed, nội suy tuyến tính; δ = 0, ρ = 1, K = 10, h_max ≤ 1/2 — tức các lựa chọn CÓ LỢI NHẤT cho định lý) và tìm trên mọi (x, ν, λ) giá trị nhỏ nhất của p(ν, λ) mà điều kiện Φ̄(x+ν) + λ ≤ x còn thoả:

   | Dataset (≥ 8 lớp, TabPFN v2) | n_train | gain âm nhất đo được | p nhỏ nhất cho MỘT trần bất kỳ |
   |---|---|---|---|
   | MiceProtein | 756 | −0,077 | 15 |
   | mfeat-factors | 1 395 | −0,071 | 9,8 |
   | semeion | 1 115 | −0,111 | 3,65 |
   | optdigits | 3 934 | −0,019 | 26,5 — và ngoài phạm vi (n − ⌈n/K⌉ = 3 540 > 2 000, context bị subsample) |

   p ≥ 1 nghĩa là định lý không chứng nhận trần nào, kể cả cho một vòng quét. Lý do: ở n_F ≈ 750–1 400, để 30p ≤ 1 cần ν ≥ 0,17–0,24 và λ ≥ 0,04–0,05 (bảng ở M2), trong khi |gain| đo được ≤ 0,11.

3. *"cannot" là quá mạnh ngay cả khi định lý áp dụng được*: (b) cho "với xác suất ≥ 1 − Hp trong H vòng quét".

**Phạm vi ảnh hưởng.** Câu trong abstract "with TabPFN v2 the same chain stalled under heavy ambiguity, which the response map explains" và mệnh đề (2) của Contributions dựa vào suy diễn này. Phần thực nghiệm (ánh xạ âm ⇒ chuỗi kẹt) vẫn là một quan sát hợp lệ; chỉ có điều nó KHÔNG phải hệ quả của Theorem 3.

**Sửa tối thiểu.** "With TabPFN v2 the gain is negative for intermediate r on all four datasets. This is the configuration in which \cref{thm:basins}(b) yields a ceiling when the upper half of \cref{ass:H} holds and n_F is large; at the sizes of these datasets the theorem's constants are not informative and, for TabPFN v2, the upper half fails in the favourable direction (chains do better than the uniform-error map says). We therefore read the map as a qualitative explanation of the stall, not as a certificate."

---

### M2 — Abstract / Introduction / Theory nói nhiều hơn điều được chứng minh

Điều định lý thật sự nói: *nếu* một TFM tất định thoả (H^∓_δ) với (Φ, h, δ), nhãn yếu cấp mẫu với thừa số cố định, context không subsample, n_F > ⌈n/K⌉, *thì* một mức x thoả điều kiện biên là sàn/trần trong H vòng quét với xác suất ≥ 1 − H·p(ν, λ), và E[τ − s] ≥ 1/p.

Các khoảng cách giữa câu chữ và định lý:

| Câu trong bài | Định lý cho | Khoảng cách |
|---|---|---|
| Abstract: "We prove … for the sampler as run with a real TFM, that a one-dimensional response map of the TFM governs where a chain ends" | Mệnh đề có điều kiện theo Assumption 1 | Không có gì được chứng minh về một TFM thật; Assumption 1 chưa được kiểm đúng dạng (M4) |
| "for the sampler as run" / Intro: "For the sampler exactly as it is run" | Nhãn cấp mẫu, context đầy đủ | Không gồm LLP (một trong hai thiết lập chính của bài); không gồm n_train − ⌈n_train/K⌉ > 2 000. Theo `tables/app_datasets.tex` và `test_size = 0.3`, 21/52 dataset của Study 1–2 có 0,7n > 2 222 |
| "with two attractors, the start decides" | Sàn và trần riêng rẽ, mỗi cái cần biên | Chỉ cho điểm xuất phát cách điểm cắt ít nhất một biên; trong H vòng, xác suất 1 − Hp |
| Intro: "for a time exponential in the number of unknown labels"; Contributions: "invariant floors and ceilings … with an exponential escape time" | E[τ − s] ≥ 1/p | Là cận dưới của KỲ VỌNG; sàn/trần không "invariant" mà giữ với xác suất cao; mũ theo n_F chỉ khi ρ, K, biên cố định (m11) |
| Theory dòng 93: "leaving a basin takes at least 1/p sweeps" | như trên | Thiếu "in expectation" |
| Theory dòng 94: "Thirty sweeps cannot repair a start in the wrong basin" | 1 − 30p | Cần p < 1/30. Bảng dưới |

Biên cần thiết để định lý có nội dung trong 30 vòng quét (K = 10, ρ = 1, h_max ≤ 1/2, chia đều p cho hai số hạng; tự tính):

| n_F | 30p ≤ 1: ν / λ | 30p ≤ 0,05: ν / λ | Biên gain cần (Φ hệ số góc ≈ 1), chưa kể δ |
|---|---|---|---|
| 500 | 0,289 / 0,064 | 0,341 / 0,084 | 0,35 – 0,43 |
| 756 | 0,235 / 0,052 | 0,278 / 0,068 | 0,29 – 0,35 |
| 1 115 | 0,193 / 0,043 | 0,229 / 0,056 | 0,24 – 0,29 |
| 1 400 | 0,172 / 0,038 | 0,204 / 0,050 | 0,21 – 0,25 |
| 2 222 (lớn nhất chưa bị cap context) | 0,137 / 0,030 | 0,162 / 0,040 | 0,17 – 0,20 |

Gain đo được trên các dataset của Fig. `fig:mechanism` nằm trong [−0,11; +0,15], và phần dư có dấu (trung vị, các vòng quét muộn) của TabPFN v2 so với ánh xạ gieo đều lên tới +0,093 (`th-delta-pfn-res-hi` trong `numbers_v2.tex`; giá trị này không được dùng trong bài). Vậy ở mọi cỡ dữ liệu mà định lý áp dụng được trong bài, câu "Thirty sweeps cannot repair…" không phải hệ quả của định lý. Bài có nói "The constants are loose" và Remark (iii) "informative only for large n_F", nhưng abstract, intro và câu dòng 94 không mang theo cảnh báo đó.

**Sửa**: xem câu chữ đề xuất ở §5.

---

### M3 — Câu tóm tắt `prop:predbias` trong thân bài sai ở ba điểm

**Câu trong bài** (`theory.tex` dòng 45–48): "in a conjugate model with unconstrained unknown labels, the stationary mean of the predictor equals the exact posterior mean under a prior concentration inflated by K(n−1)/((K−1)n), a bias of order 1/(nK) (\cref{prop:predbias})."

**(i) "predictor … equals" là sai.** Mệnh đề (c) nói E_π̃[k_c] = U(L_c+α′)/(N+Cα′): trung bình của SỐ ĐẾM nhãn tự do bằng trung bình hậu nghiệm dưới Dir(α′). Predictor là f_c(k) = (L_c + k_c + α)/(n + Cα) — vẫn dùng α gốc — nên với m = (L_c+α)/(N+Cα), m′ = (L_c+α′)/(N+Cα′):

  π̃(f_c) = m′ + (N+Cα)/(n+Cα) · (m − m′),

một tổ hợp lồi của m và m′, không bằng m′ (trừ khi α′ = α hoặc CL_c = N). Phản ví dụ nhỏ nhất, tính bằng ma trận chuyển đầy đủ trên vectơ nhãn:

- n = 8, N = 3, L = (2, 1), C = 2, α = 1, K = 4: π̃(f₀) = 0,596875; dự báo hậu nghiệm chính xác dưới α′ = 7/6 là 0,59375; hậu nghiệm đúng (α) là 0,6.
- n = 8, N = 3, L = (2, 1, 0), C = 3, α = 0,7, K = 2: π̃(f₀) = 0,50651; dưới α′ là 0,48315; hậu nghiệm đúng 0,52941.

Công thức bias trong phụ lục thì đúng (đã kiểm tới 1e-16); chỉ câu diễn giải trong thân bài sai.

**(ii) "bias of order 1/(nK)" thiếu điều kiện.** Từ công thức chính xác, bias = U(α′−α)(N−CL_c)/((n+Cα)(N+Cα′)(N+Cα)). Với λ = N/n cố định: ≈ α(1−λ)(1−Cp_c)/(λ(K−1)n) — đúng bậc 1/(nK). Nhưng với SỐ điểm có nhãn N cố định, n → ∞: bias → α(N−CL_c)/((K−1)(N+Cα′)(N+Cα)) ≠ 0. Tính số (N = 10, L = 8, α = 1, K = 10, C = 2): n = 10², 10³, 10⁴, 10⁵, 10⁶ cho bias = −0,00362; −0,00445; −0,00454; −0,004544; −0,004545 (n·K·bias tăng tuyến tính theo n). Bậc đúng là 1/((K−1)·N): bias tỉ lệ nghịch với số điểm CÓ NHÃN.

**(iii) Kiểu quét.** Mệnh đề (c) rút một khối ngẫu nhiên mới ở mỗi bước. Algorithm 1 phân hoạch một hoán vị. Phụ lục có nói ở "What the identity does not cover (i)", thân bài thì không, và câu đứng ngay sau "The default sampler skips that step". Tôi dựng nhân vòng quét chính xác của Alg. 1 (trung bình trên mọi phân hoạch có thứ tự) cho bốn trường hợp nhỏ: trung bình dừng của k_c ở cuối vòng quét KHÔNG bằng công thức (c); độ dịch so với hậu nghiệm lớn hơn công thức 1,7–3,8%:

| n, N, L, C, α, K | E k₀ (Alg. 1, chính xác) | công thức (c) | hậu nghiệm | tỉ số độ dịch |
|---|---|---|---|---|
| 8, 3, (2,1), 2, 1, 4 | 2,967548 | 2,968750 | 3 | 1,038 |
| 9, 4, (3,1), 2, 0,5, 3 | 3,435257 | 3,437500 | 3,5 | 1,036 |
| 8, 3, (2,1,0), 3, 0,7, 2 | 2,407523 | 2,415730 | 2,647059 | 1,035 |
| 10, 5, (3,1,1), 3, 2, 5 | 2,233399 | 2,234043 | 2,272727 | 1,017 |

Vậy với sampler mặc định, "exactly" không đúng; chỉ đúng tới số hạng dẫn đầu (như phụ lục đã ghi).

**Câu thay thế đề xuất**: "Predictions behave better: in a conjugate model with unconstrained unknown labels and a fresh random block per step, the stationary mean of the free-label counts equals their exact posterior mean under a prior concentration inflated by K(n−1)/((K−1)n); the resulting bias of the predictor is of order 1/((K−1)N), N being the number of labelled points (\cref{prop:predbias}). For the permutation scan of \cref{alg:icg} the same holds to leading order, numerically."

---

### M4 — Assumption 1: dạng phát biểu quá mạnh, có chỗ chưa xác định, và phép "kiểm" trong bài kiểm một thứ khác

1. *Lượng từ.* "for every state y and every block B with B∩F ≠ ∅" là sup trên C^{n_F} trạng thái và mọi tập con B. (a) r_{−B}(y) không xác định khi F ⊆ B; cỡ của B không giới hạn (định lý chỉ cần |B| ∈ {⌊n/K⌋, ⌈n/K⌉}). (b) Khi ρ < 1 có khối chỉ chứa 1 điểm tự do; áp giả thiết cho các khối đó cho |c_i(y,B) − h_i − Φ(r_{−B}(y))| ≤ δ với TỪNG điểm, tức đồng nhất theo điểm chứ không chỉ theo trung bình khối — mạnh hơn nhiều so với lời diễn giải "depends on how many context labels are right … up to a tolerance δ for which ones are right". (c) Một δ nhỏ đều trên mọi (y, B) là khó tin với TFM thật: có thể chọn y đối nghịch (ví dụ mọi lỗi dồn vào một lớp).

2. *Chứng minh dùng ít hơn thế.* `lem:sweep` chỉ áp giả thiết tại các cặp (Y_{s,k−1}, B_k) trong vòng quét với r_{−B_k} ≥ x − ν (sàn) hoặc ≤ x + ν (trần). Có thể phát biểu giả thiết trên một tập 𝒢 các cặp (y, B) và kết luận "cho tới khi chuỗi rời 𝒢" mà không đổi chứng minh.

3. *Phép kiểm thực nghiệm.* `experiments.tex` dòng 75: "We checked \cref{ass:H} on 234 cells". Cái được kiểm (`V2/analysis/w1_delta.py`, `v1c_chain_map.py`) là: điểm bất động của ánh xạ đo bằng cách gieo lỗi (đều / theo biên của chuỗi) dự đoán độ chính xác cuối với sai số trung vị 0,073 / 0,016. Đó là kiểm một HỆ QUẢ ở mức quỹ đạo, với một họ trạng thái gieo cụ thể; nó không ước lượng δ = sup_{y,B}|…| và không kiểm bất đẳng thức của Assumption 1. Thêm nữa, chính đoạn đó cho thấy với lỗi gieo đều thì dự đoán sai mức ("right about direction, not about level"), tức δ của ánh xạ gieo đều không nhỏ.

**Sửa tối thiểu.** (i) Phát biểu: "for every block B with |B| ≤ ⌈n/K⌉ and F ⊄ B, and every state y in a set 𝒴 ⊆ [C]^n" và cho định lý kết luận tới thời điểm đầu tiên một trạng thái trong vòng quét rời 𝒴. (ii) Đổi câu thực nghiệm thành "We checked the one-dimensional description that \cref{ass:H} formalises …; this is a check of a consequence along the chains, not of the assumption for all states." (iii) Nêu rõ (H_δ) theo điểm khi khối có một điểm tự do, hoặc loại các khối đó bằng điều kiện |B∩F| ≥ f_min.

Tôi xếp MAJOR vì một referee sẽ hỏi ngay "δ là bao nhiêu và đo thế nào", và hiện bài không trả lời được câu đó cho giả thiết như đã viết. Tính đúng của định lý (như một mệnh đề có điều kiện) không bị ảnh hưởng.

---

## 4. Những gì đã kiểm, bằng cách nào, và mức tin cậy

Script (chỉ ghi trong scratchpad, không sửa repo):
- `/tmp/claude-0/-home-claude/643477cb-9e2c-5e3b-9dd1-1dba7a8f3cd0/scratchpad/review_math/predbias_exact.py`
- `/tmp/claude-0/-home-claude/643477cb-9e2c-5e3b-9dd1-1dba7a8f3cd0/scratchpad/review_math/predbias_lumped.py`
- `/tmp/claude-0/-home-claude/643477cb-9e2c-5e3b-9dd1-1dba7a8f3cd0/scratchpad/review_math/moments_exact.py`
- `/tmp/claude-0/-home-claude/643477cb-9e2c-5e3b-9dd1-1dba7a8f3cd0/scratchpad/review_math/lemma_sweep_sim.py`
- `/tmp/claude-0/-home-claude/643477cb-9e2c-5e3b-9dd1-1dba7a8f3cd0/scratchpad/review_math/vacuity_real_maps.py`

### 4.1 Lemma "One sweep" và Theorem 3 — PROVEN (có điều kiện theo Assumption 1); tin cậy ≈ 95%

Kiểm tay từng bước:
- Đẳng thức (n_F − f_k) r_k = Σ_{U_k} ξ_i + Σ_{V_k} o_i: đúng (mỗi điểm tự do được rút lại đúng một lần; |U_k| + |V_k| = n_F − f_k).
- Quy nạp theo k: dùng (H⁻) tại (Y_{s,j−1}, B_j) với r_j ≥ x − ν, Φ(r_j) ≥ Φ̲(x−ν), Φ̲(x−ν) − δ ≥ x; khối không có điểm tự do bị bỏ qua và không ảnh hưởng; n không chia hết cho K: f_k ≤ m = ⌈n/K⌉ là đủ; n_F > m bảo đảm n_F − f_k > 0.
- Cuối vòng quét: Σ_F h_i = 0 và Σ_k f_k = n_F được dùng đúng.
- Đo được/điều kiện hoá: hoán vị độc lập với 𝓕_s (trong mã: `rng.permutation` rút mới mỗi vòng); cho trước 𝓕_s, U_k và V_k là mẫu không hoàn lại cỡ tất định (cỡ khối của `np.array_split` là tất định); quần thể {h_i 1[i∈F]} có tổng 0, khoảng ≤ 2h_max; quần thể {(o_i − R_s)1[i∈F]} có tổng 0, khoảng ≤ 1 (giá trị trong [−R_s, 1−R_s], kể cả số 0 của điểm ngoài F). Hoeffding (1963, Thm 4) áp dụng được.
- Martingale: cho trước hoán vị và 𝓕_s, xếp các lần rút theo khối; trong một khối các lần rút độc lập có điều kiện theo trạng thái trước khối (mã: `_draw` dùng một uniform độc lập cho mỗi hàng; TFM tất định), nên ξ_i − c_i là hiệu martingale trong khoảng dự báo được độ dài 1. Cận exp(−2t²/n_F) đúng với bản Hoeffding có điều kiện (m5).
- Hằng số: t = νa n_F/3. Phần h: t²/(2n h²_max) = a²ρ n_F ν²/(18h²_max). Phần o: 2t²/n = (2/9)a²ρ n_F ν². Phần Z_k: 2t²/n_F = (2/9)a² n_F ν² ≥ (2/9)a²ρ n_F ν². Cả ba ≤ exp(−cρ n_F ν²) với c = a² min{2/9, 1/(18h²_max)} — ĐÚNG. Union bound 3K đúng (thừa một chút: k = 1 và k = K mỗi cái có phần bằng 0). h_max = 0: đúng, c = 2a²/9.
- Định lý (a): lặp điều kiện hoá cho ≥ max(0,1−p)^H ≥ 1 − Hp; E[τ−s] = Σ_H P(τ−s > H) ≥ 1/p (m3 về trường hợp p > 1). (b): đối xứng, đã kiểm lại dấu từng bước. (c): q_j tất định, J ≤ ⌈(b−r₀)/γ⌉, điều kiện bổ đề tại x = q_j thoả vì Φ̲(q_j−ν) − δ = ψ(q_j) + λ ≥ q_j; quy nạp cho 1 − Jp. Đúng.
- Khớp thân bài ↔ phụ lục: p(ν, λ), c, a, điều kiện n_F > ⌈n/K⌉ trùng nhau. Remark (ii): g(x) ≥ δ + ℓν + λ là đủ khi Φ không giảm và ℓ-Lipschitz; p ≤ (1+3K)exp(−cρ n_F ν²) vì 2 ≥ cρ — đúng.
- Khớp mã: `sampler.py::_sweep_instance` với `block_over="all"` (mặc định), `mh_correct=False`: hoán vị của mọi điểm, `array_split` thành K khối, context = mọi điểm ngoài khối, chỉ rút điểm tự do, khối rỗng bị bỏ qua. Trùng "Setting". Khác biệt: `max_context=2000` (ngoài phạm vi khi n − m > 2 000) và `weak.update_params` (m10).

Tấn công đã thử và không phá được: Φ không đơn điệu, gián đoạn; h_i = ±h_max; nhiễu ±δ phụ thuộc trạng thái; n không chia hết cho K; có điểm cố định (ρ < 1); khối không có điểm tự do; trạng thái xuất phát đúng tại biên R_s = x.

Kiểm số (mô phỏng, `lemma_sweep_sim.py`): kiểm THEO TỪNG QUỸ ĐẠO hàm ý tất định "trên 𝓔 thì r_k ≥ x−ν với mọi k và R_{s+1} ≥ Φ̲(x−ν) − δ + Z/n_F" (và bản đối xứng):

| n, n_F, K, h_max, δ, x, ν, λ, phía | vi phạm trên 𝓔 | tần suất 𝓔ᶜ (cận) | tần suất hỏng (cận p) |
|---|---|---|---|
| 503, 400, 10, 0,05, 0,01, 0,60, 0,02, 0,03, sàn | 0 / 20 000 | 0,399 (1) | 0,0009 (29,8) |
| 503, 400, 10, 0,05, 0,01, 0,60, 0,10, 0,03, sàn | 0 / 20 000 | 0,0001 (1) | 0,0006 (18,0) |
| 2003, 1500, 7, 0,10, 0,02, 0,62, 0,15, 0,04, sàn | 0 / 4 000 | 0 (0,533) | 0 (0,542) |
| 4000, 3000, 10, 0,20, 0, 0,60, 0,14, 0,03, sàn | 0 / 3 000 | 0 (0,019) | 0 (0,024) |
| 503, 400, 10, 0,05, 0,01, 0,30, 0,02, 0,03, trần | 0 / 20 000 | 0,416 (1) | 0,0305 (29,8) |
| 2003, 1500, 7, 0,10, 0,02, 0,28, 0,06, 0,04, trần | 0 / 4 000 | 0 (1) | 0 (11,7) |

0 vi phạm trong 71 000 vòng quét. Lưu ý: mô phỏng không thể bác bỏ cận xác suất vì cận rất lỏng (đây là kiểm logic, không phải chứng minh).

### 4.2 Proposition "Stationary moments for an affine map" — PROVEN; tin cậy ≈ 98%

Kiểm tay: (R_s) là xích Markov trên {0, 1/n, …, 1} (phân hoạch đều ⇒ (O_1..O_K) siêu bội đa biến chỉ phụ thuộc S); trung bình affine và hiệp phương sai bậc hai theo S truyền qua đệ quy N_k = mc₀ + g_kᵀW_k + ε_k (cần E[p_k(1−p_k)] = E p_k − (E p_k)² − Var p_k); α₁ ≠ 1 vì c₀ > 0; điểm bất động r* = c₀/(1−b) (dùng n − m = (K−1)m, cần K ∣ n); Cov(R_{s+1}, R_s) = α₁ Var R; phương sai toàn phần ⇒ α₁² < 1 và đẳng thức cuối. Đúng.

Kiểm số bằng ma trận chuyển CHÍNH XÁC trên S (`moments_exact.py`), kèm mô phỏng 200 000 vòng quét của xích chỉ báo thật:

| n, K, c₀, b | sai số khớp affine / bậc hai | α₁ | E R (= r*) | tự tương quan bậc 1 | Var R·(1−α₁²−q₂) vs q₀+q₁r*+q₂r*² | sd chính xác / mô phỏng |
|---|---|---|---|---|---|---|
| 12, 3, 0,3, 0,5 | 4e-16 / 2e-16 | 0,3645833 | 0,6000000000 | 0,3645833333 | 0,034383341471 = 0,034383341471 | 0,19577 / 0,19600 |
| 12, 3, 0,05, 0,9 | 2e-16 / 2e-16 | 0,8482500 | 0,5000000000 | 0,8482500000 | 0,058041058420 = 0,058041058420 | 0,35344 / 0,35377 |
| 12, 4, 0,2, 0 | 2e-16 / 2e-17 | 0 | 0,2000000000 | 0 | 0,013333333333 = 0,013333333333 | 0,11547 / 0,11546 |
| 20, 4, 0,45, 0,5 | 3e-16 / 3e-16 | 0,3605324 | 0,9000000000 | 0,3605324074 | 0,007792249134 = 0,007792249134 | 0,09372 / 0,09353 |
| 12, 2, 0,1, 0,7 | 3e-16 / 2e-16 | 0,5950000 | 0,3333333333 | 0,5950000000 | 0,041053942200 = 0,041053942200 | 0,23564 / 0,23549 |

α₁ tính bằng đệ quy trung bình d_k = b(Σ_{j<k}d_j + K − k)/(K−1) trùng tới 10 chữ số. Đối chứng: Φ không affine (0,2 + 0,6r²) cho sai số khớp affine 0,082 — giả thiết affine là cần.

### 4.3 Proposition "Bias of the predictor under uncorrected blocks" — PROVEN (với khối ngẫu nhiên mới mỗi bước, K ∣ n); tin cậy ≈ 99%

Dựng lại đại số: E u = MU/n; E[ul] = M²NU/n² − V; E u² = V + M²U²/n²; V/N = M(U/n)(n−M)/(n(n−1)); ngoặc thứ nhất = (MU/n)[L_c(n−M)/(n−1) + α]; ngoặc thứ hai = (MU/n)[N(n−M)/(n−1) + Cα] (dùng N − M + MU/n = N(n−M)/n); điểm bất động U[L_c + α(n−1)/(n−M)]/[N + Cα(n−1)/(n−M)]; (n−1)/(n−M) = K(n−1)/((K−1)n) khi M = n/K; α′ − α = α(n−K)/((K−1)n); hiệu f_c và số hạng dẫn đầu α(1−λ)(1−Cp_c)/(λ(K−1)n)·(n−K)/n·(1+O(Cα/N)). Tất cả ĐÚNG. (a): một dòng, đúng. (b): a = (u/U)(1 − (U−u)/(n−u+Cα)) > 0, đúng; đúng cả khi u = U.

Kiểm số độc lập 1 — xích đầy đủ trên vectơ nhãn y ∈ [C]^U, liệt kê mọi khối, KHÔNG dùng công thức siêu bội (`predbias_exact.py`):

| n, N, L, C, α, K | max_c |E_π̃k_c − công thức| | max |bias − công thức| | |π_P − hậu nghiệm|₁ | (a): π̃(f)−π(f) vs π̃(d)/a | (b) khối tự do, u = 1..U: max |E_π̃k − E_πk| |
|---|---|---|---|---|---|
| 8, 3, (2,1), 2, 1, 4 | 2,7e-15 | 4,6e-16 | 5,9e-16 | −0,03125 = −0,03125 | ≤ 8,9e-16 |
| 9, 4, (3,1), 2, 0,5, 3 | 1,3e-15 | 2,6e-16 | 2,9e-15 | −0,0625 = −0,0625 | ≤ 1,3e-15 |
| 8, 3, (2,1,0), 3, 0,7, 2 | 4,9e-15 | 9,4e-16 | 5,0e-16 | −0,2313284865 = −0,2313284865 | ≤ 3,1e-15 |
| 10, 5, (3,1,1), 3, 2, 5 | 2,2e-15 | 6,4e-16 | 3,0e-15 | −0,0386847195 = −0,0386847195 | ≤ 4,0e-15 |
| 6, 0, (0,0), 2, 1, 3 (N = 0) | 5,8e-15 | 1,8e-15 | 4,2e-15 | 0 = 0 | ≤ 1,5e-14 |

Drift của nhân chính xác là affine theo k_c (sai số khớp ≤ 7e-15) — giả thiết của (a) thoả. Với khối tự do, trung bình trùng nhưng TV(π̃, π) không nhỏ khi u lớn (u = 5: 0,12–0,20) — đúng tinh thần "bias ở mức trung bình ≠ khoảng cách phân phối".

Kiểm số độc lập 2 — xích gộp C = 2 tự viết (`predbias_lumped.py`), α = 1, 10% có nhãn, 30% dương (cấu hình SSL-asym của bài): n ∈ {50, 100, 200}, K ∈ {5, 10, 20}: |E_π̃k (ma trận) − công thức| ≤ 5e-12 ở 8/9 ô; ô n = 50, K = 20 lệch 9,7e-3 vì M = n//K = 2 ≠ n/K (m8) — công thức theo M khớp. α khác 1: (n, N, L, K, α) = (60, 12, 2, 6, 0,3): 8,8914069877 = 8,8914069877; (60, 12, 9, 3, 3): 30,9064748201 trùng; (90, 9, 1, 10, 0,5): 12,4272616137 trùng; (40, 20, 3, 4, 2): 4,4444444444 trùng. TV(π̃, π) ở K = 10: 0,0051; 0,0093; 0,0104 cho n = 50, 100, 200 — khớp \val{th-e7b-tv-k10} (0,005; 0,009; 0,010) và với khoảng [0,0012; 0,0248] của bài.

Tiệm cận: λ = 0,1, p_c = 0,8, α = 1, K = 10: tỉ số bias/số hạng dẫn đầu = 0,603; 0,948; 0,995; 0,9995 cho n = 10², 10³, 10⁴, 10⁵.

Các đoạn văn: "Reading" — đúng về toán (tỉ lệ 1/n, 1/(K−1), U/N; triệt tiêu khi p_c = 1/C; cơ chế: context mất l ≈ N/K điểm có nhãn ⇒ α′ ≈ αN/(N−l)); câu về "15 configurations" sai nguồn (m9). "What the identity does not cover" — trung thực; tôi xác nhận (i) bằng tính chính xác (bảng ở M3). "Why label-level bounds are vacuous" — số TV khớp; kết luận hợp lý.

### 4.4 Cách thân bài mô tả — xem M1–M3, m1, m2, m13 và §5.

### 4.5 Bảng verdict theo kết quả

| Kết quả | Verdict | Ghi chú |
|---|---|---|
| Assumption 1 | Cần sửa phát biểu (M4, m4) | Miền của B và r_{−B}; sup trên mọi (y, B) |
| Lemma `lem:sweep` (i), (ii) | PROVEN | Sống sót: Φ không đơn điệu/gián đoạn, h ≠ 0, ρ < 1, K ∤ n, khối rỗng |
| Theorem 3 (a), (b), (c) | PROVEN (có điều kiện) | Thêm quy ước p > 1, "for every H"; thừa số nhãn yếu cố định |
| Remarks (i)–(v) | Đúng; (i) và (iii) cần chính xác hơn | m6, m12 |
| `prop:moments` (a), (b) | PROVEN | Giả thiết là "mỗi lần rút đúng với xác suất đúng bằng c₀+br_k", mạnh hơn (H₀) |
| `prop:predbias` (a) | PROVEN | Tầm thường, đúng |
| `prop:predbias` (b) | PROVEN | Đúng cả khi u = U |
| `prop:predbias` (c) | PROVEN UNDER ADDITIONAL ASSUMPTIONS: K ∣ n, K ≥ 2, khối ngẫu nhiên mới mỗi bước | Với phân hoạch của Alg. 1: KHÔNG chính xác (lệch 1,7–3,8% ở ví dụ nhỏ) |
| Câu tóm tắt trong `theory.tex` dòng 45–48 | FALSE như đã viết | M3 |
| Câu `experiments.tex` dòng 71 | Suy diễn không hợp lệ | M1 |

---

## 5. Các câu nói quá trong thân bài / abstract và câu chữ đề xuất

1. **Abstract, dòng 8–9.**
   Hiện tại: "We prove that its exact variants target the Bayes posterior when the TFM is exact and, for the sampler as run with a real TFM, that a one-dimensional response map of the TFM governs where a chain ends: with two attractors, the start decides."
   Đề xuất: "We prove that its exact variants target the Bayes posterior when the TFM is exact. For the default sampler on instance-level weak labels we show that, if the TFM's block response depends on its context mainly through the fraction of correct labels, a one-dimensional response map yields floors and ceilings for the label accuracy that hold with high probability: with two attractors, the start decides for many sweeps."

2. **Abstract, dòng 12**: "which the response map explains" → "which a measured response map accounts for qualitatively".

3. **Introduction, dòng 26–28.**
   Hiện tại: "We answer with a response map, the expected label accuracy after one sweep as a function of the accuracy before it. For the sampler exactly as it is run, the map bounds where a chain can go (Theorem 3): with two attractors the start decides, for a time exponential in the number of unknown labels."
   Đề xuất: "We answer with a response map: the expected accuracy of a block redraw as a function of the accuracy of the labels in its context. Under a homogeneity assumption on the TFM, and for the default sampler on instance-level weak labels, the map bounds where a chain can go (Theorem 3): with two attractors the start decides, and the expected time to leave a basin grows exponentially with the number of unknown labels."

4. **Contributions (2)**: "for the default sampler with a real TFM, invariant floors and ceilings of the label accuracy with an exponential escape time" → "for the default sampler on instance-level weak labels and any deterministic TFM with a homogeneous response, floors and ceilings of the label accuracy that hold with high probability, with an expected escape time exponential in the number of unknown labels".

5. **Theory, dòng 90–95** (đoạn "Read with the gain"). Đề xuất viết lại:
   "For a non-decreasing Φ, read with the gain g(r) = Φ(r) − r. Where g exceeds the margins, the chain climbs (c) and, with probability at least 1 − Hp over H sweeps, does not fall back (a); where g is below minus the margins, it does not climb (b). If g is negative on an interval and positive above it, there are two basins: a chain that starts sufficiently far above the crossing stays up and one that starts sufficiently far below stays down, and the expected time to leave a basin is at least 1/p, which grows exponentially with n_F when the fraction of free points and the margins are fixed. The constants are loose: at the sizes of our datasets p is not small, so the theorem describes the mechanism and does not certify individual runs. This is why we use several starts: a learner start usually lies above the crossing even when the warm start does not."
   (Bỏ "Thirty sweeps cannot repair a start in the wrong basin" và "every start reaches the same band" — câu sau cần bản đi xuống của (c).)

6. **Theory, dòng 96**: "for affine Φ the stationary mean and variance of R_s have a closed form" → "for affine Φ and all points free, the stationary mean of R_s has a closed form and its variance an exact K-step recursion".

7. **Theory, dòng 45–48**: xem câu thay thế ở M3.

8. **Theory, dòng 44–45**: xem m13.

9. **Experiments `sec:tfm`, dòng 71**: xem câu thay thế ở M1. **Dòng 75**: "We checked \cref{ass:H}" → "We checked the one-dimensional description that \cref{ass:H} formalises".

10. **Caption Fig. `fig:mechanism`**: "gain Φ(r) − r of one Gibbs sweep started from a fraction r of correct labels" → "gain Φ(r) − r of one block redraw given a context with a fraction r of correct labels (all blocks conditioned on the same planted assignment)".

11. **Conclusion**: "a response map of the TFM governs where a chain ends" → "a response map of the TFM describes where a chain ends".

12. **Limitations** (đã trung thực): thêm "and at the sizes of our datasets its probability bound is not informative".

---

## 6. Va chạm ký hiệu (COSMETIC, c1)

| Ký hiệu | Nghĩa 1 | Nghĩa 2 (và 3) | Đề xuất |
|---|---|---|---|
| N_k | `lem:sweep`: tổng nhiễu Σh + Z_k + Σ(o−R_s) | `prop:moments` (cùng tiểu mục): số lần rút đúng trong khối k; còn N = số điểm có nhãn (`prop:predbias`), N ~ Bin (`prop:naive`) | Lemma: 𝒩_k hoặc G_k; moments: A_k |
| x | mức độ chính xác trong Thm 3 | đặc trưng x_i, x_* | mức: θ hoặc ℓ₀ |
| δ | dung sai của (H_δ) | hệ số Dobrushin δ(P^t) trong Thm 2 (cùng thân bài) | hệ số Dobrushin: δ_D hoặc κ(·) |
| τ | thời điểm thoát | τ(P) thời gian trộn tích phân (Thm 2) | thời điểm thoát: T_x |
| ν | biên | độ đo có dấu ν = π̃(P̃ − P) trong chứng minh Thm 2 | độ đo: η |
| λ | biên | λ = N/n (`prop:predbias`); λ_k hệ số góc (`prop:naive`) | tỉ lệ có nhãn: ϱ_L |
| a | 1 − ⌈n/K⌉/n_F | hệ số drift (`prop:predbias`); E[1/(1+N)] (`prop:naive`) | drift: κ |
| c | hằng số của p(ν, λ) | chỉ số lớp c; c_i(y, B); c₀ | hằng số: c_H |
| p | p(ν, λ) | p(·) mật độ/xác suất khắp bài; p_c = L_c/N | p(ν, λ) → ϖ(ν, λ) hoặc ε_sw |
| b | mức đích trong Thm 3(c) | hệ số góc trong `prop:moments`; b_k (`prop:naive`) | mức đích: x₁ |
| q | q_j mức trong chứng minh (c) | q₀, q₁, q₂ hệ số phương sai; q xác suất distractor PLL | hệ số phương sai: v₀, v₁, v₂ |
| H | số vòng quét | tên giả thiết (H_δ) | số vòng: T′ hoặc h |
| U, u | số điểm tự do / trong khối (`prop:predbias`) | U_k tập đã cập nhật (lemma); u điểm cắt (Remark ii); n_F cũng là số điểm tự do | dùng n_F thống nhất |
| m, M | ⌈n/K⌉ (lemma) | M = n/K (`prop:predbias`); m cỡ khối trong `prop:blocks`(b) | một ký hiệu |
| g | gain g(r) | chỉ số túi g (B_g, w_g); g_k vectơ (`prop:moments`) | gain: γ(r) (và đổi γ của (c)) |
| S | S = nR_s | số mẫu S; tập ứng viên S_i; luật tính điểm S | số đúng: A |
| α | Dirichlet α | α₀, α₁ hệ số affine | hệ số affine: β₀, β₁ |

---

## 7. Thứ tự sửa ưu tiên

1. M1: sửa câu `experiments.tex` dòng 71 (và "which the response map explains" trong abstract).
2. M3: viết lại câu `theory.tex` dòng 45–48.
3. M2: chỉnh abstract, intro, Contributions, đoạn "Read with the gain".
4. M4 + m1: phát biểu lại Assumption 1 (miền của B, tập trạng thái), định nghĩa response map một lần, đổi "We checked Assumption 1".
5. m3, m4, m5, m7, m8, m9, m10: các chỉnh sửa một dòng trong phụ lục.
6. c1: đổi tên ký hiệu trong `app:dynamics` (ít nhất N_k, b, q).
