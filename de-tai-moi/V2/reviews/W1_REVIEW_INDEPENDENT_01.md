# W1 — Rà độc lập §1–§2 của `W1_R1A_REPORT.md` (Mệnh đề 4, Bổ đề 1–3, Mệnh đề 4′)

05/11/2026. Người rà không đọc bất kỳ tệp nào trong `V2/reviews/`. Khung: skill dm-paper-math-review (thang L0–L5, mức BLOCKING / MAJOR / MINOR / COSMETIC).

## Kết luận một dòng

**BLOCKING**: gạch đầu dòng thứ nhất của Hệ quả (c), "chuỗi vào [r* − η, r* + η] trong T₀ vòng quét", **sai như đã viết** khi 1/(ρK) > 2η. Có phản ví dụ thoả mọi giả thiết: K = 2, L = 4, xác suất mệnh đề hứa ≥ 0,99, xác suất thực bằng 0. Sửa lỗi này rẻ.

Ngoài ra có 2 MAJOR:
- (H_δ) lượng từ trên mọi khối, nên δ của TFM giả ≈ 0,05 chứ không phải ≤ 0,001.
- §2.6 bỏ qua `max_context = 2000` của cấu hình chạy thật.

Phần lõi đứng vững: Bổ đề 1, Bổ đề 2, Mệnh đề 4(a)(b), Bổ đề 3 và Mệnh đề 4′ (a′)–(d′).

## Phạm vi

- **Đã đọc kỹ:** §1, §2.1–§2.6, kể cả bảng "Kiểm hằng số" của §2.6, và §0 ở phần nói về §2.
- **Đọc lướt:** §3–§4, chỉ để kiểm các khẳng định của §2 có trỏ tới đó.
- **Mã đã đối chiếu:**
  - `G4/code/icgibbs/sampler.py`: `_ctx` (dòng 74–80), `run` (118–168, `update_params` ở dòng 151), `_sweep_instance` (170–212);
  - `weak.py`, `conditional.py`;
  - `G4/code/experiments/w1_drift_check.py`: TFM giả và `partition_bound`;
  - `G4/code/icgibbs/g7/methods.py:17`: cấu hình GIBBS.
- **Không kiểm:** các tài liệu được trích ngoài Hoeffding (1963) và Azuma, kiểm theo phát biểu chuẩn; số liệu §4–§5.
- **Kiểm bằng máy** (là bằng chứng, không phải chứng minh): 4 script trong `/tmp/claude-0/-home-claude/643477cb-9e2c-5e3b-9dd1-1dba7a8f3cd0/scratchpad/review/`, xem cuối tệp.

---

## Sổ lỗi

### [BLOCKING-1] Hệ quả (c), gạch 1: lần chạm biên gần có thể nhảy qua cả cửa sổ. Vị trí: §2.4, Hệ quả (c), phát biểu và dòng đầu của chứng minh

**Phát biểu như đã viết.** Từ mọi r₀, với xác suất ≥ 1 − e^{−ρn_Fλ²/(c_wT₀)} − 2(HK+1)e^{−θ*λ} − 2HK·e^{−Kρn_Fλ′²/c_w} − ε, "chuỗi vào [r* − η, r* + η] trong T₀ vòng quét".

**Vì sao sai.**
- Chứng minh chỉ áp (a) với b = r* − η. (a) cho ∃t ≤ T₀K: R_t ≥ r* − η, tức chuỗi *chạm hoặc vượt* biên gần. (a) không cho R_t ≤ r* + η.
- Bước cắt có thể nhảy qua r* + η: bước nhảy chỉ bị chặn bởi 1/(ρK), và g có thể dốc tới L + 1.
- Không giả thiết nào của (c) loại trừ điều này.

**Bằng chứng** (`c1_overshoot.py`).
- Thiết lập:
  - lịch quét khối ngẫu nhiên, K = 2, ρ = 1, n = n_F = 10⁶;
  - h_i = 0 và δ = 0, với c_i = Φ(r_{−B}) đúng, nên (H_δ) đúng;
  - Φ(r) = clip(0,5 − 4(r − 0,5); 0,1; 0,9), tức L = 4, r* = 0,5, η = 0,05;
  - g > 0 trên [0; 0,5) và g < 0 trên (0,5; 1].
- Hằng số: γ_η = 0,25; δ′ = δ_ctx = 0,0112 (N = 12, ε = 0,01); T₀ = 5. Với λ = λ′ = 0,01 và H = 1, chặn của (c) cho **≥ 0,990**.
- Động học trung bình là R ← (R + Φ(R))/2: 0,3 → 0,6 → 0,35 → 0,625 → … Chuỗi hội tụ về chu trình 2 điểm {0,367; 0,633} và không bao giờ vào [0,45; 0,55].
- Mô phỏng đếm chính xác (hypergeometric + binomial): từ r₀ ∈ {0,30; 0,35; 0,65}, **0/2000** lần chạy vào cửa sổ trong T₀ vòng quét.
- Gạch 2 (dải có β ∋ ḡ/K ≈ 0,43) không bị phản ví dụ này bác.

**Tầm ảnh hưởng.**
- Câu "(c)" ở §0.
- Chữ "vào dải quanh r₂" của (d), nếu hiểu là [r₂ − η, r₂ + η].
- Không ảnh hưởng số liệu §3: ở đó K = 10, ρ ≈ 1, η = 0,1, nên 1/(ρK) ≤ 2η, và gạch 1 đúng nhờ chặn bước nhảy hầu chắc chắn.

**Sửa tối thiểu.** Chọn một trong hai:
- thêm giả thiết 1/(ρK) ≤ 2η. Khi đó R_τ < r* − η + 1/(ρK) ≤ r* + η hầu chắc chắn;
- đổi gạch 1 thành "chạm r* − η (đi từ dưới lên), và tại lúc đó R ≤ r* + η + β".

---

### [MAJOR-1] (H_δ) lượng từ ∀y ∀B, nên δ "≤ 0,001" của TFM giả sai. Vị trí: §2.1, mục "Đọc giả định", gạch cuối; liên quan §0 mục 3 và §2.6 "Kiểm hằng số"

**Phát biểu như đã viết.**
- "Trong TFM giả ở §3: δ ≤ 0,001 (chỉ do clip), h_max = 0,1."
- "Trung bình h_i trên một khối dao động. Dao động đó đi vào nhiễu (ζ_2), không đi vào δ."
- "Giả định áp cho mọi khối."

**Vì sao sai.**
1. (H_δ) phải đúng cho **mọi** y và **mọi** B. Vì vậy mọi sai lệch theo điểm có phụ thuộc trạng thái, mà h_i cố định không hấp thụ được, bị tính trên **khối xấu nhất**, không được lấy trung bình.
   - Clip trong TFM giả đúng là loại sai lệch này: c_i − h_i − Φ(r) = −(Φ(r) + h_i − 0,99)₊ phụ thuộc cả r lẫn i.
   - Khối xấu nhất là m điểm tự do có h_i lớn nhất.
2. Khi có khối với f = |B∩F| nhỏ, tức n − n_F ≥ m − 1, xấp xỉ 1 − ρ ≥ 1/K, (H_δ) thực chất **theo từng điểm**. Ví dụ SSL hoặc PU với ≥ 10% điểm cố định khi K = 10. Cách đọc "δ là phần còn lại sau trung bình" khi đó không đúng.

**Bằng chứng** (`c2_mock_delta.py`, cùng `problem()` seed 0 như `w1_drift_check.py`).

| Ánh xạ | sup_r δ⁻ khối xấu nhất (n = 500 / 3.000) | δ⁻ khối xấu nhất, r trong ±0,05 quanh điểm hút trên | q99 với khối ngẫu nhiên, sup_r |
|---|---|---|---|
| 0,9 | 0,049 / 0,050 | 0,019 / 0,020 | 0,014 / 0,011 |
| 0,6 | 0 / 0 (không clip, Φ ≤ 0,84) | 0 | 0 |
| hai điểm hút | 0,051 / 0,052 | 0,051 / 0,052 | 0,015 / 0,012 |

- Ngay cả với khối ngẫu nhiên, δ vẫn lên tới 0,011–0,015 khi r ≈ 1. Lý do: sai số clip trung bình tổng thể phụ thuộc trạng thái (≈ 0,009 tại r = 1 với ánh xạ 0,9) và không nằm trong Φ.
- Chỉ phía (H⁻) bị ảnh hưởng: clip dưới không bao giờ kích hoạt, nên δ⁺ = 0.

**Tầm ảnh hưởng.** Các định lý không sai. Thứ sai là các khẳng định "giả thiết thoả" dùng δ = 0:
- **§3, ánh xạ 0,9, n = 3.000:** δ′ = 0,008, trong khi δ⁻ khối xấu nhất quanh điểm hút đã là 0,02.
- **§3, hai điểm hút, n = 3.000:** câu "các điều kiện của (d) thoả … min|g| ≈ 0,028–0,035 > δ′ = 0,021" không đứng được. Với (H) toàn cục, δ′ ≈ 0,07. Kể cả bản cục bộ hoá, xét r_{−B} ∈ [0,55; 0,90], δ⁻ vẫn là 0,035, đã ≥ γ.
- **§2.6, bảng "Kiểm hằng số":** đặt δ = 0. Với ánh xạ 0,9, sàn lùi thêm δ/(1 − b) ≈ 0,04.
- **§4 và §0 mục 4:** "phần dư dọc vết" lấy trung bình trên toàn F mỗi vòng quét. Nó chỉ là chặn dưới của δ theo ∀B, không phải ước lượng của nó. Vậy "(H⁻) đúng với δ⁻ ≈ 0,02" chưa kiểm (H⁻_δ) như đã phát biểu.

**Sửa tối thiểu.**
- (A) Giữ định lý và báo δ trung thực (sup theo y, B). Hoặc bỏ clip khỏi TFM giả, tức chọn Φ và h_max sao cho Φ + h ∈ [0,01; 0,99]; khi đó δ = 0 thật.
- (B) Tốt hơn: làm yếu (H_δ) về đúng mức chứng minh cần.
  1. **Cục bộ theo trạng thái.** Chứng minh của (a), (b) và Bổ đề 3 chỉ dùng (H) tại các trạng thái đi qua trước thời điểm dừng. Mgf của ζ không cần (H).
  2. **"Khối điển hình".** Với mọi y, P_B(|dev(y, B)| > δ) ≤ ε_B, theo đúng luật mà lịch quét dùng cho B.
     - Bổ đề 1: cộng ε_B vào P(G_t^c | 𝓕_t). Điều này chính xác vì B_t ⊥ 𝓕_t.
     - Bổ đề 3: luật của B_k biết 𝓖_{s,k−1} là đều trên các tập con của những điểm *chưa dùng*. Phát biểu "điển hình" phải theo luật có điều kiện này; cần viết thêm một đoạn, rồi cộng Kε_B mỗi vòng.

---

### [MAJOR-2] §2.6 không mô hình hoá ngữ cảnh lấy mẫu con của cấu hình chạy thật. Vị trí: §2.6 "Lịch quét"; §1 "c_i(y,B) … chỉ phụ thuộc y_{−B}"

**Phát biểu như đã viết.** "với ngữ cảnh là trạng thái hiện tại Y_{s,k−1} trên [n]∖B_k". Mục 2.6 được trình bày là mô hình của bộ lấy mẫu thật.

**Vì sao chưa đúng.**
- `sampler._ctx` lấy mẫu con không hoàn lại `max_context` điểm từ [n]∖B khi |[n]∖B| > max_context.
- Cấu hình chạy thật đặt `max_context = 2000` (`g7/methods.py:17`). Chuỗi nhiều điểm xuất phát của V0 "giống hệt gibbs_warm (K, số vòng quét, burn-in, ngữ cảnh, seed)".
- Khi n − ⌈n/K⌉ > 2000, tức n ≳ 2.223 với K = 10, có ba hệ quả:
  - (i) c_i phụ thuộc mẫu con ngẫu nhiên C_k, chứ không chỉ y_{−B};
  - (ii) mọi điểm của B_k dùng chung C_k, nên ξ_i chỉ độc lập *có điều kiện theo C_k*. Định nghĩa c_i phải điều kiện theo C_k, nếu không lập luận ζ_1 / Z_k không còn đúng;
  - (iii) TFM thấy r_{C_k}. Đại lượng này lệch r_{−B_k} một dao động hypergeometric cỡ √(r(1−r)/(2000ρ)), **không giảm theo n_F**.
- Hậu quả: Bổ đề 3 cần thêm mảnh nhiễu thứ tư, với số mũ ∝ max_context·ε² thay cho ρn_Fε². Mọi khẳng định "mũ theo n_F" bão hoà ở cỡ max_context. Đó là thời gian thoát của (d′) và "Dải giới hạn … n_F → ∞ … co về G_δ". Lịch quét khối ngẫu nhiên có vấn đề tương tự, với δ_ctx theo cỡ mẫu 2000.

**Bằng chứng.** Đọc mã: `sampler.py` dòng 74–80 và `g7/methods.py` dòng 17, 156. TFM giả ở §3 chạy với `max_context = None`, nên không kiểm được điểm này.

**Tầm ảnh hưởng.**
- Khẳng định §2.6 mô hình hoá trung thực lịch quét thật.
- (d′) "thời gian thoát tăng theo hàm mũ của n_F", và các nhận xét tiệm cận.
- Mọi dataset của V0/V2 có n_train vượt ngưỡng. Người rà không kiểm dataset nào vượt.

**Sửa tối thiểu.** Chọn một trong hai:
- phát biểu cho `max_context = None`, hoặc thêm điều kiện n − ⌈n/K⌉ ≤ max_context;
- đưa C_k vào mô hình: (H) điều kiện theo C_k, với Φ tính tại r_{C_k}; thêm mảnh |r_{C_k} − r_{−B_k}| (hypergeometric) vào E; thay n_F bằng min(n_F, max_context) ở số mũ.

---

### [MINOR-1] Hệ quả (c): dùng tính Markov mạnh tại τ − 1 và đếm xác suất thiếu. Vị trí: §2.4, chứng minh của (c), gạch 2

**Phát biểu như đã viết.** "dùng (b) … kể từ bước ngay trước lần chạm, nhờ tính Markov mạnh". Các thừa số đi kèm là 2(HK+1) và 2HK.

**Vì sao sai.**
- τ − 1 **không** là thời điểm dừng: biết τ − 1 = t cần biết bước t → t+1.
- Hơn nữa, ζ⁺_{τ−1} chính là nhiễu của bước cắt, nên bị lệch theo kích thước. Không thể chặn P(ζ⁺_{τ−1} ≥ λ′) bằng e^{−λ′²/(2s²)} qua điều kiện hoá.
- Lập luận của (b) không cần "khởi động lại", vì G và E′ là biến cố toàn cục. Nhưng khi đó E′ phải phủ cả pha đi tới, [0, (T₀+H)K).
- Số lần vượt có thể giữ HK + 1 nếu đánh chỉ số theo thời điểm bắt đầu t₀ ∈ [τ, τ + HK], vì {τ ≤ t₀} ∈ 𝓕_{t₀}.

**Sửa.**
- Thay 2HK·e^{−Kρn_Fλ′²/c_w} bằng (2H + T₀)K·e^{−Kρn_Fλ′²/c_w}.
- Bỏ câu "Markov mạnh".
- Sửa lệch một ở (HK+1).

### [MINOR-2] Bản tiệm cận của (c): một λ gánh hai vai. Vị trí: §2.4, "Bản tiệm cận"

- Với λ ~ log(HKn_F)/(ρn_F), số hạng đi tới e^{−ρn_Fλ²/(c_wT₀)} = e^{−O(log²/n_F)} → 1. Chặn của (c) khi đó ≤ 0, tức rỗng.
- Cần tách hai tham số: λ_a cho pha đi tới, hằng hoặc ≫ n_F^{−1/2}, dùng trong T₀; và λ_b cho dải β.
- Thêm hai điểm:
  - Dải là đối xứng r* ± η, nên giới hạn là khoảng đối xứng nhỏ nhất quanh r* chứa G_δ, không phải "G_δ". Nếu G_δ có thành phần xa r*, giới hạn là bao lồi.
  - "ḡ = O(1/K) nhờ chặn bước nhảy xác suất cao" mới được khẳng định, chưa chứng minh. Mức L1.

### [MINOR-3] Hệ quả (d), thời gian thoát: bỏ qua việc δ′ phụ thuộc N. Vị trí: §2.4, (d), "Thời gian thoát (phác thảo)"

- Muốn thời gian thoát ≥ e^{c′n_F}/K thì N = HK ≈ e^{c′n_F}. Khi đó δ_ctx → L√(c′/(2K))/(ρ^{3/2}(1 − 1/(ρK))), một hằng số, không → 0.
- Chặn cũng thiếu số hạng +ε, và ε không thể nhỏ theo hàm mũ mà không làm δ_ctx bằng Θ(1).
- Kết luận chỉ đứng được nếu c′ đủ nhỏ, cỡ c′ ≲ 2Kρ³(1 − 1/(ρK))²(γ − δ)²/L², và điều này phải ghi ra.
- (d′) chỉ thay phần này cho **lịch quét thật**. Với lịch quét khối ngẫu nhiên, (d) vẫn ở mức L2. Vì vậy câu "Chứng minh đầy đủ cho lịch quét khối ngẫu nhiên" ở §0 nói quá.

### [MINOR-4] "Σh_i = 0 không làm mất tổng quát" có cái giá. Vị trí: §2.1, "Đọc giả định"; §1, "Φ: [0,1] → [0,1]"

- Định tâm h_i = h_i⁰ − h̄ cho |h_i| ≤ h_max + |h̄|, tức tới 2h_max. Đồng thời Φ + h̄ có thể ra khỏi [0, 1], trái với §1.
- Các hằng số w, c_w, s² và c phải dùng h_max sau khi định tâm.
- Bổ đề 1 bước 6 (dùng "Φ ≤ 1") vẫn đúng với h_max sau định tâm. Người rà đã kiểm: đầu dưới ≤ −h_max⁰ − δ ≤ 0. Nhưng phải nói rõ.
- Về số, không đáng kể ở §3 (|h̄| ≤ 0,003). Mô phỏng `c3` vấp đúng chỗ này: định tâm đẩy |h| vượt h_max.

### [MINOR-5] Cỡ khối ⌈n/K⌉ và hằng số c. Vị trí: §2.6, "Lịch quét", "Hằng số", Bổ đề 3 bước 5

- Chứng minh cần a = (n_F − m)/n_F với m = ⌈n/K⌉. Bài viết lại dùng 1 − 1/(ρK) = (n_F − n/K)/n_F, lớn hơn a khi K ∤ n. Vậy c bị khai **lớn hơn** một chút.
- Cũng cần n_F > ⌈n/K⌉, mà ρK > 1 không kéo theo điều này. Ví dụ n = 11, K = 2, n_F = 6: khi đó F∖B có thể rỗng.
- Ở §1, m = n/K ngầm giả sử K | n.
- Sửa: thay bằng a = (n_F − ⌈n/K⌉)/n_F > 0.

### [MINOR-6] Phạm vi: tham số phiền nhiễu, tính tất định của TFM, dòng mở rộng. Vị trí: §1, "Phạm vi" và "c_i chỉ phụ thuộc y_{−B}"

- `run` gọi `weak.update_params` sau mỗi vòng quét (`sampler.py` dòng 151).
  - Với PU-SCAR học c và PU-SAR học η, ℓ_i đổi giữa các vòng, nên c_i đổi theo, và trạng thái thật là (y, θ). (H_δ) phải đều theo θ. Khi đó Bổ đề 3 vẫn đúng, vì θ_s ∈ 𝓕_s.
  - Phạm vi "PU" ở §1 nên ghi rõ là c hoặc η đã biết.
- Giả định ẩn: TFM tất định theo (ngữ cảnh, truy vấn). Nếu không, (H) phải đúng cho từng hiện thực, và ζ_1 phải điều kiện theo nguồn ngẫu nhiên nội tại của TFM.
- Các dòng `extend` (`n_total`) cũng là điểm tự do và nằm trong F. Nên ghi ra.

### [MINOR-7] Nhất quán §0 / §2.5 / §2.6 / §3. Vị trí: §0 mục 1 và 3; §2.5(2); §2.6 "So với Mệnh đề 4(c)"; §3

- **Mâu thuẫn trạng thái Bổ đề 3.**
  - §0 viết: "Lịch quét thật … mới có phác thảo (Bổ đề 3)" và "Còn mở: Bổ đề 3 đầy đủ, hằng số của thời gian thoát".
  - Trong khi §2.5(2) viết "Đã chứng minh đầy đủ ở §2.6", và (d′) đã có hằng số tường minh.
  - §3 vẫn viết "kiểm cả phác thảo Bổ đề 3".
- **So sai đối tượng ở §0 mục 3 và §3.** Hai mục này so mô phỏng **lịch quét thật** với dải "G_{δ′} + β" của Mệnh đề 4 (lịch quét khối ngẫu nhiên), mà không tính β. Vì thế "đúng như" không bác bỏ được. Đối tượng đúng để so là 4′(c′).
- **"Chặt hơn Mệnh đề 4" ở §2.6.**
  - 4′ chỉ kiểm soát R_s ở ranh giới vòng quét, còn 4 kiểm soát mọi bước.
  - Lε trong điều kiện sàn đóng đúng vai của δ_ctx, nên "không cần δ_ctx" chỉ là đổi tên.
  - "Phần định tính đã chặt ở mọi n" sai ở n = 500 với ánh xạ 0,6, nơi dải là [0; 1], tức rỗng.

### [MINOR-8] Bảng "Kiểm hằng số" của §2.6 dùng kế toán của (b′), không phải (c′). Vị trí: §2.6, cuối mục; `w1_drift_check.py::partition_bound`

- p = 0,05/(2H) bỏ số hạng 2⌈1/γ′⌉p của (c′).
- Tại sàn lớn nhất x̲ = r* − (bε + λ)/(1 − b), ψ(x) − x → 0 khi x ↑ x̲. Vậy không có γ′ > 0 cho phần "đi vào" của (c′).
- Dải trong bảng là **dải bất biến** (b′): đúng khi chuỗi đã ở trong dải. Nó không phải dải "từ mọi r₀".
- Docstring ghi "(c′)".
- Sửa: lấy x̲ thấp hơn một chút để có γ′ > 0, rồi cộng thêm 2⌈1/γ′⌉p.

### [COSMETIC-1] Trùng ký hiệu

- ρ_s (dãy so sánh) với ρ = n_F/n.
- E (biến cố) với E[·].
- N_k với chân trời N.
- H (chân trời) với (H_δ).
- u (đầu mút trên của đoạn ở (b)) với u (điểm không ổn định ở (d)).
- x̄ ở (b) (chặn điểm bắt đầu) với x̄ ở (c′) (trần).
- c, c_i, c_w, c′.
- ε: xác suất thất bại ở §2.2–2.4, nhưng là lề ngữ cảnh ở §2.6.

### [COSMETIC-2] Định nghĩa vòng tròn ở (c) và (a)

T₀ phụ thuộc δ′, δ′ phụ thuộc N, và N = (T₀ + H)K. Nên cố định N trước rồi kiểm điều kiện.

### [COSMETIC-3] c′ = min(c, 2) ở (d′)

Luôn bằng c, vì c ≤ 2/9. Còn e^{−2n_Fλ²} ≤ e^{−cρn_Fλ²} chỉ cần ρ ≤ 1.

### [COSMETIC-4] (d′) "≥ 1/p" thiếu một dòng

Cần viết: P(τ − s₀ > H) ≥ (1 − p)^H, nên E[τ − s₀] ≥ Σ_H (1 − p)^H = 1/p. Dùng Hp thì chỉ được cỡ 1/(2p).

### [COSMETIC-5] Nhận xét 4

- ρ² vô nghĩa khi đã giả sử ρ = 1.
- Tính lại: E S_B² = m·R(1−R)·(n−m)/(n−1), nên độ lệch mỗi vòng ≈ ‖Φ″‖/(8(K−1)n), không phải ‖Φ″‖/(8Kn). Nhận xét ở mức L1 và dùng "≲", nên chỉ là hằng số.

### [COSMETIC-6] Phát biểu theo ranh giới vòng quét

- (b′), (c′), (d′) và Bổ đề 3(i) chỉ nói về R_s ở ranh giới vòng quét. Nên ghi rõ.
- Bổ đề 3(i) dùng trong quy nạp của (a′) nên viết là "P(·|𝓕_s) ≥ 1 − p trên biến cố {x ≤ R_s}".
- "Dải giới hạn … co về G_δ quanh r*" nên là bao lồi của G_δ.

---

## Phán quyết từng kết quả

| Kết quả | Mức | Phán quyết | Ghi chú |
|---|---|---|---|
| (H_δ), §2.1 | — | Đặt vấn đề đúng | Hai vế chỉ phụ thuộc (y_{−B}, B). WLOG tốn hằng số (MINOR-4). Lượng từ ∀B mạnh hơn cách đọc và cách kiểm (MAJOR-1). |
| Bổ đề 1 | L5 | **PROVEN** | Đã thử: đoạn chứa 0 cho b_i và a_i ở hai phía, kể cả Φ sát 0/1 và δ > h_max; điều kiện Hoeffding không hoàn lại; ghép mgf bằng kỳ vọng lặp; tính lại δ_ctx (hai dạng bằng nhau tới 10⁻⁶); mô phỏng 2·10⁵ bước có 0 vi phạm; mgf thực nghiệm < chặn. |
| Bổ đề 2 | L4 | **PROVEN** | Ville cho siêu martingale mũ; bản dừng đúng vì 1[u<σ] ∈ 𝓕_u. |
| Mệnh đề 4(a) | L4 | **PROVEN** | Phân trường hợp đủ. Dùng bao hàm biến cố {không chạm} ∩ G ⊂ {min M ≤ −κ}, với M định nghĩa trên toàn không gian và **không** điều kiện theo G, là hợp lệ. |
| Mệnh đề 4(b) | L4 | **PROVEN** | θ* = 2(γ−δ′)/(Ks²) = 4ρn_F(γ−δ′)/c_w đúng. Lần vượt đánh chỉ số theo thời điểm bắt đầu tất định, nên có ≤ HK+1 siêu martingale, mỗi cái qua Ville. Cửa sổ ḡ_ℓ và chặn 1/(ρK) đúng. Bao hàm "mọi t có R_t ≥ x̄+λ thuộc một lần vượt chạm x+λ trước σ" đúng. |
| Hệ quả (c), gạch 1 | — | **FALSE** | Phản ví dụ ở BLOCKING-1. Đúng khi thêm 1/(ρK) ≤ 2η. |
| Hệ quả (c), gạch 2 | L2 | **INCOMPLETE** | Cấu trúc đúng; kế toán xác suất sai (MINOR-1); bản tiệm cận rỗng với một λ (MINOR-2). |
| Hệ quả (d), lưu vực | L2 | **INCOMPLETE** | "Cùng dạng như (c)", nên thừa hưởng MINOR-1. Phần κ = min(λ, η) và "phía trên u + η" đúng. |
| Hệ quả (d), thời gian thoát | L2 | **INCOMPLETE** | δ_ctx(N), +ε (MINOR-3). |
| Nhận xét 1, 4 | L1 | — | Chỉ là nhận xét; hằng số ở COSMETIC-5. |
| Bổ đề 3 | L5 | **PROVEN UNDER ADDITIONAL ASSUMPTIONS** | Cần thêm: n_F > ⌈n/K⌉ và a = (n_F − ⌈n/K⌉)/n_F trong c (MINOR-5); mô hình không có max_context (MAJOR-2); θ cố định (MINOR-6). Đã thử: đẳng thức (n_F − f_k)r_k, quy nạp với Φ̲ bị kẹp, bước cuối, ba mảnh, Azuma với rút đồng thời trong khối; mô phỏng ~9.000 vòng quét có 0 vi phạm. |
| Mệnh đề 4′ (a′) | L4 | **PROVEN** | Cùng các lưu ý của Bổ đề 3. Quy nạp so sánh và Jp đúng. |
| Mệnh đề 4′ (b′) | L4 | **PROVEN** | — |
| Mệnh đề 4′ (c′) | L4 | **PROVEN** | 2(⌈1/γ′⌉ + H)p đúng: gồm Jp, (J+H)p và Hp. Không mắc lỗi vượt cửa sổ, vì trần bất biến ngay từ s = 0. "Dải giới hạn" ở mức L2 (COSMETIC-6). |
| Mệnh đề 4′ (d′) | L3 | **PROVEN** | 1/p chưa viết ra (COSMETIC-4); hằng số (1 + 3K) đúng. "Mũ theo n_F" chỉ đúng khi không có max_context (MAJOR-2). |
| Bảng "Kiểm hằng số", §2.6 | — | Số tái lập được | ε, λ và các dải khớp tới 3 chữ số. Nhưng bảng dùng kế toán (b′) (MINOR-8) và δ = 0 (MAJOR-1). |

## Bảng giả định

| Giả định | Nêu ở đâu | Dùng ở đâu | Cần không? | Mất nó thì sao |
|---|---|---|---|---|
| (H⁻_δ) / (H⁺_δ) với ∀y, ∀B | §2.1 | Bổ đề 1 bước 2; Bổ đề 3 bước 3–4 | Chỉ cần tại trạng thái đi qua, và chỉ cần cho các khối có xác suất dương | Drift không chặn được |
| Σ_F h_i = 0 | §2.1 | Bổ đề 1 bước 4; Bổ đề 3 bước 4 và mảnh h | Có (hoặc gộp vào Φ, tốn hằng số) | Drift lệch h̄ |
| Φ L-Lipschitz | §2.1 | Chỉ Bổ đề 1 bước 2 | Không cần cho Bổ đề 3 | δ_ctx không định nghĩa được |
| Φ(·) ∈ [0, 1] | §1 | Bổ đề 1 bước 6 | Có thể nới, đổi w | w đổi |
| ρK > 1 (thật ra n_F > ⌈n/K⌉) | §1 | r_{−B} xác định; n_F − m > 0 | Có (**chưa nêu đúng**) | Chia cho 0 |
| B_t ⊥ 𝓕_t, đều | §1 | Hoeffding cho S_B và ζ_2 | Có | — |
| Rút độc lập trong khối, có điều kiện | §1 | ζ_1, Z_k | Có | **Vỡ nếu max_context dùng chung C_k mà không điều kiện theo C_k** |
| TFM tất định, θ cố định | **Chưa nêu** | "c_i chỉ phụ thuộc y_{−B}" | Có, hoặc điều kiện hoá | (H) không đặt được |
| 1/(ρK) ≤ 2η | **Chưa nêu** | (c), gạch 1 | Có | BLOCKING-1 |

## Đã tấn công nhưng không phá được

- **Bổ đề 1.**
  - Tính lại mọi bất đẳng thức.
  - r_{−B} − R_t = S_B/(n_F − f): đúng.
  - Kiểm hai phía (H⁻) và (H⁺) ở các trường hợp biên: Φ = 0, Φ = 1, δ > h_max. Đều có độ dài đoạn ≤ w.
  - Proxy s²/2 = m(1+w²)/(8n_F²): đúng.
  - P(G_t^c|𝓕_t) = 2e^{−ln(2N/ε)} = ε/N: đúng.
  - Mô phỏng `c4_lemma1.py` (ρ = 0,7, độ lệch khối đối nghịch): 0 vi phạm theo đường đi; Ê e^{θζ} < e^{θ²s²/2} với θ ∈ {±300, 50, 150, 600}.
- **Bổ đề 2 và Mệnh đề 4(a)(b).**
  - Hợp lệ khi trội theo đường đi trên G và dùng siêu martingale của tổng trội định nghĩa trên toàn không gian.
  - Ville áp từ thời điểm bắt đầu ngẫu nhiên, qua đánh chỉ số bằng thời điểm tất định.
  - Kiểm đại số của θ*.
  - Kiểm cửa sổ ḡ khi R_{t₀−1} sát 0.
- **Bổ đề 3.**
  - U_k và V_k là M_k phần tử đầu / n − M_{k+1} phần tử cuối của một hoán vị đều. Cỡ của chúng tất định (array_split) và chúng độc lập với 𝓕_s.
  - Tổng thể h·1_F và (o − R_s)·1_F có tổng 0 và biên độ 2h_max / 1.
  - Z_k: rút đồng thời trong khối không phá Azuma, vì độc lập có điều kiện cho phép lộ dần từng điểm, và c_i khả đoán theo 𝓖_{s,j−1}.
  - Tính lại c: (2/9)a², a²/(18h_max²), và (2/9)a²n_F ≥ (2/9)a²ρn_F.
  - Ca cơ sở k = 1; Φ̲ bị kẹp khi x − ε < 0.
  - Mô phỏng `c3_lemma3.py`: lịch quét hoán vị và array_split thật, Φ không đơn điệu, ρ = 0,8, độ lệch khối ∈ [−δ, 0]. Kết quả: 0 vi phạm đẳng thức, quy nạp hay bước cuối trên 8.935 vòng quét. p(ε, λ) rỗng ở n_F = 1.600, đúng như dự đoán.
- **Mệnh đề 4′.**
  - Kế toán Jp + (J+H)p + Hp.
  - (1 + 3K) và c′.
  - Thời gian thoát qua (1 − p)^H.
- **Đối chiếu mã.**
  - `sampler.py` dòng 182–183: hoán vị trên [n_total], array_split thành min(K, n) khối.
  - Ngữ cảnh loại cả khối, kể cả điểm cố định trong khối.
  - Các uniform trong khối độc lập.
  - `mh_correct` mặc định False trong cấu hình GIBBS.
  - TFM giả thật sự dùng r trên phần tự do của ngữ cảnh, khớp r_{−B}.

## Thứ tự sửa ưu tiên

1. **BLOCKING-1.** Thêm 1/(ρK) ≤ 2η hoặc làm yếu gạch 1 của (c). Rà lại chữ "vào dải" của (d) và câu tương ứng ở §0.
2. **MAJOR-1.**
   - Sửa câu "δ ≤ 0,001" của TFM giả: tính δ trung thực, hoặc bỏ clip khỏi TFM giả.
   - Viết (H_δ) bản cục bộ hoặc bản "khối điển hình".
   - Kiểm lại "các điều kiện của (d) thoả" ở §3.
   - Ghi rõ §4 chỉ đo một đại lượng trung bình.
3. **MAJOR-2.** Giới hạn §2.6 vào max_context = None (hoặc n − ⌈n/K⌉ ≤ max_context), hoặc đưa C_k vào mô hình. Sửa các câu "mũ theo n_F".
4. **MINOR-1 đến 3.** Kế toán của (c), tách λ ở bản tiệm cận, δ_ctx(N) ở (d).
5. **MINOR-4, 5, 8.** Hằng số: h_max sau định tâm; a = (n_F − ⌈n/K⌉)/n_F; bảng §2.6 theo (c′).
6. **MINOR-6, 7, COSMETIC.** Phạm vi; nhất quán giữa §0, §2.5 và §3; ký hiệu.

## Kiểm bằng máy

Thư mục `/tmp/claude-0/-home-claude/643477cb-9e2c-5e3b-9dd1-1dba7a8f3cd0/scratchpad/review/`:
- `c1_overshoot.py`: phản ví dụ cho (c), gạch 1. Lịch quét khối ngẫu nhiên, đếm chính xác, n = 10⁶, 2.000 lần chạy cho mỗi r₀.
- `c2_mock_delta.py`: δ của TFM giả §3 dưới ∀B so với khối ngẫu nhiên, n ∈ {500, 3.000}, 3 ánh xạ, lưới r 101 điểm.
- `c3_lemma3.py`: kiểm theo đường đi các bước tất định của Bổ đề 3 (i) trên lịch quét thật; n = 2.000, K = 10, 3 cặp (ε, λ).
- `c4_lemma1.py`: Bổ đề 1 trên lịch quét khối ngẫu nhiên; n = 1.000, K = 5, ρ = 0,7, 2·10⁵ bước tại một trạng thái cố định.

Phủ: chỉ là các ví dụ hữu hạn nói trên. Chúng là bằng chứng, không thay chứng minh.

---

## Phản hồi của tác giả (em, 05/11/2026 chiều)

Mọi mục đã sửa trong `V2/W1_R1A_REPORT.md`. Chỗ sửa trong bài được đánh nhãn [IR-…].

| Mục | Cách sửa |
|---|---|
| BLOCKING-1 | (c): gạch 1 viết lại thành "chạm biên gần". Khẳng định "nằm trong cửa sổ" chỉ khi thêm 1/(ρK) ≤ 2η. Bài ghi phản ví dụ K = 2. (d): sửa tương tự. §0 sửa theo. |
| MAJOR-1 | §2.1 nói rõ (H) theo mọi y và mọi B là đòi hỏi mạnh, và thêm hai cách làm yếu: cục bộ, và khối điển hình (chỉ viết cho lịch quét ngẫu nhiên). TFM giả được chạy lại với h_max = 0,02, không clip (tỉ lệ clip đo được bằng 0), nên δ = 0 đúng. Lượt h = 0,1 giữ lại với δ trung thực. §3 và bảng §2.6 dùng lượt mới. §4 ghi phép đo chỉ là bản trung bình. |
| MAJOR-2 | §1 và Bổ đề 3 giới hạn phạm vi ở ngữ cảnh đầy đủ. Nhận xét 7 mới nêu cách mở rộng và hệ quả bão hoà ở cỡ max_context. Đã kiểm V0: chỉ optdigits (n_train 3.934) vượt ngưỡng. |
| MINOR-1 | Bỏ "Markov mạnh"; dùng biến cố toàn cục, thừa số (2H + T₀)K. |
| MINOR-2 | Tách λ_a và λ_b. Giới hạn là khoảng đối xứng hoặc bao lồi. Khẳng định ḡ = O(1/K) ghi là L1. |
| MINOR-3 | (d) ghi rõ chỗ hở δ_ctx(N) và số hạng +ε, vẫn ở mức L2. Bản chặt là (d′) cho lịch quét thật. |
| MINOR-4 | Nêu cái giá của việc định tâm h_i trong §2.1. |
| MINOR-5 | Dùng m = ⌈n/K⌉, a_m = (n_F − m)/n_F, giả sử n_F > m. Code `partition_bound` sửa theo. |
| MINOR-6 | §1, phạm vi: PU với tham số đã biết, TFM tất định, dòng `extend` thuộc F. |
| MINOR-7 | §0, §2.5, §2.6 và §3 sửa cho nhất quán. "So với 4(c)" thừa nhận chỉ kiểm soát ở ranh giới vòng quét, và Lν đóng vai δ_ctx. |
| MINOR-8 | Bảng §2.6 ghi rõ là dải bất biến (b′), kèm cách lấy dải (c′). Docstring sửa theo. |
| COSMETIC-1 | Ở §2.6: ε → ν, ρ_s → q_s, E → 𝓔. Các trùng ký hiệu khác ghi nhận, sẽ xử lý khi viết bản tiếng Anh. |
| COSMETIC-2 | (c): cố định N trước. |
| COSMETIC-3 | c′ = c. |
| COSMETIC-4 | Thêm dòng (1 − p)^H ⇒ E ≥ 1/p. |
| COSMETIC-5 | Nhận xét 4: ‖Φ″‖/(8(K−1)n), bỏ ρ². |
| COSMETIC-6 | Bổ đề 3 và 4′ ghi "ở ranh giới vòng quét" và "trên biến cố {x ≤ R_s}". |
