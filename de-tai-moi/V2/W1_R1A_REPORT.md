# W1 và R1a: mệnh đề drift cho chuỗi có ánh xạ phản ứng, và độ lệch của khối không sửa

05/11/2026. Làm trên CPU trong lúc chờ V1A/V1B. **Toàn bộ là thăm dò.** Tài liệu này không đổi giao thức đã đóng băng và không dùng thêm số test mới. δ trên TFM thật được đo lại từ dữ liệu V0 đã có.

## 0. Tóm tắt

1. **Mệnh đề 4 (drift), §2.**
   - Nội dung: giả sử độ chính xác của TFM trên một khối chỉ phụ thuộc vào *tỉ lệ* nhãn đúng trong ngữ cảnh và độ khó riêng của từng điểm, với dung sai δ. Đây là giả định (H_δ); Φ được giả sử Lipschitz.
   - Đặt δ′ = δ + δ_ctx, với δ_ctx = O(√(log N / n_F)). Khi đó, với lịch quét khối ngẫu nhiên:
     - (a) nơi gain Φ(r) − r ≥ γ > δ′, chuỗi đi lên với xác suất ≥ 1 − exp(−ρ n_F κ²/(c_w T)) − ε;
     - (b) nơi gain ≤ −γ, chuỗi không vượt lên quá một dải hẹp, và xác suất vượt nhỏ theo hàm mũ của n_F;
     - (c) nếu Φ có một điểm hút, điểm cuối không phụ thuộc điểm xuất phát;
     - (d) nếu Φ có hai điểm hút, điểm xuất phát quyết định lưu vực, và thời gian thoát lưu vực dài theo hàm mũ của n_F.
   - Chứng minh đầy đủ cho lịch quét khối ngẫu nhiên, ở bài rút độc lập (PLL, SSL, PU). Chưa phủ LLP.
   - **Lịch quét thật** (hoán vị rồi chia K khối mỗi vòng, đúng như `sampler.py`): chứng minh đầy đủ ở §2.6 (Bổ đề 3, Mệnh đề 4′, bổ sung 05/11 trưa).
     - Không cần Φ đơn điệu hay Lipschitz, vì chứng minh dùng bao đơn điệu của Φ.
     - Chuỗi trội một dãy lặp tất định kiểu Jacobi.
     - Sàn và trần bất biến; thời gian thoát lưu vực tăng mũ theo n_F, có hằng số.
     - Hằng số lỏng khoảng 10 lần so với mô phỏng.
   - **Tự rà bằng skill dm-paper-math-review** tìm ra 3 lỗi MAJOR ở bản đầu, đã sửa (`V2/reviews/W1_SELF_REVIEW_01.md`). Bản đầu so Φ tại r của cả trạng thái, đòi dung sai đều cho cả độ khó riêng của điểm, và ngụ ý phủ LLP. Bổ đề 3 và hằng số của thời gian thoát đã viết xong ở §2.6. Lượt rà độc lập đang chạy (§6).
2. **Mỗi phía của mệnh đề chỉ cần một phía của (H).**
   - Muốn kết luận "đi lên, ở lại trên", chỉ cần Φ_chuỗi ≥ Φ − δ.
   - Muốn kết luận "sụp, ở lại dưới", cần Φ_chuỗi ≤ Φ + δ.
3. **Kiểm bằng TFM giả biết trước Φ, §3.** Mô phỏng chạy bằng đúng `InContextGibbs`, tức lịch quét thật.
   - Với một điểm hút, điểm cuối không phụ thuộc điểm xuất phát: các điểm xuất phát lệch nhau ≤ 0,005.
   - Dải dừng co theo 1/√n và rộng hơn G_{δ′} = {r : |g(r)| ≤ δ′} một chút, đúng như dải "G_{δ′} + β" của mệnh đề.
   - Với hai điểm hút, ở n = 3.000, các điểm xuất phát cách u ≥ 0,2 thoả điều kiện của (d), và lưu vực đúng 100%.
   - Sát u (±0,05) mệnh đề không nói gì. Quan sát thấy lẫn lưu vực ở n = 500 (4/20 và 19/20 vào lưu vực trên) nhưng tách hẳn ở n = 3.000. Chênh này do nhiễu.
4. **δ đo trên TFM thật (dữ liệu V0), §4.**
   - TabICL: (H) gần đúng ở 2/4 dataset, sát biên ở 2 dataset còn lại. Ánh xạ gieo đều dự đoán điểm cuối với sai số 0,01–0,06.
   - TabPFN: (H) hai phía **sai**; trên semeion, δ ≈ gain. Nhưng phía dưới (H⁻) **gần đúng**: phần dư lệch về phía dương (trung vị +0,002 đến +0,093) và p10 ≥ −0,02, tức chuỗi làm tốt hơn hoặc bằng ánh xạ gieo đều báo, sai không quá 0,02 ở 90% bước.
   - Vậy ánh xạ gieo đều là **chặn dưới thận trọng**. Nó bảo đảm được kết luận "đi lên", nhưng không bảo đảm được kết luận "sụp". Đây đúng là cách P2 của V0 sai: dự đoán sụp, thực tế chuỗi dừng ở 0,78.
5. **R1a, §5.**
   - Total correlation (TC) của một khối có giới hạn đóng (d/2)[1/(K−1) − ln(K/(K−1))] ≤ d/(4(K−1)²), không phụ thuộc n. Giá trị chính xác luôn nằm dưới giới hạn, kiểm trên 80 ô.
   - Định lý 2 với hệ số Dobrushin τ cho chặn **rỗng** ở 22/24 ô, và lỏng 60–630 lần.
   - Phân rã cho thấy phần lỏng chủ yếu nằm ở ε̄. Sai số theo từng hàng triệt tiêu khi lấy trung bình theo π̃, nhỏ đi 10–210 lần. Về thực nghiệm, TV(π̃, π)/ε̄ chỉ nằm trong 0,49–1,17.
   - TV trên vector nhãn cỡ 1/K (thực nghiệm K^−1,13) và **không giảm theo n**. Sai lệch dự đoán chỉ cỡ n^−0,9·K^−1,15.
   - R2 vì vậy nên phát biểu ở mức dự đoán, không ở mức TV của vector nhãn.
   - **LLP (§5.6).** Bức tranh khác:
     - TV giảm theo n khi bài nhận dạng được;
     - sai lệch dự đoán lớn ở dữ liệu nhỏ: 0,02–0,05 ở n = 160, K = 10;
     - TV/ε̄ = 3–18;
     - khi bài không nhận dạng được (hai loại đối xứng), TV không giảm.

## 1. Thiết lập và ký hiệu

**Dữ liệu và phạm vi**
- n điểm huấn luyện.
- F là tập điểm tự do, gồm các điểm có nhãn chưa biết: với PLL là điểm có tập ứng viên ≥ 2; với SSL và PU là điểm chưa có nhãn.
- n_F = |F|, ρ = n_F/n. Giả sử ρK > 1, tức mỗi khối nhỏ hơn F.
- y* là nhãn thật. Nó **chỉ dùng trong phân tích**, phương pháp không dùng.
- **Phạm vi:** khối mà các điểm tự do được rút **độc lập**, tức PLL, SSL, PU. LLP rút cả túi theo ràng buộc số đếm nên chưa được phủ (Nhận xét 5).

**Đại lượng theo dõi**
- r(y) = (1/n_F) Σ_{i∈F} 1[y_i = y*_i], và R_t = r(Y_t).
- r_{−B}(y) là tỉ lệ đúng trên F∖B, tức trên phần tự do của *ngữ cảnh*.

**Lịch quét khối ngẫu nhiên** (đối tượng của mệnh đề)
- Ở mỗi bước t, chọn B_t là tập con đều cỡ m = n/K của [n], độc lập với quá khứ.
- Các điểm tự do trong B_t được rút lại độc lập. Mỗi điểm rút từ xác suất có điều kiện của TFM nhân với khả năng nhãn yếu, với ngữ cảnh là mọi điểm ngoài B_t.
- Một vòng quét gồm K bước.

**Ký hiệu khác**
- c_i(y, B) là xác suất lần rút mới của i ∈ B ∩ F bằng y*_i. Nó chỉ phụ thuộc y_{−B}, vì TFM chỉ thấy ngữ cảnh.
- Φ: [0, 1] → [0, 1] là ánh xạ phản ứng; g(r) = Φ(r) − r là gain.
- 𝓕_t = σ(Y_0, B_0, Y_1, …, B_{t−1}, Y_t). Lọc phải chứa B_u, vì một điểm rút lại đúng nhãn cũ không làm lộ khối.

**Hằng số**
- w = 1 + 2h_max + δ, c_w = (1 + w²)/2 và s² = c_w·m/(2n_F²) = c_w/(2Kρ n_F). Ở đây h_max và δ là các hằng số của (H) ở §2.1.
- Với chân trời N bước và ε ∈ (0, 1):
  - x_N = √(m ln(2N/ε)/2);
  - δ_ctx = K·L·m·x_N/(n_F(n_F − m)) = L/(ρ^{3/2}(1 − 1/(ρK)))·√(ln(2N/ε)/(2K n_F));
  - δ′ = δ + δ_ctx.

## 2. Mệnh đề 4

*Đây là bản sau lần tự rà thứ nhất. Bản đầu có ba lỗi MAJOR, ghi ở `V2/reviews/W1_SELF_REVIEW_01.md`:*
- *so Φ tại r(y) thay vì tại r_{−B};*
- *đòi dung sai đều trên mọi khối cho cả phần không đồng nhất theo điểm;*
- *ngụ ý phủ LLP.*

### 2.1 Giả định

**(H_δ), phản ứng đồng nhất.** Có các thành phần sau:
- một hàm Φ L-Lipschitz;
- các số h_i (i ∈ F) cố định, |h_i| ≤ h_max, Σ_{i∈F} h_i = 0;
- một số δ ≥ 0;

sao cho với mọi trạng thái y và mọi khối B có B ∩ F ≠ ∅:

  | (1/|B∩F|) Σ_{i∈B∩F} (c_i(y, B) − h_i) − Φ(r_{−B}(y)) | ≤ δ.

**Hai bản một phía.**
- (H⁻_δ): chỉ giữ vế "≥ Φ − δ".
- (H⁺_δ): chỉ giữ vế "≤ Φ + δ".

**Đọc giả định.**
- Hai vế chỉ phụ thuộc y_{−B}: giả định so cái TFM thấy với cái TFM thấy.
- h_i là độ khó riêng của từng điểm (điểm dễ, điểm khó), không đổi theo trạng thái.
  - Trung bình của h_i được gộp vào Φ, nên giả sử Σ h_i = 0 không làm mất tổng quát.
  - Trung bình h_i trên một khối dao động. Dao động đó đi vào **nhiễu** (ζ_2), không đi vào δ.
- δ là phần còn lại: mức mà *nhãn nào* sai còn quan trọng, ngoài *bao nhiêu* nhãn sai và độ khó riêng của điểm.
- Ánh xạ phản ứng (S2X, V0) đo Φ.
- Giả định áp cho mọi khối.
- Trong TFM giả ở §3: δ ≤ 0,001 (chỉ do clip), h_max = 0,1.

### 2.2 Bổ đề 1 (một bước)

**Phát biểu.** Cố định N và ε. Dưới (H⁻_δ), với mọi t < N có biến ζ⁻_t và biến cố G_t thoả ba điều:
1. Trên G_t:  R_{t+1} − R_t ≥ (1/K)(g(R_t) − δ′) + ζ⁻_t.
2. E[ζ⁻_t | 𝓕_t] = 0 và E[e^{θζ⁻_t} | 𝓕_t] ≤ e^{θ²s²/2} với mọi θ ∈ ℝ.
3. P(G_t^c | 𝓕_t) ≤ ε/N.

Dưới (H⁺_δ), trên cùng G_t có ζ⁺_t cùng tính chất, với R_{t+1} − R_t ≤ (1/K)(g(R_t) + δ′) + ζ⁺_t.

Ngoài ra |R_{t+1} − R_t| ≤ m/n_F = 1/(ρK) luôn đúng.

**Chứng minh** (phía dưới; phía trên đổi dấu δ và số hạng ngữ cảnh). Viết f = |B∩F| với B = B_t.

1. **Tách nhiễu rút.**
   - Gọi ξ_i = 1[lần rút mới của i đúng]. Khi biết (𝓕_t, B_t), các ξ_i độc lập, có kỳ vọng c_i.
   - Điểm ngoài B giữ nguyên, nên R_{t+1} − R_t = ζ_1 + (1/n_F) Σ_{i∈B∩F} (c_i − 1[Y_{t,i} đúng]), với ζ_1 = (1/n_F) Σ_{i∈B∩F} (ξ_i − c_i).
2. **Áp (H⁻) và Lipschitz.**
   Σ_{i∈B∩F} c_i ≥ Σ_{i∈B∩F} h_i + f(Φ(r_{−B}) − δ) ≥ Σ_{i∈B∩F} h_i + f(Φ(R_t) − δ) − fL·|r_{−B} − R_t|.
3. **Số hạng ngữ cảnh.**
   - Gọi k là số điểm đúng trong B∩F. Khi đó r_{−B} = (n_F R_t − k)/(n_F − f), nên r_{−B} − R_t = S_B/(n_F − f).
   - Ở đây S_B = Σ_{i∈B} b_i, với b_i = (R_t − 1[Y_{t,i} đúng])·1[i∈F].
   - Mọi b_i nằm trong [R_t − 1, R_t], một đoạn dài 1 chứa 0. Trung bình tổng thể của b_i bằng (1/n) Σ_F (R_t − 1[đúng]) = 0.
   - Hoeffding (1963) cho mẫu không hoàn lại: P(|S_B| > x_N | 𝓕_t) ≤ 2e^{−2x_N²/m} = ε/N.
   - Đặt G_t = {|S_{B_t}| ≤ x_N}. Vì f ≤ m, trên G_t: (f/n_F)·L·|r_{−B} − R_t| ≤ (m/n_F)·L·x_N/(n_F − m) = δ_ctx/K.
4. **Hằng số hoá.**
   - Đặt a_i = (Φ(R_t) − δ + h_i − 1[Y_{t,i} đúng])·1[i∈F]. Theo bước 2–3, trên G_t: (1/n_F) Σ_{B∩F} (c_i − 1[đúng]) ≥ (1/n_F) Σ_{i∈B} a_i − δ_ctx/K.
   - Vì Σ_F h_i = 0: E_B[(1/n_F) Σ_{i∈B} a_i] = (m/n)(1/n_F) Σ_F a_i = (1/K)(g(R_t) − δ).
   - Đặt ζ_2 = (1/n_F) Σ_{i∈B} a_i − E_B[·] và ζ⁻_t = ζ_1 + ζ_2.
5. **Proxy của ζ_1.** Khi biết (𝓕_t, B_t), ζ_1 là tổng của ≤ m biến độc lập có kỳ vọng 0, mỗi biến trong một đoạn dài 1/n_F. Bổ đề Hoeffding cho proxy m/(4n_F²).
6. **Proxy của ζ_2.**
   - Với i ∈ F, a_i ∈ [Φ − δ − h_max − 1, Φ − δ + h_max]. Với i ∉ F, a_i = 0.
   - Đoạn nhỏ nhất chứa các giá trị này và 0 dài ≤ max(1 + 2h_max, 1 + h_max + δ) ≤ w. Lý do: đầu dưới ≤ 0, vì Φ ≤ 1; nếu đầu trên < 0 thì độ dài là 1 + h_max + δ − Φ.
   - Hoeffding cho mẫu không hoàn lại cho proxy m·w²/(4n_F²).
7. **Ghép.**
   - ζ_2 đo được theo (𝓕_t, B_t), và E[ζ_1 | 𝓕_t, B_t] = 0. Lấy kỳ vọng lặp: E[e^{θ(ζ_1+ζ_2)} | 𝓕_t] ≤ exp(θ²·m(1 + w²)/(8n_F²)) = e^{θ²s²/2}.
   - Chặn |ΔR| ≤ f/n_F ≤ m/n_F là hiển nhiên. ∎

### 2.3 Bổ đề 2 (cực đại của martingale)

**Phát biểu.** Giả sử ζ_t thích nghi với 𝓕_{t+1} và E[e^{θζ_t} | 𝓕_t] ≤ e^{θ²s²/2}. Đặt M_t = Σ_{u<t} ζ_u. Khi đó P(min_{t≤N} M_t ≤ −κ) ≤ exp(−κ²/(2Ns²)), và chặn tương tự đúng cho max. Kết quả vẫn đúng cho tổng dừng Σ_{u<t∧σ} ζ_u, với σ là thời điểm dừng theo 𝓕, vì 1[u < σ] ∈ 𝓕_u.

**Chứng minh.** exp(−θM_t − θ²ts²/2) là siêu martingale không âm. Áp bất đẳng thức Ville trên [0, N], rồi chọn θ = κ/(Ns²). ∎

### 2.4 Mệnh đề 4

Các chặn dưới đây dùng chung một chân trời N và một ε. Biến cố G = ∩_{t<N} G_t có P(G^c) ≤ ε. Trên G, Bổ đề 1 đúng **theo từng đường đi**.

**(a) Đi tới, chỉ cần (H⁻_δ).**

*Phát biểu.* Giả sử:
- g ≥ γ trên [a, b), với γ > δ′;
- R_0 = r_0 ∈ [a, b);
- λ > 0, T ≥ (b − r_0 + λ)/(γ − δ′) vòng quét, và TK ≤ N.

Khi đó

  P(∃ t ≤ TK: R_t ≥ b) ≥ 1 − exp(−ρ n_F κ²/(c_w T)) − ε,

với κ = min(λ, r_0 − a). Nếu a = 0 thì κ = λ.

Bản đối xứng cho chiều đi xuống dùng (H⁺_δ).

*Chứng minh.*
1. Gọi σ là lần đầu R_t ≥ b hoặc R_t < a.
2. Trên G, với t < σ: R_{t+1} − R_t ≥ (γ − δ′)/K + ζ⁻_t. Do đó R_t ≥ r_0 + t(γ − δ′)/K + M_t với mọi t ≤ σ ∧ TK, trong đó M là tổng dừng của ζ⁻.
3. Nếu không chạm b, thì xảy ra một trong hai trường hợp:
   - σ > TK, nên M_{TK} < b − r_0 − T(γ − δ′) ≤ −λ;
   - R_σ < a, nên M_σ < −(r_0 − a). Trường hợp này không xảy ra khi a = 0.
4. Vậy {không chạm} ∩ G ⊂ {min_{t≤TK} M_t ≤ −κ}.
5. Bổ đề 2, với TK·s² = c_w T/(2ρ n_F), cho chặn của mệnh đề. ∎

**(b) Ở lại dưới, chỉ cần (H⁺_δ).**

*Phát biểu.* Giả sử g ≤ −γ trên [ℓ, u], với γ > δ′. Đặt:
- θ* = 4ρ n_F (γ − δ′)/c_w;
- ḡ_ℓ = max(0, sup{g(r) : ℓ − 1/(ρK) ≤ r < ℓ});
- x̄ = ℓ + (ḡ_ℓ + δ′)/K + λ′.

Giả sử thêm x̄ + λ ≤ u và R_0 ≤ x̄. Khi đó, với HK ≤ N,

  P(∃ t ≤ HK: R_t ≥ x̄ + λ) ≤ (HK + 1)·e^{−θ* λ} + HK·e^{−Kρ n_F λ′²/c_w} + ε.

Bản đối xứng: "ở lại trên" khi g ≥ γ trên một đoạn, dùng (H⁻_δ).

*Chứng minh.*
1. **Điểm bắt đầu của một lần vượt.**
   - Một lần vượt bắt đầu tại t₀ = 0 nếu R_0 ≥ ℓ, hoặc tại một lần cắt lên R_{t₀−1} < ℓ ≤ R_{t₀}.
   - Ở lần cắt lên, vì mỗi bước nhảy ≤ 1/(ρK), ta có R_{t₀−1} ∈ [ℓ − 1/(ρK), ℓ). Trên G, Bổ đề 1 cho R_{t₀} < ℓ + (ḡ_ℓ + δ′)/K + ζ⁺_{t₀−1}.
   - Biến cố E′ = {∃ t < HK: ζ⁺_t ≥ λ′} có xác suất ≤ HK·e^{−λ′²/(2s²)} = HK·e^{−Kρ n_F λ′²/c_w}.
   - Ngoài E′, mọi lần vượt bắt đầu tại x ≤ x̄.
2. **Trong một lần vượt.**
   - Gọi σ là lần đầu sau t₀ mà R < ℓ hoặc R ≥ x + λ.
   - Đặt D_v = (1/K)(g(R_v) + δ′) + ζ⁺_v. Trên G, với t₀ ≤ t ≤ σ: R_t − x ≤ Σ_{v=t₀}^{t−1} D_v.
   - Với v < σ, R_v ∈ [ℓ, u], nên D_v ≤ −(γ − δ′)/K + ζ⁺_v. Do đó E[e^{θ* D_v} | 𝓕_v] ≤ exp(−θ*(γ − δ′)/K + θ*²s²/2) = 1, vì θ* = 2(γ − δ′)/(Ks²).
   - Vậy W_t = exp(θ* Σ_{v=t₀}^{t−1} D_v) là siêu martingale không âm trên [t₀, σ].
   - Trên G, chạm x + λ trước σ kéo theo sup W ≥ e^{θ*λ}. Theo Ville, xác suất ≤ e^{−θ* λ}.
3. **Ghép lại.** Mọi t có R_t ≥ x̄ + λ đều nằm trong một lần vượt nào đó. Có ≤ HK + 1 lần vượt. Cộng các xác suất lại được chặn của mệnh đề. ∎

**Hệ quả (c): một điểm hút, cần cả hai phía của (H).**

*Phát biểu.* Giả sử g > 0 trên [0, r*) và g < 0 trên (r*, 1]. Với η > 0, đặt

  γ_η = min{min_{[0, r*−η]} g, min_{[r*+η, 1]} (−g)},

và giả sử γ_η > δ′. Đặt T₀ = ⌈(1 + λ)/(γ_η − δ′)⌉ và N = (T₀ + H)K. Khi đó, **từ mọi r_0**, với xác suất

  ≥ 1 − e^{−ρ n_F λ²/(c_w T₀)} − 2(HK + 1)·e^{−θ*λ} − 2HK·e^{−Kρ n_F λ′²/c_w} − ε,

với θ* = 4ρ n_F(γ_η − δ′)/c_w, hai điều xảy ra:
- chuỗi vào [r* − η, r* + η] trong T₀ vòng quét;
- sau đó chuỗi ở trong [r* − η − β, r* + η + β] suốt H vòng quét, với β = (ḡ + δ′)/K + λ′ + λ và ḡ là giá trị lớn hơn của hai ḡ_ℓ ở hai biên.

*Chứng minh.*
- Dùng (a) với biên ngoài là 0 hoặc 1, nên κ = λ, để đi tới biên gần.
- Sau đó dùng (b) và bản đối xứng của nó ở hai biên r* ± η, kể từ bước ngay trước lần chạm, nhờ tính Markov mạnh.
- G dùng chung cho mọi bước, nên chỉ trừ ε một lần.

*Khi 1/(ρK) < η:* cửa sổ của ḡ nằm trong vùng gain ngược dấu, nên ḡ = 0.

*Bản tiệm cận.*
- Cho n_F → ∞ với ρ, K, h_max cố định, H đa thức theo n_F, và ε = 1/n_F. Khi đó δ_ctx → 0.
- Chọn λ ~ log(HK n_F)/(ρ n_F) và λ′ ~ √(log(HK n_F)/(Kρ n_F)).
- Dải co về **G_δ ± (ḡ + δ)/K**, với G_δ = {r : |g(r)| ≤ δ}.
- Dùng chặn bước nhảy xác suất cao, cỡ (max|g| + δ)/K + λ′, thay cho 1/(ρK), thì ḡ = O(1/K), nên phần thêm vào là O(1/K²).
- Khi g tuyến tính với hệ số góc −α quanh r*, ta có G_δ = r* ± δ/α: sai lệch của điểm dừng bằng độ không đồng nhất chia cho độ co của ánh xạ.

*Ý nghĩa cho V2.*
- Nếu Φ có một điểm hút, mọi điểm xuất phát cho cùng một dải. Nhiều điểm xuất phát không đổi được điểm cuối; chúng chỉ giảm biến thiên Monte Carlo khi lấy trung bình hoặc stack.
- Điều này khớp với V0 ở các ô mà các chuỗi về cùng chỗ (V0_REPORT, P4).

**Hệ quả (d): hai điểm hút.**

*Phát biểu.* Giả sử:
- r₁ < u < r₂;
- g > 0 trên [0, r₁) ∪ (u, r₂), và g < 0 trên (r₁, u) ∪ (r₂, 1];
- R_0 ≥ u + 2η;
- γ = min{min_{[u+η, r₂−η]} g, min_{[r₂+η, 1]} (−g)} > δ′;
- r₂ − u > 2η + β, với β như ở (c).

Khi đó, với xác suất có cùng dạng như ở (c), với κ = min(λ, η) ở phần đi tới, chuỗi vào dải quanh r₂ và ở đó, phía trên u + η, suốt H vòng quét. Đối xứng, nếu R_0 ≤ u − 2η thì chuỗi vào dải quanh r₁.

*Thời gian thoát (phác thảo, mức L2).*
- Chọn λ = λ′ cỡ (r₂ − u − 2η − (ḡ + δ′)/K)/2 trong bản đối xứng của (b).
- Xác suất rời lưu vực trong H vòng quét ≤ C·HK·exp(−c ρ n_F min{(γ − δ′)λ, Kλ²}).
- Vậy thời gian thoát ≥ e^{c′n_F}/K vòng quét. Hằng số chưa viết ra.

*Ý nghĩa.*
- Điểm xuất phát quyết định lưu vực, và chạy thêm vòng quét không sửa được điều đó.
- Base learner khởi động có ích khi r_base ≥ u + 2η, và không có ích khi r_base ≤ u − 2η. Trong khoảng giữa, mệnh đề không nói gì.
- Điều này khớp về chiều với P1 của V0: TabPFN, PLL q 0,6, khởi động ấm ở r ≈ 0,17–0,21, còn base ở 0,96.
- Nhưng §4 cho thấy chuỗi TabPFN thật thoát một phần khỏi lưu vực dưới của ánh xạ gieo đều. Vậy lưu vực đó không phải bẫy thật đối với TFM thật.

### 2.5 Nhận xét

1. **Hằng số lỏng, cấu trúc đúng.**
   - Chặn thời gian của (a) với γ hằng lỏng 3–20 lần (§3). Ghép (a) trên nhiều đoạn con cho thời gian ≈ ∫ dr/(g(r) − δ′); cách ghép này hợp lệ khi g không tăng trên [0, b].
   - Chặn (b) phải cộng HK lần.
   - Phần đáng tin là cấu trúc:
     - các vùng chế độ, theo dấu của g và điều kiện γ > δ′;
     - sự độc lập với điểm xuất phát;
     - dải G_δ;
     - thang mũ theo n_F.
2. **Lịch quét thật.** Đã chứng minh đầy đủ ở §2.6 (Bổ đề 3, Mệnh đề 4′). Bản đó chặt hơn Mệnh đề 4 về cấu trúc, nhưng hằng số lỏng hơn.
3. **Hai phía của (H) dùng cho hai việc khác nhau.**
   - (a) và "ở lại trên" chỉ cần (H⁻).
   - (b) và kết luận "sụp" cần (H⁺).
   - Điều này quan trọng với TabPFN (§4).
4. **Số hạng ngữ cảnh không có log** (nhận xét, mức L1).
   - Giả sử Φ ∈ C² và ρ = 1, tức f = m cố định.
   - Phần tuyến tính Φ′(R)·S_B/(n_F − m) có kỳ vọng 0 theo B, nên có thể đưa vào ζ_2. Cái giá là proxy lớn lên.
   - Phần còn lại cho độ lệch mỗi vòng quét ≲ ‖Φ″‖/(8ρ²K n_F).
   - Với Φ tuyến tính, như hai ánh xạ một điểm hút ở §3, số hạng ngữ cảnh không làm lệch drift. Chưa viết chứng minh.
5. **LLP (chưa phủ).**
   - Hướng mở rộng: lấy túi làm đơn vị, cả khi chọn khối lẫn khi tính nhiễu rút.
   - ζ_1 thành tổng độc lập theo túi, mỗi túi có biên độ ≤ cỡ túi/n_F. Proxy nhân thêm khoảng b_max, cỡ túi lớn nhất.
   - c_i được thay bằng xác suất biên dưới quy hoạch động có ràng buộc số đếm.
6. **So với các phần khác của bài.**
   - Định lý 2 và Mệnh đề 3 nói về khoảng cách từ π̃ đến posterior.
   - Mệnh đề 4 nói về **động học** của chính chuỗi xấp xỉ: nó đi tới đâu và ở đó bao lâu. Đây là phần lý thuyết cho ánh xạ phản ứng (C3 trong V2_PLAN §3).

### 2.6 Lịch quét thật: Bổ đề 3 và Mệnh đề 4′

**Lịch quét.**
- Bộ lấy mẫu (`sampler.py`, dòng 182–183) làm như sau ở mỗi vòng s: lấy một hoán vị đều của [n], độc lập với quá khứ, rồi chia thành K khối liên tiếp B_1, …, B_K. Mỗi khối có ≤ m = ⌈n/K⌉ điểm.
- Với k = 1, …, K, các điểm tự do của B_k được rút lại độc lập, với ngữ cảnh là trạng thái hiện tại Y_{s,k−1} trên [n]∖B_k.
- Mỗi điểm tự do được rút **đúng một lần** mỗi vòng.

**Ký hiệu.**
- f_k = |B_k ∩ F|.
- r_k = r_{−B_k}(Y_{s,k−1}): tỉ lệ đúng của ngữ cảnh lúc cập nhật khối k.
- U_k = F ∩ (B_1 ∪ … ∪ B_{k−1}): các điểm đã cập nhật trong vòng.
- V_k = F ∩ (B_{k+1} ∪ … ∪ B_K): các điểm chưa cập nhật.
- o_i = 1[Y_{s,0,i} đúng]: độ đúng ở đầu vòng.
- **Bao đơn điệu của Φ:** Φ̲(x) = inf_{r∈[x,1]} Φ(r) và Φ̄(x) = sup_{r∈[0,x]} Φ(r). Ngoài [0, 1] thì kẹp về biên.
  - Hai hàm này không giảm.
  - Nếu Φ không giảm thì Φ̲ = Φ̄ = Φ.
  - **Không cần giả sử Φ đơn điệu.**

**Hằng số.** Với ε, λ > 0:

  p(ε, λ) = e^{−2n_Fλ²} + 3K·e^{−c ρ n_F ε²},  c = (1 − 1/(ρK))²·min{2/9, 1/(18 h_max²)}.

Nếu h_max = 0, lấy c = (1 − 1/(ρK))²·2/9.

**Bổ đề 3 (một vòng quét).** Giả sử (H_δ) ở §2.1 và ρK > 1. Không cần Φ Lipschitz.

- **(i) Sàn, chỉ cần (H⁻_δ).** Lấy x là một mức đo được theo 𝓕_s, với x ≤ R_s và Φ̲(x − ε) − δ ≥ x. Khi đó

  P(R_{s+1} ≥ Φ̲(x − ε) − δ − λ | 𝓕_s) ≥ 1 − p(ε, λ).

- **(ii) Trần, chỉ cần (H⁺_δ).** Lấy x ≥ R_s với Φ̄(x + ε) + δ ≤ x. Khi đó

  P(R_{s+1} ≤ Φ̄(x + ε) + δ + λ | 𝓕_s) ≥ 1 − p(ε, λ).

**Chứng minh (i).**

1. **Phân tích r_k.** Vì |U_k| + |V_k| = n_F − f_k, ta có

   (n_F − f_k)·r_k = Σ_{i∈U_k} ξ_i + Σ_{i∈V_k} o_i,

   với ξ_i là độ đúng của lần rút mới.
2. **Ba mảnh nhiễu.** Đặt N_k = Σ_{i∈U_k} h_i + Z_k + Σ_{i∈V_k} (o_i − R_s), với Z_k = Σ_{i∈U_k} (ξ_i − c_i). Đặt E = {N_k > −ε(n_F − m) với mọi k}.
3. **Quy nạp theo k: trên E, r_k ≥ x − ε với mọi k.**
   - Giả sử r_j ≥ x − ε với mọi j < k. Khi đó Φ(r_j) ≥ Φ̲(x − ε).
   - Theo (H⁻) cho từng khối j < k:
     Σ_{U_k} c_i ≥ Σ_{U_k} h_i + Σ_{j<k} f_j(Φ(r_j) − δ) ≥ Σ_{U_k} h_i + |U_k|·(Φ̲(x − ε) − δ) ≥ Σ_{U_k} h_i + |U_k|·x.
   - Cộng với Σ_{V_k} o_i = |V_k|·R_s + Σ_{V_k}(o_i − R_s) ≥ |V_k|·x + Σ_{V_k}(o_i − R_s), được (n_F − f_k)·r_k ≥ (n_F − f_k)·x + N_k.
   - Trên E: r_k > x − ε(n_F − m)/(n_F − f_k) ≥ x − ε.
   - Trường hợp k = 1 (U_1 = ∅) là đúng bước này.
4. **Cuối vòng.** Mọi điểm tự do được rút đúng một lần, nên
   R_{s+1} = (1/n_F)(Σ_F c_i + Z), với Z = Σ_F (ξ_i − c_i).
   Từ đó
   R_{s+1} ≥ (1/n_F)(Σ_F h_i + Σ_k f_k(Φ(r_k) − δ)) + Z/n_F ≥ Φ̲(x − ε) − δ + Z/n_F.
   Bất đẳng thức cuối dùng Σ_F h_i = 0, Σ_k f_k = n_F, và Φ(r_k) ≥ Φ̲(x − ε) trên E.
5. **Xác suất.** Ở mỗi k, mỗi mảnh được cho phần ε(n_F − m)/3:
   - **Σ_{U_k} h_i.** U_k được quyết định bởi hoán vị, độc lập với 𝓕_s. Mảnh này là tổng mẫu không hoàn lại cỡ ≤ n, từ tổng thể {h_i·1[i∈F]} có tổng 0 và biên độ 2h_max. Hoeffding (1963, §6) cho xác suất ≤ exp(−(ε(n_F − m)/3)²/(2n h_max²)).
   - **Σ_{V_k}(o_i − R_s).** Mảnh này là tổng mẫu không hoàn lại, từ tổng thể {(o_i − R_s)·1[i∈F]} có tổng 0 và biên độ 1. Xác suất ≤ exp(−2(ε(n_F − m)/3)²/n).
   - **Z_k.** Với lọc trong vòng, Z_k là martingale có ≤ n_F bước, mỗi bước trong một đoạn dài 1. Azuma–Hoeffding cho xác suất ≤ exp(−2(ε(n_F − m)/3)²/n_F).
   - **Gộp.** Vì n_F − m = n_F(1 − 1/(ρK)) và n = n_F/ρ, cả ba chặn đều ≤ e^{−cρ n_F ε²}. Cộng trên k ≤ K và ba mảnh được P(E^c) ≤ 3K·e^{−cρ n_F ε²}.
   - **Bước cuối.** Azuma cho Z: P(Z/n_F ≤ −λ) ≤ e^{−2n_Fλ²}. ∎

(ii) chứng minh đối xứng, dùng Φ̄, vế (H⁺) và các mảnh nhiễu với dấu ngược lại.

**Mệnh đề 4′ (lịch quét thật).** Đặt ψ(x) = Φ̲(x − ε) − δ − λ và ψ̄(x) = Φ̄(x + ε) + δ + λ. Cả hai không giảm. Viết p = p(ε, λ).

- **(a′) Đi tới, chỉ cần (H⁻).**
  - *Phát biểu.* Giả sử ψ(x) ≥ x + γ′ trên [r_0, b), với γ′ > 0. Đặt ρ_0 = r_0, ρ_{s+1} = ψ(ρ_s), và J là s đầu tiên có ρ_s ≥ b. Ta có J ≤ ⌈(b − r_0)/γ′⌉. Khi đó P(R_s ≥ ρ_s với mọi s ≤ J) ≥ 1 − J·p, và nói riêng P(R_J ≥ b) ≥ 1 − J·p.
  - *Chứng minh.* Quy nạp theo s, áp (i) với x = ρ_s:
    - x ≤ R_s theo giả thiết quy nạp;
    - Φ̲(ρ_s − ε) − δ = ψ(ρ_s) + λ ≥ ρ_s;
    - do đó R_{s+1} ≥ ψ(ρ_s) = ρ_{s+1}. ∎
- **(b′) Sàn và trần bất biến.**
  - Nếu ψ(x̲) ≥ x̲ và R_{s₀} ≥ x̲, thì P(R_s ≥ x̲ với mọi s ∈ [s₀, s₀ + H]) ≥ 1 − H·p. Áp (i) với x = x̲ ở mỗi vòng. Chỉ cần (H⁻).
  - Đối xứng: nếu ψ̄(x̄) ≤ x̄ và R_{s₀} ≤ x̄, thì chuỗi ở dưới x̄. Chỉ cần (H⁺).
- **(c′) Một điểm hút.**
  - *Phát biểu.* Lấy hai mức x̲ < x̄ sao cho:
    - ψ(x̲) ≥ x̲ và ψ̄(x̄) ≤ x̄;
    - ψ(x) ≥ x + γ′ trên [0, x̲);
    - ψ̄(x) ≤ x − γ′ trên (x̄, 1].
  - *Kết luận.* Từ mọi r_0, với xác suất ≥ 1 − 2(⌈1/γ′⌉ + H)·p, chuỗi vào [x̲, x̄] trong ⌈1/γ′⌉ vòng quét, rồi ở đó H vòng quét.
  - *Chứng minh.* Nếu r_0 < x̲:
    - đi tới x̲ theo (a′);
    - trong lúc đó trần x̄ đã giữ từ đầu theo (b′), vì R_0 < x̄;
    - sau đó dùng (b′) cho cả sàn lẫn trần.
    Trường hợp r_0 > x̄ đối xứng. ∎
  - **So với Mệnh đề 4(c):**
    - không có số hạng vượt O(1/K);
    - không có thừa số HK;
    - không cần δ_ctx, vì Bổ đề 3 làm việc thẳng với r_k của ngữ cảnh.
  - **Dải giới hạn.** Khi Φ không giảm và L-Lipschitz, mọi x̲ < r* có g(x̲) ≥ δ + Lε + λ đều là sàn, và tương tự cho trần. Cho n_F → ∞ với ε, λ ~ √(log/n_F), dải co về G_δ quanh r*.
- **(d′) Hai điểm hút: lưu vực và thời gian thoát (chặt).**
  - *Phát biểu.* Giả sử ψ(u + η) ≥ u + η, tức u + η là một sàn, và R_{s₀} ≥ u + η. Khi đó P(chuỗi xuống dưới u + η trong H vòng quét) ≤ H·p.
  - *Thời gian thoát.* Thời gian kỳ vọng để rời lưu vực trên ≥ 1/p.
  - *Với Φ không giảm và L-Lipschitz:* điều kiện sàn thoả khi g(u + η) ≥ δ + Lε + λ. Chọn ε = λ = (g(u + η) − δ)/(L + 1) cho p ≤ (1 + 3K)·exp(−c′ρ n_F (g(u + η) − δ)²/(L + 1)²), với c′ = min(c, 2).
  - Vậy thời gian thoát tăng theo hàm mũ của n_F. Điểm xuất phát quyết định lưu vực.
  - Bản này thay phần "phác thảo" của Hệ quả (d).

**Kiểm hằng số với mô phỏng ở §3.**

Cách tính (`w1_drift_check.py partition`, bảng `results/w1/w1_partition_bound.csv`):
- Lấy H = 60, xác suất tổng ≤ 0,05, và chia đều p cho hai số hạng.
- Ánh xạ tuyến tính Φ = a + b·r có r* = a/(1 − b). Sàn lớn nhất là r* − (bε + λ)/(1 − b), trần nhỏ nhất là r* + (bε + λ)/(1 − b).

| Ánh xạ | n | ε | λ | Dải chặn được | Khoảng 99% quan sát |
|---|---|---|---|---|---|
| 0,9 | 500 | 0,363 | 0,092 | [0,35; 1] | [0,850; 0,943] |
| 0,6 | 500 | 0,363 | 0,092 | [0; 1] (rỗng) | [0,514; 0,689] |
| 0,9 | 3.000 | 0,148 | 0,038 | [0,68; 1] | [0,880; 0,918] |
| 0,6 | 3.000 | 0,148 | 0,038 | [0,28; 0,92] | [0,564; 0,636] |

Đọc bảng:
- Chặn **đúng** nhưng **lỏng**: ở n = 3.000, nửa dải chặn được là 0,22, trong khi quan sát chỉ cỡ 0,02, tức lỏng khoảng 10 lần.
- Muốn nửa dải ≤ 0,05 thì cần n_F ≈ 6·10⁴ với ánh xạ 0,9 và ≈ 1,2·10⁵ với ánh xạ 0,6 (cột `n_F_for_halfwidth_0_05`).
- **Nguồn lỏng thứ nhất:** ε trả cho trường hợp xấu nhất, khi mọi khối trong vòng cùng thấy ngữ cảnh lệch. Thật ra lệch của r_k có kỳ vọng 0 và được lấy trung bình qua K khối.
- **Nguồn lỏng thứ hai:** Hoeffding dùng biên độ, không dùng phương sai.
- Bản bậc hai, giống Nhận xét 4, dùng Bernstein và cách chia ε tối ưu, có thể thu dải về cỡ nhiễu √(r(1−r)/n_F). Bản đó chưa viết.
- Phần định tính đã chặt ở mọi n: độc lập với điểm xuất phát, sàn và trần bất biến, thời gian thoát mũ.

## 3. Kiểm bằng TFM giả biết trước Φ

**Thiết kế** (`G4/code/experiments/w1_drift_check.py`)
- **Bài toán:** PLL với C = 10 lớp, q = 0,6. Có n_F ≈ n điểm tự do.
- **"TFM" giả:** xác suất rút đúng là clip(Φ(r_{−B}) + h_i, 0,01, 0,99), với h_i ~ U(−0,1; 0,1) cố định theo điểm.
  - Như vậy (H_δ) đúng với δ ≤ 0,001 (chỉ do clip), h_max = 0,1, |h̄| ≤ 0,003.
  - Hằng số Lipschitz L = 0,5 / 0,6 / 1,36 cho ánh xạ 0,9 / 0,6 / hai điểm hút.
- **Bộ lấy mẫu:** chạy bằng **đúng** `InContextGibbs`, K = 10, T = 60. Đây là lịch quét thật, nên mô phỏng kiểm cả phác thảo Bổ đề 3.
- **Lưới:** 6 điểm xuất phát × 20 lần lặp × 3 ánh xạ × n ∈ {500, 3.000}.
- **δ′ = δ_ctx** với N = 600 và ε = 0,01, ở n = 500 / 3.000:
  - ánh xạ 0,9: 0,019 / 0,008;
  - ánh xạ 0,6: 0,023 / 0,009;
  - hai điểm hút: 0,052 / 0,021.
- **δ̂_blk:** phân vị 99 của |trung bình h trên một khối|, bằng 0,019 / 0,008. Đại lượng này là thang **nhiễu** do h gây ra, không phải δ.
  - Bản đầu của report dùng δ̂_blk như δ; đó là MAJOR-2 của lần tự rà.
  - δ̂_blk trùng gần đúng với δ′ của ánh xạ 0,9, nhưng đó là trùng hợp.

**Dải dừng và tính độc lập với điểm xuất phát** (vòng quét ≥ 20; `results/w1/w1_theory_check.csv`)

| Ánh xạ | n | Chênh giữa 6 điểm xuất phát | G_{δ′} | Khoảng 99% quan sát | sd × √n |
|---|---|---|---|---|---|
| một điểm hút 0,9 | 500 | 0,002 | [0,862; 0,938] | [0,850; 0,943] | 0,397 |
| một điểm hút 0,9 | 3.000 | 0,0006 | [0,885; 0,916] | [0,880; 0,918] | 0,393 |
| một điểm hút 0,6 | 500 | 0,005 | [0,543; 0,657] | [0,514; 0,689] | 0,719 |
| một điểm hút 0,6 | 3.000 | 0,002 | [0,577; 0,623] | [0,564; 0,636] | 0,740 |
| hai điểm hút 0,2 / 0,95 | 500 | 0,736 | — | — | — |
| hai điểm hút 0,2 / 0,95 | 3.000 | 0,739 | — | — | — |

**Đọc bảng, ánh xạ một điểm hút**
- Điểm cuối không phụ thuộc điểm xuất phát, đúng như (c).
- Dải quan sát rộng hơn G_{δ′} một chút. Điều này khớp với (c), vì dải của mệnh đề là G_{δ′} cộng β.
- sd × √n gần như hằng: dải co theo 1/√n.
- Trong mô phỏng này δ ≈ 0 và Φ tuyến tính, nên theo Nhận xét 4 độ rộng dải là **nhiễu thuần**, cỡ √(log/n). G_{δ′} chỉ là chặn.
- sd quan sát lớn hơn dự đoán AR(1) kiểu Jacobi, √(r*(1−r*)/(n(1−Φ′²))), khoảng 11–21%.

**Thời gian tới** (`results/w1/w1_hitting.csv`; đích là r* ∓ 0,1; đơn vị là vòng quét; các cột lý thuyết dùng δ′)

| Ánh xạ, điểm xuất phát, n | Quan sát (trung vị, max) | Quét hệ thống tất định | Jacobi | ODE quét ngẫu nhiên | Chặn (a), γ hằng |
|---|---|---|---|---|---|
| 0,9, từ 0,1, n = 500 | 2, 3 | 3 | 4 | 5,0 | 45 |
| 0,9, từ 0,1, n = 3.000 | 2, 3 | 3 | 4 | 4,5 | 33 |
| 0,6, từ 0,95, n = 3.000 | 2, 2 | 2 | 3 | 3,6 | 16 |
| hai điểm hút, từ 0,70, n = 3.000 | 2, 2 | 3 | 4 | 3,4 | 8,4 |

- Ở mọi ô mà γ > δ′, trung vị quan sát ≤ quét hệ thống tất định ≤ Jacobi.
- ODE của quét ngẫu nhiên cùng cỡ với Jacobi.
- Chặn (a) với γ hằng lỏng 3–20 lần.
- Xác suất thất bại của chặn, đã gồm ε = 0,01, là 0,02–0,86 ở n = 500 và 0,010–0,16 ở n = 3.000.

**Ánh xạ hai điểm hút, điểm không ổn định u = 0,5**

Độ lệch chuẩn nhiễu của một vòng quét ≈ √(0,25/n_F), tức 0,022 ở n = 500 và 0,009 ở n = 3.000.

| Điểm xuất phát | g(r_0) | Vào lưu vực trên, n = 500 | Vào lưu vực trên, n = 3.000 |
|---|---|---|---|
| 0,10 và 0,30 | g(0,1) > 0, g(0,3) < 0: cùng về 0,2 | 0/20 và 0/20 | 0/20 và 0/20 |
| 0,45 | −0,016 | **4/20** | **0/20** |
| 0,55 | +0,018 | **19/20** | **20/20** |
| 0,70 và 0,95 | g ≥ 0: về 0,95 | 20/20 và 20/20 | 20/20 và 20/20 |

- **n = 3.000.** Với η = 0,1, các điều kiện của (d) thoả cho r_0 ≤ 0,3 và r_0 ≥ 0,7, vì min|g| trên các đoạn liên quan ≈ 0,028–0,035 > δ′ = 0,021. Chặn xác suất thì yếu: ví dụ phần đi tới từ 0,7 có xác suất thất bại ≈ 0,25. Quan sát đúng 100%.
- **n = 500.** δ′ = 0,052, nên mệnh đề không khẳng định gì.
- **Sát u (0,45 và 0,55).** |g| < δ′ ở cả hai n, nên mệnh đề không nói gì.
  - Chênh giữa hai n đến từ **nhiễu**: khoảng cách tới u là 0,05, tức ≈ 2,2 độ lệch chuẩn nhiễu ở n = 500 và ≈ 5,5 ở n = 3.000.
  - Kết quả quan sát đúng chiều mà mệnh đề nói ở giới hạn n lớn. Nhưng chặn định lượng chưa đủ chặt để phủ hai ô này.

## 4. δ đo trên TFM thật

Script `V2/analysis/w1_delta.py`, chỉ đọc dữ liệu V0: PLL q 0,6, 4 dataset có chuỗi nhiều điểm xuất phát. Bảng đầy đủ ở `V2/results/w1_delta.csv`.

**Các đại lượng:**
- **δ̂_plant:** |Φ_đều − Φ_cấu_trúc| tại cùng r, tức hai cách gieo lỗi khác nhau. Nếu (H) đúng thì δ̂_plant ≤ 2δ.
- **Phần dư có dấu:** (r_{t+1} − r_t) − g_đều(r_t) dọc vết thật của chuỗi, ở các vòng quét t ≥ 5.
  - Dương nghĩa là chuỗi làm tốt hơn ánh xạ báo.
  - Phần dư gồm cả nhiễu Monte Carlo của một vòng quét, cỡ √(r(1−r)/n_F) ≈ 0,005–0,01. Vì vậy nó là chặn trên của δ.
- **Sai số điểm cuối:** sai số khi dự đoán r cuối bằng cách lặp ánh xạ gieo đều 30 lần (như E2 của V0).

| TFM | Dataset | δ̂_plant (trung vị) | Phần dư muộn: trung vị | Phần dư muộn: p10 | γ̂ (trung vị \|g\|) | Sai số điểm cuối (có dấu) |
|---|---|---|---|---|---|---|
| TabICL | mfeat-factors | 0,005 | +0,011 | +0,002 | 0,045 | −0,017 |
| TabICL | optdigits | 0,002 | +0,008 | +0,007 | 0,066 | −0,010 |
| TabICL | semeion | 0,004 | +0,025 | +0,011 | 0,025 | +0,057 |
| TabICL | MiceProtein | 0,004 | −0,036 | −0,059 | 0,044 | −0,027 |
| TabPFN | mfeat-factors | 0,005 | +0,002 | −0,010 | 0,007 | +0,036 |
| TabPFN | optdigits | 0,003 | +0,018 | −0,020 | 0,019 | +0,080 |
| TabPFN | semeion | 0,003 | **+0,093** | −0,001 | 0,100 | **+0,405** |
| TabPFN | MiceProtein | 0,003 | +0,027 | −0,012 | 0,019 | +0,140 |

**Đọc kết quả** (thăm dò):

1. **Hai cách gieo cho cùng một Φ.** Trung vị δ̂_plant bằng 0,002–0,005 ở mọi ô. Hai cách gieo lỗi khác nhau cho cùng một ánh xạ, nên lỗi gieo có cấu trúc kiểu V0 **không** mô phỏng được lỗi của chuỗi.
2. **TabICL.**
   - Phần dư nhỏ so với gain ở mfeat-factors và optdigits. Hai ô semeion và MiceProtein sát biên: |phần dư| ≈ γ̂.
   - Sai số điểm cuối 0,01–0,06.
   - Mọi điểm xuất phát đều về cùng một chỗ: cold 0,16 → 0,98, base 0,96 → 0,98. Đây là chế độ (c).
3. **TabPFN.**
   - Phần dư lệch hẳn về phía dương: trung vị +0,002 đến +0,093, trong khi p10 ≥ −0,02.
   - Nghĩa là **lỗi do chuỗi tự tạo ít hại hơn lỗi gieo đều**. Chuỗi rút nhãn sai ở những điểm mà chính TFM thấy mơ hồ. Lỗi gieo đều thì làm hỏng cả những điểm dễ.
   - (H) hai phía sai: trên semeion, phần dư 0,093 ≈ γ̂ 0,100. Còn (H⁻) đúng với δ⁻ ≈ 0,02.
   - Theo Nhận xét 3, ánh xạ gieo đều là chặn dưới thận trọng:
     - dự đoán "đi lên hoặc ở lại trên" vẫn được bảo đảm;
     - dự đoán "sụp" thì không.
   - P2 của V0 sai đúng theo cách này: ánh xạ báo sụp, thực tế chuỗi dừng ở 0,78. Chuỗi khởi động ấm ở lưu vực dưới cũng thoát lên một phần, tới 0,36–0,72 ở 3/4 dataset, trong khi ánh xạ báo ở lại khoảng 0,17–0,21.
4. **Hàm ý cho W1.**
   - Muốn dự đoán định lượng (tiêu chí "Xong khi" của V2_PLAN W1: sai số trung vị ≤ 0,1), phải đo Φ trên **lỗi giống lỗi của chuỗi**, chứ không chỉ "có cấu trúc".
   - Một cách là gieo lỗi bằng chính phân phối dự đoán của TFM: rút nhãn sai theo Q̄ của một chuỗi đã chạy, ở mức r đã định.
   - Q̄ của V1 đang được lưu (npz). Vì vậy phép đo này chạy được trên GPU mà không cần chạy thêm chuỗi.

## 5. R1a: độ lệch của khối không sửa trên bài giải được chính xác

Script `G4/code/experiments/r1a_block_bias.py`; kết quả ở `G4/results/r1a/`.

### 5.1 E7a: total correlation của một khối

**Thiết lập.**
- Prior Dirichlet(1) trên C lớp, d = C − 1.
- Ngữ cảnh gồm N = n − m nhãn. Khối có m = n/K điểm.
- TC = KL(p(y_B | ngữ cảnh) ‖ Π_i p(y_i | ngữ cảnh)) = KL(Dirichlet-multinomial ‖ multinomial).
- Tính chính xác qua biên Beta-binomial của từng lớp.

**Kết quả** (`e7a_block_tc.csv`, 80 ô: C ∈ {2, 3, 5, 10}, n từ 100 đến 10.000, K ∈ {5, 10, 20, 40}):
- **Giới hạn đóng.** Khi N → ∞ với m/N = β = 1/(K−1), xấp xỉ Gauss cho vector đếm chỉ khác ở phương sai, theo hệ số 1 + β. Do đó

  TC → (d/2)[β − ln(1 + β)] = (d/2)[1/(K−1) − ln(K/(K−1))] ≤ d/(4(K−1)²).

- **Kiểm số.**
  - TC chính xác **luôn ≤ giới hạn** ở cả 80 ô. Tỉ số chính xác/giới hạn bằng 0,27–0,999, và ≥ 0,937 khi n ≥ 1.000.
  - Tỉ số TC/[d/(4(K−1)²)] bằng 0,26–0,98.
- **Kết luận.** Thang d/K², không phụ thuộc n, của hướng R2 **đúng**. Giới hạn đóng là chặn số học chặt hơn d/(4(K−1)²).
- **Chưa chứng minh:** TC ≤ giới hạn với mọi n hữu hạn. Hiện mới kiểm bằng số.

### 5.2 E7b: Định lý 2 với nhân khối chính xác làm tham chiếu

**Thiết lập.**
- Chuỗi gộp một ô, gồm hai bối cảnh: SSL-asym (10% có nhãn, 30% trong đó dương) và PU (c = 0,3).
- Khối cỡ n/K, chọn trên mọi điểm, như bộ lấy mẫu thật.
- **Nhân tham chiếu** rút **cùng nhau** các điểm tự do của khối, khi biết mọi nhãn còn lại.
  - Nhân này bất biến với π: sai số ≤ 1e-12 ở cả 24 ô.
  - Nhân xấp xỉ trùng nhân đã công bố ở E5c: chênh bằng 0 ở n ≤ 200.

**Kết quả** (`e7b_thm2_block_bound.csv`; n ∈ {50, 100, 200, 500}, K ∈ {5, 10, 20}):

| Đại lượng | Khoảng |
|---|---|
| TV(π̃, π) thật | 0,0012–0,0248 |
| ε̄ (sai số một bước, trung bình theo π̃) | 0,0021–0,0224 |
| τ(P), theo bước khối | 52–431 (n ≤ 200) |
| τ₁(Z) | 52–532 (τ₁(Z)/τ = 0,98–1,00) |
| Chặn τ₁(Z)·ε̄ | 0,69–2,60. **Rỗng (> 1) ở 22/24 ô** |
| Độ lỏng | 60–630 lần |

- **Chặn rỗng vì τ.** τ đếm số *bước khối* để trộn, và tăng theo cả K lẫn n.
- **TV không giảm theo n.** Ở K = 10, TV bằng 0,005, 0,009, 0,010, 0,011 khi n = 50, 100, 200, 500. E5c ở n = 1.000 cho 0,0115.

### 5.3 E7d: chặn lỏng ở đâu (thăm dò)

**Phân rã.** Có đẳng thức chính xác π̃ − π = wZ, với:
- w = π̃(P̃ − P) là khuyết một bước của π̃ dưới nhân chính xác (tổng bằng 0);
- Z = (I − P + 1π)⁻¹.

Do đó TV = ½‖wZ‖₁ ≤ τ₁(Z)·½‖w‖₁ ≤ τ₁(Z)·ε̄.

**Kết quả** (`e7d_bound_decomposition.csv`, 18 ô, n ≤ 200):

| Thừa số | Khoảng |
|---|---|
| Triệt tiêu: ε̄ / ½‖w‖₁ | 10–212 |
| Khuếch đại thật: TV / ½‖w‖₁ | 12–104, bằng 0,41–0,65 lần thời gian hồi phục 1/(1 − λ₂) |
| Chặn hậu kiểm τ₁(Z)·½‖w‖₁ so với TV | lỏng 2,9–8,4 lần |

**Đọc kết quả.**
- Phần lỏng chính **không** nằm ở τ. Nó nằm ở bước ‖w‖ ≤ ε̄.
- Sai số theo từng hàng của nhân, tức tích các biên so với phân phối chung, triệt tiêu gần hết khi lấy trung bình theo π̃.
- Hai thừa số triệt tiêu và khuếch đại có cùng bậc, nên **TV(π̃, π)/ε̄ ∈ [0,49; 1,17]** trên cả 24 ô của E7b.
- **Phỏng đoán:** TV(π̃, π) ≤ C·ε̄ với C = O(1), không cần τ. Đây là đích nên nhắm cho R2.
- **Hướng chứng minh:** dùng cấu trúc của P̃ − P. Hai nhân có cùng biên, chỉ khác ở mô-men bậc hai của số đếm, nên P̃ − P gần như trực giao với mode chậm nhất. Đặt cận cho ‖w‖ theo ε̄/(thời gian hồi phục).

### 5.4 E7c: π̃ lệch π ở đâu (thăm dò)

Kết quả ở `e7c_moments.csv`, 18 ô.

- **Độ rộng.** π̃ **rộng hơn** π: sd lớn hơn 0,2–5%, và K·(sd lệch) ≈ 0,04–0,25.
- **Vị trí.** Trung bình lệch ≤ 0,046 sd.
- **Dự đoán.** Dự đoán của một điểm mới tuyến tính theo số dương k, nên chỉ phần lệch vị trí tác động lên dự đoán. Sai lệch dự đoán ≤ 0,0056.
- **Thang theo n và K** (hồi quy log trên E5c, n ≥ 200):
  - TV ∝ n^{0,04…0,08}·K^{−1,13…−1,14};
  - sai lệch dự đoán ∝ n^{−0,86…−0,93}·K^{−1,15…−1,16}.

### 5.6 E7e: LLP, bag-block không sửa (thăm dò, 05/11 trưa)

**Mô hình** (`r1a_block_bias.py llp`, hàm `llp_chain`).
- **Dữ liệu:**
  - hai loại feature x ∈ {0, 1}, mỗi loại có p_x ~ Beta(5, 5) riêng;
  - B túi giống nhau; mỗi túi có s0 điểm loại 0, s1 điểm loại 1, và đúng c dương.
- **Trạng thái:** cấu hình một túi tóm bằng k, số dương loại 0. Vì các túi giống nhau, chuỗi gộp chính xác được trên histogram túi.
- **Khối:** M = B/K túi, chọn đều mỗi bước.
- **Nhân xấp xỉ** (đúng bag-block của bộ lấy mẫu): mỗi túi trong khối rút độc lập. Cách rút là quy hoạch động có ràng buộc số đếm, dùng xác suất biên p̂_x của ngữ cảnh, tức mọi túi ngoài khối.
- **Nhân tham chiếu:** rút chung mọi túi của khối. Nhân này bất biến với π, sai số ≤ 7·10⁻¹⁰.
- **Thiết lập:**
  - túi 4 với (3, 1), c = 1, cho B từ 20 đến 320 (n = 80–1.280);
  - túi 8 với (6, 2), c = 2, cho B ∈ {20, 40, 60} (n = 160–480);
  - túi 8 với (4, 4), c = 2, là trường hợp **đối xứng**;
  - K ∈ {5, 10, 20}, tổng cộng 33 ô.
- **Vì sao chọn Beta(5, 5):** với Beta(1, 1), hậu nghiệm dồn hết về cấu hình thuần (mọi dương cùng một loại), hoặc thành hai mode đối xứng. Em đã thử và thấy không đại diện.

**Kết quả** (`results/r1a/e7e_llp_bag_block.csv`).

1. **Hai thiết lập bất đối xứng, tức nhận dạng được (24 ô):**
   - **TV giảm theo n**, cỡ n^−0,6 đến n^−1,05, và theo K^−1,13 đến K^−1,24. Điều này **khác với SSL và PU** (§5.2), nơi TV không giảm theo n.
   - Nguyên nhân có thể: hậu nghiệm của mô hình đồ chơi co về một cấu hình biên khi n tăng; sd(K0) giảm từ 3,1 xuống 0,29.
   - **Sai lệch dự đoán lớn khi dữ liệu ít.** Ở K = 10:
     - n = 80 (túi 4): 0,013 cho loại 0 và 0,031 cho loại 1;
     - n = 160 (túi 8): 0,020 và 0,052;
     - từ n ≥ 320 trở đi: ≤ 0,0024.
     So với SSL và PU, nơi sai lệch luôn ≤ 0,0056, mức này lớn hơn hẳn ở dữ liệu nhỏ, rồi giảm nhanh theo n.
   - **π̃ rộng hơn π:** sd(K0) lớn hơn 5–91%.
   - **TV/ε̄ = 3–18, không còn ≈ 1 như ở SSL và PU.** Với bag-block, sai số theo hàng triệt tiêu ít hơn.
   - **Định lý 2 với τ₁(Z):** chặn rỗng ở 21/24 ô, lỏng 10–12.000 lần.
   - Từ B ≥ 160 (túi 4), hậu nghiệm có hai mode. Ở B = 320, τ₁(Z) ≈ 0,5–2·10⁵.
2. **Thiết lập đối xứng (4, 4), 9 ô:**
   - Hai loại hoán đổi được cho nhau, nên hậu nghiệm có hai mode, và sai lệch dự đoán bằng 0 do đối xứng.
   - TV **không giảm** theo n, ở mức 0,04–0,20.
   - τ₁(Z) tăng tới 7.000. Khi bài không nhận dạng được, độ lệch của bag-block không biến mất.

**Hàm ý.**
- Với LLP, độ lệch của bag-block **đáng kể ở dữ liệu nhỏ**: ít túi mỗi khối (M ≤ 2–4) và n ≤ 160.
- Ở cỡ dữ liệu của bài (n_train hàng trăm đến hàng nghìn, túi 8, K = 10, tức hàng chục túi mỗi khối), mô hình đồ chơi cho sai lệch dự đoán ≤ 10⁻³.
- Mô hình đồ chơi có hai loại feature, nên còn rất xa TFM thật. Đây không phải bằng chứng cho dữ liệu thật; R1b (GPU) mới đo được điều đó.
- Mệnh đề 3(b) cho bag-block (hệ số κ_B) đang rỗng. Bằng chứng ở đây ủng hộ một phát biểu ở mức dự đoán có điều kiện "đủ túi mỗi khối", chứ không phải một chặn đều theo n.

### 5.5 Hàm ý cho R2 và Mệnh đề 3(b)

1. **Phát biểu ở mức dự đoán.**
   - Ở mức vector nhãn, độ lệch của khối không sửa là **Θ(1/K) và không giảm theo n**. Không chặn TV nào trên vector nhãn có thể giảm theo n.
   - Mệnh đề 3(b) nên được viết lại ở **mức dự đoán**: sai lệch dự đoán cỡ 1/(nK) trong mô hình gộp. Hoặc ghi rõ rằng độ lệch trên vector nhãn không giảm theo n.
2. **Hai mảnh chứng minh cho R2.**
   - Mảnh TC trên khối (§5.1) đã có giới hạn đóng.
   - Mảnh chuyển từ một khối sang π̃ phải tránh τ: dùng phỏng đoán TV ≲ C·ε̄ (§5.3), hoặc làm thẳng ở mức mô-men (§5.4).
   - Định lý 2 với Dobrushin τ chỉ nên giữ như kết quả tổng quát. Bài không nên dựa vào nó cho khối.
3. **R1a cho LLP đã làm (§5.6).**
   - Kết luận khác với SSL và PU: TV giảm theo n khi bài nhận dạng được, nhưng sai lệch dự đoán lớn ở dữ liệu nhỏ, và TV/ε̄ ≫ 1.
   - Vì vậy R2 cho LLP cần đi đường riêng.
   - Bản có feature cho PLL và SSL không cần làm riêng: không có ràng buộc túi, hai loại feature tách thành hai chuỗi độc lập, nên kết quả như §5.2.

## 6. Việc tiếp theo

1. **Toán của §2.**
   - Lần tự rà thứ nhất đã xong và đã sửa 3 lỗi MAJOR (`V2/reviews/W1_SELF_REVIEW_01.md`).
   - Bổ đề 3 và Mệnh đề 4′ (lịch quét thật) đã chứng minh đầy đủ ở §2.6.
   - Rà độc lập: một agent chưa thấy bản nháp, kết quả ở `V2/reviews/W1_REVIEW_INDEPENDENT_01.md`.
   - Còn mở: bản bậc hai cho dải hẹp hơn (Nhận xét 4 và cuối §2.6), và LLP.
2. **Đo Φ_chuỗi trên GPU.** Gieo lỗi theo Q̄ đã lưu của V1, rồi so với Φ_đều. Mục đích là kiểm δ⁺ và tiêu chí sai số điểm cuối ≤ 0,1. Cần thầy duyệt trước khi viết code: khoảng 1 phiên Kaggle.
3. **R1a:** đã làm cả phần LLP (§5.6). Còn R1b trên GPU, theo kế hoạch của REVIEW_RESPONSE_PLAN.
4. **Đưa vào bài, sau V1:**
   - Mệnh đề 4 vào mục lý thuyết, chứng minh ở phụ lục;
   - bảng δ̂ vào phần ánh xạ phản ứng;
   - viết lại Mệnh đề 3(b) như §5.5.

## 7. Tệp

| Tệp | Nội dung |
|---|---|
| `V2/reviews/W1_SELF_REVIEW_01.md` | tự rà toán lần 1: sổ lỗi, phán quyết theo kết quả |
| `G4/code/experiments/w1_drift_check.py` | TFM giả biết Φ; `theory` so với Mệnh đề 4 (bản sửa, δ′ = δ_ctx) |
| `G4/results/w1/w1_drift_runs.csv`, `w1_drift_summary.csv` | 720 chuỗi, vết từng vòng quét |
| `G4/results/w1/w1_theory_check.csv`, `w1_hitting.csv` | dải dừng, G_δ, sd, thời gian tới |
| `V2/analysis/w1_delta.py`, `V2/results/w1_delta.csv` | δ trên TFM thật, từ dữ liệu V0 |
| `G4/code/experiments/r1a_block_bias.py` | E7a `tc`, E7b `thm2`, E7c `moments`, E7d `decomp`, E7e `llp` |
| `G4/results/r1a/e7a_block_tc.csv`, `e7b_thm2_block_bound.csv`, `e7c_moments.csv`, `e7d_bound_decomposition.csv`, `e7e_llp_bag_block.csv`, `e7b_log.txt`, `e7e_log.txt` | kết quả R1a |
| `G4/results/w1/w1_partition_bound.csv` | hằng số của Mệnh đề 4′ so với mô phỏng |

**Tài liệu dẫn** (đều là kết quả kinh điển; trước khi đưa vào bài phải kiểm theo REFERENCE_CHECK):
- Hoeffding, W. (1963), *Probability inequalities for sums of bounded random variables*, JASA 58:13–30, §6 (mẫu không hoàn lại).
- Ville, J. (1939), bất đẳng thức cực đại cho siêu martingale.
- Serfling, R. J. (1974), *Probability inequalities for the sum in sampling without replacement*, Ann. Statist. 2:39–48.
- Seneta (1988) và Cho & Meyer (2001) đã có trong bài, cho τ₁(Z).
