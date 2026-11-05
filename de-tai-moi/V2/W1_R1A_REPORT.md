# W1 và R1a: mệnh đề drift cho chuỗi có ánh xạ phản ứng, và độ lệch của khối không sửa

05/11/2026. Làm trên CPU trong lúc chờ V1A/V1B. **Toàn bộ là thăm dò.** Tài liệu này không đổi giao thức đã đóng băng và không dùng thêm số test mới. δ trên TFM thật được đo lại từ dữ liệu V0 đã có.

## 0. Tóm tắt

1. **Mệnh đề 4 (drift), §2.**
   - Nội dung: giả sử độ chính xác của TFM trên một khối chỉ phụ thuộc vào *tỉ lệ* nhãn đúng trong ngữ cảnh và độ khó riêng của từng điểm, với dung sai δ. Đây là giả định (H_δ); Φ được giả sử Lipschitz.
   - Đặt δ′ = δ + δ_ctx, với δ_ctx = O(√(log N / n_F)). Khi đó, với lịch quét khối ngẫu nhiên:
     - (a) nơi gain Φ(r) − r ≥ γ > δ′, chuỗi đi lên với xác suất ≥ 1 − exp(−ρ n_F κ²/(c_w T)) − ε;
     - (b) nơi gain ≤ −γ, chuỗi không vượt lên quá một dải hẹp, và xác suất vượt nhỏ theo hàm mũ của n_F;
     - (c) nếu Φ có một điểm hút, điểm cuối không phụ thuộc điểm xuất phát. Khẳng định "vào cửa sổ quanh r*" cần thêm 1/(ρK) ≤ 2η;
     - (d) nếu Φ có hai điểm hút, điểm xuất phát quyết định lưu vực.
   - Chứng minh đầy đủ cho lịch quét khối ngẫu nhiên, trừ thời gian thoát ở (d), vẫn là phác thảo.
   - **Phạm vi:**
     - bài rút độc lập: PLL, SSL, PU với tham số nhãn yếu đã biết; chưa phủ LLP;
     - TFM tất định;
     - ngữ cảnh: §2.1–2.6 cho ngữ cảnh đầy đủ; §2.7 mở rộng cho lấy mẫu con max_context = 2000 của cấu hình chạy thật.
   - **Lịch quét thật** (hoán vị rồi chia K khối mỗi vòng, đúng như `sampler.py`): chứng minh đầy đủ ở §2.6 (Bổ đề 3, Mệnh đề 4′, bổ sung 05/11 trưa).
     - Không cần Φ đơn điệu hay Lipschitz, vì chứng minh dùng bao đơn điệu của Φ.
     - Chuỗi trội một dãy lặp tất định kiểu Jacobi.
     - Sàn và trần bất biến ở ranh giới vòng quét; thời gian thoát lưu vực tăng mũ theo n_F, có hằng số.
     - Hằng số lỏng khoảng 10 lần so với mô phỏng.
   - **Mở rộng (§2.7, 05/11 chiều):** Bổ đề 3* phủ ngữ cảnh lấy mẫu con max_context.
     - Thời gian thoát chứng minh được bão hoà ở cỡ e^{c·M_c}.
     - Mô phỏng ở n = 3.000 cho thấy ảnh hưởng thật nhỏ hơn chặn. Ở cỡ dữ liệu của bài, đây chỉ là **dự đoán theo xấp xỉ AR(1)**, chưa kiểm.
     - Bổ đề 3* cũng dùng một giả định "tập điểm xấu". Giả định này yếu hơn về logic, nhưng với hằng số hiện có chỉ rẻ hơn khi n_F ≳ 1,2·10⁴ [IR2-MAJOR-2].
     - Hằng số tinh chỉnh giảm n_F cần thiết 1,7–1,8 lần (chặt), và khoảng 3 lần với bản Bernstein/Freedman (đã bổ sung ba bước thiếu theo IR2-MAJOR-1).
   - **Mệnh đề 6 (§2.8, 05/11 tối):** khi Φ affine, trung bình và phương sai dừng của R ở lịch quét thật có **dạng đóng chính xác**, tính bằng một đệ quy K bước.
     - Khớp mô phỏng trong ±1,5% (sd) ở cả 4 ô; dải 99% theo Gauss khớp trong ±0,005 (±0,001 ở n = 3.000). Rà độc lập lần 2 xác nhận ở mức L5.
     - Xấp xỉ Jacobi cũ thấp hơn 18–21%; Mệnh đề 6 giải thích vì sao.
   - **Tự rà bằng skill dm-paper-math-review** tìm ra 3 lỗi MAJOR ở bản đầu, đã sửa (`V2/reviews/W1_SELF_REVIEW_01.md`). Bản đầu so Φ tại r của cả trạng thái, đòi dung sai đều cho cả độ khó riêng của điểm, và ngụ ý phủ LLP. Bổ đề 3 và hằng số của thời gian thoát đã viết xong ở §2.6.
   - **Rà độc lập** (agent chưa thấy bản nháp, `V2/reviews/W1_REVIEW_INDEPENDENT_01.md`) tìm 1 BLOCKING, 2 MAJOR, 8 MINOR. **Đã sửa hết**; xem nhãn [IR-…] trong bài.
     - BLOCKING: ở (c), bước chạm có thể nhảy qua cả cửa sổ. Phản ví dụ: K = 2. Sửa bằng điều kiện 1/(ρK) ≤ 2η.
     - MAJOR-1: δ của TFM giả cũ phải tính theo khối xấu nhất, nên không phải "≤ 0,001". Em chạy lại mô phỏng không có clip, để δ = 0 đúng.
     - MAJOR-2: chưa mô hình hoá max_context; em giới hạn phạm vi.
   - Lõi được người rà xác nhận: Bổ đề 1 (L5), Bổ đề 2, Mệnh đề 4(a)(b), Bổ đề 3 (L5, với các giả định đã thêm), Mệnh đề 4′ (a′)–(d′).
2. **Mỗi phía của mệnh đề chỉ cần một phía của (H).**
   - Muốn kết luận "đi lên, ở lại trên", chỉ cần Φ_chuỗi ≥ Φ − δ.
   - Muốn kết luận "sụp, ở lại dưới", cần Φ_chuỗi ≤ Φ + δ.
3. **Kiểm bằng TFM giả biết trước Φ, §3.** Mô phỏng chạy bằng đúng `InContextGibbs`, tức lịch quét thật.
   - Lượt chính có h_max = 0,02 và không clip, nên (H) đúng với δ = 0. Đối tượng so đúng là Mệnh đề 4′.
   - Với một điểm hút, điểm cuối không phụ thuộc điểm xuất phát: các điểm xuất phát lệch nhau ≤ 0,005.
   - Dải dừng co theo 1/√n và có cùng bậc với G_{δ′} = {r : |g(r)| ≤ δ′}. Chặn chính thức của 4′ rộng hơn khoảng 10 lần.
   - Quan sát số vòng quét để tới đích ≤ dãy Jacobi trội, đúng như (a′).
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
   - **R2, Mệnh đề 5 (§5.7, 05/11 tối):** trong mô hình liên hợp gộp **SSL**, với lịch khối ngẫu nhiên, sai lệch dự đoán = (lệch kỳ vọng mỗi bước)/(tốc độ hồi phục của trung bình), không phụ thuộc total correlation.
     - Khối chỉ gồm điểm tự do: dự đoán **đúng tuyệt đối**, dù TV đo được không giảm theo n.
     - Khối trên mọi điểm: chuỗi cho **đúng trung bình hậu nghiệm với prior α′ = αK(n−1)/((K−1)n)**, tức sai lệch ≈ α(1−λ)(1−Cp_c)/(λ(K−1)n).
     - Phạm vi chỉ là SSL: với PU, khối free lệch tới 0,07. Vì vậy Mệnh đề 5 là một mệnh đề riêng, **không thay** Mệnh đề 3(b) tổng quát; nó thay đoạn "Why this is vacuous" [IR2-BLOCKING-1].
   - **LLP (§5.5).** Bức tranh khác:
     - TV giảm theo n khi bài nhận dạng được;
     - sai lệch dự đoán lớn ở dữ liệu nhỏ: 0,02–0,05 ở n = 160, K = 10;
     - TV/ε̄ = 3–18;
     - khi bài không nhận dạng được (hai loại đối xứng), TV không giảm.

## 1. Thiết lập và ký hiệu

**Dữ liệu và phạm vi**
- n điểm huấn luyện.
- F là tập điểm tự do, gồm các điểm có nhãn chưa biết: với PLL là điểm có tập ứng viên ≥ 2; với SSL và PU là điểm chưa có nhãn.
- n_F = |F|, ρ = n_F/n. Cỡ khối lớn nhất là m = ⌈n/K⌉; giả sử n_F > m, để F∖B luôn khác rỗng, và đặt a_m = (n_F − m)/n_F. Khi K chia hết n thì a_m = 1 − 1/(ρK).
- y* là nhãn thật. Nó **chỉ dùng trong phân tích**, phương pháp không dùng.
- **Phạm vi:**
  - khối mà các điểm tự do được rút **độc lập**, tức PLL, SSL, PU. LLP rút cả túi theo ràng buộc số đếm nên chưa được phủ (Nhận xét 5);
  - PU chỉ khi c (SCAR) hoặc η (SAR) **đã biết**. Bộ lấy mẫu gọi `weak.update_params` sau mỗi vòng quét (`sampler.py` dòng 151). Khi tham số được học, ℓ_i đổi giữa các vòng, và (H) phải đúng đều theo tham số;
  - TFM **tất định** theo (ngữ cảnh, truy vấn);
  - **ngữ cảnh đầy đủ**: max_context = None, hoặc n − m ≤ max_context. Cấu hình chạy thật đặt max_context = 2000 (`g7/methods.py` dòng 17), xem Nhận xét 7;
  - các dòng `extend` (n_total) cũng là điểm tự do, nên thuộc F.

**Đại lượng theo dõi**
- r(y) = (1/n_F) Σ_{i∈F} 1[y_i = y*_i], và R_t = r(Y_t).
- r_{−B}(y) là tỉ lệ đúng trên F∖B, tức trên phần tự do của *ngữ cảnh*.

**Lịch quét khối ngẫu nhiên** (đối tượng của mệnh đề)
- Ở mỗi bước t, chọn B_t là tập con đều cỡ m của [n], độc lập với quá khứ. Với lịch quét này, giả sử K chia hết n.
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

*Sau đó một lượt rà độc lập (`V2/reviews/W1_REVIEW_INDEPENDENT_01.md`, agent chưa thấy bản nháp) tìm 1 lỗi BLOCKING, 2 MAJOR, 8 MINOR. Tất cả đã sửa trong bản này; mục sửa ghi bằng nhãn [IR-…].*

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
  - Trung bình của h_i được gộp vào Φ, nên có thể giả sử Σ h_i = 0. [IR-MINOR-4] Việc này có cái giá:
    - sau khi định tâm, |h_i| ≤ h_max⁰ + |h̄| (tới 2h_max⁰);
    - Φ + h̄ có thể ra khỏi [0, 1].
    Các hằng số w, c_w, s², c dùng h_max sau khi định tâm. Bước 6 của Bổ đề 1 vẫn đúng vì đầu dưới ≤ −h_max⁰ − δ ≤ 0.
  - Trung bình h_i trên một khối dao động. Dao động đó đi vào **nhiễu** (ζ_2), không đi vào δ.
- δ là phần còn lại: mức mà *nhãn nào* sai còn quan trọng, ngoài *bao nhiêu* nhãn sai và độ khó riêng của điểm.
- Ánh xạ phản ứng (S2X, V0) đo Φ.
- **Giả định áp cho mọi y và mọi khối.** [IR-MAJOR-1] Đây là đòi hỏi mạnh:
  - mọi sai lệch phụ thuộc trạng thái mà h_i cố định không hấp thụ được đều bị tính trên **khối xấu nhất**, không được lấy trung bình;
  - khi có khối với |B∩F| nhỏ (1 − ρ ≳ 1/K, ví dụ SSL hoặc PU có ≥ 10% điểm cố định khi K = 10), (H) gần như đòi đúng **theo từng điểm**.
- **Hai cách làm yếu** (đã đối chiếu từng bước với chứng minh):
  - **Cục bộ:** các chứng minh chỉ dùng (H) tại trạng thái mà chuỗi đi qua trước thời điểm dừng tương ứng. Ví dụ, (a) chỉ cần (H) tại các y có r(y) ∈ [a, b). Vì vậy "mọi y" có thể thay bằng "mọi y trong vùng liên quan".
  - **Khối điển hình (H_{δ, ε_B})**, cho lịch quét khối ngẫu nhiên: với mọi y, P_B(|dev(y, B)| > δ) ≤ ε_B, với dev là vế trái của (H) và B theo đúng luật mà lịch quét dùng. Bổ đề 1 khi đó đúng với P(G_t^c | 𝓕_t) ≤ ε/N + ε_B. Lý do: B_t ⊥ 𝓕_t, và ζ_2 dựng từ a_i không dùng (H). Với lịch quét thật (§2.6), bản này cần luật có điều kiện của B_k biết các khối trước, nên **chưa viết**.
- **Trong TFM giả ở §3:**
  - **lượt chính, h_max = 0,02:** không có clip, vì Φ + h ∈ [0,16; 0,97]. Vậy (H) đúng với **δ = 0** cho mọi y và mọi B;
  - **lượt đầu, h_max = 0,1:** clip làm δ⁻ ở khối xấu nhất ≈ 0,05 với ánh xạ 0,9 và hai điểm hút, và ≈ 0,02 quanh điểm hút trên của ánh xạ 0,9. Ánh xạ 0,6 không bị clip nên δ = 0;
  - bản đầu của report viết "δ ≤ 0,001" cho lượt đầu; câu đó sai.

### 2.2 Bổ đề 1 (một bước)

**Phát biểu.** Cố định N và ε. Dưới (H⁻_δ), với mọi t < N có biến ζ⁻_t và biến cố G_t thoả ba điều:
1. Trên G_t:  R_{t+1} − R_t ≥ (1/K)(g(R_t) − δ′) + ζ⁻_t.
2. E[ζ⁻_t | 𝓕_t] = 0 và E[e^{θζ⁻_t} | 𝓕_t] ≤ e^{θ²s²/2} với mọi θ ∈ ℝ.
3. P(G_t^c | 𝓕_t) ≤ ε/N. Với bản khối điển hình, cộng thêm ε_B; xem §2.1.

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

và giả sử γ_η > δ′.
- **Cố định trước** N = (T₀ + H)K, rồi mới tính δ′ theo N. [IR-COSMETIC-2] Cách làm: chọn T₀ ≥ (1 + λ_a)/(γ_η − δ′), rồi kiểm lại điều kiện.
- Dùng **hai tham số riêng**: λ_a cho pha đi tới, λ_b cho dải. [IR-MINOR-2]
- Đặt θ* = 4ρ n_F(γ_η − δ′)/c_w.

Khi đó, **từ mọi r_0**, với xác suất

  ≥ 1 − e^{−ρ n_F λ_a²/(c_w T₀)} − 2(HK + 1)·e^{−θ*λ_b} − (2H + T₀)K·e^{−Kρ n_F λ′²/c_w} − ε,

hai điều xảy ra:
- **chuỗi chạm biên gần** (r* − η nếu đi từ dưới, r* + η nếu đi từ trên) trong T₀ vòng quét. Nếu thêm **1/(ρK) ≤ 2η**, thì tại lúc chạm chuỗi nằm **trong** [r* − η, r* + η]. [IR-BLOCKING-1]
- sau đó chuỗi ở trong [r* − η − β, r* + η + β] suốt H vòng quét, với β = (ḡ + δ′)/K + λ′ + λ_b và ḡ là giá trị lớn hơn của hai ḡ_ℓ ở hai biên.

*Vì sao cần 1/(ρK) ≤ 2η.*
- Thiếu điều kiện này, bước chạm có thể nhảy qua cả cửa sổ.
- Phản ví dụ của lượt rà độc lập: K = 2, L = 4. Khi đó chuỗi trung bình dao động trên chu trình 2 điểm {0,367; 0,633} quanh r* = 0,5, không bao giờ vào [0,45; 0,55], dù chặn cũ hứa xác suất ≥ 0,99.
- Với cấu hình thật (K = 10, ρ ≈ 1) và η = 0,1, điều kiện thoả.

*Chứng minh.*
- **Đi tới:** dùng (a) với biên ngoài là 0 hoặc 1, nên κ = λ_a. Chạm từ dưới thì R_τ ≥ r* − η. Vì mỗi bước nhảy ≤ 1/(ρK) ≤ 2η và R_{τ−1} < r* − η, nên R_τ < r* + η.
- **Ở lại:** lập luận (b) và bản đối xứng ở hai biên r* ± η. Không dùng tính Markov mạnh tại τ − 1, vì τ − 1 không phải thời điểm dừng. [IR-MINOR-1] Thay vào đó:
  - G và E′ là các biến cố toàn cục; E′ phủ cả pha đi tới, tức các bước [0, (T₀ + H)K), nên có thừa số (2H + T₀)K;
  - các lần vượt được đánh chỉ số theo thời điểm bắt đầu t₀ tất định, và {τ ≤ t₀} ∈ 𝓕_{t₀};
  - G dùng chung, nên chỉ trừ ε một lần.

*Khi 1/(ρK) < η:* cửa sổ của ḡ nằm trong vùng gain ngược dấu, nên ḡ = 0.

*Bản tiệm cận.*
- Cho n_F → ∞ với ρ, K, h_max cố định, H đa thức theo n_F, và ε = 1/n_F. Khi đó δ_ctx → 0.
- Chọn λ_a hằng (ví dụ 1), λ_b ~ log(HK n_F)/(ρ n_F), λ′ ~ √(log(HK n_F)/(Kρ n_F)).
- Dải co về **khoảng đối xứng nhỏ nhất quanh r* chứa G_δ, cộng ± (ḡ + δ)/K**, với G_δ = {r : |g(r)| ≤ δ}. Nếu G_δ có thành phần xa r*, giới hạn là bao lồi.
- Nhận định "dùng chặn bước nhảy xác suất cao, cỡ (max|g| + δ)/K + λ′, thay cho 1/(ρK), thì ḡ = O(1/K)" mới ở mức L1, chưa chứng minh.
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

Khi đó, với xác suất có cùng dạng như ở (c), với κ = min(λ_a, η) ở phần đi tới, chuỗi chạm r₂ − η, rồi ở trong dải quanh r₂, phía trên u + η, suốt H vòng quét. Nếu thêm 1/(ρK) ≤ 2η thì lúc chạm chuỗi nằm trong [r₂ − η, r₂ + η]. Đối xứng, nếu R_0 ≤ u − 2η thì chuỗi vào dải quanh r₁.

*Thời gian thoát (phác thảo, mức L2; chỉ cho lịch quét khối ngẫu nhiên).*
- Chọn λ_b = λ′ cỡ (r₂ − u − 2η − (ḡ + δ′)/K)/2 trong bản đối xứng của (b).
- Xác suất rời lưu vực trong H vòng quét ≤ C·HK·exp(−c ρ n_F min{(γ − δ′)λ_b, Kλ_b²}) + ε.
- **Chỗ hở** [IR-MINOR-3]: muốn thời gian thoát cỡ e^{c′n_F} thì N = HK cũng cỡ e^{c′n_F}. Khi đó δ_ctx(N) → L√(c′/(2K))/(ρ^{3/2}a_m), một hằng số. Kết luận chỉ đứng được khi c′ ≲ 2Kρ³a_m²(γ − δ)²/L², và ε cũng không nhỏ theo hàm mũ được.
- Với **lịch quét thật**, (d′) ở §2.6 đã có chứng minh chặt kèm hằng số.

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
2. **Lịch quét thật.** Đã chứng minh đầy đủ ở §2.6 (Bổ đề 3, Mệnh đề 4′). Bản đó chỉ kiểm soát R ở **ranh giới vòng quét**, không ở mọi bước, và hằng số lỏng hơn. Bù lại, nó không có số hạng vượt O(1/K), không có thừa số HK, và có thời gian thoát chặt.
3. **Hai phía của (H) dùng cho hai việc khác nhau.**
   - (a) và "ở lại trên" chỉ cần (H⁻).
   - (b) và kết luận "sụp" cần (H⁺).
   - Điều này quan trọng với TabPFN (§4).
4. **Số hạng ngữ cảnh không có log** (nhận xét, mức L1).
   - Giả sử Φ ∈ C² và ρ = 1, tức f = m cố định.
   - Phần tuyến tính Φ′(R)·S_B/(n_F − m) có kỳ vọng 0 theo B, nên có thể đưa vào ζ_2. Cái giá là proxy lớn lên.
   - Phần còn lại cho độ lệch mỗi vòng quét ≲ ‖Φ″‖/(8(K−1)n), vì E S_B² = m·R(1−R)·(n−m)/(n−1). [IR-COSMETIC-5]
   - Với Φ tuyến tính, như hai ánh xạ một điểm hút ở §3, số hạng ngữ cảnh không làm lệch drift. Chưa viết chứng minh.
5. **LLP (chưa phủ).**
   - Hướng mở rộng: lấy túi làm đơn vị, cả khi chọn khối lẫn khi tính nhiễu rút.
   - ζ_1 thành tổng độc lập theo túi, mỗi túi có biên độ ≤ cỡ túi/n_F. Proxy nhân thêm khoảng b_max, cỡ túi lớn nhất.
   - c_i được thay bằng xác suất biên dưới quy hoạch động có ràng buộc số đếm.
6. **So với các phần khác của bài.**
   - Định lý 2 và Mệnh đề 3 nói về khoảng cách từ π̃ đến posterior.
   - Mệnh đề 4 nói về **động học** của chính chuỗi xấp xỉ: nó đi tới đâu và ở đó bao lâu. Đây là phần lý thuyết cho ánh xạ phản ứng (C3 trong V2_PLAN §3).

7. **Ngữ cảnh lấy mẫu con (max_context)** [IR-MAJOR-2].
   - Cấu hình chạy thật đặt max_context = 2000 (`g7/methods.py` dòng 17). Khi |[n]∖B| > 2000, `sampler._ctx` (dòng 74–80) lấy mẫu con không hoàn lại C_k cỡ 2000, dùng chung cho cả khối.
   - Khi đó ba điều thay đổi:
     - c_i phụ thuộc C_k;
     - các ξ_i chỉ độc lập khi đã điều kiện theo C_k;
     - TFM thấy r_{C_k}, lệch r_{−B_k} một dao động siêu bội cỡ √(r(1−r)/(2000ρ)). Dao động này **không giảm theo n_F**.
   - **Phạm vi hiện tại:** mọi kết quả ở §2 được phát biểu cho ngữ cảnh đầy đủ, tức n − ⌈n/K⌉ ≤ max_context. Với K = 10, điều này nghĩa là n ≲ 2.223.
   - Trong 4 dataset PLL có chuỗi nhiều điểm xuất phát ở V0, chỉ **optdigits** (n_train = 3.934) vượt ngưỡng. mfeat-factors (1.395), semeion (1.115) và MiceProtein (756) đều nằm trong ngưỡng.
   - **Đã mở rộng ở §2.7** (Bổ đề 3*): (H) điều kiện theo C_k, với Φ tính tại r_{C_k}, và thêm biến cố 𝓔_ctx.
     - Thời gian thoát chứng minh được **bão hoà ở cỡ e^{c·M_c}**.
     - Mô phỏng cho thấy ảnh hưởng thật nhỏ hơn chặn nhiều, vì mỗi vòng quét lấy trung bình qua K mẫu con.

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

**Hằng số.** Với ν, λ > 0, trong đó ν là lề của ngữ cảnh trong vòng (đổi tên từ ε để khỏi trùng với ε ở §2.2–2.4) [IR-COSMETIC-1]:

  p(ν, λ) = e^{−2n_Fλ²} + 3K·e^{−c ρ n_F ν²},  c = a_m²·min{2/9, 1/(18 h_max²)},  a_m = (n_F − ⌈n/K⌉)/n_F. [IR-MINOR-5]

Nếu h_max = 0, lấy c = a_m²·2/9.

**Bổ đề 3 (một vòng quét).** Giả sử (H_δ) ở §2.1, n_F > ⌈n/K⌉, và ngữ cảnh đầy đủ (Nhận xét 7). Không cần Φ Lipschitz. Các khẳng định chỉ nói về R_s ở **ranh giới vòng quét**. [IR-COSMETIC-6]

- **(i) Sàn, chỉ cần (H⁻_δ).** Lấy x là một mức đo được theo 𝓕_s, với Φ̲(x − ν) − δ ≥ x. Khi đó, trên biến cố {x ≤ R_s},

  P(R_{s+1} ≥ Φ̲(x − ν) − δ − λ | 𝓕_s) ≥ 1 − p(ν, λ).

- **(ii) Trần, chỉ cần (H⁺_δ).** Lấy x ≥ R_s với Φ̄(x + ν) + δ ≤ x. Khi đó

  P(R_{s+1} ≤ Φ̄(x + ν) + δ + λ | 𝓕_s) ≥ 1 − p(ν, λ).

**Chứng minh (i).**

1. **Phân tích r_k.** Vì |U_k| + |V_k| = n_F − f_k, ta có

   (n_F − f_k)·r_k = Σ_{i∈U_k} ξ_i + Σ_{i∈V_k} o_i,

   với ξ_i là độ đúng của lần rút mới.
2. **Ba mảnh nhiễu.** Đặt N_k = Σ_{i∈U_k} h_i + Z_k + Σ_{i∈V_k} (o_i − R_s), với Z_k = Σ_{i∈U_k} (ξ_i − c_i). Đặt 𝓔 = {N_k > −ν(n_F − m) với mọi k}.
3. **Quy nạp theo k: trên 𝓔, r_k ≥ x − ν với mọi k.**
   - Giả sử r_j ≥ x − ν với mọi j < k. Khi đó Φ(r_j) ≥ Φ̲(x − ν).
   - Theo (H⁻) cho từng khối j < k:
     Σ_{U_k} c_i ≥ Σ_{U_k} h_i + Σ_{j<k} f_j(Φ(r_j) − δ) ≥ Σ_{U_k} h_i + |U_k|·(Φ̲(x − ν) − δ) ≥ Σ_{U_k} h_i + |U_k|·x.
   - Cộng với Σ_{V_k} o_i = |V_k|·R_s + Σ_{V_k}(o_i − R_s) ≥ |V_k|·x + Σ_{V_k}(o_i − R_s), được (n_F − f_k)·r_k ≥ (n_F − f_k)·x + N_k.
   - Trên 𝓔: r_k > x − ν(n_F − m)/(n_F − f_k) ≥ x − ν.
   - Trường hợp k = 1 (U_1 = ∅) là đúng bước này.
4. **Cuối vòng.** Mọi điểm tự do được rút đúng một lần, nên
   R_{s+1} = (1/n_F)(Σ_F c_i + Z), với Z = Σ_F (ξ_i − c_i).
   Từ đó
   R_{s+1} ≥ (1/n_F)(Σ_F h_i + Σ_k f_k(Φ(r_k) − δ)) + Z/n_F ≥ Φ̲(x − ν) − δ + Z/n_F.
   Bất đẳng thức cuối dùng Σ_F h_i = 0, Σ_k f_k = n_F, và Φ(r_k) ≥ Φ̲(x − ν) trên 𝓔.
5. **Xác suất.** Ở mỗi k, mỗi mảnh được cho phần ν(n_F − m)/3:
   - **Σ_{U_k} h_i.** U_k được quyết định bởi hoán vị, độc lập với 𝓕_s. Mảnh này là tổng mẫu không hoàn lại cỡ ≤ n, từ tổng thể {h_i·1[i∈F]} có tổng 0 và biên độ 2h_max. Hoeffding (1963, §6) cho xác suất ≤ exp(−(ν(n_F − m)/3)²/(2n h_max²)).
   - **Σ_{V_k}(o_i − R_s).** Mảnh này là tổng mẫu không hoàn lại, từ tổng thể {(o_i − R_s)·1[i∈F]} có tổng 0 và biên độ 1. Xác suất ≤ exp(−2(ν(n_F − m)/3)²/n).
   - **Z_k.** Với lọc trong vòng, Z_k là martingale có ≤ n_F bước, mỗi bước trong một đoạn dài 1. Azuma–Hoeffding cho xác suất ≤ exp(−2(ν(n_F − m)/3)²/n_F).
   - **Gộp.** Vì n_F − m = a_m·n_F, với m = ⌈n/K⌉, và n = n_F/ρ, cả ba chặn đều ≤ e^{−cρ n_F ν²}. Cộng trên k ≤ K và ba mảnh được P(𝓔^c) ≤ 3K·e^{−cρ n_F ν²}.
   - **Bước cuối.** Azuma cho Z: P(Z/n_F ≤ −λ) ≤ e^{−2n_Fλ²}. ∎

(ii) chứng minh đối xứng, dùng Φ̄, vế (H⁺) và các mảnh nhiễu với dấu ngược lại.

**Mệnh đề 4′ (lịch quét thật).** Đặt ψ(x) = Φ̲(x − ν) − δ − λ và ψ̄(x) = Φ̄(x + ν) + δ + λ. Cả hai không giảm. Viết p = p(ν, λ).

- **(a′) Đi tới, chỉ cần (H⁻).**
  - *Phát biểu.* Giả sử ψ(x) ≥ x + γ′ trên [r_0, b), với γ′ > 0. Đặt q_0 = r_0, q_{s+1} = ψ(q_s), và J là s đầu tiên có q_s ≥ b. Ta có J ≤ ⌈(b − r_0)/γ′⌉. Khi đó P(R_s ≥ q_s với mọi s ≤ J) ≥ 1 − J·p, và nói riêng P(R_J ≥ b) ≥ 1 − J·p.
  - *Chứng minh.* Quy nạp theo s, áp (i) với x = q_s:
    - x ≤ R_s theo giả thiết quy nạp;
    - Φ̲(q_s − ν) − δ = ψ(q_s) + λ ≥ q_s;
    - do đó R_{s+1} ≥ ψ(q_s) = q_{s+1}. ∎
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
  - **So với Mệnh đề 4(c)** [IR-MINOR-7]:
    - không có số hạng vượt O(1/K), không có thừa số HK, và không mắc lỗi vượt cửa sổ, vì trần bất biến ngay từ vòng 0;
    - nhưng chỉ kiểm soát R ở ranh giới vòng quét;
    - Lν trong điều kiện sàn đóng đúng vai của δ_ctx. Đó là đổi tên, không phải bớt được giả định.
  - **Dải giới hạn.** Khi Φ không giảm và L-Lipschitz, mọi x̲ < r* có g(x̲) ≥ δ + Lν + λ đều là sàn, và tương tự cho trần. Cho n_F → ∞ với ν, λ ~ √(log/n_F), dải co về bao lồi của phần G_δ quanh r*. Mức L2.
- **(d′) Hai điểm hút: lưu vực và thời gian thoát (chặt).**
  - *Phát biểu.* Giả sử ψ(u + η) ≥ u + η, tức u + η là một sàn, và R_{s₀} ≥ u + η. Khi đó P(chuỗi xuống dưới u + η trong H vòng quét) ≤ H·p.
  - *Thời gian thoát.* Gọi τ là vòng quét đầu tiên chuỗi xuống dưới u + η. Do tính Markov tại các ranh giới vòng quét, P(τ − s₀ > H) ≥ (1 − p)^H. Vậy E[τ − s₀] ≥ Σ_{H≥0} (1 − p)^H = 1/p. [IR-COSMETIC-4]
  - *Với Φ không giảm và L-Lipschitz:* điều kiện sàn thoả khi g(u + η) ≥ δ + Lν + λ. Chọn ν = λ = (g(u + η) − δ)/(L + 1) cho p ≤ (1 + 3K)·exp(−cρ n_F (g(u + η) − δ)²/(L + 1)²), vì c ≤ 2/9 < 2 và ρ ≤ 1. [IR-COSMETIC-3]
  - Vậy thời gian thoát tăng theo hàm mũ của n_F. Điểm xuất phát quyết định lưu vực.
  - Bản này thay phần "phác thảo" của Hệ quả (d).

**Kiểm hằng số với mô phỏng ở §3** (dải bất biến (b′), không phải dải đi vào của (c′)) [IR-MINOR-8].

Cách tính (`w1_drift_check.py partition --h 0.02`, bảng `results/w1/w1_partition_bound_h002.csv`; lượt này có δ = 0 đúng):
- Lấy H = 60, xác suất tổng ≤ 0,05, và chia đều p cho hai số hạng.
- Ánh xạ tuyến tính Φ = a + b·r có r* = a/(1 − b). Sàn lớn nhất là r* − (bν + λ)/(1 − b), trần nhỏ nhất là r* + (bν + λ)/(1 − b).

| Ánh xạ | n | ν | λ | Dải bất biến chặn được | Khoảng 99% quan sát |
|---|---|---|---|---|---|
| 0,9 | 500 | 0,363 | 0,092 | [0,35; 1] | [0,844; 0,944] |
| 0,6 | 500 | 0,363 | 0,092 | [0; 1] (rỗng) | [0,512; 0,686] |
| 0,9 | 3.000 | 0,148 | 0,038 | [0,68; 1] | [0,880; 0,919] |
| 0,6 | 3.000 | 0,148 | 0,038 | [0,28; 0,92] | [0,563; 0,636] |

Đọc bảng:
- Chặn **đúng** nhưng **lỏng**: ở n = 3.000, nửa dải chặn được là 0,22, trong khi quan sát chỉ cỡ 0,02, tức lỏng khoảng 10 lần.
- Muốn nửa dải ≤ 0,05 thì cần n_F ≈ 6·10⁴ với ánh xạ 0,9 và ≈ 1,2·10⁵ với ánh xạ 0,6 (cột `n_F_for_halfwidth_0_05`).
- **Nguồn lỏng thứ nhất:** ν trả cho trường hợp xấu nhất, khi mọi khối trong vòng cùng thấy ngữ cảnh lệch. Thật ra lệch của r_k có kỳ vọng 0 và được lấy trung bình qua K khối.
- **Nguồn lỏng thứ hai:** Hoeffding dùng biên độ, không dùng phương sai.
- Bản bậc hai, giống Nhận xét 4, dùng Bernstein và cách chia ν tối ưu, có thể thu dải về cỡ nhiễu √(r(1−r)/n_F). Bản đó chưa viết.
- Phần định tính (độc lập với điểm xuất phát, sàn và trần bất biến, thời gian thoát mũ) là định lý ở mọi n. Nhưng ở n nhỏ, ví dụ ánh xạ 0,6 với n = 500, chặn rỗng.
- Dải trong bảng là **dải bất biến** (b′): đúng khi chuỗi đã ở trong dải. Muốn có dải "từ mọi r_0" của (c′), phải lấy sàn thấp hơn một chút để có γ′ > 0, rồi cộng 2⌈1/γ′⌉p.

### 2.7 Mở rộng Bổ đề 3: ngữ cảnh lấy mẫu con, tập điểm xấu, hằng số tinh chỉnh (05/11 chiều)

Mục này xử lý ba việc mở ở §6:
- max_context [IR-MAJOR-2];
- bản "điển hình" của (H) cho lịch quét thật [IR-MAJOR-1];
- hằng số của Bổ đề 3.

Mục này đã qua rà độc lập lần 2 (`V2/reviews/W1_REVIEW_INDEPENDENT_02.md`): 2 MAJOR và một số MINOR, đã sửa; mỗi chỗ sửa ghi nhãn [IR2-…].

**Thiết lập.** Lịch quét như §2.6, với hai thay đổi:
- **Ngữ cảnh.** Ở khối k, ngữ cảnh là C_k ⊆ [n]∖B_k.
  - Nếu |[n]∖B_k| ≤ M_c, thì C_k = [n]∖B_k.
  - Nếu không, C_k là tập con đều cỡ M_c của [n]∖B_k, rút độc lập với mọi thứ khác (`sampler._ctx`, dòng 74–80).
- **Tỉ lệ đúng của ngữ cảnh.** Gọi r_C(y) là tỉ lệ đúng trên C ∩ F.

**Giả định (H*_{δ, η₀, D}), phía dưới.** Có các thành phần sau:
- Φ và các h_i như §2.1;
- δ ≥ 0;
- với mỗi trạng thái y, một tập Bad(y) ⊆ F với |Bad(y)| ≤ η₀ n_F. **Bad chỉ phụ thuộc y, không phụ thuộc B hay C.**

Đòi hỏi: với mọi y, mọi B và mọi C hợp lệ (C ∩ F ≠ ∅),

  Σ_{i∈B∩F} (c_i(y, B, C) − h_i) ≥ |B∩F|·(Φ(r_C(y)) − δ) − D·|B ∩ Bad(y)|.

Ghi chú:
- Phía trên đối xứng.
- Với η₀ = 0 và ngữ cảnh đầy đủ, (H*) chính là (H⁻_δ) của §2.1.

**(H*) yếu hơn về logic, nhưng chưa chắc rẻ hơn về hằng số** [IR2-MAJOR-2].
- **Yếu hơn về logic:** (H⁻_δ) kéo theo (H*) với Bad = ∅.
- **Giá phải trả:**
  - (H) theo mọi khối buộc δ chịu cả khối xấu nhất. Nếu một tỉ lệ η₀ điểm có thể lệch tới D, thì δ ≥ D·min(1, η₀n_F/m), và dải bị nới δ/(1 − b) với Φ tuyến tính độ dốc b.
  - (H*) trả β_bad = D(η₀H̃_K + ν_b) **hai lần**: một lần ngoài Φ̲, và một lần bên trong qua ν′. Dải bị nới β_bad(1 + b/a_m)/(1 − b).
  - Số hạng D·ν_b không nhỏ ở n vừa phải: ν_b phải cỡ √(K ln(K/q)/(2ρn_F)).
- **Ví dụ** (2% điểm lệch tới D = 1,1; K = 10; b = 0,5; tính theo lượt rà độc lập):

  | n_F | Giá trên dải, (H*) | Giá trên dải, (H) |
  |---|---|---|
  | 3.000 | 0,68 | 0,44 |
  | 10.000 | 0,46 | 0,44 |
  | 30.000 | 0,35 | 0,44 |
  | ∞ | 0,20 | 0,44 |

  Vậy (H*) chỉ có lợi khi n_F ≳ 1,2·10⁴, và lợi khoảng 2,2 lần ở giới hạn.
- **Cách cải thiện** (chưa làm): thay Hoeffding cho |B_k ∩ Bad| bằng Bernstein hoặc Chernoff siêu bội. Phương sai thật cỡ m·η₀H̃_K/K, nhỏ hơn nhiều biên độ 1 mà Hoeffding dùng, nên ν_b sẽ giảm hẳn.
- **Phạm vi của chặn H̃_K:** lượt rà dựng được một tập Bad thích nghi, luôn nằm trong các điểm chưa dùng, đạt gần đúng chặn này. Vậy thừa số H̃_K không lỏng ở trường hợp xấu nhất.

**Hằng số.**
- m = ⌈n/K⌉, m₋ = ⌊n/K⌋, a_m = (n_F − m)/n_F.
- H̃_K = (m/m₋)·Σ_{j=1}^{K} 1/j. Với K = 10, H̃_K ≈ 2,93.
- Với ν, ν_C, ν_b, λ > 0:
  - β_bad = D·(η₀ H̃_K + ν_b);
  - ν′ = ν + ν_C + β_bad/a_m;
  - p* = p(ν, λ) + K·e^{−2ν_b² n_F²/(K² m)}·1[η₀ > 0] + K·(e^{−M_c ρ² a_m² ν_C²/2} + e^{−M_c ρ² a_m²/2})·1[có lấy mẫu con]. Ở đây 2ν_b²n_F²/(K²m) ≈ 2ρn_Fν_b²/K.

**Bổ đề 3* (một vòng quét).** Giả sử (H*), phía dưới, và n_F > m. Lấy x đo được theo 𝓕_s, với Φ̲(x − ν′) − δ ≥ x. Khi đó, trên biến cố {x ≤ R_s},

  P(R_{s+1} ≥ Φ̲(x − ν′) − δ − β_bad − λ | 𝓕_s) ≥ 1 − p*.

Bản trần đối xứng: dùng phía trên của (H*) và Φ̄(x + ν′) + δ ≤ x. Bản này chưa viết ra; tập Bad của phía trên có thể khác của phía dưới (mức L2).

**Chứng minh.** Chứng minh giống Bổ đề 3, chỉ thêm hai biến cố. Gọi 𝓖_{s,k−1} là thông tin đến trước khối k: 𝓕_s, B_1, …, B_{k−1}, các ngữ cảnh C_1, …, C_{k−1} và mọi lần rút trước đó [IR2-COSMETIC-1]. Thật ra chỉ cần Bad(Y_{s,k−1}) đo được theo 𝓖_{s,k−1}, không nhất thiết chỉ phụ thuộc y.

1. **Luật của khối và của ngữ cảnh.**
   - Biết 𝓖_{s,k−1}, B_k là tập con đều cỡ |B_k| của tập điểm chưa dùng A_k, với |A_k| ≥ (K − k + 1)·m₋. Lý do: array_split của một hoán vị đều.
   - Biết thêm B_k, C_k là mẫu con đều của [n]∖B_k.
   - Bad(Y_{s,k−1}) đo được theo 𝓖_{s,k−1}.
2. **Biến cố 𝓔_bad.**
   - |B_k ∩ Bad(Y_{s,k−1})| có luật siêu bội, với kỳ vọng ≤ |B_k|·η₀n_F/|A_k| ≤ (m/m₋)·η₀n_F/(K − k + 1).
   - Hoeffding cho mẫu không hoàn lại: phần vượt kỳ vọng > x_b có xác suất ≤ e^{−2x_b²/m}.
   - Cộng trên k, với x_b = ν_b n_F/K: ngoài một biến cố xác suất ≤ K·e^{−2ν_b²n_F²/(K²m)}, ta có Σ_k D·|B_k ∩ Bad_k| ≤ n_F β_bad.
3. **Biến cố 𝓔_ctx** (chỉ khi lấy mẫu con).
   - Đặt e_j = (1[j đúng] − r_k)·1[j∈F] cho j ∈ [n]∖B_k, với r_k = r_{−B_k}(Y_{s,k−1}). Mọi e_j nằm trong [−r_k, 1 − r_k], một đoạn dài 1 chứa 0. Tổng của chúng bằng 0, và Σ_{j∈C_k} e_j = |C_k ∩ F|·(r_{C_k} − r_k).
   - Tỉ lệ điểm tự do ngoài khối là ρ_D = |F∖B_k|/|[n]∖B_k| ≥ ρ a_m.
   - Hoeffding (không hoàn lại) cho hai điều:
     - |C_k ∩ F| ≥ M_c ρ_D/2, với xác suất thất bại ≤ e^{−M_c ρ² a_m²/2};
     - Σ_{C_k} e_j ≥ −ν_C M_c ρ_D/2, với xác suất thất bại ≤ e^{−M_c ρ² a_m² ν_C²/2}.
   - Hai điều này cho r_{C_k} ≥ r_k − ν_C. Cộng trên k.
4. **Quy nạp.**
   - Giả thiết: r_{C_j} ≥ x − ν′ với mọi j < k.
   - Khi đó, như bước 3 của Bổ đề 3, trên 𝓔 ∩ 𝓔_bad:
     (n_F − f_k)·r_k ≥ (n_F − f_k)·x + N_k − n_F β_bad,
     nên r_k ≥ x − ν − β_bad/a_m.
   - Trên 𝓔_ctx thì r_{C_k} ≥ r_k − ν_C ≥ x − ν′.
5. **Cuối vòng.** R_{s+1} ≥ Φ̲(x − ν′) − δ − β_bad + Z/n_F. ∎

**Hệ quả cho Mệnh đề 4′.**
- (a′)–(d′) giữ nguyên, với ψ*(x) = Φ̲(x − ν′) − δ − β_bad − λ thay cho ψ, và p* thay cho p.
- **Khi có lấy mẫu con, p* ≥ K·e^{−M_cρ²a_m²ν_C²/2}.** Thời gian thoát chứng minh được ở (d′) do đó **bão hoà ở cỡ e^{c·M_c}**, không tăng mãi theo n_F. Đây chính là điều lượt rà độc lập chỉ ra; giờ nó đã thành định lượng.

**Mô phỏng: ảnh hưởng thật của max_context nhỏ hơn chặn** (`w1_drift_check.py context --h 0.02`, n = 3.000, 10 lần lặp, `results/w1/w1_context_cap_h002.csv`)

| M_c | sd theo thời gian, ánh xạ 0,9 | so với ngữ cảnh đầy đủ | Dự đoán √(1 + n_F·Φ′²/(K·M_c)) | Dự đoán có hệ số tổng thể hữu hạn (1 − M_c/(n − m)) | Hai điểm hút, từ 0,55: vào lưu vực trên |
|---|---|---|---|---|---|
| 100 | 0,0098 | ×1,32 | ×1,32 | ×1,31 | 9/10 |
| 300 | 0,0082 | ×1,11 | ×1,12 | ×1,10 | 9/10 |
| 1.000 | 0,0081 | ×1,09 | ×1,04 | ×1,02 | 10/10 |
| đầy đủ (3.000) | 0,0074 | ×1 | ×1 | ×1 | 10/10 |

**Đọc bảng.**
- **Dự đoán** (xấp xỉ AR(1), mức L1). Mỗi vòng quét lấy trung bình qua K khối, mỗi khối có một mẫu con riêng. Vì vậy phương sai thêm vào chỉ cỡ Φ′²·r(1−r)/(K·M_c), so với r(1−r)/n_F của nhiễu rút.
  - Dự đoán khớp ở M_c = 100 và 300.
  - Ở M_c = 1.000 thì không khớp: dự đoán ×1,02–1,04, quan sát ×1,09. Có thể do nhiễu với 10 lần lặp, nhưng chưa kiểm [IR2-MINOR-9].
- **Lưu vực.** Mẫu con nhỏ làm chuỗi sát u lẫn lưu vực hơn: 1/10 chuỗi rơi xuống lưu vực dưới khi M_c ≤ 300.
- **Hàm ý cho cấu hình thật** (M_c = 2.000, K = 10): phần thêm vào là n_F·Φ′²/20.000.
  - Với n_train ≤ 7.000, như Study 2, độ rộng dải tăng ≤ 16% ngay cả khi Φ′ = 1, và chỉ khoảng 4% khi Φ′ = 0,5. Đây là dự đoán theo xấp xỉ AR(1), chưa chứng minh.
  - Như vậy max_context gần như không đổi động học ở cỡ dữ liệu của bài. Chặn chứng minh được (theo từng khối, xấu nhất) bảo thủ hơn nhiều.

**Hằng số tinh chỉnh của Bổ đề 3** (`w1_drift_check.py refined`, `results/w1/w1_partition_bound_refined.csv`). Có hai cải tiến:
1. **Chia xác suất thất bại cho từng mảnh, dùng Ville cho mảnh Z** (chặt, mức L3). Z_k là các tiền tố của một martingale trong vòng, nên không cần thừa số K. Xác suất thành
   P(𝓔^c) ≤ e^{−2a_m²n_Fν_Z²} + K·e^{−2a_m²ρn_Fν_o²} + K·e^{−a_m²ρn_Fν_h²/(2h_max²)},  với ν = ν_Z + ν_o + ν_h.
   Code chia đều xác suất thất bại cho các mảnh; đây chưa phải phép chia ν tối ưu [IR2-MINOR-10].
2. **Bernstein/Freedman thay Hoeffding** [IR2-MAJOR-1]. Bản đầu thiếu ba bước, giờ đã viết đủ (mức L3, chưa rà lại):
   - **(i) Dừng.** Dừng martingale Z tại khối đầu tiên mà r_k < x − ν. Martingale dừng có biến thiên bậc hai khả đoán ≤ n_F·v_Z luôn đúng, nên Freedman (1975) áp được. Trên giao các biến cố còn lại, việc dừng không xảy ra.
   - **(ii) Jensen.** (H) chỉ khống chế trung bình khối, không khống chế từng c_i. Vì c(1−c) lõm, Σ_{i∈B∩F} c_i(1−c_i) ≤ f·c̄(1−c̄), với c̄ là trung bình khối. Trước thời điểm dừng, c̄ ≥ Φ̲(x − ν) − δ − h_max.
   - **(iii) Phương sai một phía.** v_Z = max c(1−c) trên c ∈ [Φ̲(x − ν) − δ − h_max, 1]. Mảnh o là mẫu không hoàn lại có phương sai tổng thể ρR_s(1−R_s), với R_s ≥ x; ta lấy v_o = max R(1−R) trên [x, 1]. Bernstein cho mẫu không hoàn lại đúng nhờ định lý trội lồi của Hoeffding (1963, Định lý 4).
   - Code đã sửa theo (iii): bản đầu lấy v trên dải hai phía và bỏ lề ν. Số gần như không đổi (19.100 → 19.800).

| Ánh xạ | Cách lấy hằng số | Nửa dải ở n_F = 3.000 | n_F cần để nửa dải ≤ 0,05 |
|---|---|---|---|
| 0,9 | bản gốc (§2.6) | 0,224 | 60.000 |
| 0,9 | chia tối ưu + Ville | 0,172 | 36.000 |
| 0,9 | + Bernstein/Freedman | 0,155 | 19.800 |
| 0,6 | bản gốc | 0,316 | 122.000 |
| 0,6 | chia tối ưu + Ville | 0,239 | 69.000 |
| 0,6 | + Bernstein/Freedman | 0,245 (kém bản trên; lấy min của hai bản) | 69.000 (v = 1/4, không lợi) |

**Kết luận về hằng số.**
- Phần đã chặt từ trước (chia + Ville) cắt n_F cần thiết 1,7–1,8 lần. Bản Bernstein/Freedman cắt khoảng 3 lần với ánh xạ 0,9.
- So sánh đúng đối tượng [IR2-MINOR-3]: dải của 4′ là dải **đều theo thời gian** trên H = 60 vòng quét. Đối chứng phải là khoảng [min, max] quan sát trên 60 vòng quét, ≈ [0,874; 0,924], tức nửa dải ≈ 0,025. Vậy ở n_F = 3.000, chặn 0,155 rộng hơn khoảng 6 lần.
- Phần lỏng còn lại **có cấu trúc, không phải do hằng số**: chặn trả lề ν cho *mọi* khối cùng lúc, trong khi lệch của r_k có kỳ vọng 0 và được lấy trung bình qua K khối.
- Mô-men dừng chính xác cho Φ affine đã có ở §2.8 (Mệnh đề 6) [IR2-COSMETIC-5]. Bài báo nên dùng chặn hiện có cho phần định tính (sàn, trần, lưu vực, thời gian thoát) và Mệnh đề 6 cho độ rộng dải.

### 2.8 Mệnh đề 6: mô-men dừng chính xác ở lịch quét thật khi Φ affine (05/11 tối)

Mục này là lập luận bậc hai đã hứa ở §2.7. Thay vì chặn đuôi, mục này tính **đúng** trung bình và phương sai dừng của R.

Rà độc lập lần 2 xác nhận ở mức L5. Lượt rà kiểm bằng ba cách: một quy hoạch động độc lập khớp đệ quy tới 2·10⁻¹⁵ trên 20 cấu hình; ma trận chuyển đầy đủ khớp tới 10 chữ số; và `InContextGibbs` thật với 40.000 vòng quét khớp trong sai số Monte Carlo. Các sửa đổi ghi nhãn [IR2-…].

**Thiết lập.**
- Lịch quét thật, ngữ cảnh đầy đủ, ρ = 1 (mọi điểm tự do), K khối bằng nhau cỡ m = n/K.
- Xác suất rút đúng của mỗi điểm trong khối k là c_k = c₀ + b·r_k, với r_k = r_{−B_k}(Y_{s,k−1}). Nghĩa là (H) đúng với h ≡ 0, δ = 0, và Φ(r) = c₀ + b·r affine **trên toàn [0, 1]**, với Φ([0, 1]) ⊆ [0, 1]. Đòi hỏi trên toàn đoạn là cần thiết: chuỗi bất khả quy, nên r_k có thể đạt mọi giá trị trên lưới {j/(n − m)} [IR2-MINOR-4].
- Đây đúng là hai ánh xạ một điểm hút của TFM giả §3, chỉ khác là bỏ h. Lượt chính có h_max = 0,02, nên ảnh hưởng của h ở cỡ h_max² ≈ 4·10⁻⁴.

**Mệnh đề 6.**
- **(a) Mô-men có điều kiện.** E[R_{s+1} | 𝓕_s] = α₀ + α₁R_s là affine, và Var[R_{s+1} | 𝓕_s] = q₀ + q₁R_s + q₂R_s² là bậc hai. Các hệ số tính đúng bằng một đệ quy K bước (dưới đây).
- **(b) Mô-men dừng.** Dưới phân phối dừng của chuỗi theo vòng quét:
  - E R = α₀/(1 − α₁) = r* = c₀/(1 − b);
  - Var R = (q₀ + q₁r* + q₂r*²)/(1 − α₁² − q₂);
  - tự tương quan trễ 1 bằng α₁.
- Không có xấp xỉ, không có hằng số chưa xác định.
- Các mẫu số đều dương [IR2-COSMETIC-3]: α₁ ≠ 1, và 1 − α₁² − q₂ > 0 vì Var R·(1 − α₁² − q₂) = q(r*) > 0 khi c_k ∈ (0, 1).

**Chứng minh.**
1. **Ký hiệu.** Gọi S = nR_s. Đặt O_j là số điểm đúng *cũ* trong khối j, và N_k là số điểm đúng *mới* rút ở khối k.
2. **Luật của O.** Vì phân hoạch là đều, (O_1, …, O_K) có luật siêu bội đa biến:
   - E O_j = m·S/n;
   - Var O_j = m·p(1−p)(n−m)/(n−1), Cov(O_j, O_l) = −m²·p(1−p)/(n−1), với p = S/n.
   Trung bình là affine theo S, hiệp phương sai là bậc hai theo S.
3. **Đệ quy theo khối.**
   - r_k = (Σ_{j<k} N_j + Σ_{j>k} O_j)/(n − m), nên N_k = m·c₀ + g_k·Z + ε_k, với Z = (N_{<k}, O), g_k là vector hằng (mb/(n − m) ở các vị trí tương ứng), và ε_k = N_k − m·c_k.
   - Biết quá khứ trong vòng, ε_k có kỳ vọng 0 (nên không tương quan với mọi biến trước nó) và phương sai m·c_k(1 − c_k). Đại lượng này là bậc hai theo Z, nên kỳ vọng của nó chỉ cần trung bình và hiệp phương sai của Z.
   - Quy nạp theo k: vector trung bình của (N_{≤k}, O) affine theo S, ma trận hiệp phương sai bậc hai theo S. Hiệp phương sai của N_k với các biến trước nó bằng g_k·Σ.
4. **Kết luận (a).** R_{s+1} = Σ_k N_k/n kế thừa hai tính chất trên.
5. **Kết luận (b).**
   - Chuỗi theo vòng quét có không gian trạng thái hữu hạn, nên dưới phân phối dừng E R_{s+1} = E R_s và E R_{s+1}² = E R_s².
   - Thay (a) vào và giải hai phương trình tuyến tính.
   - r* là điểm bất động của ánh xạ trung bình: nếu R_s = r* thì quy nạp cho E r_k = r* và E c_k = Φ(r*) = r* với mọi k. ∎

**Kiểm với mô phỏng** (`w1_drift_check.py second --h 0.02`, `results/w1/w1_second_order_h002.csv`; sd gộp mọi chuỗi và mọi vòng quét ≥ 20)

| Ánh xạ | n | sd chính xác (Mệnh đề 6) | sd quan sát | Tỉ số | sd Jacobi (cũ) | α₁ chính xác | α₁ quan sát (trong ngoặc: α₁ chính xác trừ chệch chuỗi ngắn) | Dải 99% theo Gauss | Dải 99% quan sát |
|---|---|---|---|---|---|---|---|---|---|
| 0,9 | 500 | 0,01898 | 0,01908 | 1,005 | 0,0155 | 0,355 | 0,285 (dự đoán 0,296) | [0,851; 0,949] | [0,844; 0,944] |
| 0,9 | 3.000 | 0,00775 | 0,00772 | 0,996 | 0,0063 | 0,355 | 0,300 (0,296) | [0,880; 0,920] | [0,880; 0,919] |
| 0,6 | 500 | 0,03465 | 0,03412 | 0,985 | 0,0274 | 0,456 | 0,380 (0,387) | [0,511; 0,689] | [0,512; 0,686] |
| 0,6 | 3.000 | 0,01416 | 0,01432 | 1,011 | 0,0112 | 0,456 | 0,400 (0,387) | [0,564; 0,636] | [0,563; 0,636] |

Cột α₁ quan sát là số thô, ước lượng trên chuỗi dài L = 41 vòng quét. Số trong ngoặc là α₁ chính xác trừ đi phần chệch (1 + 4α₁)/L. Đó là công thức chệch của ước lượng Yule–Walker; code dùng hệ số Pearson theo từng chuỗi, nên chệch khác một chút ở chữ số thứ ba [IR2-COSMETIC-2].

**Đọc bảng.**
- **Độ khớp.** sd chính xác khớp mô phỏng trong ±1,5% ở cả 4 ô. Dải 99% theo Gauss khớp dải quan sát trong ±0,005, và trong ±0,001 ở n = 3.000 [IR2-MINOR-2]. Lệch lớn nhất ở đầu dưới của ô (0,9; 500), nơi phân phối hơi lệch.
- **Vì sao xấp xỉ Jacobi cũ thấp hơn 18–21%.** Có hai hiệu ứng ngược chiều trong một vòng quét:
  - quét theo khối (Gauss–Seidel) làm tự tương quan giữa các vòng **nhỏ hơn** Jacobi: 0,355 so với b = 0,5;
  - nhưng nhiễu của các khối đầu lan qua ngữ cảnh sang các khối sau, nên nhiễu mỗi vòng **lớn hơn**.
  Hiệu ứng thứ hai thắng.
- **So với chặn của Mệnh đề 4′** [IR2-MINOR-3]. Hai kết quả nói về hai đối tượng khác nhau.
  - 4′ cho dải đều theo thời gian (nửa dải 0,155–0,224 ở n = 3.000, ánh xạ 0,9, §2.7).
  - Mệnh đề 6 cho σ chính xác của phân phối dừng tại một thời điểm. Dải 2,576·σ = 0,020 là phân vị Gauss, tức xấp xỉ.
  - Hai kết quả bổ sung cho nhau:
  - 4′ bảo đảm chuỗi **tới** và **ở lại**, từ mọi điểm xuất phát, không cần Φ affine;
  - 6 cho **đúng độ rộng** của phân phối dừng.

**Giới hạn.**
1. Chỉ chính xác khi Φ affine trên vùng chuỗi đi qua, h ≡ 0, δ = 0 và ρ = 1.
   - Với ρ < 1, kết luận vẫn đúng nhưng **không** vì lý do bản đầu viết ("điểm cố định chỉ thêm hằng số vào r_k") [IR2-MINOR-1]. r_k chỉ tính trên F∖B_k, nên điểm cố định làm cỡ tự do f_k của khối ngẫu nhiên, và mẫu số n_F − f_k của r_k cũng ngẫu nhiên. Lý do đúng: khi điều kiện theo vector (f_1, …, f_K), vốn độc lập với trạng thái, đệ quy affine–bậc hai đúng với các hệ số phụ thuộc f; trộn theo f vẫn cho E affine và Var bậc hai. Lượt rà đã kiểm bằng quy hoạch động chính xác; em chưa cài vào code.
   - Khi K không chia hết n, dạng đóng như đã viết không áp dụng. Mô phỏng của lượt rà cho thấy sd đổi trơn theo n.
   - Với h ≠ 0, cần theo dõi thêm Σ h_i·o_i; ảnh hưởng ở cỡ h_max².
2. Với Φ không affine, tuyến tính hoá quanh r* cho xấp xỉ có sai số do độ cong, cỡ ‖Φ″‖·σ² ở trung bình. Mức L2, chưa chặn.
3. Mệnh đề 6 cho mô-men, không cho đuôi.
   - Chặn đuôi chặt chẽ lấy từ Chebyshev, P(|R − r*| ≥ t) ≤ σ²/t², yếu.
   - Dải 99% theo Gauss là xấp xỉ, dựa trên định lý giới hạn trung tâm; khớp tốt trong mô phỏng nhưng không phải định lý.

**Ý nghĩa cho bài.** Bài có thể nói: "khi ánh xạ phản ứng affine (không bị cắt), độ dao động dừng của độ chính xác nhãn có dạng đóng; khi gần affine, dạng đóng là xấp xỉ bậc hai chưa có chặn sai số" [IR2-MINOR-4]. Công thức này có thể kiểm trên TFM thật khi có Φ_chuỗi đo trên GPU: đo b = Φ′(r*) rồi so sd dự đoán với sd quan sát của các chuỗi V1.

## 3. Kiểm bằng TFM giả biết trước Φ

**Thiết kế** (`G4/code/experiments/w1_drift_check.py`)
- **Bài toán:** PLL với C = 10 lớp, q = 0,6. Có n_F ≈ n điểm tự do.
- **"TFM" giả:** xác suất rút đúng là clip(Φ(r_{−B}) + h_i; 0,01; 0,99), với h_i ~ U(−h_max, h_max) cố định theo điểm.
- **Hai lượt:**
  - **Lượt chính, h_max = 0,02.** Mọi Φ + h ∈ [0,16; 0,97], nên không có clip; tỉ lệ bị clip đo được là 0. Vì vậy (H_δ) đúng với **δ = 0** cho mọi y và mọi B, sau khi gộp h̄ (|h̄| ≤ 0,001) vào Φ. Kết quả ở `results/w1/*_h002.csv`.
  - **Lượt đầu, h_max = 0,1.** Có clip ở ánh xạ 0,9 và hai điểm hút. Ở các ánh xạ này, δ⁻ của khối xấu nhất ≈ 0,05; xem §2.1 và [IR-MAJOR-1]. Lượt này giữ lại để so độ bền (`results/w1/*.csv` không có hậu tố).
- **Hằng số Lipschitz:** L = 0,5 / 0,6 / 1,36 cho ánh xạ 0,9 / 0,6 / hai điểm hút.
- **Bộ lấy mẫu:** chạy bằng **đúng** `InContextGibbs`, K = 10, T = 60, ngữ cảnh đầy đủ. Đây là lịch quét thật, nên đối tượng so đúng là Mệnh đề 4′ (§2.6). Mệnh đề 4 chỉ dùng để tham khảo.
- **Lưới:** 6 điểm xuất phát × 20 lần lặp × 3 ánh xạ × n ∈ {500, 3.000}.
- **δ′ = δ_ctx** với N = 600 và ε = 0,01, ở n = 500 / 3.000:
  - ánh xạ 0,9: 0,019 / 0,008;
  - ánh xạ 0,6: 0,023 / 0,009;
  - hai điểm hút: 0,052 / 0,021.

**Dải dừng và tính độc lập với điểm xuất phát** (lượt chính; vòng quét ≥ 20; `w1_theory_check_h002.csv`)

| Ánh xạ | n | Chênh giữa 6 điểm xuất phát | G_{δ′} | Khoảng 99% quan sát | sd × √n |
|---|---|---|---|---|---|
| một điểm hút 0,9 | 500 | 0,003 | [0,862; 0,938] | [0,844; 0,944] | 0,411 |
| một điểm hút 0,9 | 3.000 | 0,0008 | [0,885; 0,916] | [0,880; 0,919] | 0,410 |
| một điểm hút 0,6 | 500 | 0,004 | [0,543; 0,657] | [0,512; 0,686] | 0,726 |
| một điểm hút 0,6 | 3.000 | 0,002 | [0,577; 0,623] | [0,563; 0,636] | 0,749 |
| hai điểm hút 0,2 / 0,95 | 500 | 0,746 | — | — | — |
| hai điểm hút 0,2 / 0,95 | 3.000 | 0,749 | — | — | — |

**Đọc bảng, ánh xạ một điểm hút**
- Điểm cuối không phụ thuộc điểm xuất phát, đúng như (c′).
- sd × √n gần như hằng, nên dải co theo 1/√n.
- Ở đây δ = 0 và Φ tuyến tính, nên theo Nhận xét 4, độ rộng dải là **nhiễu thuần**, cỡ √(log/n). G_{δ′} chỉ cho cùng bậc độ lớn.
- Chặn chính thức của (b′)–(c′) lỏng hơn nhiều; xem cuối §2.6.
- sd quan sát lớn hơn dự đoán AR(1) kiểu Jacobi, √(r*(1−r*)/(n(1−Φ′²))), khoảng 6–22%.
- Lượt h_max = 0,1 cho cùng hình ảnh: chênh giữa điểm xuất phát ≤ 0,005, cùng sd × √n. Điều này gợi ý δ của khối xấu nhất là bi quan.

**Thời gian tới** (lượt chính; `w1_hitting_h002.csv`; đích là r* ∓ 0,1; đơn vị là vòng quét; cột lý thuyết dùng δ′)

| Ánh xạ, điểm xuất phát, n | Quan sát (trung vị, max) | Quét hệ thống tất định | Jacobi | ODE quét ngẫu nhiên | Chặn (a), γ hằng |
|---|---|---|---|---|---|
| 0,9, từ 0,1, n = 500 | 2, 3 | 3 | 4 | 5,0 | 45 |
| 0,9, từ 0,1, n = 3.000 | 2,5, 3 | 3 | 4 | 4,5 | 33 |
| 0,6, từ 0,95, n = 3.000 | 2, 2 | 2 | 3 | 3,6 | 16 |
| hai điểm hút, từ 0,70, n = 3.000 | 2, 2 | 3 | 4 | 3,4 | 8,4 |

- Ở mọi ô có γ > δ′: trung vị quan sát ≤ quét hệ thống tất định ≤ Jacobi. Điều này khớp với (a′): chuỗi thật trội một dãy lặp kiểu Jacobi.
- Chặn (a) với γ hằng lỏng 3–20 lần.
- Xác suất thất bại của chặn, đã gồm ε = 0,01, là 0,015–0,84 ở n = 500 và 0,010–0,12 ở n = 3.000.

**Ánh xạ hai điểm hút, điểm không ổn định u = 0,5** (cả hai lượt cho cùng tỉ lệ)

Độ lệch chuẩn nhiễu một vòng quét ≈ √(0,25/n_F), tức 0,022 ở n = 500 và 0,009 ở n = 3.000.

| Điểm xuất phát | g(r_0) | Vào lưu vực trên, n = 500 | Vào lưu vực trên, n = 3.000 |
|---|---|---|---|
| 0,10 và 0,30 | g(0,1) > 0, g(0,3) < 0: cùng về 0,2 | 0/20 và 0/20 | 0/20 và 0/20 |
| 0,45 | −0,016 | **4/20** | **0/20** |
| 0,55 | +0,018 | **19/20** | **20/20** |
| 0,70 và 0,95 | g ≥ 0: về 0,95 | 20/20 và 20/20 | 20/20 và 20/20 |

- **n = 3.000, lượt chính (δ = 0).** Với η = 0,1, các điều kiện của (d) thoả cho r_0 ≤ 0,3 và r_0 ≥ 0,7:
  - min|g| trên các đoạn liên quan ≈ 0,028–0,035, lớn hơn δ′ = 0,021;
  - 1/(ρK) = 0,1 ≤ 2η.
  Chặn xác suất thì yếu: phần đi tới từ 0,7 có xác suất thất bại ≈ 0,25. Quan sát đúng 100%.
- **Lượt h_max = 0,1.** δ⁻ của khối xấu nhất ≈ 0,05 ≥ γ, nên (H) toàn cục **không** đủ để khẳng định điều trên [IR-MAJOR-1]. Quan sát vẫn y hệt.
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

**Lưu ý về phép đo** [IR-MAJOR-1]:
- Phần dư lấy trung bình trên toàn F mỗi vòng quét. Vì vậy nó đo **bản trung bình** của δ, tức là chặn dưới của δ theo mọi khối. Nó không kiểm được (H) như đã phát biểu.
- optdigits (n_train = 3.934) chạy với ngữ cảnh lấy mẫu con 2.000 điểm, ngoài phạm vi của §2 (Nhận xét 7).

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

### 5.5 E7e: LLP, bag-block không sửa (thăm dò, 05/11 trưa)

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

### 5.6 Hàm ý cho R2 và Mệnh đề 3(b)

1. **Phát biểu ở mức dự đoán.**
   - Ở mức vector nhãn, độ lệch của khối không sửa là **Θ(1/K) và không giảm theo n**. Không chặn TV nào trên vector nhãn có thể giảm theo n.
   - Mệnh đề 3(b) nên được viết lại ở **mức dự đoán**: sai lệch dự đoán cỡ 1/(nK) trong mô hình gộp. Hoặc ghi rõ rằng độ lệch trên vector nhãn không giảm theo n.
2. **Hai mảnh chứng minh cho R2.**
   - Mảnh TC trên khối (§5.1) đã có giới hạn đóng.
   - Mảnh chuyển từ một khối sang π̃ phải tránh τ: dùng phỏng đoán TV ≲ C·ε̄ (§5.3), hoặc làm thẳng ở mức mô-men (§5.4).
   - Định lý 2 với Dobrushin τ chỉ nên giữ như kết quả tổng quát. Bài không nên dựa vào nó cho khối.
3. **R1a cho LLP đã làm (§5.5).**
   - Kết luận khác với SSL và PU: TV giảm theo n khi bài nhận dạng được, nhưng sai lệch dự đoán lớn ở dữ liệu nhỏ, và TV/ε̄ ≫ 1.
   - Vì vậy R2 cho LLP cần đi đường riêng.
   - Bản có feature cho PLL và SSL không cần làm riêng: không có ràng buộc túi, hai loại feature tách thành hai chuỗi độc lập, nên kết quả như §5.2.

### 5.7 R2 ở mức dự đoán: Mệnh đề 5 (05/11 tối)

Mục này thay hướng "total correlation + TV" của REVIEW_RESPONSE_PLAN §3.R2 bằng một kết quả **ở mức dự đoán**. Kết quả chính xác trong mô hình liên hợp **SSL**.

Rà độc lập lần 2 xác nhận (a)–(c) ở mức L4–L5, nhưng câu đề xuất cho bài ở bản đầu **sai** [IR2-BLOCKING-1]; đã sửa ở cuối mục.

**Thiết lập.**
- **Mô hình:** mô hình gộp Dirichlet–categorical C lớp, prior Dir(α, …, α).
- **Dữ liệu:** n_lab điểm có nhãn, với L_c điểm thuộc lớp c, và U điểm tự do không bị nghiêng (SSL). Đặt λ = n_lab/n và p_c = L_c/n_lab.
- **Trạng thái:** vector số đếm k = (k_c) của các điểm tự do theo lớp.
- **Lịch quét:** **khối ngẫu nhiên mỗi bước**, không phải hoán vị và chia K khối như bộ lấy mẫu thật [IR2-MINOR-5]. Khối theo một trong hai kiểu:
  - **free:** khối gồm u điểm tự do;
  - **all:** khối là tập con đều cỡ M = n/K của mọi điểm. Kiểu khối này giống mặc định `block_over="all"` của bộ lấy mẫu: ngữ cảnh bỏ cả khối, kể cả các điểm có nhãn nằm trong khối.
- **Hai nhân:**
  - P: Gibbs khối chính xác, rút chung các điểm tự do của khối khi biết mọi nhãn còn lại;
  - P̃: không sửa, rút độc lập theo dự đoán của ngữ cảnh.
- **Dự đoán:** dự đoán hậu nghiệm của một điểm mới là f_c(y) = (L_c + k_c + α)/(n + Cα), tuyến tính theo k.

**Mệnh đề 5.**

- **(a) Đồng nhất thức tổng quát.** Xét bất kỳ hai nhân nào với cùng không gian trạng thái hữu hạn. Nếu dưới nhân chính xác E_P[k_c′ − k_c | y] = A_c − a·k_c là affine với a > 0, và hai nhân lệch nhau về kỳ vọng có điều kiện một lượng d_c(y) = E_P̃[k_c′ | y] − E_P[k_c′ | y], thì

    E_π̃[k_c] − E_π[k_c] = π̃(d_c)/a.

  - **Chứng minh.** π̃ là phân phối dừng của P̃, nên E_P̃[k′ − k] = 0 dưới π̃, tức π̃(A_c − a·k_c + d_c) = 0. Ngoài ra E_π k_c = A_c/a. Chỉ cần a ≠ 0; không cần π̃ duy nhất. ∎
  - **Ý nghĩa.** Sai lệch dự đoán bằng π̃(d_c)/(a(n + Cα)), tức **(lệch kỳ vọng mỗi bước) × (thời gian hồi phục của trung bình, 1/a)**, chia cho n + Cα [IR2-COSMETIC-4]. Đây là bản ở mức dự đoán của Định lý 2: TV được thay bằng mô-men bậc nhất, và τ(P) được thay bằng 1/a. Nó **không** phụ thuộc vào việc phân phối chung của khối khác tích các biên bao nhiêu, tức total correlation.
- **(b) Khối free: dự đoán đúng tuyệt đối.**
  - Với kiểu free, cả hai nhân rút u điểm với cùng ngữ cảnh cỡ n − u.
  - Dirichlet–multinomial và tích các categorical có cùng biên. Do đó E[j′_c | ngữ cảnh] = u(L_c + k_c − j_c + α)/(n − u + Cα) cho cả hai, tức d ≡ 0.
  - Vậy **π̃(f_c) = π(f_c) với mọi n, K**. Điều này đúng dù TV(π̃, π) đo được không giảm theo n: 0,003–0,023 ở SSL-asym (E5c, cột tv_free) [IR2-COSMETIC-5].
  - Lượt rà thấy bằng số rằng kết luận này cũng đúng cho lịch quét hệ thống (mức L2).
- **(c) Khối all (mặc định): dạng đóng và tiệm cận.**
  - Cả hai drift đều affine. Ví dụ, với nhân chính xác:
    A_c = E_l[u(L_c + α)/(n − u + Cα)] và a = E_l[u/U − u(1 − u/U)/(n − u + Cα)],
    với l ~ siêu bội là số điểm có nhãn trong khối và u = M − l.
  - Với nhân không sửa, mẫu số là n − M + Cα, và L_c được thay bằng L_c − l·L_c/n_lab.
  - Vậy E_π̃ k_c = Ã_c/ã chính xác.
  - **Dạng đóng gọn** (rút gọn đại số của lượt rà độc lập, Phụ lục A của `W1_REVIEW_INDEPENDENT_02.md`; em đã kiểm lại):

    E_π k_c = U(L_c + α)/(n_lab + Cα),  E_π̃ k_c = U(L_c + α′)/(n_lab + Cα′),  **α′ = α·K(n − 1)/((K − 1)n)**.

    Nghĩa là **chuỗi không sửa có đúng trung bình hậu nghiệm của một prior mạnh hơn khoảng K/(K − 1) lần**. Từ đó, với N = n_lab:

    π̃(f_c) − π(f_c) = U(α′ − α)(N − C·L_c) / ((n + Cα)(N + Cα′)(N + Cα)),  với α′ − α = α(n − K)/((K − 1)n).

  - **Tiệm cận:**

    **π̃(f_c) − π(f_c) = α(1 − λ)(1 − C·p_c)/(λ(K − 1)·n)·[1 − (K + Cα + Cα/λ + Cα′/λ)/n + O(n⁻²)].**

    Hằng số của phần dư lớn khi λ nhỏ [IR2-MINOR-6]: khoảng −54/n khi λ = 0,1, K = 10. Nghĩa là phần dư ≈ 5% ở n = 1.000. Ở n nhỏ nên dùng dạng đóng gọn, không dùng tiệm cận.
  - **Kiểm** [IR2-MINOR-7]:
    - `G4/results/r1a/e7f_alpha_prime_check.csv`: dạng đóng gọn khớp `ssl_affine_means` tới ≤ 1,5·10⁻¹¹ với α ∈ {0,5; 1; 3} và 5 tổ hợp (n, λ, p, K);
    - `V2/analysis/r2_asymptotic.py` (sympy, chạy lại được) cho số hạng chính với C tổng quát;
    - lượt rà độc lập kiểm thêm C = 3 bằng ma trận chuyển đầy đủ, khớp tới 10 chữ số.
  - **Đọc công thức.**
    - Bỏ điểm có nhãn trong khối khỏi ngữ cảnh làm chuỗi **co về trung bình prior 1/C**, với trọng số ∝ α/(λ(K − 1)).
    - Sai lệch ∝ 1/n, ∝ 1/(K − 1), ∝ độ mạnh của prior, và ∝ tỉ lệ điểm tự do trên điểm có nhãn.
    - Khi p_c = 1/C, sai lệch bằng 0, vì dữ liệu trùng prior.

**Kiểm số** (`r1a_block_bias.py affine`, `results/r1a/e7f_affine_check.csv`, `e7f_affine_closed_form.csv`):
- Trung bình dừng dạng đóng khớp trung bình tính từ ma trận chuyển (E7b, E7c) **tới mọi chữ số in ra**, ở 9 ô SSL-asym, cho cả π lẫn π̃, cả kiểu all lẫn free.
- Ở kiểu free, π̃ và π có cùng trung bình ở mọi ô.
- Hồi quy trên dạng đóng với n từ 1.000 đến 10⁶ cho sai lệch ∝ n^{−0,97…−1,00}·K^{−1,10}. Số mũ của K bằng −1,10 chứ không phải −1, vì công thức có (K − 1).
- Con số thực nghiệm cũ của **SSL-asym** (E5c, n ≤ 1.000) là n^{−0,86}·K^{−1,15}. Nó là đoạn đầu chưa tiệm cận của cùng công thức. Con số của PU (n^{−0,94}) không thuộc phạm vi của dạng đóng [IR2-MINOR-8].

**Phạm vi và giới hạn.**
1. Kết quả chỉ chính xác cho **mô hình liên hợp gộp** và **nhãn yếu không nghiêng** (SSL).
2. Với PLL và PU, nhãn yếu nghiêng phân phối rút, nên drift của nhân chính xác không còn affine. Đồng nhất thức (a) vẫn đúng dưới dạng π̃(f) − π(f) = π̃((P̃ − P)u_f), với u_f là nghiệm phương trình Poisson của nhân chính xác, nhưng không có dạng đóng.
   - Con số đo được cho PU: 8·10⁻⁵ đến 4·10⁻³ (E5c, kiểu all).
   - Cơ chế dự kiến: các nhãn yếu khác trong khối mang thông tin về θ mà tích các biên bỏ qua. Đó là lượng I(y_i; w_{B∖i} | D_{−B}, w_i) trong REVIEW_RESPONSE_PLAN.
3. **LLP** (§5.5) cho bức tranh khác, vì nhân không sửa của bag-block lệch cả ở mô-men bậc nhất.
4. **TFM thật** không liên hợp. Cơ chế "ngữ cảnh mất các điểm có nhãn của khối, nên co về prior" vẫn có, nhưng α hiệu dụng của TFM không biết.
5. Kết quả cho TFM chính xác (q = p). Khi q ≠ p, cộng thêm sai số của Định lý 2.

**Đề xuất cho bài** (bản sửa theo IR2-BLOCKING-1).
- **Bản đầu sai.** Bản đầu viết "trong mô hình liên hợp, khối free cho dự đoán đúng tuyệt đối" và "đủ để thay Mệnh đề 3(b)". Hai câu đó sai:
  - mô hình gộp Beta–Bernoulli với PU **cũng** liên hợp, và ở đó khối free lệch dự đoán 0,02–0,07 (E5c);
  - Mệnh đề 3(b) là phát biểu cho mọi loại nhãn yếu, kể cả bag-block của LLP.
- **Giữ 3(b) tổng quát.** Chỉ thay đoạn "Why this is vacuous" bằng E7b và E7d, tức giải thích **vì sao** chặn TV rỗng.
- **Thêm một mệnh đề riêng có phạm vi rõ.** Câu đề xuất: "Trong mô hình liên hợp gộp **SSL** (điểm tự do không có nhãn yếu), với lịch khối ngẫu nhiên: (i) khối chỉ gồm điểm tự do cho trung bình dừng của dự đoán đúng tuyệt đối; (ii) khối trên mọi điểm cho đúng trung bình hậu nghiệm với prior α′ = αK(n−1)/((K−1)n), tức sai lệch α(1−λ)(1−Cp_c)/(λ(K−1)n)·(1 + O(1/n)). Đồng nhất thức (a) cho thấy cái quyết định là mô-men bậc nhất của bước cập nhật, không phải total correlation của khối."
- **Nhận xét thực nghiệm, tách khỏi mệnh đề** [IR2-MINOR-11]: TV(π̃, π) đo được cỡ K^{−1,13} và không giảm theo n.
- **Hệ quả nên nói thẳng.** Với SSL, kiểu `free` tốt hơn mặc định ở mức dự đoán. Với PU thì ngược lại; đó là lý do mặc định là `all`.
- **Với lịch quét thật:** số hạng chính trùng với lịch khối ngẫu nhiên theo số (tỉ số 1,000–1,010, lượt rà độc lập). Đó là bằng chứng số, không phải chứng minh.

## 6. Việc tiếp theo

1. **Toán của §2.**
   - **Đã xong:**
     - tự rà lần 1, đã sửa 3 lỗi MAJOR (`V2/reviews/W1_SELF_REVIEW_01.md`);
     - Bổ đề 3 và Mệnh đề 4′ cho lịch quét thật (§2.6);
     - rà độc lập: 1 BLOCKING, 2 MAJOR, 8 MINOR, 6 COSMETIC, đã sửa hết (`V2/reviews/W1_REVIEW_INDEPENDENT_01.md`).
   - **Đã làm thêm 05/11 chiều (§2.7, chưa rà độc lập):**
     - Bổ đề 3* cho ngữ cảnh lấy mẫu con;
     - giả định tập điểm xấu (H*) thay cho dung sai đều theo khối;
     - hằng số tinh chỉnh (chia tối ưu, Ville, Bernstein/Freedman);
     - mô phỏng max_context.
   - **Đã làm thêm 05/11 tối:** Mệnh đề 6, mô-men dừng chính xác (§2.8); Mệnh đề 5, R2 ở mức dự đoán (§5.7).
   - **Rà độc lập lần 2** cho §2.7, §2.8 và §5.7 (`V2/reviews/W1_REVIEW_INDEPENDENT_02.md`): 1 BLOCKING (câu đề xuất cho bài ở §5.7), 2 MAJOR, 11 MINOR, 5 COSMETIC. **Đã sửa hết.** Các kết quả toán đều đứng vững; Mệnh đề 6 và Mệnh đề 5(a)–(c) ở mức L4–L5.
   - **Còn mở, không phụ thuộc V1:**
     - thời gian thoát ở (d) cho lịch quét khối ngẫu nhiên (đã có bản chặt (d′) cho lịch quét thật);
     - Mệnh đề 6 với h ≠ 0 và ρ < 1;
     - R2 cho PLL, PU (nhãn yếu nghiêng) và LLP;
     - LLP trong Mệnh đề 4′.
   - Nếu đưa Mệnh đề 4′ vào bài, nên có một lượt rà độc lập thứ hai trên bản tiếng Anh.
2. **Đo Φ_chuỗi trên GPU.** Gieo lỗi theo Q̄ đã lưu của V1, rồi so với Φ_đều. Mục đích là kiểm δ⁺ và tiêu chí sai số điểm cuối ≤ 0,1. Cần thầy duyệt trước khi viết code: khoảng 1 phiên Kaggle.
3. **R1a:** đã làm cả phần LLP (§5.5). Còn R1b trên GPU, theo kế hoạch của REVIEW_RESPONSE_PLAN.
4. **Đưa vào bài, sau V1:**
   - Mệnh đề 4′ (lịch quét thật) vào mục lý thuyết, chứng minh ở phụ lục. Mệnh đề 4 (lịch quét ngẫu nhiên) chỉ cần nếu bài muốn chặn ở từng bước;
   - bảng δ̂ vào phần ánh xạ phản ứng;
   - giữ Mệnh đề 3(b) tổng quát, thay đoạn "Why this is vacuous" bằng E7b và E7d, và thêm Mệnh đề 5 như một mệnh đề riêng có phạm vi SSL (§5.7);
   - Mệnh đề 6 vào phần ánh xạ phản ứng, kèm phép kiểm trên TFM thật khi có Φ_chuỗi.

## 7. Tệp

| Tệp | Nội dung |
|---|---|
| `V2/reviews/W1_SELF_REVIEW_01.md` | tự rà toán lần 1: sổ lỗi, phán quyết theo kết quả |
| `V2/reviews/W1_REVIEW_INDEPENDENT_01.md` | rà độc lập §1–§2: sổ lỗi, phản ví dụ, phán quyết theo kết quả |
| `G4/code/experiments/w1_drift_check.py` | TFM giả biết Φ (`--h`); `theory`, `partition` |
| `G4/results/w1/*_h002.csv`, `w1_h002_log.txt` | **lượt chính**, h_max = 0,02, không clip: vết, tóm tắt, dải, thời gian tới, chặn 4′ |
| `G4/results/w1/w1_drift_runs.csv`, `w1_drift_summary.csv`, `w1_theory_check.csv`, `w1_hitting.csv`, `w1_partition_bound.csv` | lượt đầu, h_max = 0,1, có clip |
| `G4/results/w1/w1_context_cap_h002.csv`, `w1_context_cap_runs_h002.csv`, `w1_context_cap_log.txt` | mô phỏng max_context (§2.7) |
| `G4/results/w1/w1_partition_bound_refined.csv` | hằng số tinh chỉnh của Bổ đề 3 (§2.7) |
| `G4/results/w1/w1_second_order_h002.csv` | Mệnh đề 6 so với mô phỏng (§2.8) |
| `G4/results/r1a/e7f_affine_check.csv`, `e7f_affine_closed_form.csv`, `e7f_alpha_prime_check.csv` | Mệnh đề 5: kiểm dạng đóng, dạng α′ và tiệm cận (§5.7) |
| `V2/analysis/r2_asymptotic.py` | sympy: số hạng chính của Mệnh đề 5(c), C tổng quát |
| `V2/reviews/W1_REVIEW_INDEPENDENT_02.md` | rà độc lập lần 2 (§2.7, §2.8, §5.7) kèm phản hồi |
| `V2/analysis/w1_delta.py`, `V2/results/w1_delta.csv` | δ trên TFM thật, từ dữ liệu V0 |
| `G4/code/experiments/r1a_block_bias.py` | E7a `tc`, E7b `thm2`, E7c `moments`, E7d `decomp`, E7e `llp` |
| `G4/results/r1a/e7a_block_tc.csv`, `e7b_thm2_block_bound.csv`, `e7c_moments.csv`, `e7d_bound_decomposition.csv`, `e7e_llp_bag_block.csv`, `e7b_log.txt`, `e7e_log.txt` | kết quả R1a |

**Tài liệu dẫn** (đều là kết quả kinh điển; trước khi đưa vào bài phải kiểm theo REFERENCE_CHECK):
- Hoeffding, W. (1963), *Probability inequalities for sums of bounded random variables*, JASA 58:13–30, §6 (mẫu không hoàn lại).
- Ville, J. (1939), bất đẳng thức cực đại cho siêu martingale.
- Azuma, K. (1967), *Weighted sums of certain dependent random variables*, Tôhoku Math. J. 19:357–367 (dùng ở Bổ đề 3).
- Freedman, D. A. (1975), *On tail probabilities for martingales*, Ann. Probab. 3:100–118 (dùng ở §2.7).
- Bản phác thảo trước của Bổ đề 3 định dùng Serfling (1974). Bản chứng minh hiện tại cộng trên K khối, nên không cần.
- Seneta (1988) và Cho & Meyer (2001) đã có trong bài, cho τ₁(Z).
