# Trả lời reviewer ngoài 01: đánh giá và kế hoạch sửa (lập 04/11/2026)

- **Nguồn:** [reviews/REVIEW_EXTERNAL_01.md](reviews/REVIEW_EXTERNAL_01.md). Nguyên văn, bản nháp dựng từ commit `2ff12b4`.
- **Trạng thái:**
  - Mục R9 và R10 đã sửa trong bài ngày 04/11.
  - R1–R8 **chờ thầy chọn** (§4).
- **Quy tắc khi làm:**
  - Mọi lượt mới đều là **thăm dò** (exploratory). Study 2 đã xong và không được chạy lại theo nghĩa xác nhận.
  - Mỗi lượt có script phân tích viết trước khi xem số, kèm quy tắc diễn giải ghi trước (§3). Chạy xong thì ghi vào `S2/EXPERIMENT_LOG.md` hoặc một nhật ký mới.
  - Không bỏ, không giấu kết quả bất lợi.

## 1. Tóm tắt đánh giá

1. **Điểm chính của reviewer là đúng.**
   - Định lý chỉ bao phủ các bộ lấy mẫu chính xác: một mẫu, hoán đổi, khối có sửa MH.
   - Kết quả trong bài lại dùng khối không sửa, K = 10, kèm bag-block.
   - So sánh với bản MH ở cùng 30 vòng quét không tách được hai nguồn sai: độ lệch của bản xấp xỉ và độ trộn chậm.
   - Bài **đã tự nói** các ý này: đoạn "Blocks and scope" ở mục 5, mục Limitations, và phụ lục A mục "Why this is vacuous". Nhưng nói ra không trả lời được câu "lợi ích có phải do xấp xỉ không". Muốn trả lời phải làm thí nghiệm R1, và nếu được thì có lý thuyết R2.
2. **Có điểm reviewer nói chưa đủ bối cảnh.**
   - Hai TFM **đã** chạy trên cùng 24 dataset của Study 1, cùng giao thức (phụ lục B, Bảng app_s1_tabicl), cùng với ánh xạ phản ứng. Chỉ Study 2 là thiếu TabPFN.
   - Chênh hai chuỗi 0,018–0,021 lấy từ **3 dataset nhỏ nhất mỗi nhóm của Study 1**.
     - Tập test có 150–221 điểm, nên một điểm test bằng 0,005–0,007 accuracy.
     - Đây không phải số đo trên Study 2.
     - Mỗi seed của Study 2 chạy một chuỗi riêng, nên phương sai Monte Carlo đã nằm trong kiểm định Wilcoxon: nó làm giảm lực kiểm định, không đẩy trung bình về phía dương.
     - Dù vậy vẫn nên đo trực tiếp (R4).
3. **Một đề xuất em không theo:** đưa khối có sửa MH vào phương pháp chính.
   - Khẳng định xác nhận của Study 2 là về **cấu hình mặc định đã đăng ký**. Đổi phương pháp chính sau khi có kết quả thì Study 2 mất tư cách xác nhận.
   - Bản MH lại không dự đoán tốt hơn mà chậm hơn trung vị 7 lần, có ô tới 206 lần.
   - Hướng đúng: trình bày bản mặc định là xấp xỉ nhanh của bản chính xác, rồi **đo** độ lệch của xấp xỉ đó (R1, R2).
4. **Reviewer phát hiện đúng một lỗi trình bày:** checklist ghi giấy phép **[Yes]** trong khi còn TODO xác nhận. Đã sửa (R9).
5. **Lỗi em tự phát hiện khi rà lại:** câu "ở Lost bốn phương pháp đầu cách nhau < 0,003" sai.
   - Thực tế chỉ **ba** phương pháp: Gibbs 0,7496; PRODEN tune 0,7478; GLWS tune 0,7469. CC tune 0,7230 kém hơn 0,027.
   - Reviewer chép lại câu sai này. Đã sửa trong bài (nay sinh bằng macro `s1-real-lost-top3`) và trong G8_REPORT.

## 2. Đối chiếu từng ý

| # | Ý của reviewer | Đánh giá | Bài đang viết gì | Việc |
|---|---|---|---|---|
| 1 | Bổ đề 1 chỉ đúng khi TFM là PPD chính xác; TFM thật chỉ xấp xỉ | Đúng | Bổ đề phát biểu cho PPD; Định lý 2 lo trường hợp xấp xỉ | — |
| 2 | Định lý 1(c): chỉ bản chuyển nạp chính xác; bản quy nạp lệch tối đa TV(·), bằng 0 khi prior tách được | Đúng, bài đã viết đúng như vậy | Định lý 1(c). Bản chuyển nạp chính xác lại kém hơn (−0,013 AUC LLP, phụ lục F) | — |
| 3 | π̃ duy nhất ≠ posterior Bayes khi xác suất có điều kiện không tương thích | Đúng; bài chưa nói thẳng | — | **R10, đã sửa:** thêm câu sau Định lý 2 |
| 4 | Chặn của Định lý 2 phụ thuộc τ(P); chưa kiểm trên dữ liệu thật | Đúng | RQ1 kiểm trên bài nhỏ: chặn đúng 42/42 cấu hình, lỏng 1,9–15 lần. Trên dữ liệu thật không tính được τ, ε̄ | R2 (lý thuyết); ghi rõ trong Limitations (đã có) |
| 5 | Đề xuất khối là tích các biên, khác xác suất có điều kiện chung của khối, kể cả khi TFM chính xác | Đúng; đây chính là Mệnh đề 3(b) | Phụ lục A: chặn hiện tại O(n/K²), log₁₀κ_B ≈ 15,8, rỗng | **R1, R2** |
| 6 | Bản MH không tốt hơn ≠ bản mặc định không lệch; bị trộn lẫn với độ trộn | Đúng; Limitations đã thừa nhận "bias not isolated" | §6.6 và Limitations | **R1** |
| 7 | So sánh ở cùng ngân sách tính toán và ở cùng mức hội tụ | Đồng ý | — | **R1** (đường theo số forward pass; chuỗi dài) |
| 8 | Study 1 thất bại và ánh xạ phản ứng là đóng góp phụ có giá trị; nên có lý thuyết thống nhất (độ bền với nhiễu, hội tụ, chất lượng suy luận) | Đồng ý; trùng D2 | §6.4; TODO D2 | D2 (hạn 20/12); R2 |
| 9 | Lợi thế PLL nhỏ (+0,012), gần oracle; có ý nghĩa thực tiễn không | Đúng một phần | Bài nêu chênh tới oracle. Số mô tả em tính thêm (chưa đưa vào bài): Gibbs **thu hẹp 60% (q 0,3) và 39% (q 0,6) khoảng cách từ CC tune tới oracle**; so với TFM ngây thơ: +0,022 và +0,205; LLP thu hẹp 57% | Tuỳ thầy: thêm cách đọc "phần khoảng cách tới oracle được thu hẹp" như số mô tả |
| 10 | Chỉ 3 bộ PLL thật; không có LLP thật | Đúng | Limitations (Scope) | **R6** |
| 11 | Cơ chế sinh nhãn yếu lý tưởng hoá (distractor độc lập, túi ngẫu nhiên cỡ 8) | Đúng. Cách đọc có lợi: Mệnh đề 2(a) nói TFM ngây thơ nhất quán về xếp hạng **đúng ở các cơ chế này**. Với cơ chế phụ thuộc feature, (a) không còn đúng, nên lý thuyết dự đoán lợi thế lớn hơn | Kết luận đã nêu là hướng mở | **R6(b)** |
| 12 | R̂ > 1,1 ở 6–31% nhãn; dự đoán ổn định ≠ posterior hội tụ; cần ESS, tự tương quan, MCSE, trace, nhiều điểm khởi động, split-R̂ chuẩn hoá theo hạng, độ nhạy theo T và K | Đúng | §6.6 và Limitations ("not draws from a converged posterior") | **R3** |
| 13 | Chênh hai chuỗi 0,018–0,021 ≥ lợi thế PLL 0,012 | Cần đo; nhưng số đó lấy từ 3 dataset nhỏ nhất của Study 1 (xem §1.2) | **R10, đã sửa:** §6.6 nói rõ số lấy từ 3 dataset nhỏ nhất và kiểm định của Study 2 đã gồm biến thiên Monte Carlo | **R4** |
| 14 | Lấy mẫu con ngữ cảnh 2.000 điểm làm lệch Bổ đề 1, chưa phân tích | Đúng. Có ở **14/28** dataset của Study 2 (10/18 LLP, 4/10 PLL; 13 trong bước lấy mẫu, jasmine chỉ ở bước dự đoán) và 9/24 của Study 1 (4/10 LLP, 5/14 PLL) | **R10, đã sửa:** Limitations nói rõ số dataset và rằng Định lý 2 chỉ phủ qua ε̄ | **R7**. Số mô tả ở §3.R7 |
| 15 | So sánh TabPFN/TabICL chưa đối xứng | Đúng một phần (xem §1.2) | Limitations: "not a controlled comparison of TFMs" | **R5** |
| 16 | Chưa đánh giá định lượng rủi ro riêng tư của LLP | Đúng | Broader impacts chỉ nêu định tính | **R9, đã sửa:** thêm câu "chưa đo". **R8** |
| 17 | Còn TODO (D2, LLM, giấy phép); checklist ghi giấy phép [Yes] khi chưa xác nhận | Đúng | — | **R9, đã sửa:** `\pending{}`. Bản final **không biên dịch được** khi còn mục này |
| 18 | Tính mới lý thuyết trung bình; dạng chặn là cổ điển | Bài đã tự nói "Its form is classical"; cái mới là xác suất có điều kiện không tương thích, có bước MH, không tiệm cận | §5 | R2 sẽ nâng phần này |
| 19 | Thí nghiệm giai thừa TFM × bộ lấy mẫu × ngữ cảnh | Đồng ý với câu hỏi; đề xuất tách đôi (xem dưới) | — | **R1** (TFM × bộ lấy mẫu, dữ liệu nhỏ) + **R7** (TFM × trần ngữ cảnh, dữ liệu lớn) |
| 20 | Đừng chỉ thêm 20–30 benchmark | Đồng ý | — | R1, R2, R4 trước R6 |
| 21 | Đưa MH vào phương pháp chính | Không làm (xem §1.3) | — | Bản MH là biến thể chính xác; đo độ lệch của bản mặc định |

**Lý do tách thí nghiệm giai thừa (ý 19):**
- Bản MH chậm hơn trung vị 7 lần, có ô tới 206 lần.
- Nếu chạy MH với ngữ cảnh đầy đủ trên dataset 7.000 điểm thì mỗi ô mất nhiều giờ.
- Vì vậy bộ lấy mẫu được so ở dữ liệu nhỏ, nơi ngữ cảnh vốn đã đầy đủ (R1), còn trần ngữ cảnh được so ở dữ liệu lớn với bộ mặc định (R7).
- Nếu cần một ô giao (MH × ngữ cảnh lớn), chạy trên 1–2 dataset.

## 3. Kế hoạch R1–R10

Chi phí GPU ước tính từ thời gian thật (T4):
- TabICL v2, `gibbs_warm` trên Study 2: tổng 8.139 giây cho 123 ô (khoảng 2,3 giờ); trung vị 36 giây mỗi ô.
- Bản MH trên dataset nhỏ: trung vị 130–290 giây mỗi ô ở T = 30.
- TabPFN v2 trên Study 1: trung vị 205 giây mỗi ô.

Mọi ước tính đều thô; sẽ đo lại bằng một lượt thử nhỏ.

### R1 (P0) — Tách độ lệch của khối không sửa khỏi độ trộn

**Câu hỏi:** ở trạng thái gần dừng, bản mặc định có dự đoán khác bản chính xác không? Ở cùng ngân sách tính toán, bản nào tốt hơn?

**R1a — trên bài giải tích (CPU, rẻ, không cần TFM):**
- Mở rộng RQ1: thêm nhân khối không sửa và bag-block không sửa, dùng PPD chính xác.
- Đo TV(π̃_khối, π) và sai số dự đoán theo n và K.
- Dùng bài gộp (pooled) giải được chính xác, đã có E5: ở n = 1.000, K = 10, sai dự đoán là 1,7·10⁻⁴. Thêm prior hỗn hợp có feature mang thông tin.
- Mục đích: kiểm thang O(d/K²) của hướng R2.

**R1b — trên dữ liệu thật (GPU):**
- **Dataset:** 6 dataset R̂ của Study 1 (n_train 350–515) và 4 dataset nhỏ nhất của Study 2 (profb, socmob cho LLP; website_phishing, anneal cho PLL). Ở các dataset này ngữ cảnh đầy đủ, không lấy mẫu con.
- **Bối cảnh:** LLP túi 8; PLL q 0,3 và 0,6.
- **TFM:** TabICL v2 và TabPFN v2.
- **Bộ lấy mẫu:**
  - mặc định (K = 10, không sửa);
  - khối có sửa MH;
  - một mẫu (single-site), chỉ trên mẫu con n_train ≈ 200, vì mỗi vòng quét tốn n forward pass.
- **Chuỗi:** 4 chuỗi mỗi bộ lấy mẫu, với điểm khởi động khác nhau: khởi động ấm; TFM ngây thơ; ngẫu nhiên thoả ràng buộc; khởi động ấm seed khác.
  - Bản mặc định chạy T = 300.
  - Bản MH chạy cùng số forward pass với bản mặc định T = 300, và dài hơn nếu ngân sách cho phép.
- **Lưu:**
  - vết nhãn mỗi vòng quét (int8);
  - xác suất test mỗi vòng quét (float16), hoặc ít nhất là chỉ số test mỗi vòng quét.
- **Đo:**
  1. **Δ_eq** = chỉ số test của trung bình Rao–Blackwell trên các vòng quét 100–300 (mặc định) trừ cùng chỉ số của bản MH. MCSE tính qua các chuỗi.
  2. **Độ lệch ở mức nhãn:** trung bình theo nhãn huấn luyện của TV giữa biên của hai bộ lấy mẫu, so với cùng đại lượng giữa các chuỗi của cùng một bộ lấy mẫu (sàn nhiễu).
  3. **Đường cân bằng tính toán:** chỉ số theo số forward pass và theo giây.
  4. R̂ và ESS, dùng cho R3.
- **Quy tắc diễn giải** (bản nháp, thầy duyệt trước khi chạy):
  - **"Xấp xỉ không đáng kể":** khoảng tin cậy 90% của trung bình Δ_eq nằm trong ±0,01 ở cả ba bối cảnh, và độ lệch mức nhãn không vượt sàn nhiễu quá 2 lần MCSE. Khi đó viết "đã đo, không đáng kể".
  - **Δ_eq > 0 rõ:** bản mặc định tốt hơn bản chính xác ở trạng thái dừng. Lợi ích có phần đến từ xấp xỉ. Phải báo cáo và đổi cách viết, ví dụ "bản mặc định nhắm một phân phối khác, dự đoán tốt hơn", và đưa vào mục cơ chế.
  - **Δ_eq < 0 rõ nhưng bản mặc định thắng khi cân bằng tính toán:** đây là đánh đổi tốc độ và chính xác. Báo cáo đúng như vậy.
- **Chi phí:** khoảng 15–30 giờ GPU cho cả hai TFM, tức 3–4 phiên Kaggle (2 × T4). Phần TabPFN chiếm nhiều nhất. Có thể chạy TabICL trước.
- **Code cần viết:**
  - một phần mới trong `run_g7.py` (ví dụ `--part equil`): nhiều chuỗi, chọn điểm khởi động, lưu vết; dùng lại `GibbsSampler` với cờ MH;
  - lượt mới trong `run_s2_lot.sh`;
  - script phân tích viết trước, ví dụ `S2/analysis/e7_report.py`;
  - test.

### R2 (P0, lý thuyết) — Chặn khác rỗng cho cập nhật khối không sửa

**Hiện trạng:**
- Mệnh đề 3(b) cộng TV theo từng cặp nhãn (lập luận lai, hybrid). Với β = O(1/n), chặn là m_f(m−1)β = O(n/K²), tăng theo n.
- Với bag-block, hệ số κ_B làm chặn rỗng.

**Hướng chứng minh** (phác thảo, **chưa kiểm**):
1. **Đổi lập luận.** Thay lập luận lai TV bằng quy tắc dây chuyền của KL: KL(p(y_B | D_−B) ‖ ∏ p(y_i | x_i, D_−B)) bằng total correlation TC = Σ_i I(y_i ; y_{B,<i} | D_−B).
2. **Prior tham số đều, d chiều.**
   - Theo tiệm cận Clarke–Barron, I(y_mới ; θ | D_M) ≈ (d/2) log((M+1)/M).
   - Suy ra I(y_i ; y_{<i} | D_N) ≈ d(i−1)/(2N²), và TC ≈ d·m²/(4N²) = **d/(4(K−1)²), không phụ thuộc n**.
   - Pinsker cho TV ≲ √(d/8)/(K−1). Ví dụ d = 20, K = 10: TC ≈ 0,06 nat, TV ≲ 0,18.
3. **Nhãn yếu.** Với nhãn yếu dạng tích (PLL), làm trực tiếp trên phân phối đã nghiêng theo likelihood: posterior dưới nhãn yếu vẫn i.i.d. có điều kiện theo θ. Như vậy tránh được κ, thứ làm đường TV rỗng.
   - Phần chênh giữa biên của phân phối chung và biên của tích là Σ I(y_i ; W_{B∖i} | D, w_i), cũng O(d·m²/N²).
4. **LLP (điều kiện theo số đếm).** Cần lập luận riêng, vì điều kiện hoá bằng hàm chỉ thị chia cho xác suất khớp số đếm (lại ra κ_B). Một hướng là chặn qua I(θ ; c_B | D_−B), lượng thông tin số đếm của khối mang về θ.

**Rủi ro:**
- Prior của TFM không hữu hạn chiều, nên phải thay d bằng một đại lượng information gain γ_N, có thể tăng theo N.
- Định lý 2 còn nhân với τ(P), có thể lớn. Một lập luận ở mức dự đoán (trung bình Rao–Blackwell) có thể tốt hơn.
- Phần trên chỉ cho TFM chính xác. Khi q ≠ p phải cộng thêm sai số ε của Định lý 2.

**Kiểm bằng số:** R1a.

**Hạn:** song song với D2, tới 20/12. Không xong thì giữ đoạn "Scope" trung thực như hiện tại.

### R3 (P1) — Chẩn đoán trộn đầy đủ

- Dùng vết của R1 và R4. Tính:
  - split-R̂ chuẩn hoá theo hạng (Vehtari et al. 2021), ESS bulk và tail cho từng nhãn (one-hot theo lớp) và cho các đại lượng vô hướng: chỉ số test của bộ dự đoán một trạng thái ở mỗi vòng quét, số nhãn đổi mỗi vòng quét;
  - hàm tự tương quan;
  - MCSE của chỉ số test sau trung bình Rao–Blackwell (batch means hoặc qua các chuỗi);
  - trace plot cho vài nhãn đại diện: bất định nhất và trung vị;
  - độ nhạy theo T ∈ {30, 60, 120, 300} và K ∈ {5, 10, 20}.
- **Ghi trong bài:** với TFM không tương thích, các chẩn đoán này đo hội tụ về π̃, không phải về π.
- **Trình bày:** phụ lục, cộng 1–2 câu ở §6.6.

### R4 (P1) — Chạy lại Study 2 với seed chuỗi khác

**Công cụ có sẵn:**
- `run_rhat` (`icgibbs/g7/methods.py`) đã chạy hai chuỗi cùng lần chia, cùng nhãn yếu, cùng điểm khởi động, chỉ khác seed lấy mẫu (seed và seed + 1000).
- Cần gỡ chốt chặn `--part rhat` trên `--split s2` (`run_g7.py` khoảng dòng 170), nhưng **chỉ cho lượt thăm dò này**, bằng một cờ riêng.
- Lưu thêm vết, dùng cho R3 và R8.

**Phạm vi:** mọi ô của Study 2, tức 114 ô: 18 LLP × 3 seed, cộng 10 PLL × 2 mức × 3 seed.

**Phân tích** (script viết trước):
- (a) tính lại H2′ và H4′ với chuỗi b thay chuỗi a;
- (b) |a − b| theo ô, và MCSE của chênh trung bình theo dataset;
- (c) số dataset đổi dấu chênh so với đối thủ chính;
- (d) phân bố R̂ trên Study 2.

**Chi phí:** khoảng 4,5 giờ GPU nếu chạy hai chuỗi, hoặc khoảng 2,3 giờ nếu chỉ chạy chuỗi b. Chia hai GPU thì xong trong **một phiên**.

**Giá trị:** cao nhất so với chi phí. Trả lời thẳng ý 13 trên chính dữ liệu xác nhận.

### R5 (P1) — TabPFN v2 trên 28 dataset của Study 2

**Cách làm:**
- Chỉ chạy phần TFM: `gibbs_warm`, `naive`, `oracle`; thêm `hard_em_warm` nếu cần H3.
- Baseline không phụ thuộc TFM và đã có, nên dùng lại.
- Mọi dataset của Study 2 có ≤ 10 lớp, vừa giới hạn của TabPFN v2.
- Token chỉ qua Kaggle Secrets `TABPFN_TOKEN`.

**Kết quả thu được:** so sánh đối xứng hai TFM trên 52 dataset (24 + 28).

**Ghi trước dự đoán** (nên làm; commit có dấu thời gian trước khi chạy):
- Theo ánh xạ phản ứng, TabPFN v2 sẽ không đạt H4′ ở q 0,6.
- Phần thua tập trung ở 4 dataset có ≥ 8 lớp: MIC 8, JapaneseVowels 9, cardiotocography 10, yeast 10.
- Nếu dự đoán đúng, cơ chế C4 chuyển từ quan sát sau sự kiện thành dự đoán đã được kiểm.

**Chi phí:**
- `gibbs_warm` khoảng 7–8 giờ GPU; `naive` và `oracle` khoảng 1 giờ; `hard_em` thêm khoảng 4 giờ.
- Tổng 2 phiên × 2 GPU.
- Cần một lượt mới (ví dụ `S2P`) với `--backend tabpfn`, cấu hình giống Study 1.

### R6 (P1) — Nhãn yếu thật và cơ chế thực tế hơn

**(a) PLL thật thêm.**
- FG-NET (78 lớp), Soccer Player (171 lớp, 17.472 hàng) và Yahoo! News (219 lớp) đã bị loại theo tiêu chí viết trước: ≤ 26 lớp, n ≤ 10.000, d ≤ 500 (G5/DATASET_CARDS).
- Mirflickr có d = 1.536 > 500, cũng không đạt (cần kiểm lại số).
- Muốn dùng phải nới tiêu chí. Việc này là sai lệch so với giao thức, chỉ chấp nhận như thăm dò.
- **Ứng viên cần kiểm** (chưa đối chiếu):
  - dữ liệu bảng có nhiều người gán nhãn, với tập ứng viên là hợp các nhãn của người gán, ví dụ dữ liệu thể loại nhạc của Rodrigues et al.;
  - phải kiểm nguồn, kích thước, giấy phép, và đưa vào REFERENCE_CHECK trước khi trích.

**(b) Cơ chế phụ thuộc feature** (bán thật, nên làm trước (a)):
- **PLL:** distractor phụ thuộc mẫu, lấy từ các lớp dễ nhầm theo một mô hình phụ.
- **LLP:** túi gom theo một thuộc tính phân loại (vùng, nhóm) hoặc theo k-means.
- Mệnh đề 2(a) không còn áp dụng, nên lý thuyết dự đoán lợi thế so với TFM ngây thơ tăng. DLLP vẫn hợp lệ, nên so với DLLP là phép thử công bằng.

**(c) LLP thật.** LLP-Bench (Criteo) bị bỏ vì không có dữ liệu (deviation 14). Thử lại nếu tải được; không thì dùng (b).

**Ưu tiên:** sau R1, R4, R5.

### R7 (P2) — Ảnh hưởng của trần ngữ cảnh

**Số mô tả có sẵn** (thăm dò, tính 04/11, chưa đưa vào bài; nhóm "lấy mẫu con" là n_train > 2.000):

| Bối cảnh | Ngữ cảnh đầy đủ | Có lấy mẫu con |
|---|---|---|
| LLP túi 8, so DLLP tuyến tính | 8 dataset, TB +0,058, trung vị +0,024, thắng/thua 7/1 | 10 dataset, TB +0,047, trung vị +0,022, 9/1 |
| PLL q 0,3, so CC tune | 6 dataset, TB +0,012, 4/0 | 4 dataset, TB +0,013, 4/0 |
| PLL q 0,6, so CC tune | 6 dataset, TB +0,007, 4/1 | 4 dataset, TB +0,020, 4/0 |

- Lợi ích có ở cả hai nhóm.
- Nhưng nhóm trùng với cỡ dữ liệu, nên **không tách được** ảnh hưởng của trần ngữ cảnh.

**Thí nghiệm:**
- 14 dataset của Study 2 có n_train > 2.000; trần ∈ {1.000, 2.000 (hiện tại), 4.000, đầy đủ ≤ 7.000}; bộ mặc định; 3 seed; TabICL, thêm TabPFN nếu chạy R5.
- Theo Định lý 2, lấy mẫu con ngẫu nhiên là một xác suất có điều kiện không tương thích, sai số nằm trong ε̄.

**Chi phí:** khoảng 6–10 giờ GPU (ngữ cảnh đầy đủ đắt hơn 2–4 lần). Cần cờ `--max_context`.

### R8 (P2) — Định lượng rủi ro riêng tư của LLP

**Mô hình đe doạ:**
- Bên phát hành công bố feature của từng người cùng tỉ lệ nhãn theo nhóm.
- Kẻ tấn công có các dữ liệu đó và một TFM công khai, muốn suy ra nhãn của từng người.

**Đo:**
- AUC hoặc accuracy khi khôi phục nhãn **huấn luyện** từ biên posterior.
- So với: (i) chỉ dùng tỉ lệ của túi; (ii) TFM ngây thơ; (iii) DLLP tuyến tính.
- Lợi thế = chênh so với (i). Xem theo cỡ túi 8 / 16 / 32.
- Tuỳ chọn: thêm nhiễu label-DP vào số đếm.

**Chi phí:** túi 8 gần như miễn phí nếu R4 lưu vết. Túi 16 và 32 cần chạy thêm (Study 1 có các mức này với TabPFN).

**Trình bày:** một đoạn phụ lục, và sửa câu ở Broader impacts.

### R9 (đã làm 04/11) — TODO và checklist

- Thêm macro `\pending{}` trong `main.tex`. Bản nháp hiện chữ đỏ. Bản final dừng biên dịch với lỗi "Unresolved checklist answer", nên không thể nộp khi chưa xác nhận.
- Checklist mục 12 (giấy phép): đổi **[Yes]** thành `\pending{…}`, giữ câu trả lời nháp. "TabICL" sửa thành "TabICL code".
- Broader impacts: thêm "We have not measured how much individual-level information the sampler recovers from bag counts", kèm `\todo` trỏ R8.
- **Còn mở:** D2 (ngưỡng sụp, hạn 20/12); chính sách LLM của NeurIPS 2027 (khi có call); xác nhận giấy phép từng tài sản.

### R10 (đã làm 04/11) — Viết cho đúng phạm vi

- **Mục 5, sau Định lý 2:** "When the conditionals are incompatible, π̃ is in general not the posterior π; the theorem bounds how far it is."
- **Giới thiệu, ý đóng góp lý thuyết:** "the sampler targets the Bayes posterior" sửa thành "the exact variants of the sampler target the Bayes posterior".
- **Kết luận:** "exact with an exact TFM" sửa thành "with exact variants for an exact TFM".
- **§6.6:** chênh hai chuỗi là số trên 3 dataset nhỏ nhất mỗi nhóm; mỗi seed của Study 2 có chuỗi riêng, nên kiểm định xác nhận đã gồm biến thiên Monte Carlo. Câu "thirty sweeps give stable predictions" ở §6.6 đã bỏ; Limitations vẫn giữ.
- **Limitations:** lấy mẫu con ngữ cảnh có ở 14/28 dataset của Study 2 (macro `s2-ctxsub-n`), là một xấp xỉ khác, Định lý 2 chỉ phủ qua ε̄, chưa tách được.
- **§6.5:** sửa "top four" thành "top three" (macro `s1-real-lost-top3`).
- **Phụ lục B:** "oracle within 0.005" nay sinh bằng macro `s1-tabicl-oracle-maxdiff`. Giá trị thật là 0,0052, làm tròn 0,005; câu viết là "about equally accurate".
- **Bố cục:** phần chính vẫn vừa 9 trang (kết luận hết ở trang 9), không có overfull box. Để lấy lại dòng, đã rút gọn câu ở giới thiệu, công trình liên quan, mục 5 và §6.6.

## 4. Đề xuất thứ tự và việc cần thầy quyết

| Thứ tự | Việc | GPU | Vì sao trước |
|---|---|---|---|
| 1 | **R4** (+ lưu vết cho R3, R8) | ~2,5–4,5 h, 1 phiên | Rẻ nhất; trả lời ý MCSE trên chính dữ liệu xác nhận |
| 2 | **R1a** (CPU, trong container) + **R2** song song với D2 | 0 | Kiểm hướng chặn O(d/K²) trước khi tốn GPU |
| 3 | **R1b** (TabICL trước, rồi TabPFN) | ~15–30 h, 3–4 phiên | Điểm P0 của reviewer |
| 4 | **R5** (ghi trước dự đoán) | ~8–13 h, 2 phiên | So sánh TFM đối xứng; kiểm dự đoán của cơ chế |
| 5 | **R7** | ~6–10 h | P2; nên chạy cùng đợt với R5 |
| 6 | **R6(b)**, rồi **R6(a)/(c)** | tuỳ | Tổng quát hoá; theo lời reviewer, không ưu tiên hơn P0 |
| 7 | **R8** | ~0 (từ vết R4) | P2 |

**Cần thầy quyết:**
- **Q1.** Chọn những việc nào trong R1–R8, theo thứ tự trên hay thứ tự khác.
- **Q2.** Duyệt quy tắc diễn giải của R1 (biên ±0,01, KTC 90%).
- **Q3.** R5: có ghi trước dự đoán từ ánh xạ phản ứng không (em đề xuất **có**).
- **Q4.** Có thêm cách đọc "thu hẹp X% khoảng cách tới oracle" (số mô tả) vào §6.2 không.
- **Q5.** R6(a): có nới tiêu chí chọn bộ PLL thật (≤ 26 lớp, d ≤ 500) cho phần thăm dò không.

## 5. Tiến độ

| Ngày | Việc |
|---|---|
| 05/11/2026 | **R4 xong** (lượt V0A của V2; V2/V0_REPORT §3): chuỗi thứ hai trên 114/114 ô của Study 2. H2′ và H4′ q 0,3 bền; **H4′ q 0,6 không đạt với chuỗi b (Holm 0,074)**, đạt khi trung bình hai chuỗi. Đã đưa vào §6.2, §6.6, Limitations, Bảng tab:r4 (thầy duyệt 05/11). Từ 04/11, R1–R8 được thay bằng V2 W6 ([V2_PLAN](../V2/V2_PLAN.md)) |
| 04/11/2026 | Nhận nhận xét. Lưu nguyên văn. Đối chiếu từng ý. Sửa R9, R10 và lỗi "top four" ở Lost. Tính số mô tả về trần ngữ cảnh và về phần khoảng cách tới oracle. Lập kế hoạch R1–R8 |
