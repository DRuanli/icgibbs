# Phản biện kiểu NeurIPS 2027 (main track) — "In-Context Gibbs: Learning from Weak Labels with Frozen Tabular Foundation Models"

- **Bản được đọc:** `paper/main.pdf` biên dịch ngày 12/11/2026, 35 trang (chính văn trang 1–9, tài liệu tham khảo 10–12, phụ lục A–N 12–34, checklist 35). Số dòng trích dưới đây là số dòng in ở lề PDF (L…).
- **Cách đọc:** chỉ đọc PDF và, khi PDF mơ hồ, mã LaTeX trong `paper/sections/` (cùng `tables/`, `numbers.tex` để kiểm tra một con số trùng lặp). Không đọc kế hoạch, báo cáo hay bản phản biện trước của dự án. Chỗ nào bài không nói rõ thì được ghi là một phát hiện.
- **Tự kiểm của người phản biện:** đã đọc lại chứng minh Bổ đề 1, Định lý 1–3, Bổ đề 2; đã tính lại hằng số của Định lý 3 ở cỡ dữ liệu thật của bài (mục 3.2); đã cộng lại Bảng 11 để đối chiếu Bảng 2 (khớp). Không có internet: các nhận định về tài liệu sau giữa năm 2026 và về một số trị số trong văn liệu PLL được đánh dấu "không chắc".

---

## 1. Tóm tắt bài báo (theo cách hiểu của người phản biện)

Bài đề xuất dùng một tabular foundation model (TFM) **đóng băng** để học từ nhãn yếu, cụ thể là partial labels (PLL, mỗi điểm có một tập ứng viên) và label proportions (LLP, chỉ biết số đếm nhãn trong từng bag). Quan sát gốc: dưới prior khả hoán của PFN, phân phối của một nhãn ẩn khi biết mọi nhãn khác chính là posterior predictive với các điểm còn lại làm context (Bổ đề 1), nên một forward pass của TFM là full conditional của một Gibbs sampler trên nhãn ẩn; nhãn yếu chỉ đi vào qua likelihood. Thuật toán chạy thực tế ("In-Context Gibbs") chia ngẫu nhiên dữ liệu thành K=10 khối, mỗi sweep vẽ lại nhãn của từng khối từ dự đoán của TFM với context là các khối còn lại (không có bước Metropolis–Hastings), chạy 30 sweep, rồi lấy trung bình dự đoán trên 10 mẫu nhãn.

Lý thuyết gồm ba phần: (i) Định lý 1: các biến thể *chính xác* (single-site, swap-MH) có phân phối dừng là posterior Bayes khi TFM đúng bằng PPD; (ii) Định lý 2: cận nhiễu loạn kiểu Dobrushin cho TFM xấp xỉ, kể cả khi các conditional không tương thích; (iii) Giả thiết 1 + Định lý 3: nếu xác suất vẽ đúng nhãn của một khối chỉ phụ thuộc vào *tỉ lệ* nhãn đúng trong context qua một "response map" Φ (sai số δ), thì tỉ lệ nhãn đúng của chain bị chặn dưới/chặn trên bởi các điểm bất động của Φ, và thời gian thoát khỏi một basin là mũ theo số nhãn ẩn.

Thực nghiệm là ba nghiên cứu "pre-registered": Study 1 (24 dataset CC18, TabPFN v2) không đạt tiêu chí vì chain "stall" ở PLL q=0.6 nhiều lớp; Study 2 (28 dataset mới, TabICL v2) đạt: một chain hơn linear DLLP +0.052 AUC (LLP, 18 dataset) và hơn tuned CC +0.012 accuracy (PLL, 10 dataset); Study 3 (20 dataset mới, bốn TFM) kiểm định hai bước thêm vào sau: trung bình ba chain từ ba điểm khởi đầu (một trong đó là dự đoán của learner chuyên biệt L đã tune) và trung bình qua các TFM; hai giả thuyết đạt, hai tiêu chí đăng ký thất bại (chẩn đoán stalling không cần nhãn; tỉ lệ thua L khi chỉ dùng một TFM là 14.1%).

## 2. Điểm mạnh

1. **Ý tưởng gọn và đúng.** Bổ đề 1 đơn giản nhưng là một cách nhìn hữu ích: nó cho một thủ tục duy nhất cho mọi loại nhãn yếu có likelihood biết trước, không huấn luyện lại TFM, và nối TFM với data augmentation / multiple imputation. Theo hiểu biết của tôi, chưa có bài nào dùng TFM đóng băng như Gibbs conditional cho PLL/LLP (không chắc về các preprint 2026).
2. **Trung thực hiếm thấy về kết quả âm.** Study 1 trượt được in ngay trong Bảng 1; hai tiêu chí trượt của Study 3 có trong abstract; phụ lục G liệt kê 13 dự đoán viết trước, 5 cái sai; Bảng 14 tự phân rã và cho thấy lợi ích multi-start đến từ learner start; L306–309 tự báo rằng H4′ ở q=0.6 không còn đạt khi đổi seed lấy mẫu. Các bài nộp NeurIPS thông thường không làm vậy.
3. **Thiết kế tái lập trên dữ liệu mới.** Study 2 là một replication thật sự (dataset mới, đổi đúng một yếu tố), baseline được tune 20 cấu hình bằng validation chỉ có nhãn yếu, phương pháp không tune theo dataset. Bảng 7 so với mọi baseline chứ không chỉ comparator chính.
4. **Kiểm tra bằng số cho lý thuyết.** Liệt kê toàn bộ không gian trạng thái (n ≤ 14) cho Định lý 1–2, mô hình liên hợp cho Mệnh đề 3, mô phỏng cho Mệnh đề 4; bài tự nói thẳng cận (4) là vô nghĩa (vacuous) cho sampler mặc định.
5. **Mệnh đề 5 có ích cho người dùng:** nói trước khi nào naive TFM là đủ (PU-SCAR, nhiễu nhãn, SSL) và điều đó khớp với các setting phụ của Study 1 (Bảng 27).
6. **Kết quả C5″** (trung bình bốn TFM ngang với TFM tốt nhất của từng dataset, gap +0.000, chỉ 1/20 kém hơn 0.02) là một kết quả thực dụng tốt, dù không bất ngờ với ensemble.

## 3. Điểm yếu (xếp theo mức độ nghiêm trọng)

### 3.1. Phương pháp cuối cùng không còn "training-free", và bằng chứng xác nhận (confirmatory) không phủ phương pháp được khuyến nghị

- **LLP: ngay chain đơn đã khởi đầu từ một learner được huấn luyện.** L111–113: warm start cho LLP là *linear DLLP*, chính là comparator chính của H2/H2′. Vậy kết quả đầu bảng "+0.052 AUC over DLLP" thực chất là "tinh chỉnh nhãn của DLLP bằng TFM thì hơn DLLP". Điều này không sai, nhưng abstract ("without updating a weight"), related work L70 ("We replace training by inference") và Bảng 17 (cột "no training ✓") gây ấn tượng khác. Bài không nói linear DLLP ở warm start là bản mặc định hay đã tune.
- **Multi-start cần L đã tune 20 cấu hình** (DLLP-MLP, CC). L124–125 có nói, nhưng abstract chỉ viết "averaging chains from several starts", không nói một start là learner chuyên biệt.
- **Lợi ích multi-start đến hoàn toàn từ L** (Bảng 14): chain khởi từ L một mình = trung bình ba start (+0.001); warm+cold không có L = chain đơn (PLL 0.601 so với 0.589); *trung bình chain đơn với dự đoán của L, không chạy thêm chain nào*, cũng bằng (+0.008, p=0.26). Tức H1″ ("multi-start > single") gần như là "thêm baseline vào thì tốt hơn khi chain hỏng". Theorem 3 không cần thiết để giải thích điều đó.
- **Với một TFM, phương pháp đăng ký không hơn L có ý nghĩa thống kê** (+0.018, p=0.14, L286–287) và thua L hơn 0.02 ở 9/71 cặp. Chain đơn thua L hơn 0.02 ở 14/71 cặp (L783–784). Đây là kết quả quan trọng nhất của Study 3 đối với người dùng và nó nằm ở cuối một đoạn "Not confirmed".
- **So sánh multi-TFM với L (20/20, +0.064) là exploratory và không công bằng:** ensemble 3–4 foundation model × 3 chain (trong đó có chain khởi từ L) so với *một* MLP/linear đã tune. Không có ensemble của learner chuyên biệt, không có naive-TFM ensemble, không có ngân sách tính toán ngang nhau.
- **Hệ quả:** không có kiểm định xác nhận nào cho câu "phương pháp được khuyến nghị (multi-start/multi-TFM) hơn learner chuyên biệt". Xác nhận chỉ có cho chain đơn + TabICL v2 (Study 2), mà chính cấu hình đó lại stall ở Study 3 (soybean 0.589 so với L 0.788; seismic-bumps AUC 0.554, và multi-start còn xuống 0.491 < 0.5).

### 3.2. Định lý 3 và "response map": giả thiết không được kiểm, định lý vô nghĩa ở cỡ dữ liệu của bài, và phần "giải thích" mâu thuẫn với chính hình của bài

- **Giả thiết 1 gần như là kết luận.** Nó khẳng định động lực của label accuracy là một chiều (xác suất đúng của khối chỉ phụ thuộc r của context, với mọi trạng thái y và mọi khối B). Từ đó Định lý 3 chỉ còn là Hoeffding/Azuma + union bound quanh ánh xạ r ↦ Φ(r). Chứng minh đúng (tôi đã kiểm Bổ đề 2 và ba phần của định lý), nhưng nội dung toán học mỏng: "nếu động lực là 1 chiều thì nó có điểm bất động và basin".
- **"Với mọi trạng thái y" là quá mạnh và không kiểm được.** Hai labelling cùng r nhưng lỗi có cấu trúc khác nhau (ví dụ lỗi dồn vào một cặp lớp bị hoán đổi so với lỗi rải đều) chắc chắn cho xác suất đúng rất khác; δ dạng sup do đó có thể lớn tới mức làm định lý rỗng. **Bài không ước lượng δ ở đâu cả.** Mục "Does the assumption hold?" (L262–271) không kiểm giả thiết mà kiểm một hệ quả khác (điểm bất động của gain nội suy so với accuracy cuối).
- **Tính lại hằng số ở cỡ thật.** Với K=10, ρ≈1, h_max ≤ 0.5 (c = a²·2/9), để p(ν,λ) < 1/30 (tức cận "ở yên 30 sweep" không rỗng) cần biên ν+λ tối thiểu: mfeat-factors (n_F≈1396) 0.21; semeion (≈1115) 0.23; MiceProtein (≈756) 0.29; optdigits (≈3934, bỏ qua việc context bị cắt ở 2000) 0.13; n_F=350 là 0.42. Điều kiện của định lý đòi gain tại x∓ν vượt ν+λ+δ. **Gain đo được ở Hình 1 có trị tuyệt đối ≤ 0.15.** Vậy ngay cả với δ=0, Định lý 3 không chứng nhận được floor hay ceiling nào trên bất kỳ dataset nào trong bốn dataset minh hoạ. Phụ lục có thừa nhận "the proven invariant band has half-width 0.22 and the simulated 99% band 0.02" (L662–664), nhưng chính văn chỉ nói "The constants are loose". Câu "exponential in n_F" (L193, abstract L17–18, intro L52–53) đúng về hình thức nhưng không có trọng lượng định lượng.
- **Phạm vi:** không phủ LLP (một nửa thực nghiệm, và là nơi có lỗi hoán đổi lớp AUC<0.5), không phủ context bị subsample (>2000 điểm, tức phần lớn dataset lớn), chỉ cho TFM tất định.
- **Mâu thuẫn nội tại ở §6.3.** L258–259: "With TabPFN v2 it is negative for intermediate r on all four datasets: by Theorem 3(b), a chain that starts there cannot climb." Nhưng (a) L265–267 thừa nhận với TabPFN v2 "only the lower half of the assumption holds", tức đúng nửa (H⁺) mà Định lý 3(b) cần lại *không* đúng dưới uniform map; (b) Hình 2 (hàng dưới) cho thấy chain TabPFN v2 **có leo** qua vùng gain âm: optdigits 2/3 seed lên ~0.95, MiceProtein lên 0.5–0.65, semeion từ 0.17 lên ~0.38. Chỉ mfeat-factors là đứng yên. Vậy uniform map "giải thích" 1–2 trên 4 dataset. Tiêu đề §6.3 ("what the response map explains") và abstract L21–22 ("which the response map explains") nói quá.
- **Kiểm chứng bằng chain-seeded map là vòng tròn.** Lỗi được đặt theo marginal nhãn của chính chain cần dự đoán; ở r bằng accuracy cuối, trạng thái trồng vào gần như là trạng thái dừng của chain, nên một sweep từ đó ở yên (gain≈0) *theo tính dừng*, bất kể mô tả 1 chiều có đúng hay không. Sai số trung vị 0.016 (Hình 5a gần như nằm trên đường chéo) do đó chỉ kiểm rằng chain gần dừng theo R_s. Bài có câu "it is not a forecast" nhưng vẫn viết "This checks the one-dimensional description" (L268–269); tôi cho rằng nó không kiểm cả điều đó. Ngưỡng dự đoán P8 (≤0.10) quá lỏng để bị bác.
- **Không dự báo được gì trước khi chạy.** Map cần nhãn thật; chẩn đoán không nhãn (H3″) trượt; dự đoán P11, P14, P15 dựa trên map đều sai. Một lý thuyết "giải thích" stalling mà ba dự đoán tiền nghiệm của nó sai và không cho quy tắc hành động nào thì là mô tả hậu nghiệm.
- **Không có cơ chế:** vì sao TabICL v2 có gain dương còn TabPFN v2/TabPFN-3 thì không? Map mô tả, không giải thích.

### 3.3. Định lý 1–2 nói về sampler không được chạy

- Sampler mặc định là khối không hiệu chỉnh, 30 sweep, context cắt 2000; bài tự viết cận (4) "is vacuous" cho nó (L157) và chỉ có Mệnh đề 3 trong *một* mô hình liên hợp không feature. Bản MH chính xác chậm 7 lần và **không dự đoán tốt hơn** (L303–306), tức thực nghiệm không ủng hộ rằng "nhắm đúng posterior Bayes" là lý do phương pháp hoạt động.
- τ(P)=Σδ(Pᵗ) và ε̄ đều không tính được cho bài toán thật; τ(P) có thể mũ theo n khi posterior đa mode (chính là tình huống hai basin của §5.2).
- **Tuyên bố mới ở L152–154 yếu:** "a chain built from an incompatible TFM still has a unique stationary distribution" là tính chất sơ cấp của chain hữu hạn bất khả quy, và đã quen thuộc trong văn liệu dependency networks / pseudo-Gibbs và fully conditional specification. Người phản biện biết mảng này sẽ bắt lỗi.
- Bảng 17 đánh dấu "inference over latent labels with guarantees ✓" cho phương pháp: bảo đảm chỉ có cho biến thể không chạy.
- "the gain comes from sampling the labels jointly" (L247–248) không được chứng minh: so sánh H3 là log loss với hard-EM, trong khi dự đoán (3) là trung bình 10 mẫu (hiệu ứng ensemble làm giảm log loss bất kể "jointly"). Trên thước đo chính, hard-EM bằng Gibbs ở PLL q=0.3 (0.862 so với 0.862), và **"Stochastic EM (same TFM)" — không được định nghĩa ở đâu trong bài — chỉ kém 0.004–0.008** (Bảng 10: 0.835/0.858/0.843 so với 0.840/0.862/0.851). Không có kiểm định Gibbs so với hai biến thể EM trên AUC/accuracy.

### 3.4. Baseline và phạm vi thực nghiệm

- **Baseline PLL là các phương pháp deep learning sinh ra cho ảnh, chạy với MLP/linear** (PRODEN, RC, CC, GLWS/ILL). Không có phương pháp PLL cổ điển cho dữ liệu bảng (dựa trên kNN, SVM, ECOC, lan truyền nhãn), không có biến thể cây tăng cường với loss PLL/LLP. Dấu hiệu baseline yếu: L = tuned CC đạt **0.210 trên vowel** (11 lớp), 0.477 trên micro-mass; PRODEN mặc định 0.373 trên Lost. RC/CC "use a linear model with the PRODEN space" (L859–860): không rõ bản tune có được dùng tầng ẩn không.
- **LLP:** [15] (belief propagation trên nhãn ẩn rồi bootstrap learner có giám sát, cho dữ liệu bảng) được trích nhưng không so, dù gần ý tưởng nhất. Không có phương pháp LLP cổ điển (mean-map, ∝SVM), không có benchmark LLP dạng bảng quy mô lớn (Bảng 4 nói đã bỏ một benchmark ngoài vì không tải được dữ liệu).
- **Comparator đầu bảng không phải cái mạnh nhất.** Abstract L19–20 viết "outperformed the strongest specialised learners for label proportions (+0.052 AUC over DLLP)", nhưng +0.052 là so với *linear DLLP*; so với baseline mạnh nhất (tuned DLLP-MLP) là +0.041 (Bảng 7). "A rule applied to development datasets" chọn comparator chính (L218–219) không được mô tả.
- **Phạm vi hẹp:** LLP chỉ nhị phân, bag ngẫu nhiên cỡ 8 (bag 16/32 không có ý nghĩa ở Study 1; bag 32 "hard for every TFM"); PLL chỉ đa lớp, distractor đều, và Study 3 chỉ q=0.6 trên **8** dataset; nhãn yếu tổng hợp trừ ba dataset PLL thật nhỏ; dataset ≤10,000 dòng, context cắt 2000.
- **Dữ liệu PLL thật:** L302 viết "registered in Study 1", nhưng TFM đăng ký của Study 1 là TabPFN v2 và phụ lục D nói mọi kết quả TabICL v2 của Study 1 là exploratory. Kết quả TabPFN v2 trên ba dataset này không được báo (có thể do giới hạn 10 lớp, nhưng bài không nói). Không có kiểm định; thiếu các benchmark PLL thật lớn hơn.
- **Study 3 thiếu naive TFM, hard-EM, oracle.** Chain đơn TabPFN-3 đạt 0.074 (soybean, 19 lớp), 0.047 (micro-mass, 20 lớp), 0.205 (texture): có tệ hơn naive TFM không? Người đọc không biết sampler có *gây hại* hay không.
- **Cấu trúc dataset đổi giữa các study:** Study 1–2 có 3–10 lớp; Study 3 có 5/8 dataset PLL với C ≥ 9 (tới 19–20), đúng vùng stalling đã biết từ Study 1, và việc chọn có "manual review". Điều này làm H1″ dễ đạt hơn; bài không bình luận.
- **Rò rỉ tiền huấn luyện:** chỉ loại tập của TabDPT. CC18 (Study 1) là benchmark phát triển quen thuộc của họ TabPFN; baseline MLP huấn luyện từ đầu không có lợi thế đó. Bài có nêu ở Limitations nhưng không có phân tích độ nhạy (ví dụ theo ngày công bố dataset).
- **Chi phí:** 310 forward pass mỗi chain; 12 chain + tune L, trung vị 37 phút cho mỗi dataset×seed trên T4 với dữ liệu ≤10k dòng. Không báo thời gian baseline, không có đường cong chi phí–chất lượng.

### 3.5. Thống kê

- **"Pre-registered" không kiểm chứng được từ bên ngoài:** đăng ký là commit trong kho riêng, không có registry công khai hay dấu thời gian bên thứ ba, PDF không có đường dẫn. Toàn bộ ba study diễn ra trong 17 ngày (26/10–12/11/2026); Bảng 4 ghi phân tích Study 1 ngày 30/10 trong khi run kéo tới 1/11. Từ "registered" xuất hiện khoảng 40 lần; người phản biện sẽ coi đây là lời tự khai.
- **PLL của Study 2 mỏng:** n=10 dataset, hiệu ứng +0.012 accuracy; với Holm trên toàn bộ baseline thì không baseline PLL đã tune nào bị vượt có ý nghĩa (mọi p=0.070, Bảng 7); bỏ dataset gắn cờ thì p=0.062; đổi seed lấy mẫu thì H4′ q=0.6 có p=0.074. Tuyên bố "outperformed … partial labels" trong abstract không có rào chắn nào.
- **CI của PLL q=0.3 và q=0.6 trùng nhau từng chữ số** (+0.012 [+0.006, +0.019], Bảng 1; hai macro riêng trong `numbers.tex` cùng giá trị). Có thể trùng hợp; cần kiểm lại script.
- **H1″ gộp chênh lệch AUC (LLP) và accuracy (PLL) vào một Wilcoxon:** hạng bị chi phối bởi 8 dataset PLL. Trung bình +0.058 nhưng trung vị +0.003, LLP +0.007. p<0.001 trong abstract là một phía và không ghi là một phía.
- **δ₂ là ngưỡng thấp:** "ensemble hơn trung bình các thành viên" gần như luôn đúng. So sánh có ý nghĩa là với TFM tốt nhất chọn trước (TabICL v2): +0.016, p=0.08, không có ý nghĩa (L297–299).
- **Tie không được định nghĩa:** 13/6/1 ở H1″ không thể là bằng nhau tuyệt đối trên trung bình nhiều TFM; hẳn có ngưỡng. Wilcoxon xử lý tie/zero thế nào?
- **Tiêu chí không có kiểm định:** H3″ = 0.766 với CI [0.59, 0.92] chứa ngưỡng 0.80; C4″ đếm 71 cặp không độc lập (cùng dataset qua các TFM). "Cell fails" trong H3″ không được định nghĩa trong chính văn.
- **Đơn vị "cell" không độc lập:** H3 dùng 38 và 128 cell (dataset × mức) trong Bayesian signed-rank như thể độc lập.
- **Không có phương sai theo seed** ở bất kỳ bảng nào; bootstrap chỉ trên dataset. Một split 70/30 cho mỗi seed.
- **Interim look** (60/71 cặp) không có quy tắc dừng nên không thổi phồng sai lầm loại I; chỉ cần nói rõ vì sao xem.
- **Lỗi nhỏ về số:** Bảng 13 ghi TabPFN-3/micro-mass = 0.046, Bảng 11 ghi 0.047; "in 2 of them behind the single chain and in 9 behind L" (2+9≠10, có một cặp tính hai lần, nên viết rõ); "regressed in 0 of 71 pairs" cho một đại lượng theo dataset (multi-TFM) là khó hiểu.

### 3.6. Abstract, giới thiệu và kết luận nói quá so với bằng chứng

| Câu trong bài | Vấn đề |
|---|---|
| "We prove that … a one-dimensional response map of the TFM governs where a chain ends" (L15–18) | Chỉ dưới Giả thiết 1, mà bài tự thừa nhận chỉ đúng một nửa với TabPFN v2; không phủ LLP và context cắt; rỗng ở cỡ dữ liệu của bài. |
| "which the response map explains" (L21–22) | Xem 3.2: mâu thuẫn với L265–267 và Hình 2. |
| "the strongest specialised learners … (+0.052 AUC over DLLP)" (L19–20, L48–49) | +0.052 là so với linear DLLP; mạnh nhất là +0.041. |
| "averaging chains from several starts (+0.058, p<0.001)" (L22–23) | Giấu rằng một start là learner đã tune, rằng trung vị là +0.003, và rằng toàn bộ lợi ích đến từ L. |
| "One chain with a robust TFM" (tiêu đề §6.2) | TabICL v2 stall ở Study 3 (soybean, vowel, micro-mass, seismic-bumps). |
| "with the failures reported as prominently as the confirmations" (L64–65) | Đúng với hai tiêu chí trượt; không đúng với "+0.018, p=0.14 so với L" và "lợi ích đến từ L". |
| "No weight of the TFM is updated" / "We replace training by inference" | Đúng theo chữ với TFM; sai theo tinh thần với LLP (warm start = DLLP) và multi-start. |

### 3.7. Người dùng không biết nên chạy gì

Bài đưa ra ba cấu hình (chain đơn; ba start; ba start × bốn TFM) và không có khuyến nghị. Chain đơn rẻ nhưng có thể sụp xuống dưới ngẫu nhiên (AUC<0.5 ở 4 cell LLP) mà không có chẩn đoán. Cấu hình đầy đủ cần 12 chain và một learner đã tune. Bảng 14 gợi ý một công thức rẻ hơn nhiều ("chains started from L only", 0.823/0.814 so với 0.828/0.808) nhưng đó là hậu nghiệm. Cần một đoạn "khuyến nghị thực hành" có ghi rõ mức bằng chứng.

### 3.8. Tính mới

Phần mới thật sự: dùng PPD của TFM làm Gibbs conditional với likelihood nhãn yếu, và bộ thực nghiệm. Phần không mới hoặc mỏng: Gibbs với conditional học được và không tương thích; block-Gibbs song song không hiệu chỉnh; self-training/EM với pseudo-label (biến thể EM trong bài gần ngang về accuracy); ensemble nhiều TFM; cận nhiễu loạn chain Markov. Người phản biện khó tính sẽ tóm lại: "self-training ngẫu nhiên với TFM, khởi đầu bằng baseline, cộng ensemble".

## 4. Trình bày và độ rõ

**Đánh giá chung.** Người chưa theo dự án đọc 9 trang này rất vất vả: bài nén ba study, ba định lý, năm mệnh đề, hơn mười mã giả thuyết, và dựa nặng vào phụ lục. Văn phong gọn và chính xác ở từng câu, nhưng mật độ quá cao và nhiều thứ dùng trước khi định nghĩa.

**Những chỗ tôi phải đoán:**

1. Thẻ xanh **[C2], [C14][C15], [C16]…** in khắp chính văn (L60–64, L144, L151, L193, L207, L244–250, L261, L268, L274–311): không được giải thích, là dấu vết bản nháp.
2. **"cell"** dùng ở L247 trước khi định nghĩa và có ít nhất hai nghĩa: (dataset, mức nhãn yếu) ở Bảng 22; (TFM, dataset, seed) ở "71 pairs (213 cells)" và "234 cells".
3. **"pair"** = (TFM, dataset): định nghĩa lướt ở L236.
4. **H2, H4, H3** xuất hiện theo thứ tự đó; **H1 và H5 không được định nghĩa ở đâu** (Bảng 4 nói "H3 and H5 met"). H2′–H4′, rồi H1″–H3″, C4″, C5″: H3 (hard-EM) và H3″ (chẩn đoán) không liên quan nhau. Nên đặt tên theo nội dung.
5. **"registered"** (ablation, rule, tier, descriptive comparison, expectations…): không nói đăng ký ở đâu, ai kiểm được.
6. **"primary comparator"** và "a rule applied to development datasets": quy tắc và danh sách development dataset không có trong bài.
7. **"five curated suites [e.g. 13]"**: không nêu tên năm suite.
8. **"the five flagged datasets"** (L279): gắn cờ vì gì thì chỉ có ở phụ lục.
9. **"uninformative labelling"**, **"oracle"**, **"hard-EM"** (chỉ có "arg max instead of sampling"), **"Stochastic EM"** (không có định nghĩa nào), **"naive TFM"** (có), **"R̂"** (Bảng 4, 20: không định nghĩa), "Chains do not mix at the level of individual labels" (L306–307) không có ngữ cảnh.
10. **"within its published limits"** (TabPFN v2): không nêu giới hạn. **"TabPFN-3 was preferred to its successor"** (L809): successor nào?
11. **"cold start drawn from the normalised likelihood"**: với LLP nghĩa là gì?
12. **Tie** trong W/T/L: không định nghĩa.
13. L254–255 **"it was ahead on ten datasets by +0.033"** đọc như thắng 10 dataset, trong khi Bảng 1 ghi 7/0/7; ý là trung bình trên 10 dataset dưới 8 lớp.
14. L270–271: câu về TabPFN-3 và TabDPT treo lơ lửng, không có hình/bảng.
15. **Ký hiệu trùng:** p (PPD) và p(ν,λ); q (TFM), q (xác suất distractor), q_j, q₀–q₂; τ(P) và thời điểm τ; δ (Dobrushin), δ (dung sai), δ₁, δ₂; λ (biên), λ_k (Mệnh đề 5), λ=N/n (Mệnh đề 3); S (tập ứng viên, số mẫu, scoring rule, nR_s); ℓ (likelihood) và ℓ-Lipschitz; K (khối), k (lớp, khối, cỡ bag); H (horizon) và H (giả thuyết); C (số lớp), C4″, [C#]; r (accuracy) và r (tỉ lệ nhiễu, Bảng 27).
16. Phương trình (1): hai đối tượng trên một dòng, π′ định nghĩa sau; (A0) chèn giữa câu.

**Bảng và hình:**

- **Bảng 2** ghép hai bảng cạnh nhau ở cỡ chữ ~6pt, cột "single/multi" và hàng "Multi-TFM" khó đọc; đây là bảng chính của Study 3.
- **Bảng 1** đọc được nhưng cột Study 1 không có CI.
- **Hình 1:** 3 seed không có dải sai số; không nói Φ được đo thế nào (bao nhiêu lần trồng lỗi, lưới r, có cắt context không).
- **Bảng 5, 17:** tiêu đề cột lệch (ô "– (pretrains a PFN)" nằm sai cột). **Bảng 25** không đọc được (~4pt) và chứa cả baseline PU/SSL.
- Trang 1 trống gần nửa trang sau 4 dòng đầu của Introduction.

**Nội dung thiết yếu đang nằm ở phụ lục:**

- **Algorithm 1** (Định lý 3 viết "Run Algorithm 1" nhưng thuật toán ở phụ lục B). Phải đưa vào chính văn.
- **Hình 2 hàng dưới** (quỹ đạo label accuracy): là bằng chứng trực tiếp duy nhất của "stalling", được trích ở L256.
- **Hình 4** (Study 3 theo dataset): cho thấy ngay lợi ích tập trung ở 4–5 dataset.
- Mệnh đề 1, 2, 3, 5 được dùng trong lập luận chính văn nhưng chỉ phát biểu ở phụ lục; Mệnh đề 5 được trích ngay L35.
- Định nghĩa chẩn đoán H3″, cách chọn dataset Study 3, định nghĩa các baseline cùng TFM.

**Nên cắt hoặc chuyển:**

- Định lý 2: còn một câu + phụ lục. Đoạn "Uncorrected blocks": còn 2 câu.
- Đoạn Mệnh đề 5 (L198–207): còn 3 dòng.
- §6.5: chuyển MH và R̂ xuống phụ lục; giữ PLL thật, chi phí, và cảnh báo H4′ phụ thuộc seed (đưa cảnh báo này lên §6.2).
- Sổ sách đăng ký (mã H/C, "registered" lặp lại): thay bằng một bảng nhỏ "khẳng định | xác nhận hay khám phá | kiểm định | kết quả".
- Cân nhắc giảm Định lý 3 xuống "mệnh đề + minh hoạ" và dùng chỗ trống cho Algorithm 1, Hình 2 dưới, Hình 4.

## 5. Câu hỏi cho tác giả (những câu có thể đổi điểm)

1. Ở LLP, warm start linear DLLP là mặc định hay đã tune? Kết quả H2′ với cold start (không learner nào) trên dataset Study 2 là bao nhiêu? Nếu vẫn hơn tuned DLLP-MLP thì tuyên bố "frozen/training-free" đứng được.
2. δ của Giả thiết 1 bằng bao nhiêu trên dữ liệu thật (ít nhất là cận dưới từ chênh lệch giữa các cách trồng lỗi ở cùng r)? Có dataset nào trong bài mà Định lý 3 cho một floor/ceiling không rỗng không?
3. Giải thích mâu thuẫn giữa L258–259 (dùng Định lý 3(b) cho TabPFN v2) với L265–267 (nửa trên không đúng) và với quỹ đạo optdigits/MiceProtein ở Hình 2.
4. Có kiểm chứng response map nào không vòng tròn không: map đo từ chain/seed/start A dự đoán điểm cuối của chain B?
5. "Stochastic EM (same TFM)" là gì chính xác, và Gibbs khác nó ở bước nào? Gibbs có hơn nó và hơn hard-EM có ý nghĩa trên AUC/accuracy không? Hard-EM có được trung bình trên nhiều vòng lặp như (3) không?
6. Ở Study 3, naive TFM và oracle của từng TFM đạt bao nhiêu? Chain đơn có tệ hơn naive ở các cell stall không?
7. Tuned CC/RC có được dùng tầng ẩn không? Vì sao không có baseline PLL/LLP phi tuyến cho dữ liệu bảng (kNN, cây tăng cường), và vì sao không so với [15]?
8. Vì sao comparator chính của LLP là linear DLLP chứ không phải baseline mạnh nhất? Quy tắc chọn là gì?
9. Một ensemble của L (nhiều seed/cấu hình) hoặc ensemble naive-TFM qua bốn TFM đạt bao nhiêu so với multi-TFM?
10. Kết quả H1″, H2″ tách riêng LLP và PLL, và khi bỏ TabPFN-3 (nguồn chính của stalling)?
11. Vì sao Study 3 có dataset tới 19–20 lớp trong khi Study 1–2 dừng ở 10? Quyết định này có trước khi biết stalling phụ thuộc số lớp không?
12. Đăng ký có dấu thời gian bên thứ ba nào không? Người phản biện xem kho ẩn danh ở đâu?
13. TabPFN v2 trên ba dataset PLL thật: không chạy được hay không báo?
14. Chạy dài hơn (T=100, 300) có thoát basin không? Đây là kiểm chứng trực tiếp duy nhất cho "exponential escape time".
15. Chain LLP khôi phục đúng bao nhiêu phần trăm nhãn cá nhân từ số đếm bag (liên quan tuyên bố về quyền riêng tư)?

## 6. Tài liệu và so sánh còn thiếu

Tôi nêu theo dòng nghiên cứu; chỗ có tên bài là chỗ tôi khá chắc bài tồn tại, còn lại chỉ mô tả chủ đề.

- **Gibbs với conditional học được, không tương thích:** dependency networks (Heckerman và cộng sự, JMLR 2000, "pseudo-Gibbs sampler"); fully conditional specification / MICE (van Buuren); masked language model như Gibbs sampler/MRF. Cần trích để đặt lại tuyên bố mới của Định lý 2.
- **Block-Gibbs song song không hiệu chỉnh:** Hogwild/asynchronous Gibbs và phân tích bias của nó (Johnson và cộng sự 2013; De Sa và cộng sự 2016); đây chính là "uncorrected blocks".
- **Data augmentation, stochastic EM, Monte Carlo EM, multiple imputation** (Tanner & Wong 1987; Celeux & Diebolt 1985; Wei & Tanner 1990; Rubin 1987): dự đoán (3) là multiple imputation của nhãn.
- **Martingale posterior / predictive resampling** (Fong, Holmes & Walker) và việc áp dụng cho PFN/TabPFN, cùng các bài chỉ ra mô hình in-context không thoả tính martingale/khả hoán: liên quan trực tiếp tới (A1) và tính không tương thích.
- **Nhiễu loạn chain Markov** hiện đại (Mitrophanov 2005; Rudolf & Schweizer 2018) bên cạnh các trích dẫn cổ điển.
- **MCMC/Bayes cho LLP:** Kück & de Freitas (UAI 2005, "Learning about individuals from group statistics"); LLP cổ điển: mean-map (Quadrianto và cộng sự 2009), ∝SVM (Yu và cộng sự 2013); benchmark LLP dạng bảng quy mô lớn (không chắc tên chính xác).
- **PLL cổ điển cho dữ liệu bảng:** PL-kNN (Hüllermeier & Beringer 2006), CLPL (Cour và cộng sự 2011), IPAL (Zhang & Yu 2015), PL-ECOC (Zhang và cộng sự 2017); PLL phụ thuộc instance (VALEN, Xu và cộng sự 2021). Đây là các comparator tự nhiên trên Lost/BirdSong/MSRCv2.
- **Self-training, pseudo-labelling và confirmation bias** trong học bán giám sát: hiện tượng "stall" là dạng đã biết của nó.
- **Phân tích động lực một chiều của thuật toán lặp:** EXIT chart / density evolution trong giải mã lặp, state evolution cho message passing; metastability kiểu mean-field. "Response map" là cùng một công cụ và nên được nối với văn liệu này.
- **TFM với nhãn nhiễu/thiếu, bán giám sát, và fine-tuning TFM** như một baseline thay thế (không chắc bài cụ thể năm 2026).
- **Ensemble nhiều TFM / post-hoc ensembling** trong benchmark bảng (bài có trích TabArena nhưng không bàn rằng kết quả C5″ là điều đã được kỳ vọng).
- **Label switching** trong MCMC mô hình hỗn hợp: lỗi AUC<0.5 ở LLP.

Ghi chú về danh mục: các mục arXiv ghi "In arXiv preprint …" sai định dạng; [11] có mã arXiv 2510.xxxxx nhưng ghi năm 2026. Tôi không kiểm được các tài liệu 2026 ([14], [19], [20], [23], [29], [32]).

## 7. Điểm số

Thang Overall 1–10 theo nghĩa NeurIPS (10 award; 8 strong accept; 7 accept; 6 weak accept; 5 borderline accept; 4 borderline reject; 3 reject; 2 strong reject). Nếu NeurIPS 2027 dùng thang 1–6 như năm 2025 thì 4/10 ≈ 3/6 (borderline reject), 6/10 ≈ 4/6 (borderline accept).

| | Hiện tại | Nếu sửa các điểm yếu hàng đầu (không thêm thí nghiệm GPU) |
|---|---|---|
| Soundness (1–4) | **2** (fair) | 3 (good) |
| Presentation (1–4) | **2** (fair) | 3 (good) |
| Contribution (1–4) | **2** (fair), sát 3 | 3 (good) |
| Overall (1–10) | **4** (borderline reject) | **6** (weak accept); thực tế dao động 5–6 |
| Confidence (1–5) | **4** | 4 |

**Lý do cho điểm hiện tại.** Ý tưởng tốt và sự trung thực là có thật, nhưng (i) lý thuyết hoặc nói về sampler không chạy, hoặc dựa trên giả thiết không kiểm và rỗng về số; (ii) phương pháp được khuyến nghị không có bằng chứng xác nhận so với baseline và phụ thuộc vào chính baseline đó; (iii) baseline PLL yếu, phạm vi hẹp; (iv) abstract nói quá ở bốn chỗ; (v) trình bày quá nén. Ngoài ra **bản PDF như hiện tại còn TODO, "Pending" và thẻ [C#]**; nếu nộp nguyên trạng có nguy cơ bị loại vì checklist chưa hoàn chỉnh, không tính vào điểm khoa học ở trên.

**Dự đoán hội đồng:** chuyên gia PLL/LLP 3–4 (baseline); chuyên gia Bayes/MCMC 4 (khoảng cách lý thuyết–thực hành, Định lý 3); người làm tabular thực nghiệm 5–6 (thích replication và kết quả âm). Trung bình khoảng 4–4.5.

**Trần sau khi sửa** bị giới hạn bởi những thứ cần thí nghiệm mới: baseline bảng phi tuyến, naive/oracle ở Study 3, cold-start cho LLP ở Study 2. Nếu có cả những thứ đó và kết quả giữ nguyên, 6–7 là khả thi.

## 8. Danh sách thay đổi theo ưu tiên

Các phân tích lại dưới đây đều làm được từ dự đoán và kết quả đã lưu, trừ khi ghi "cần chạy mới".

**[must]**

1. **Xoá mọi dấu vết bản nháp:** thẻ [C#], hai TODO ở phụ lục N (L887, L891, có lộ đường dẫn nội bộ), "Pending" ở checklist mục 12; biên dịch với `\finaltrue`; thay style placeholder bằng style chính thức và kiểm lại giới hạn 9 trang; dùng nguyên văn mẫu checklist.
2. **Viết lại abstract và các câu ở 3.6:** dùng +0.041 so với baseline LLP mạnh nhất (hoặc nói rõ "linear DLLP"); nói một start là learner đã tune và báo trung vị +0.003 cùng tách LLP/PLL; thêm "+0.018, p=0.14 so với L với một TFM"; đổi "We prove … governs" thành phát biểu có điều kiện; bỏ "which the response map explains"; ghi "một phía" cho p của Study 3; thêm rào chắn cho PLL (n=10, phụ thuộc seed).
3. **Sửa mâu thuẫn §6.3:** bỏ câu viện dẫn Định lý 3(b) cho TabPFN v2 hoặc chứng minh (H⁺) đúng ở đó; nói thẳng uniform map khớp bao nhiêu trong bốn dataset; đổi tiêu đề §6.3.
4. **Báo cáo trung thực tầm với của Định lý 3:** thêm bảng biên tối thiểu ν+λ theo n_F (như 3.2) cạnh biên độ gain đo được; ước lượng δ từ các map đã lưu (uniform, chain-seeded, structured ở cùng r); nói rõ định lý là định tính ở cỡ dữ liệu của bài.
5. **Nói đúng về "training-free":** nêu ở abstract/intro/Bảng 17 rằng LLP khởi từ DLLP và multi-start cần L; sửa L70; báo lại Bảng 23 (cold start) ngay cạnh tuyên bố chính.
6. **Định nghĩa mọi thuật ngữ ở mục 4** (ít nhất: cell, pair, tie, Stochastic EM, hard-EM, oracle, H1/H5, flagged, published limits, development datasets, quy tắc comparator chính, chẩn đoán H3″ và "fails").
7. **Đưa Algorithm 1 vào chính văn**; đưa Hình 2 hàng dưới và Hình 4 (hoặc bản rút gọn) vào chính văn; tách Bảng 2 thành hai bảng đọc được.
8. **Thu hẹp tuyên bố mới của Định lý 2** (L151–154) và trích văn liệu pseudo-Gibbs / dependency networks / asynchronous Gibbs.
9. **Bỏ hoặc chứng minh câu "the gain comes from sampling the labels jointly":** thêm kiểm định Gibbs so với hard-EM và Stochastic EM trên AUC/accuracy từ dự đoán đã lưu.
10. **Kiểm lại CI trùng nhau** ở PLL q=0.3/0.6 và các lệch số nhỏ (0.046/0.047; "2 và 9 trong 10").

**[should]**

11. Tách H1″/H2″ theo LLP và PLL, theo từng TFM, và leave-one-dataset-out; báo trung vị cạnh trung bình.
12. Một bảng "khẳng định | xác nhận/khám phá | kiểm định | kết quả" thay cho hệ mã H/C; đổi tên giả thuyết theo nội dung.
13. Kiểm chứng response map không vòng tròn từ dữ liệu đã lưu (map của chain A dự đoán chain B; hoặc uniform map dự đoán biến nhị phân "stall hay không" từ accuracy khởi đầu, báo AUC).
14. Thêm đoạn "khuyến nghị thực hành" với mức bằng chứng từng cấu hình và bảng chi phí–chất lượng từ Bảng 14 (1 chain; chain từ L; 3 start; × số TFM), kèm thời gian của baseline.
15. Báo độ lệch chuẩn theo seed và bootstrap phân cấp (dataset, seed); nói rõ cách xử lý tie trong Wilcoxon.
16. Mô tả cơ chế đăng ký một cách kiểm chứng được (kho ẩn danh có lịch sử commit trong supplementary; giải thích mốc 30/10 so với 1/11) và giảm tần suất từ "registered".
17. Baseline chỉ cần CPU (cần chạy mới nhưng không cần GPU): PLL dựa trên kNN và cây tăng cường với loss PLL/LLP; ensemble của L qua seed/cấu hình; [15] cho LLP nếu có mã.
18. Báo tỉ lệ nhãn cá nhân mà chain LLP khôi phục đúng (đã có trong dữ liệu response map) để thay TODO về quyền riêng tư bằng một con số.
19. Nói rõ trạng thái exploratory của kết quả PLL thật (L302) và số phận của TabPFN v2 ở đó.
20. Bổ sung related work theo mục 6; sửa định dạng bib.
21. Thống nhất ký hiệu (mục 4, điểm 15); rút gọn Định lý 2, đoạn uncorrected blocks, Mệnh đề 5, §6.5 như đề xuất.

**[nice]**

22. Sửa Bảng 5, 17, 25; thêm dải sai số cho Hình 1; lấp khoảng trống trang 1.
23. Phân tích độ nhạy rò rỉ: kết quả theo nhóm dataset (CC18 so với suite mới; theo năm công bố).
24. Nối "response map" với EXIT chart / state evolution trong một đoạn ngắn.
25. Cân nhắc đổi tên bài nếu giữ multi-start làm phương pháp chính: "Frozen" đúng với TFM nhưng không đúng với cả quy trình.

**Thí nghiệm GPU mà người phản biện sẽ đòi (ngoài phạm vi hiện tại, nên chuẩn bị câu trả lời):**

- Naive TFM, hard-EM, oracle cho từng TFM ở Study 3.
- Cold start (không learner) cho LLP trên dataset Study 2.
- Study 3 với q=0.3, bag 16/32, bag không ngẫu nhiên, PLL phụ thuộc instance.
- Chạy dài (T ≥ 100) và thay đổi n để kiểm "thời gian thoát mũ theo n_F".
- Response map trên dataset Study 3.
- Dataset lớn hơn 10k dòng và context không cắt; benchmark LLP thật; thêm dataset PLL thật.
- Baseline fine-tuning TFM; multi-start không cần L (khởi chéo giữa các TFM, tempering).
- Hơn 3 seed.

## 9. Đạo đức, double-blind, checklist

- **Dấu vết bản nháp trong PDF:** thẻ [C#] khắp chính văn; L887 "[TODO: quantify, or keep this sentence (G9/REVIEW_RESPONSE_PLAN R8)]"; L891 "[TODO: align with the venue's LLM policy; docs/AI_USAGE.md]"; L919–921 "[Pending: licences … not yet confirmed]". Hai TODO lộ tên tệp nội bộ (không lộ danh tính nhưng lộ quy trình).
- **Style:** dùng `neurips_placeholder.sty`, không phải style chính thức; số trang có thể đổi. Checklist viết tắt câu hỏi, không theo nguyên văn mẫu.
- **Ẩn danh:** không thấy tên, cơ quan hay URL. Rủi ro nhỏ: Bảng 4 in giờ commit chính xác (1 Nov 15:19, 10 Nov 04:21 UTC) và lịch 26/10–12/11/2026, có thể đối chiếu với kho công khai sau này; "free Kaggle sessions" không định danh. Chú thích trong mã LaTeX chứa đường dẫn nội bộ (G3/, G5/, S2/, S3/); phải xoá nếu nộp mã nguồn.
- **Checklist mục 1 (Claims) [Yes]:** chưa đúng; abstract có ít nhất bốn câu vượt bằng chứng (3.6), và "which the response map explains" dựa trên phân tích exploratory.
- **Mục 3 (Theory) [Yes]:** chứng minh đầy đủ, nhưng Giả thiết 1 chưa được kiểm ở dạng phát biểu; nên ghi rõ.
- **Mục 4–5 (Reproducibility, code) [Yes]:** PDF không có đường dẫn tới kho ẩn danh; chỉ viết "available in an anonymised repository and will be released". Cần liên kết hoặc supplementary.
- **Mục 7 (Significance) [Yes]:** không có sai số theo seed; ba tiêu chí Study 3 không có kiểm định; nên nêu trong phần giải trình.
- **Mục 12 (Licences):** chưa trả lời; TabPFN-3 có giấy phép phi thương mại cần ghi rõ điều kiện dùng cho nghiên cứu.
- **Mục 16 / tuyên bố LLM:** bài khai AI được dùng để viết mã, **soạn chứng minh**, chạy phân tích và soạn văn bản. Đây là mức sử dụng đáng kể; cần khai đúng theo chính sách của hội nghị, và người phản biện sẽ đọc chứng minh kỹ hơn (tôi không tìm thấy lỗi trong các chứng minh đã kiểm).
- **Broader impacts:** LLP khôi phục nhãn cá nhân từ số liệu tổng hợp là rủi ro riêng tư thật; bài tự nói chưa đo, trong khi dữ liệu để đo đã có (label accuracy của chain LLP). Hình 2 cho thấy với PLL chain gán đúng 93–99% nhãn huấn luyện; con số tương ứng cho LLP nên được báo.
- **Dữ liệu:** toàn dataset công khai; không có vấn đề về đối tượng nghiên cứu là người.
