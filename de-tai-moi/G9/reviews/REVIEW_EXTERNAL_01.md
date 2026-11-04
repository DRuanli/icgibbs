# Nhận xét của reviewer ngoài, số 01 (nhận 04/11/2026)

- **Người nhận xét:** reviewer ngoài, thầy chuyển lại trong chat ngày 04/11/2026. Không phải phản biện chính thức của hội nghị.
- **Bản được đọc:** bản nháp PDF (`neur.pdf`) dựng từ commit `2ff12b4`: phần chính 9 trang, tổng 24 trang.
- **Kết luận của reviewer:** "promising work, nhưng cần major revision về mặt kỹ thuật".
- **Đánh giá của em và kế hoạch trả lời:** [REVIEW_RESPONSE_PLAN.md](../REVIEW_RESPONSE_PLAN.md).

Dưới đây là nguyên văn, kể cả các dấu `neur.pdf` và `&#x20;` do công cụ của reviewer chèn vào. Không sửa.

---

Sau khi đọc kỹ hơn phần thuật toán, các định lý, bảng kết quả và phụ lục, mình cho rằng bài này có ý tưởng nghiên cứu thực sự đáng chú ý, nhưng tồn tại một khoảng cách khá nghiêm trọng giữa thuật toán được chứng minh và thuật toán được sử dụng trong thực nghiệm.

Đây là điểm cần phân tích kỹ nhất. Nó quyết định liệu bài báo chỉ là một phương pháp có kết quả tốt trên benchmark, hay thực sự là một đóng góp vững chắc về Bayesian inference kết hợp với Tabular Foundation Models.

# 1. Bài báo thực sự giải quyết vấn đề gì?

## 1.1. Vấn đề nghiên cứu

Bài báo bắt đầu từ một hạn chế của Tabular Foundation Models (TFMs), chẳng hạn TabPFN và TabICL.

Các mô hình này có thể dự đoán nhãn của một điểm dữ liệu dựa trên những điểm đã có nhãn trong context mà không cần huấn luyện lại. Tuy nhiên, chúng vẫn cần context được gán nhãn.

Trong thực tế, nhãn không phải lúc nào cũng đầy đủ hoặc chính xác. Bài báo tập trung vào hai trường hợp:

Partial Label Learning (PLL)

Một mẫu có nhiều nhãn ứng viên, nhưng chỉ một nhãn là chính xác.

Ví dụ:

| Sample    | Candidate labels | True label |
| --------- | ---------------- | ---------- |
| \\(x_1\\) | Cat, Dog, Bird   | Cat        |
| \\(x_2\\) | Dog, Bird        | Bird       |
| \\(x_3\\) | Cat, Dog         | Dog        |

Thông tin ta quan sát được chỉ là tập nhãn ứng viên. Không biết chính xác nhãn nào đúng.

Learning from Label Proportions (LLP)

Ta không biết nhãn của từng mẫu, mà chỉ biết số lượng hoặc tỷ lệ nhãn trong một nhóm.

Ví dụ, một nhóm có 8 mẫu, trong đó có 5 mẫu thuộc lớp A và 3 mẫu thuộc lớp B. Ta biết tổng số lượng từng lớp nhưng không biết mẫu nào thuộc lớp nào.

Hai vấn đề này khác nhau về cấu trúc thông tin, nhưng đều có một điểm chung: nhãn thực sự là biến ẩn cần suy luận.

Bài báo muốn tận dụng TFM để suy luận các nhãn ẩn mà không cần fine-tuning lại mô hình. Đây là phần có giá trị nhất trong ý tưởng nghiên cứu.&#x20;

neur.pdf



## 1.2. Ý tưởng cốt lõi: biến TFM thành một Gibbs sampler

Đây là điểm quan trọng nhất cần hiểu.

Thông thường, ta nghĩ TabPFN hoặc TabICL là một mô hình dự đoán:

\\[ q(y_i\mid x_i,D) \\]

Trong đó:

- \\(x_i\\): mẫu cần dự đoán.
- \\(D\\): tập dữ liệu context đã có nhãn.
- \\(q\\): phân phối xác suất dự đoán của TFM.

Nhưng tác giả nhận ra một tính chất khác.

Giả sử dữ liệu được sinh ra từ một phân phối trao đổi được (exchangeable prior). Khi đó:

\\[ p(y_i\mid y\_{-i},X) = p(y_i\mid x_i,D\_{-i}) \\]

Nói đơn giản:

Xác suất nhãn của một điểm khi biết nhãn của tất cả các điểm còn lại chính là posterior predictive mà TFM có thể ước lượng.

Đây là cầu nối toán học giữa TFM và Gibbs sampling.

Từ đó, thay vì dự đoán nhãn một lần, ta thực hiện lặp đi lặp lại:

1. Khởi tạo nhãn cho các mẫu chưa biết.
2. Giữ các nhãn còn lại làm context.
3. Dùng TFM dự đoán phân phối nhãn của mẫu đang được cập nhật.
4. Kết hợp phân phối đó với thông tin weak labels.
5. Lấy mẫu nhãn mới.
6. Lặp lại nhiều lần.
7. Tổng hợp nhiều trạng thái nhãn để dự đoán mẫu test.

Công thức mục tiêu của bài toán là:

\\[ \pi(y)=p(y\mid X,W) \propto p(y\mid X)\ell(y) \\]

Trong đó:

- \\(p(y\mid X)\\) là prior/posterior distribution của các nhãn ẩn khi xét dữ liệu đầu vào.
- \\(W\\) là thông tin weak labels quan sát được.
- \\(\ell(y)=p(W\mid X,y)\\) là likelihood của thông tin weak labels.

Đây là cách kết hợp hai nguồn thông tin: tri thức học được từ TFM và những ràng buộc đến từ weak labels.

neur.pdf

Luồng hoạt động của In-Context Gibbs

Frozen Tabular Foundation Model

TabPFN / TabICL — không cập nhật trọng số

Conditional label prediction

Ước lượng phân phối nhãn khi biết context hiện tại

Weak-label likelihood

Áp dụng candidate-set constraints hoặc bag-count constraints

Gibbs sampling

Lấy mẫu và cập nhật các nhãn ẩn qua nhiều vòng

Posterior predictive averaging

Tổng hợp các dự đoán trên nhiều trạng thái nhãn

Điểm hay là tác giả không cần huấn luyện một TFM mới cho PLL hoặc LLP. Họ biến một mô hình dự đoán có sẵn thành một thành phần trong hệ thống suy luận xác suất.

# 2. Phân tích sâu phần lý thuyết: định lý có thực sự bảo đảm thuật toán không?

Đây là phần mình đánh giá quan trọng nhất.

## 2.1. Lemma 1: đóng góp toán học nền tảng

Lemma 1 chứng minh:

\\[ p(y_i\mid y\_{-i},X) = p(y_i\mid x_i,D\_{-i}) \\]

Ý nghĩa là TFM có thể đóng vai trò như một full conditional distribution của Gibbs sampler.

Đây là kết quả hợp lý và có sức thuyết phục về mặt toán học nếu chấp nhận các giả định về exchangeability và posterior predictive của TFM.

Tuy nhiên, cần phân biệt hai trường hợp:

- Khi TFM là posterior predictive chính xác của prior: đẳng thức được bảo đảm.
- Khi dùng TabPFN hoặc TabICL thực tế: chúng chỉ xấp xỉ posterior predictive đó.

Vì thế, Lemma 1 thiết lập nền tảng cho phương pháp, nhưng chưa chứng minh rằng một TFM thực tế có thể tạo ra các conditional đủ chính xác.

Đánh giá: Đây là một insight đẹp, nhưng bản thân đẳng thức không phải một định lý quá khó. Giá trị nằm ở việc khai thác nó để thiết kế một thuật toán weak supervision.

neur.pdf

## 2.2. Theorem 1: khi TFM hoàn hảo

Theorem 1 chứng minh tính đúng đắn của các biến thể chính trong điều kiện:

\\[ q=p \\]

Tức TFM phải trả về posterior predictive chính xác.

Đối với PLL, cập nhật từng điểm theo full conditional có thể tạo thành Gibbs sampler đúng đắn.

Đối với LLP, vì tổng số nhãn trong mỗi bag phải được giữ nguyên, tác giả dùng thao tác đổi nhãn giữa hai điểm khác lớp và tính xác suất chấp nhận dựa trên Metropolis–Hastings.

Đây là hai cách xử lý khác nhau cho hai loại weak supervision.

Tuy nhiên, có một điểm rất đáng chú ý.

Theorem 1(c) phân biệt hai cách dự đoán:

Inductive prediction

Chỉ lấy mẫu các nhãn của tập training rồi dự đoán nhãn test bằng:

\\[ \hat p(y^\*\mid x^\*)= \frac{1}{S}\sum\_{s=1}^{S} q(y^\*\mid x^\*,(X,y^{(s)})) \\]

Transductive prediction

Đưa cả điểm test vào quá trình suy luận nhãn.

Theo Theorem 1(c), phiên bản transductive mới hội tụ chính xác đến posterior predictive:

\\[ p(y^\*\mid x^\*,X,W) \\]

Trong khi phiên bản inductive có thể có sai số được chặn bởi:

\\[ TV\left( p(y\mid X,W), p(y\mid x^\*,X,W) \right) \\]

Sai số này bằng 0 trong trường hợp prior có cấu trúc separable được định nghĩa trong bài báo.

Đây là một chi tiết quan trọng: không nên diễn giải Theorem 1 là mọi phiên bản mặc định của In-Context Gibbs đều chính xác với Bayes posterior trong mọi trường hợp.

neur.pdf

## 2.3. Theorem 2: điểm mạnh nhưng cũng là giới hạn

Theorem 2 nghiên cứu trường hợp TFM không chính xác:

\\[ q\neq p \\]

Đây là điều thực tế hơn rất nhiều.

Một điểm thú vị là các conditional do TFM tạo ra có thể không tương thích với bất kỳ phân phối joint nào.

Thông thường, ta muốn các conditional distributions phải xuất phát từ cùng một joint distribution. Nhưng neural model không nhất thiết bảo đảm điều này.

Tác giả chứng minh rằng, dù các conditional không tương thích, Markov chain vẫn có một stationary distribution duy nhất trong điều kiện positivity và các điều kiện liên quan của định lý.

Sai lệch giữa stationary distribution của thuật toán xấp xỉ và posterior đích được chặn theo dạng:

\\[ TV(\tilde\pi,\pi) \leq \tau(P)\bar{\epsilon} \\]

Trong đó:

- \\(\tau(P)\\): đại lượng liên quan đến tốc độ mixing của chain chính xác.
- \\(\bar{\epsilon}\\): sai số trung bình của transition kernel xấp xỉ.

Ý nghĩa khá trực quan:

> Nếu TFM tạo ra conditional distributions gần đúng và chain chính xác có khả năng mixing tốt, thì phân phối dừng của chain xấp xỉ cũng có thể gần posterior đích.

Đây là một đóng góp lý thuyết đáng ghi nhận.

Nhưng có ba giới hạn:

Thứ nhất, stationary distribution duy nhất không đồng nghĩa với posterior chính xác.

Khi các conditional không tương thích, stationary distribution \\(\tilde\pi\\) vẫn tồn tại, nhưng không thể tự động xem nó là posterior Bayes đúng.

Thứ hai, bound phụ thuộc vào mixing time.

Nếu \\(\tau(P)\\) rất lớn thì ngay cả một sai số nhỏ từ TFM cũng có thể tạo ra sai lệch đáng kể.

Thứ ba, các đại lượng trong bound chưa được kiểm chứng trên dữ liệu thực.

Bài báo chưa cho thấy bound này đủ chặt để bảo đảm rằng thuật toán mặc định đang tạo ra posterior gần đúng trong các benchmark.

neur.pdf

# 3. Vấn đề nghiêm trọng nhất: thuật toán thực nghiệm không phải thuật toán được bảo đảm chính xác

Đây là điểm mình muốn nhấn mạnh hơn đánh giá trước.

## 3.1. Sự khác biệt giữa single-site Gibbs và block Gibbs

Trong Gibbs sampling truyền thống, ta cập nhật từng biến dựa trên toàn bộ các biến còn lại.

Ví dụ:

\\[ p(y_i\mid y\_{-i},X) \\]

Nhưng nếu cập nhật một block gồm nhiều điểm cùng lúc, tác giả phải dùng context không chứa các điểm trong block đó.

Sau đó, họ lấy mẫu theo tích các conditional:

\\[ Q_B(y_B) \propto \prod\_{i\in B} q(y_i\mid x_i,D\_{-B}) \\]

Đây là một proposal tiện lợi để giảm số lần forward pass.

Nhưng vấn đề là:

\\[ p(y_B\mid y\_{-B},X) \neq \prod\_{i\in B} p(y_i\mid x_i,D\_{-B}) \\]

nói chung.

Lý do: các nhãn trong cùng block vẫn có thể phụ thuộc lẫn nhau thông qua prior và các tham số ẩn chung của mô hình.

Vì vậy, ngay cả khi TFM hoàn hảo, phép lấy mẫu theo tích các marginal distributions cũng không nhất thiết là phép cập nhật Gibbs chính xác.

## 3.2. Metropolis–Hastings giải quyết vấn đề nhưng lại rất đắt

Tác giả đưa ra MH correction để chấp nhận hoặc từ chối trạng thái mới của block.

Theo Proposition 3(a), khi dùng MH correction, block update có thể giữ nguyên phân phối đích nếu TFM chính xác.

Nhưng trong cấu hình mặc định của thực nghiệm, họ không sử dụng correction này.

Họ thừa nhận khá rõ rằng:

- Thuật toán mặc định dùng uncorrected blocks.
- Bound của Proposition 3(b) chỉ được kiểm soát dưới một giả định ổn định bổ sung.
- Với số lượng điểm tăng lên và số block cố định, bound có thể trở nên không hữu ích.

Một ví dụ được nêu trong Appendix A:

Với \\(n\approx 2,000\\), \\(K=10\\), và bag size 8, tác giả ước lượng:

\\[ \log\_{10}\kappa_B\approx15.8 \\]

Điều này cho thấy hệ số liên quan đến xác suất proposal thỏa mãn ràng buộc bag count có thể cực kỳ lớn.

Đây không còn là một tiểu tiết kỹ thuật. Nó ảnh hưởng trực tiếp đến khả năng đưa ra bảo đảm lý thuyết cho phương pháp được thực nghiệm đánh giá.

neur.pdf



Major concern

&#x20;Khoảng cách giữa lý thuyết và thực nghiệm



| Thành phần             | Tính chất                                              |
| ---------------------- | ------------------------------------------------------ |
| Single-site Gibbs      | Chính xác khi TFM chính xác và giả định được thỏa mãn  |
| LLP swap sampler       | Chính xác với MH và TFM chính xác                      |
| Block sampler + MH     | Có bảo đảm chính xác với TFM chính xác                 |
| Block sampler mặc định | Không có bảo đảm không tầm thường ở quy mô thực nghiệm |

Đây là câu hỏi mà một reviewer khó tính có thể đặt ra:

> The authors prove the correctness of an exact sampler, but the main experimental results are obtained using an approximate block sampler. How do we know that the empirical improvements are not artifacts of the approximation?

Câu hỏi này rất khó trả lời bằng các bảng accuracy hiện tại.

## 3.3. Kết quả so sánh với MH correction chưa giải quyết được vấn đề

Tác giả đã thử phiên bản MH-corrected.

Kết quả:

| Setting với TabICL v2   | MH-corrected trừ default |
| ----------------------- | ------------------------ |
| LLP, bag 8              | +0.033 AUC               |
| PLL, q = 0.3            | −0.009 accuracy          |
| PLL, q = 0.6            | −0.021 accuracy          |
| PLL, q = 0.6, nhiều lớp | −0.016 accuracy          |

Tác giả cũng ghi nhận MH sampler chậm hơn trung vị khoảng 7 lần và có tỷ lệ chấp nhận thấp ở một số trường hợp.

Nhưng có một vấn đề khi diễn giải:

Không thể kết luận rằng default sampler không bị bias chỉ vì MH-corrected sampler không đạt accuracy cao hơn.

Bởi vì MH-corrected sampler mixing chậm hơn. Nếu chỉ chạy cùng 30 sweeps, nó có thể chưa tiến đủ gần stationary distribution.

Nói cách khác, phép so sánh hiện tại đang trộn lẫn hai yếu tố:

1. Sai lệch do dùng block sampler không correction.
2. Sai số do hai sampler chưa mixing đủ.

Tác giả cũng thừa nhận chưa cô lập được hai yếu tố này trên dữ liệu thực.&#x20;

neur.pdf



Đề xuất của mình: cần một thí nghiệm so sánh theo hai tiêu chí riêng:

- Cùng ngân sách tính toán: đánh giá phương pháp nào hiệu quả hơn trong cùng thời gian.
- Cùng mức độ hội tụ: đánh giá stationary approximation của hai phương pháp có gần nhau hay không.

Hai câu hỏi này không thể được giải quyết bằng một bảng accuracy duy nhất.

# 4. Phân tích thực nghiệm: kết quả có mạnh không?

## 4.1. Thiết kế thực nghiệm là điểm mạnh

Bài báo có hai study.

|                     | Study 1        | Study 2         |
| ------------------- | -------------- | --------------- |
| Datasets            | 24 OpenML-CC18 | 28 datasets mới |
| TFM chính           | TabPFN v2      | TabICL v2       |
| Binary / Multiclass | 10 / 14        | 18 / 10         |
| Protocol            | Pre-registered | Pre-registered  |
| Seeds               | 3              | 3               |

Điểm tích cực là tác giả xác định trước hypothesis, baseline, phương pháp tuning và quy tắc thống kê trước khi sử dụng dữ liệu đánh giá.

Study 2 còn sử dụng các dataset mới, không trùng Study 1.

Đây là thiết kế nghiên cứu nghiêm túc hơn nhiều so với cách chỉ chọn một vài benchmark có lợi cho phương pháp mới.&#x20;

neur.pdf



## 4.2. Nhưng Study 1 lại không thành công

Đây là kết quả khá thú vị.

Với TabPFN v2, In-Context Gibbs không vượt qua được các tiêu chí xác nhận trước đó.

Đặc biệt:

| Setting      | Kết quả so với baseline chính |
| ------------ | ----------------------------- |
| LLP, bag 8   | +0.024 AUC                    |
| PLL, q = 0.3 | +0.022 accuracy               |
| PLL, q = 0.6 | −0.132 accuracy               |

Ở trường hợp PLL với q = 0.6, các dataset có nhiều lớp gặp hiện tượng chain bị đình trệ.

Sau đó tác giả nghiên cứu response map và phát hiện TabPFN v2 có xu hướng củng cố những nhãn context sai trong một khoảng nhất định.

Trong khi đó, TabICL v2 có response map tích cực trên các dataset được khảo sát, và chain không gặp hiện tượng tương tự.

Điểm này rất đáng chú ý vì nó cho thấy:

Hiệu quả của In-Context Gibbs không chỉ phụ thuộc vào khả năng dự đoán của TFM trên dữ liệu sạch, mà còn phụ thuộc vào cách TFM phản ứng với những context labels không chính xác.

Một mô hình có accuracy cao trên clean labels chưa chắc là mô hình tốt cho Gibbs sampling.

neur.pdf

Đây có thể là đóng góp nghiên cứu phụ rất có giá trị: đánh giá khả năng chống chịu với noisy context labels của tabular foundation models.

Tuy nhiên, nghiên cứu hiện tại chưa biến nó thành một lý thuyết hoàn chỉnh. Bài báo còn để TODO cho phần phân tích collapse threshold.

## 4.3. Study 2 cho kết quả tốt, nhưng cần nhìn đúng độ lớn của hiệu quả

Kết quả chính của Study 2

LLP — AUC improvement

# +0.052

So với linear DLLP; thắng 16/18 datasets; Holm-adjusted p < 0.001.

PLL, q = 0.3

# +0.012

Accuracy improvement so với tuned CC; 8 thắng, 2 hòa; p = 0.016.

PLL, q = 0.6

# +0.012

Accuracy improvement so với tuned CC; 8 thắng, 1 hòa, 1 thua; p = 0.016.

Nguồn: Table 2, Study 2.&#x20;

neur.pdf



Mình đánh giá kết quả này tích cực, nhưng không nên diễn giải quá mức.

LLP là kết quả thuyết phục nhất.

Cải thiện +0.052 AUC trên 18 datasets, với 16 chiến thắng và p-value sau hiệu chỉnh nhỏ hơn 0.001, là một tín hiệu thực nghiệm khá mạnh.

PLL có hiệu quả nhỏ hơn nhiều.

Cải thiện +0.012 accuracy tương đương 1.2 điểm phần trăm. Tuy nhiên, bài báo cho biết kết quả này nằm khá gần mức oracle:

- Ở q = 0.3: cách oracle khoảng 0.8 điểm phần trăm.
- Ở q = 0.6: cách oracle khoảng 1.9 điểm phần trăm.

Điều đó có thể giải thích vì sao mức cải thiện không quá lớn.

Nhưng cũng đặt ra một câu hỏi: liệu phương pháp có thực sự tốt hơn đáng kể về mặt thực tiễn, hay chỉ khai thác thêm một lượng thông tin nhỏ trong một bài toán mà baseline đã giải quyết tương đối tốt?

## 4.4. Vấn đề về tính khái quát hóa của benchmark

Có ba giới hạn đáng chú ý.

Một là, chỉ có ba bộ dữ liệu real-world PLL.

Kết quả trên BirdSong, Lost và MSRCv2 khá tích cực:

| Dataset  | In-Context Gibbs | Mức cải thiện so với baseline tốt nhất |
| -------- | ---------------- | -------------------------------------- |
| BirdSong | 0.759            | +0.028                                 |
| Lost     | 0.750            | +0.002                                 |
| MSRCv2   | 0.628            | +0.045                                 |

Đặc biệt, kết quả Lost gần như không tạo ra khác biệt đáng kể vì bốn phương pháp đứng đầu nằm trong khoảng 0.003.

Ba datasets này chưa đủ để kết luận phương pháp có khả năng xử lý tốt real-world weak labels nói chung.&#x20;

neur.pdf



Hai là, không có real-world LLP dataset.

LLP được đánh giá chủ yếu thông qua việc tạo bag ngẫu nhiên và công bố label proportions. Điều này hữu ích để đánh giá khả năng học từ thông tin aggregate, nhưng chưa đại diện cho tất cả tình huống thực tế, chẳng hạn groups được tạo ra bởi bệnh viện, trường học, khu vực địa lý hoặc các quy trình dữ liệu khác.

Ba là, quy trình tạo weak labels còn khá lý tưởng hóa.

PLL sử dụng mô hình distractor với xác suất độc lập, còn LLP dùng random bags với kích thước cố định là 8 trong setting chính.

Những giả định này thuận tiện cho kiểm soát thực nghiệm nhưng chưa mô tả đầy đủ các quy trình gán nhãn phức tạp ngoài đời.

# 5. Một vấn đề khác: mixing và độ tin cậy của posterior samples

Đây là điểm dễ bị bỏ qua nếu chỉ nhìn accuracy.

Tác giả sử dụng cấu hình:

\\[ K=10,\quad T=30,\quad T_0=10,\quad S=10 \\]

Nghĩa là có 30 sweeps, bỏ 10 sweeps đầu tiên và sử dụng 10 samples để tính dự đoán.

Tuy nhiên, trong phần mixing, tác giả ghi nhận hai chain có sự bất đồng đáng kể ở cấp độ nhãn cá nhân.

Tỷ lệ nhãn có \\(\hat R>1.1\\) nằm trong khoảng 6–31% tùy setting và TFM.

Tác giả chỉ chứng minh rằng kết quả dự đoán trên tập test giữa hai chains tương đối ổn định, chứ không chứng minh tất cả nhãn ẩn đã hội tụ đến posterior distribution.&#x20;

neur.pdf



Đây là hai khái niệm khác nhau:

| Đánh giá              | Ý nghĩa                                                       |
| --------------------- | ------------------------------------------------------------- |
| Stable test accuracy  | Các chains có thể tạo ra hiệu quả dự đoán gần nhau            |
| Posterior convergence | Các chains đã khám phá tốt phân phối xác suất của các nhãn ẩn |

Một sampler có thể dự đoán tương đối ổn định nhưng vẫn chưa khám phá hết posterior.

Mình cho rằng nên bổ sung:

- Nhiều chains với các initialization khác nhau.
- Effective Sample Size (ESS).
- Autocorrelation.
- Monte Carlo Standard Error.
- Trace plots cho một số nhãn đại diện.
- Rank-normalized split-\\(\hat R\\) và đánh giá trên nhiều biến nhãn hơn.
- Phân tích độ nhạy theo số sweeps và kích thước block.

Một chi tiết khác: median difference giữa hai chains trên một số setting PLL với TabICL khoảng 0.018–0.021 accuracy, trong khi mức tăng trung bình của phương pháp so với baseline chỉ là 0.012.

Điều đó không tự động chứng minh kết quả không đáng tin cậy, nhưng khiến việc đánh giá Monte Carlo uncertainty trở nên rất quan trọng.

neur.pdf

## Một giới hạn thực nghiệm nữa: context subsampling

Khi context lớn hơn 2,000 điểm, tác giả subsample ngẫu nhiên context xuống 2,000 điểm.

Điều này giúp kiểm soát chi phí, nhưng cũng tạo ra một khác biệt so với Lemma 1: TFM không còn điều kiện trên toàn bộ các điểm còn lại.

Về bản chất, context subsampling tạo ra một nguồn approximation error bổ sung, bên cạnh sai số vốn có của TFM và sai số block sampling.

Bài báo chưa phân tích riêng mức độ ảnh hưởng của nguồn sai số này.&#x20;

neur.pdf



# 6. Đánh giá novelty: đâu mới là đóng góp thực sự?

Mình sẽ tách novelty thành ba lớp.

## Lớp 1: Conceptual novelty — mạnh

Ý tưởng sử dụng posterior predictive của TFM như full conditional để tiến hành Bayesian inference trên latent labels là đóng góp nổi bật nhất.

Tác giả không chỉ sử dụng TFM làm classifier, mà thay đổi vai trò của mô hình:

| Cách sử dụng thông thường           | In-Context Gibbs                           |
| ----------------------------------- | ------------------------------------------ |
| Dự đoán trực tiếp nhãn              | Ước lượng conditional distribution         |
| Dùng context labels có sẵn          | Dùng các nhãn ẩn đang được lấy mẫu         |
| Một lần forward pass cho prediction | Nhiều lần forward pass trong inference     |
| Mục tiêu là dự đoán                 | Mục tiêu là suy luận posterior của nhãn ẩn |

Đây là sự thay đổi về cách tư duy khi khai thác foundation model, chứ không đơn thuần là thay một backbone bằng một model khác.

Mức novelty mình đánh giá: cao.

## Lớp 2: Algorithmic novelty — tốt nhưng còn phụ thuộc implementation

Đưa PLL và LLP vào cùng một framework là một điểm tích cực.

Tuy vậy, việc sử dụng Gibbs sampling, likelihood weighting, Metropolis–Hastings và dynamic programming đều là những thành phần đã có nền tảng lâu đời.

Đóng góp không nằm ở việc phát minh ra các thành phần riêng lẻ, mà nằm ở cách kết hợp chúng với TFM để tạo ra một framework không cần retraining.

Điểm yếu là block sampler được dùng mặc định chưa có bảo đảm mạnh ở quy mô thực nghiệm.

Mức novelty: tốt, nhưng cần củng cố tính đúng đắn của thuật toán thực tế.

## Lớp 3: Theoretical novelty — trung bình đến khá

Theorem 2 có tính liên hệ chặt chẽ với lý thuyết perturbation của Markov chains.

Bản thân dạng bound:

\\[ \text{stationary distribution error} \lesssim \text{mixing time}\times\text{transition error} \\]

không phải một ý tưởng hoàn toàn mới.

Điểm có giá trị là tác giả áp dụng nó cho sampler được xây dựng từ các conditional distributions có thể không tương thích với nhau.

Tuy nhiên, lý thuyết hiện tại chưa giải quyết được câu hỏi quan trọng nhất ở quy mô thực nghiệm: sai số của block sampler mặc định có kiểm soát được hay không?

Do đó, bài báo có thể được đánh giá là có conceptual contribution tốt và lý thuyết hỗ trợ hợp lý, nhưng chưa có một bảo đảm lý thuyết hoàn chỉnh cho phương pháp chính được sử dụng trong các benchmark.

# 7. Các vấn đề cần ưu tiên sửa trước khi gửi journal hoặc conference

Mình xếp mức độ theo tác động đến tính thuyết phục khoa học, không phải theo độ dễ thực hiện.

| Ưu tiên | Vấn đề                                              | Việc cần làm                                                              |
| ------- | --------------------------------------------------- | ------------------------------------------------------------------------- |
| P0      | Lý thuyết không bao phủ block sampler mặc định      | Phát triển bound hữu ích hơn hoặc đưa MH correction vào phương pháp chính |
| P0      | Chưa cô lập được approximation bias và mixing error | So sánh exact và default sampler ở mức hội tụ tương đương                 |
| P1      | Chưa đánh giá đầy đủ khả năng mixing                | Bổ sung ESS, autocorrelation, MCSE, nhiều chains                          |
| P1      | Chỉ ba bộ real-world PLL, không có real LLP         | Tìm thêm dữ liệu real-world và weak-label mechanisms thực tế              |
| P1      | So sánh TabPFN và TabICL chưa hoàn toàn đối xứng    | Chạy cả hai TFM trên cùng bộ dữ liệu và cùng protocol                     |
| P2      | Context subsampling chưa được phân tích             | Đo ảnh hưởng của giới hạn context 2,000 điểm                              |
| P2      | Chưa phân tích privacy risk của LLP                 | Xây dựng threat model và đánh giá nguy cơ suy luận nhãn cá nhân           |

### Thí nghiệm mình muốn tác giả làm nhất

Một thiết kế factorial experiment:

| TFM       | Sampler       | Context           |
| --------- | ------------- | ----------------- |
| TabPFN v2 | Default block | Full / subsampled |
| TabPFN v2 | MH-corrected  | Full / subsampled |
| TabICL v2 | Default block | Full / subsampled |
| TabICL v2 | MH-corrected  | Full / subsampled |

Cùng dữ liệu, cùng weak-label generation, cùng initialization và nhiều mức compute budget.

Thí nghiệm này trả lời được bốn câu hỏi:

1. TabICL thực sự tốt hơn TabPFN trong noisy-context robustness đến mức nào?
2. Default block sampler có gây ra sai lệch đáng kể không?
3. Context subsampling làm giảm hiệu quả ra sao?
4. Kết quả cải thiện đến từ mô hình TFM, thuật toán sampling hay sự kết hợp của cả hai?

Đây có thể là phần thực nghiệm có giá trị nhất để nâng cấp chất lượng paper.

# 8. Các điểm chưa hoàn thiện trong bản thảo

Ngoài vấn đề khoa học, tài liệu hiện vẫn có những TODO chưa được xử lý:

- Phần collapse-threshold analysis còn TODO ở Section 6.4.
- Phần khai báo sử dụng LLM chưa được điều chỉnh theo chính sách của venue.
- Phần xác nhận license của TabICL v2 checkpoint, OpenML datasets và PALM data còn TODO.
- Phần broader impacts có đề cập nguy cơ từ việc suy luận nhãn cá nhân bằng dữ liệu aggregate, nhưng chưa có đánh giá định lượng về privacy.

Các chi tiết này cần được xử lý trước khi nộp. Đặc biệt, checklist hiện đã đánh dấu phần license là hoàn thành trong khi vẫn có TODO xác nhận license.&#x20;

neur.pdf

neur.pdf



# 9. Đánh giá tổng thể của mình

Đánh giá theo từng khía cạnh

Research problem

8.5/10

Vấn đề có ý nghĩa và phù hợp với hướng phát triển TFM.

Conceptual novelty

8.5/10

Cách dùng TFM như conditional sampler khá thú vị.

Algorithm design

7.5/10

Framework hợp lý nhưng block approximation cần được củng cố.

Theoretical contribution

7/10

Có định lý và phân tích tốt, nhưng chưa bao phủ cấu hình mặc định.

Experimental design

8.5/10

Pre-registration, baseline và kiểm định thống kê là điểm mạnh.

Generalization

6.5/10

Thiếu real LLP và số lượng real PLL còn ít.

Submission readiness

6/10

Còn TODO và những câu hỏi kỹ thuật chưa giải quyết.

Chất lượng nghiên cứu tiềm năng

# 8/10

Đây là đánh giá độc lập của mình, không phải điểm reviewer chính thức.

## Kết luận quan trọng

Mình không đánh giá đây là một paper chỉ có novelty ở mức kết hợp các mô hình có sẵn. Ý tưởng trung tâm có chiều sâu: sử dụng khả năng posterior predictive của foundation models để thực hiện inference trên các biến ẩn, thay vì tiếp tục huấn luyện mô hình bằng weak labels.

Tuy nhiên, có một nghịch lý cần giải quyết:

- Lý thuyết cho thấy khi các điều kiện phù hợp, TFM có thể trở thành một thành phần của Gibbs sampler đúng đắn.
- Thực nghiệm cho thấy phương pháp block approximation có kết quả tốt.
- Nhưng bài báo chưa chứng minh rằng chính phương pháp block approximation ấy đang lấy mẫu gần đúng từ posterior mà lý thuyết nhắm tới.

Đây là khoảng trống giữa theoretical correctness và empirical effectiveness.

Mình sẽ không khuyên tác giả chỉ thêm 20–30 benchmark nữa. Việc cần làm trước tiên là giải quyết khoảng trống này, bởi vì thêm benchmark có thể củng cố hiệu quả thực nghiệm nhưng không giải quyết được câu hỏi về tính đúng đắn của thuật toán.

Phán quyết hiện tại của mình: promising work, nhưng cần major revision về mặt kỹ thuật trước khi có thể tự tin nhắm đến một venue hàng đầu như NeurIPS, ICML hoặc ICLR.

Một hướng có thể nâng tầm bài báo hơn nữa là nghiên cứu sâu mối quan hệ giữa noise robustness của foundation model, convergence của Gibbs sampler và chất lượng weak-label inference. Ba vấn đề này trong bài báo đang được quan sát cùng nhau, nhưng chưa được kết nối thành một lý thuyết thống nhất.