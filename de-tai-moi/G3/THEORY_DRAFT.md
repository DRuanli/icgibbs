# G3 · Bản nháp lý thuyết của đề tài A

> Trạng thái: **bản nháp G3**. Đây là phát biểu dự kiến và phác thảo chứng minh. Chứng minh đầy đủ và kiểm chứng số làm ở G4 theo [07_LY_THUYET_VA_CHUNG_MINH.md](../../research-handbook/07_LY_THUYET_VA_CHUNG_MINH.md). Mọi chỗ "cần kiểm tra" phải được giải quyết trước khi đưa vào bài.

## 0. Ký hiệu
| Ký hiệu | Nghĩa |
|---|---|
| $\phi\sim p(\phi)$ | ẩn của prior (cơ chế sinh dữ liệu) |
| $(x_i,y_i)\overset{iid}{\sim}p(x,y\mid\phi)$, $i=1..n$ | dữ liệu train; $y_i\in\{1,\dots,C\}$ là nhãn ẩn |
| $X=(x_1..x_n)$, $y=(y_1..y_n)$, $y_{-i}$ | ma trận feature, vector nhãn, nhãn trừ mẫu $i$ |
| $D_{-i}=\{(x_j,y_j)\}_{j\ne i}$ | ngữ cảnh có nhãn trừ mẫu $i$ |
| $p(y_*\mid x_*,D)$ | posterior predictive (PPD) với nhãn sạch dưới prior |
| $q(y_*\mid x_*,D)$ | dự đoán của TFM (ví dụ TabPFN) |
| $W$ | quan sát yếu |
| $\pi(y)=p(y\mid X,W)$ | posterior của nhãn ẩn, là mục tiêu của bộ lấy mẫu |

Hai loại quan sát yếu:
- **Cấp mẫu:** $w_i\sim p(w\mid y_i,x_i)$, độc lập có điều kiện khi biết $(y_i,x_i)$. Viết $\ell_i(y_i)=p(w_i\mid y_i,x_i)$.
  - PU: $w_i=1\Rightarrow\ell_i=\mathbb 1[y_i=1]$; với $w_i=0$, $\ell_i(1)=1-e(x_i)$ và $\ell_i(0)=1$, trong đó $e(x)$ là propensity. SCAR là trường hợp $e$ hằng.
  - PLL: $\ell_i(y)=\mathbb 1[y\in S_i]$ (mô hình tập ứng viên phổ biến; điều kiện "tập ứng viên không phụ thuộc nhãn đúng" cần nêu rõ).
  - Nhãn nhiễu: $\ell_i(y)=T_{y,w_i}$.
  - SSL: $\ell_i\equiv1$ với mẫu không nhãn; $\ell_i=\mathbb 1[y=w_i]$ với mẫu có nhãn.
- **Cấp nhóm (LLP):** các túi $B_1..B_G$ phân hoạch $\{1..n\}$, và $w_g=\mathrm{counts}(y_{B_g})$. Hàm hợp lý là $\prod_g\mathbb 1[\mathrm{counts}(y_{B_g})=w_g]$.

## 1. Bổ đề cốt lõi: dự đoán của PFN là xác suất có điều kiện đầy đủ

**Bổ đề 1.** Dưới prior trên, với mọi $i$:
$$p(y_i\mid y_{-i},X)=p(y_i\mid x_i,D_{-i}).$$

*Chứng minh.* $(X,y_{-i})$ và $(x_i,D_{-i})$ là cùng một thông tin. Vế phải chính là PPD của điểm truy vấn $x_i$ với ngữ cảnh $D_{-i}$. $\square$

**Lưu ý (cần nêu rõ trong bài).** Bổ đề chỉ đúng khi *mọi* mẫu khác đều có mặt trong ngữ cảnh cùng nhãn hiện tại của chúng. Nếu một mẫu $j$ chỉ có feature mà không có nhãn trong ngữ cảnh, TFM không điều kiện hoá trên $x_j$, trong khi $p(\cdot\mid X,\dots)$ thì có (feature mang thông tin về $\phi$). Vì vậy:
- **cập nhật một mẫu** (single-site) là chính xác;
- **cập nhật theo khối**, khi các mẫu trong khối bị loại khỏi ngữ cảnh, chỉ là xấp xỉ (Mệnh đề 4).

## 2. Định lý 1 (tính đúng)

**Giả định.**
- (A1) $q=p$ với mọi ngữ cảnh có kích thước tới $n-1$.
- (A2) $p(y\mid x,D)>0$ với mọi $y$ (đúng với đầu ra softmax).
- (A3) Mô hình quan sát yếu $\ell$ đã biết, và $\pi$ có ít nhất một trạng thái dương.

**Thuật toán.**
- **(a) Quan sát cấp mẫu**, Gibbs một mẫu (quét ngẫu nhiên hoặc tuần tự):
$$y_i\sim\pi(y_i\mid y_{-i})\propto q(y_i\mid x_i,D_{-i})\,\ell_i(y_i).$$
- **(b) LLP**, Metropolis–Hastings hoán đổi:
  - chọn đều một túi $g$;
  - chọn đều một cặp $(i,j)$ trong túi có $y_i\ne y_j$;
  - đề xuất $y'$ bằng cách đổi chỗ $y_i,y_j$;
  - chấp nhận với xác suất $\alpha=\min(1,R)$, trong đó
$$R=\frac{q(y_j\mid x_i,D_{-i})}{q(y_i\mid x_i,D_{-i})}\cdot\frac{q(y_i\mid x_j,D'_{-j})}{q(y_j\mid x_j,D'_{-j})},$$
  với $D'_{-j}$ là ngữ cảnh trừ $j$ sau khi đã gán $y_i\leftarrow y_j$. Mỗi đề xuất cần 2 lần forward.

**Phát biểu.** Dưới (A1)–(A3):
1. (a) và (b) có phân phối dừng $\pi$, là chuỗi bất khả quy và (với phiên bản "lazy") không tuần hoàn trên tập hỗ trợ của $\pi$.
2. Với chuỗi $y^{(1)},y^{(2)},\dots$, ước lượng Rao–Blackwell
$$\hat p_S(y_*\mid x_*)=\frac1S\sum_{s=1}^S q(y_*\mid x_*,(X,y^{(s)}))$$
hội tụ hầu chắc chắn về $p(y_*\mid x_*,X,W)$, tức posterior predictive Bayes khi chỉ có nhãn yếu.

*Phác thảo chứng minh.*
- (a): theo Bổ đề 1, $q(y_i\mid x_i,D_{-i})\,\ell_i(y_i)\propto\pi(y_i\mid y_{-i})$, vì $\pi(y)\propto p(y\mid X)\prod_j\ell_j(y_j)$. Đây là Gibbs chuẩn. Bất khả quy vì mọi xác suất có điều kiện dương trên tập hỗ trợ tích $\prod_i\{y:\ell_i(y)>0\}$ (A2).
- (b): **bổ đề Brook** cho tỉ số hai phân phối đồng thời qua các xác suất có điều kiện đầy đủ:
$$\frac{p(y'\mid X)}{p(y\mid X)}=\frac{p(y'_i\mid y_{-i},X)}{p(y_i\mid y_{-i},X)}\cdot\frac{p(y'_j\mid y'_i,y_{-ij},X)}{p(y_j\mid y'_i,y_{-ij},X)}.$$
  Áp Bổ đề 1 cho từng thừa số được $R$. Đề xuất đối xứng: số cặp khác nhãn trong túi chỉ phụ thuộc số đếm, mà số đếm bất biến khi hoán đổi. Các hoán vị trong một túi sinh ra mọi cách gán có cùng số đếm, nên chuỗi bất khả quy.
- Hội tụ của $\hat p_S$: định lý ergodic cho chuỗi Markov hữu hạn trạng thái, cộng với $p(y_*\mid x_*,X,y,W)=p(y_*\mid x_*,X,y)$ vì $W$ độc lập với $y_*$ khi biết $(X,y)$. $\square$

**Cần kiểm tra ở G4:**
- (i) Điều kiện của PLL ("tập ứng viên chứa nhãn đúng" và mô hình sinh tập ứng viên) có khớp với $\ell_i$ không.
- (ii) Dạng $\ell$ của PU-SAR khi propensity $e(x)$ chưa biết. Đề xuất: thêm tham số $\eta$ của $e(\cdot)$ vào trạng thái, cùng bước $\eta\sim p(\eta\mid y,X,W)\propto p(\eta)\prod_i\ell_i(y_i;\eta)$. Bước này không cần TFM.
- (iii) Lời giải thích không thiên vị cho hiện tượng chuỗi trộn chậm khi nhãn yếu rất thông tin.

## 3. Định lý 2 (độ bền khi TFM chỉ xấp xỉ)

**Giả định.**
- (B1) $\sup_{x,D}\mathrm{TV}\big(q(\cdot\mid x,D),p(\cdot\mid x,D)\big)\le\varepsilon$.
- (B2) Trọng số của nhãn yếu bị chặn so với khối lượng mà PPD đặt lên tập hỗ trợ:
$$\kappa:=\sup_{i,D}\frac{\max_y\ell_i(y)}{\sum_y p(y\mid x_i,D)\,\ell_i(y)}<\infty .$$
- (B3) Nhân của một vòng quét chính xác $P$ (dùng $p$) là ergodic đều: $\|P^t(y,\cdot)-\pi\|_{TV}\le C\rho^t$.

**Phát biểu dự kiến.**
1. Nhân dùng $q$, ký hiệu $\tilde P$, có phân phối dừng duy nhất $\tilde\pi$. Điều này đúng **kể cả khi các xác suất có điều kiện của $q$ không tương thích** với bất kỳ phân phối đồng thời nào, vì không gian trạng thái hữu hạn và nhân dương trên tập hỗ trợ.
2. Có chặn
$$\|\tilde\pi-\pi\|_{TV}\le\Big(\hat t+\frac{C\rho^{\hat t}}{1-\rho}\Big)\,\|\tilde P-P\|,\qquad \|\tilde P-P\|\le 2\,n\,\kappa\,\varepsilon,\qquad \hat t=\lceil\log_\rho(1/C)\rceil,$$
  theo chặn nhiễu loạn của Mitrophanov (2005).
3. Hệ quả: $\big|\mathbb E_{\tilde\pi}[q(y_*\mid\cdot)]-\mathbb E_{\pi}[p(y_*\mid\cdot)]\big|\le\varepsilon+\|\tilde\pi-\pi\|_{TV}$.

**Phác thảo chứng minh.** Với một cập nhật một mẫu, TV giữa hai phân phối đã chuẩn hoá lại theo $\ell_i$ bị chặn bởi $2\kappa\varepsilon$; một vòng quét gồm $n$ cập nhật, nên các chặn cộng lại. Sau đó áp định lý 3.1 của Mitrophanov (2005) cho chuỗi hữu hạn trạng thái ergodic đều.

**Cần kiểm tra ở G4:**
- (i) Hằng số và dạng chính xác của chặn Mitrophanov.
- (ii) Hệ số $n$ làm chặn rất lỏng khi $n$ lớn. Cần một chặn theo từng mẫu dùng hệ số co (contraction coefficient), hoặc chấp nhận chặn lỏng và đo thực nghiệm. Ghi trung thực trong phần Limitations.
- (iii) Phiên bản cho MH hoán đổi (LLP).
- (iv) So với Liu et al. (*Biometrika* 2014) về imputation lặp với các xác suất có điều kiện không tương thích (trường hợp liên tục), để định vị kết quả.

## 4. Mệnh đề 3 (Bayes-optimal hữu hạn mẫu)
Dưới (A1), $p(y_*\mid x_*,X,W)$ cực tiểu hoá log loss kỳ vọng, và rủi ro Bayes của mọi quyết định dựa trên một proper scoring rule, trong mọi bộ dự đoán là hàm của $(X,W,x_*)$, với kỳ vọng lấy theo prior.

*Chứng minh:* tính chất của posterior predictive, như Chương 7 của khóa học, áp dụng với dữ liệu quan sát là $(X,W)$.

*Ý nghĩa:* khác với các phương pháp nhãn yếu dựa trên ERM (thường chỉ nhất quán tiệm cận), đảm bảo ở đây là **hữu hạn mẫu, nhưng có điều kiện theo prior**. Phải nói rõ điều kiện này.

## 5. Mệnh đề 4 (cập nhật theo khối)
Chia $\{1..n\}$ thành $K$ khối. Cập nhật đồng thời khối $b$ theo $\prod_{i\in b}q(y_i\mid x_i,D_{-b})\,\ell_i(y_i)$. Cách này tốn $K$ forward mỗi vòng quét thay vì $n$, nhưng sai ở hai chỗ:
- (i) bỏ qua phụ thuộc giữa các nhãn *trong* khối;
- (ii) bỏ $x_j$ của các mẫu khác trong khối khỏi điều kiện hoá (Lưu ý ở Mục 1).

**Phát biểu dự kiến:** sai lệch của phân phối dừng tiến về 0 khi kích thước khối nhỏ so với $n$, với tốc độ phụ thuộc mức "ảnh hưởng" của một mẫu lên PPD. Cần một giả định kiểu ổn định (stability): thêm hay bớt một điểm khỏi ngữ cảnh làm PPD đổi tối đa $\beta_n\to0$.

**Cần kiểm tra:** đây là phần lý thuyết rủi ro nhất. Nếu không chứng minh được thì giữ ở dạng *đánh giá thực nghiệm* (RQ1) và ghi là giới hạn.

## 6. Kế hoạch kiểm chứng số (G4, trước khi tin)
| Kiểm chứng | Cách làm | Tiêu chí đạt |
|---|---|---|
| Bổ đề 1 và Định lý 1 với PPD chính xác | Prior "ô–Beta–Bernoulli": $x$ thuộc $M$ ô; $y\mid$ ô $\sim\mathrm{Bernoulli}(\theta_c)$, $\theta_c\sim\mathrm{Beta}(1,1)$. PPD đóng là $(k_c+1)/(n_c+2)$. Với $n\le14$, liệt kê toàn bộ $2^n$ cách gán để có $\pi$ chính xác. Chạy (a) cho PU và SSL, (b) cho LLP với PPD giải tích | TV giữa tần suất của chuỗi và $\pi$ giảm như $O(S^{-1/2})$; ở $S=10^5$ thì TV < 0,01 |
| Định lý 2 với TFM xấp xỉ | Huấn luyện một PFN nhỏ (kiểu `coin_pfn.py`) trên đúng prior trên; đo $\varepsilon$ thực nghiệm, và $\|\tilde\pi-\pi\|$ bằng liệt kê | $\|\tilde\pi-\pi\|_{TV}$ nằm dưới chặn; ghi lại độ lỏng |
| Mệnh đề 4 | So khối $K\in\{2,5,10\}$ với một mẫu, trên cùng prior | Báo cáo sai lệch theo $K$ |
| Tính đúng của cài đặt hoán đổi LLP | Kiểm tra cân bằng chi tiết (detailed balance) bằng số trên một túi 6 phần tử | Sai lệch dưới $10^{-12}$ |

Toàn bộ kiểm chứng ở bảng này chạy được trên CPU (MacBook M1).
