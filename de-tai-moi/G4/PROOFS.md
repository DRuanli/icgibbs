# G4 · Chứng minh đầy đủ (đề tài A: In-Context Gibbs)

> Trạng thái: **bản G4, đã qua một lượt tự rà soát phản biện ([SELF_REVIEW_MATH.md](SELF_REVIEW_MATH.md)) và một lượt kiểm duyệt độc lập (26/10/2026: 0 Critical, 4 Major, 12 Minor — [phản hồi](../G6/G6_T7_REVIEW_RESPONSE.md)); các chỗ sửa đánh dấu "sau kiểm duyệt"**. §7 thêm ở G6 (T7, 26/10/2026), có kiểm chứng số trong `code/g6_t7_naive_theory.py` và `code/tests/test_t7_naive_theory.py`. Thay cho [THEORY_DRAFT.md](../G3/THEORY_DRAFT.md). Mỗi kết quả có: giả định, phát biểu, chứng minh, và trỏ tới kiểm chứng số trong [RQ1_REPORT.md](RQ1_REPORT.md). Những chỗ khác với bản nháp G3 được đánh dấu **[đổi so với G3]** và ghi lý do.
>
> Quy ước theo [07_LY_THUYET_VA_CHUNG_MINH.md](../../research-handbook/07_LY_THUYET_VA_CHUNG_MINH.md): mỗi giả định được dùng ở đâu thì ghi ở đó; kết quả nào chỉ có bằng chứng số thì gọi là *quan sát*, không gọi là định lý.

## 0. Thiết lập và ký hiệu

| Ký hiệu | Nghĩa |
|---|---|
| $\phi\sim p(\phi)$ | ẩn của prior (cơ chế sinh dữ liệu) |
| $(x_i,y_i)\overset{iid}{\sim}p(x,y\mid\phi)$, $i=1..n$ | dữ liệu train, $y_i\in[C]=\{1..C\}$ là nhãn ẩn |
| $(x_*,y_*)\sim p(x,y\mid\phi)$ | điểm test, độc lập với train khi biết $\phi$ |
| $X,\ y,\ y_{-i},\ y_B$ | feature, vector nhãn, nhãn trừ mẫu $i$, nhãn trong tập $B$ |
| $D_{-i}=\{(x_j,y_j)\}_{j\ne i}$, $D_{-B}$ | ngữ cảnh có nhãn trừ $i$ (trừ $B$) |
| $p(y_*\mid x_*,D)$ | PPD: phân phối của nhãn điểm truy vấn khi biết ngữ cảnh $D$ **và** feature $x_*$ của chính nó |
| $p(y_*\mid x_*,D,X_u)$ | PPD tổng quát: thêm các feature không nhãn $X_u$ vào phần được điều kiện hoá (dùng ở Mệnh đề 4(b)); TFM không tính được đại lượng này |
| $q(y_*\mid x_*,D)$ | dự đoán của TFM |
| $W$ | quan sát yếu; $\ell(y)=p(W\mid X,y)$ |
| $\pi(y)=p(y\mid X,W)\propto p(y\mid X)\,\ell(y)$ | mục tiêu của bộ lấy mẫu |
| $\mathrm{TV}(\mu,\nu)=\tfrac12\|\mu-\nu\|_1$ | khoảng cách biến phân toàn phần |

**Giả định chung (A0) — cơ chế quan sát.** $W\sim p(W\mid X,y)$ (có thể kèm tham số phiền nhiễu $\theta_W$ độc lập tiên nghiệm với $(\phi,X,y)$), và $W$ độc lập với $(x_*,y_*)$ khi biết $(X,y)$.
- Cấp mẫu: $\ell(y)=\prod_i\ell_i(y_i)$, $\ell_i(k)=p(w_i\mid y_i=k,x_i)$. PU-SCAR: $\ell_i=(0,c)$ nếu $s_i=1$, $(1,1-c)$ nếu $s_i=0$ (lớp dương là lớp thứ hai). PU-SAR: thay $c$ bằng $e(x_i)$. PLL: $\ell_i(k)=\mathbb 1[k\in S_i]$, dưới giả định chuẩn $p(S_i\mid y_i=k,x_i)$ như nhau với mọi $k\in S_i$. Nhãn nhiễu: $\ell_i(k)=T_{k,w_i}$. SSL: $\ell_i\equiv1$ nếu không nhãn, $\mathbb 1[k=w_i]$ nếu có nhãn.
- Cấp nhóm (LLP): túi $B_1..B_G$ rời nhau, $\ell(y)=\prod_g\mathbb 1[\mathrm{counts}(y_{B_g})=w_g]$.
- *Điều kiện để các dạng $\ell$ trên là đúng* **[thêm sau kiểm duyệt độc lập, F8]**: việc chọn mẫu có nhãn (SSL) và việc tạo túi (LLP) độc lập với $y$ khi biết $X$; với SSL, xác suất có nhãn như nhau ở mọi lớp. Bộ sinh SSL của giao thức gán đúng $\max(2,\mathrm{round}(\text{mức}\cdot n_k))$ mẫu mỗi lớp, nên vi phạm nhẹ điều kiện này (ví dụ ở mức 0,05: lớp có 30 mẫu được gán 2/30 = 0,067, lớp có 270 mẫu được gán 14/270 = 0,052). Đây là sai lệch mô hình nhỏ đã biết của bối cảnh phụ SSL; không đổi bộ sinh vì giao thức đã đóng băng.

**(A1) — TFM chính xác.** $q(\cdot\mid x,D)=p(\cdot\mid x,D)$ với mọi ngữ cảnh $D$ mà thuật toán dùng.
**(A2) — Dương.** PPD của prior thoả $p(k\mid x,D)>0$ với mọi $k$ và mọi ngữ cảnh có xác suất dương. Đây là giả định về prior. Tính dương của softmax chỉ cho $q>0$ trên các ngữ cảnh mà thuật toán thực sự truy vấn, trong khi Hệ quả 1.1 dùng (A2) cho cả các ngữ cảnh nằm ngoài $\mathrm{supp}(\pi)$ trên đường Brook **[sửa sau kiểm duyệt, F7]**.
**(A3) — Hợp lý.** $\ell$ đã biết (hoặc tham số phiền nhiễu được lấy mẫu trong chuỗi, xem Nhận xét 1.3) và $\sum_y p(y\mid X)\ell(y)>0$.

## 1. Bổ đề 1 và hệ quả

**Bổ đề 1.** Với mọi $i$ và mọi $y$ sao cho $p(y_{-i}\mid X)>0$ (với feature liên tục: với $p(X)$-hầu hết $X$): $\;p(y_i\mid y_{-i},X)=p(y_i\mid x_i,D_{-i})$.

*Chứng minh.* Các bộ $(x_j,y_j)$ độc lập cùng phân phối khi biết $\phi$, nên trao đổi được. Do đó luật đồng thời của $(D_{-i},(x_i,y_i))$ trùng với luật của $(D_{n-1},(x_*,y_*))$, trong đó $D_{n-1}$ là ngữ cảnh $n-1$ mẫu và $(x_*,y_*)$ là truy vấn. Thông tin $(X,y_{-i})$ trùng với $(x_i,D_{-i})$. Vì vậy
$$p(y_i\mid y_{-i},X)=\frac{p\big((x_i,y_i),D_{-i}\big)}{\sum_k p\big((x_i,k),D_{-i}\big)}=p(y_i\mid x_i,D_{-i}).\qquad\square$$

*Ghi chú.* PPD ở đây điều kiện hoá cả trên $x_*$; feature của truy vấn có thể mang thông tin về $\phi$ qua $p(x\mid\phi)$. Đây đúng là đại lượng mà loss của PFN nhắm tới, vì trong lúc huấn luyện $x_*$ được sinh từ prior cùng với ngữ cảnh (Müller et al., ICLR 2022; khóa học, Chương 7). Kiểm chứng: `test_lemma1_full_conditional_equals_ppd` (sai lệch < $10^{-12}$ trên mọi $2^7$ cách gán, cả prior ô và prior hỗn hợp).

**Hệ quả 1.1 (dương).** Dưới (A2), $p(y\mid X)>0$ với mọi $y\in[C]^n$.
*Chứng minh (quy nạp dọc đường Brook; viết lại sau kiểm duyệt, F7).* Tồn tại $y^0$ với $p(y^0\mid X)>0$ vì tổng bằng 1. Xét dãy đổi từng mẫu $y^0\to y^1\to\dots\to y^K=y$, bước $k$ đổi mẫu $s_k$. Giả sử $p(y^{k-1}\mid X)>0$. Khi đó $p(y^{k-1}_{-s_k}\mid X)>0$, nên ngữ cảnh $D^{k-1}_{-s_k}$ có xác suất dương, Bổ đề 1 áp được và cho $p(\cdot\mid y^{k-1}_{-s_k},X)=p(\cdot\mid x_{s_k},D^{k-1}_{-s_k})$, dương ở mọi lớp theo (A2). Vì $y^k_{-s_k}=y^{k-1}_{-s_k}$, ta có $p(y^k\mid X)=p(y^{k-1}_{-s_k}\mid X)\,p(y^k_{s_k}\mid y^{k-1}_{-s_k},X)>0$. Quy nạp tới $k=K$. $\square$

**Nhận xét 1.2 (cập nhật khối không chính xác).** Với $|B|\ge2$, nói chung $p(y_B\mid y_{-B},X)\ne\prod_{i\in B}p(y_i\mid x_i,D_{-B})$, vì hai lý do: (i) các nhãn trong $B$ phụ thuộc nhau khi biết $D_{-B}$; (ii) vế phải không điều kiện hoá trên $x_j$, $j\in B\setminus\{i\}$, trong khi $x_j$ mang thông tin về $\phi$ trừ khi $p(x\mid\phi)$ không phụ thuộc $\phi$. Kiểm chứng: E5a (prior ô chỉ có nguồn (i); prior hỗn hợp có cả hai).

**Nhận xét 1.3 (tham số phiền nhiễu).** Với $\theta_W$ độc lập tiên nghiệm, $\pi(y,\theta_W)\propto p(y\mid X)\,p(\theta_W)\,\ell(y;\theta_W)$, nên Bổ đề 1 vẫn cho cập nhật $y_i$ khi biết $\theta_W$, và $\theta_W\mid y$ không cần TFM. Cập nhật liên hợp: PU-SCAR $c\mid y\sim\mathrm{Beta}(a+L,\,b+P-L)$ với $L=\#\{s_i=1\}$, $P=\#\{y_i=1\}$; nhãn nhiễu $T_{k,\cdot}\mid y\sim\mathrm{Dir}(\beta_k+N_{k,\cdot})$. PU-SAR: random-walk MH trên $\eta$ của $e(x)=\sigma(\eta^\top\tilde x)$.

## 2. Định lý 1 (tính đúng)

### 2(a) Gibbs một mẫu cho quan sát cấp mẫu

Gọi $F=\{i:|\{k:\ell_i(k)>0\}|\ge2\}$ là các mẫu tự do. Nhân cập nhật mẫu $i$:
$$P_i(y,y')=\pi_i(y'_i\mid y_{-i})\,\mathbb 1[y'_{-i}=y_{-i}],\qquad \pi_i(k\mid y_{-i})=\frac{q(k\mid x_i,D_{-i})\,\ell_i(k)}{\sum_{k'}q(k'\mid x_i,D_{-i})\,\ell_i(k')}.$$
Quét ngẫu nhiên: $P=\frac1{|F|}\sum_{i\in F}P_i$. Cài đặt dùng **một vòng quét theo hoán vị ngẫu nhiên**: $P_{sw}=\frac1{|F|!}\sum_{\sigma}P_{\sigma(1)}\cdots P_{\sigma(|F|)}$. Mỗi $P_i$ giữ $\pi$ nên $P_{sw}$ giữ $\pi$; $P_{sw}$ bất khả quy vì trong một vòng quét mỗi mẫu có thể nhận bất kỳ giá trị nào trong tập hỗ trợ của nó với xác suất dương, và không tuần hoàn vì $P_{sw}(y,y)>0$. Dưới (A1), $P_{sw}$ **khả nghịch** theo $\pi$: mỗi $P_i$ tự liên hợp trong $L^2(\pi)$, liên hợp của một tích là tích theo thứ tự ngược, và tập các hoán vị đóng dưới phép đảo thứ tự, nên $P_{sw}^*=P_{sw}$. Quét theo một thứ tự cố định thì nói chung không khả nghịch, và $\tilde P_{sw}$ với $q$ không tương thích cũng vậy. Định lý 2 không cần khả nghịch. **[Sửa sau kiểm duyệt, F5: bản trước viết sai rằng $P_{sw}$ nói chung không khả nghịch. Kiểm số: sai số cân bằng chi tiết của $P_{sw}$ ≤ 2e-16; quét cố định 1e-4 tới 2e-2.]**

**Định lý 1(a).** Dưới (A0)–(A3) với $\ell$ đã biết:
1. mỗi $P_i$ thoả cân bằng chi tiết với $\pi$;
2. $P$ bất khả quy và không tuần hoàn trên $\mathrm{supp}(\pi)=\prod_i\{k:\ell_i(k)>0\}$, nên $\max_y\mathrm{TV}(P^t(y,\cdot),\pi)\le C_{\rm erg}\rho^t$ với $C_{\rm erg}<\infty$, $\rho<1$.

*Chứng minh.* (1) Theo Bổ đề 1 và (A1), $\pi_i(k\mid y_{-i})\propto p(k\mid y_{-i},X)\,\ell_i(k)\propto\pi(k,y_{-i})$, tức $\pi_i$ chính là xác suất có điều kiện đầy đủ của $\pi$. Khi đó $\pi(y)P_i(y,y')=\pi(y_{-i})\,\pi_i(y_i\mid y_{-i})\,\pi_i(y'_i\mid y_{-i})$ đối xứng theo $(y,y')$.
(2) Theo Hệ quả 1.1, $\pi(y)>0\iff\ell(y)>0$, và với $\ell$ dạng tích thì tập này là tích các tập hỗ trợ. Từ $y$ tới $y'$ trong tập đó, đổi lần lượt từng mẫu khác nhau; mọi trạng thái trung gian vẫn thuộc tập tích, và mỗi bước có xác suất $\ge\pi_i(\cdot\mid\cdot)/|F|>0$. Không tuần hoàn vì $P(y,y)\ge\pi_i(y_i\mid y_{-i})/|F|>0$. Chuỗi hữu hạn trạng thái, bất khả quy, không tuần hoàn thì hội tụ đều theo cấp số nhân. $\square$

### 2(b) MH hoán đổi cho LLP

Đề xuất: chọn đều túi $g$; nếu túi thuần nhất thì đứng yên; nếu không, chọn đều một cặp **có thứ tự** $(i,j)$ trong túi với $y_i\ne y_j$ (có $N_g(y)$ cặp), đề xuất $y'$ bằng cách đổi chỗ $y_i,y_j$. Chấp nhận với $\alpha=\min(1,R_{ij})$,
$$R_{ij}(y)=\frac{q(y_j\mid x_i,D_{-i})}{q(y_i\mid x_i,D_{-i})}\cdot\frac{q(y_i\mid x_j,D'_{-j})}{q(y_j\mid x_j,D'_{-j})},$$
với $D'_{-j}$ là ngữ cảnh trừ $j$ sau khi gán $y_i\leftarrow y_j$. Bản "lười" (lazy): với xác suất $\tfrac12$ đứng yên.

**Định lý 1(b).** Dưới (A0)–(A2) với $\ell$ là ràng buộc số đếm, nhân hoán đổi thoả cân bằng chi tiết với $\pi$ và bất khả quy trên $\mathrm{supp}(\pi)$; bản lazy thì thêm không tuần hoàn.

*Chứng minh.* Theo bổ đề Brook đi qua trạng thái trung gian $(y'_i,y_{-i})$:
$$\frac{p(y'\mid X)}{p(y\mid X)}=\frac{p(y'_i\mid y_{-i},X)}{p(y_i\mid y_{-i},X)}\cdot\frac{p(y'_j\mid y'_i,y_{-ij},X)}{p(y_j\mid y'_i,y_{-ij},X)},$$
và Bổ đề 1 cùng (A1) biến vế phải thành $R_{ij}(y)$. Vì $\ell(y')=\ell(y)$ (hoán đổi giữ số đếm), $R_{ij}(y)=\pi(y')/\pi(y)$ — **không phụ thuộc thứ tự** $(i,j)$ hay $(j,i)$ khi (A1) đúng. Số cặp khác nhãn chỉ phụ thuộc số đếm nên $N_g(y')=N_g(y)$; xác suất đề xuất $y\to y'$ (qua hai thứ tự) bằng $2/(G\,N_g)$, đối xứng. Vậy đây là Metropolis với đề xuất đối xứng: $\pi(y)P(y,y')=\frac{2}{GN_g}\min(\pi(y),\pi(y'))$ đối xứng.
Bất khả quy: trong một túi, hai cách gán cùng số đếm nối với nhau bằng một dãy hoán đổi hai vị trí khác nhãn (sắp xếp lại từng vị trí), mọi trạng thái trung gian giữ số đếm; mỗi bước có xác suất dương vì đề xuất dương và $\alpha>0$ theo Hệ quả 1.1. Các túi độc lập nên tích bất khả quy. $\square$

*Lưu ý cài đặt.* Khi (A1) không đúng, $R_{ij}\ne R_{ji}$ nói chung: chuỗi vẫn là một nhân Markov hợp lệ nhưng không còn khả nghịch. Định lý 2 không cần khả nghịch.

### 2(c) Tham số phiền nhiễu chưa biết (PU-SCAR, nhãn nhiễu)

**Mệnh đề 1.4.** Với cập nhật liên hợp của Nhận xét 1.3 xen kẽ một vòng quét Gibbs của $y$, dãy $(y^{(t)})$ là một chuỗi Markov hữu hạn trạng thái với phân phối dừng là biên $\pi(y)$, bất khả quy và không tuần hoàn.
*Chứng minh.* Đặt $K_\theta$ là nhân của một vòng quét $y$ khi biết $\theta$ (giữ $\pi(\cdot\mid\theta)$ theo 2(a)). Dãy $y^{(t)}$ có nhân $\bar P(y,y')=\int\pi(\theta\mid y)K_\theta(y,y')\,d\theta$ (lược đồ data augmentation hai khối). Khi đó $\sum_y\pi(y)\bar P(y,y')=\int\pi(\theta)\sum_y\pi(y\mid\theta)K_\theta(y,y')\,d\theta=\int\pi(\theta)\pi(y'\mid\theta)\,d\theta=\pi(y')$. Với mọi $\theta$ trong miền của prior, $K_\theta$ có các phần tử dương như ở 2(a), nên $\bar P$ bất khả quy và không tuần hoàn. $\square$
PU-SAR với MH trên $\eta$: dãy $y$ không còn Markov riêng. Chuỗi trên $[C]^n\times\mathbb R^d$ gồm Gibbs cho $y$ và random-walk MH Gauss cho $\eta$; với prior Gauss và mật độ đề xuất dương khắp nơi, chuỗi $\pi$-bất khả quy và không tuần hoàn, nên hội tụ theo TV với $\pi$-hầu hết điểm đầu (Tierney 1994; số hiệu định lý cần kiểm, G4-1). **Vấn đề nặng hơn [thêm sau kiểm duyệt, F16]:** ở PU-SAR, $\eta$ định danh yếu từ $(x,s)$, nên posterior của $\eta$ do prior và TFM quyết định; điều này ảnh hưởng tốc độ trộn và ý nghĩa của "mục tiêu Bayes". PU-SAR nay là bối cảnh phụ.

### 2(d) Dự đoán: quy nạp và chuyển nạp **[đổi so với G3]**

Bản nháp G3 khẳng định ước lượng Rao–Blackwell hội tụ về $p(y_*\mid x_*,X,W)$. **Điều này không đúng tổng quát khi feature mang thông tin về $\phi$; điều kiện đủ là prior tách được (dưới đây).** (Bản trước viết "chỉ đúng khi": điều kiện cần này chưa được chứng minh và sai — ví dụ khi $W$ xác định hoàn toàn $y$ thì hai đại lượng trùng nhau với mọi prior; sửa sau kiểm duyệt, F10.) Phát biểu đúng:

**Định lý 1(c).** Đặt $\bar p(y_*\mid x_*)=\sum_y\pi(y)\,p(y_*\mid x_*,(X,y))$. Dưới (A0)–(A3):
1. *(quy nạp)* Với chuỗi của 2(a)/(b), $\hat p_S=\frac1S\sum_{s\le S}q(y_*\mid x_*,(X,y^{(s)}))\to\bar p(y_*\mid x_*)$ hầu chắc chắn, với mọi trạng thái đầu.
2. $p(y_*\mid x_*,X,W)=\sum_y p(y\mid x_*,X,W)\,p(y_*\mid x_*,(X,y))$, nên $|\bar p-p(y_*\mid x_*,X,W)|\le\mathrm{TV}\big(p(y\mid X,W),\,p(y\mid x_*,X,W)\big)$; dấu bằng $\bar p=p(y_*\mid x_*,X,W)$ xảy ra khi prior **tách được**: $\phi=(\phi_x,\phi_{y\mid x})$ độc lập tiên nghiệm, $p(x\mid\phi)=p(x\mid\phi_x)$ và $p(y\mid x,\phi)=p(y\mid x,\phi_{y\mid x})$ (trường hợp riêng: $p(x\mid\phi)$ không phụ thuộc $\phi$).
3. *(chuyển nạp)* Thêm $x_*$ vào chuỗi như một mẫu tự do ($\ell_*\equiv1$). Chuỗi mới chính xác với mục tiêu $p(y,y_*\mid X,x_*,W)$ — theo 2(a) với quan sát cấp mẫu; với LLP, chuỗi gồm bước hoán đổi (2(b)) cộng Gibbs một mẫu cho $x_*$, mỗi thành phần giữ $\pi$ và tích bất khả quy. Ước lượng Rao–Blackwell $\frac1S\sum_s\pi_*(y_*\mid y^{(s)})=\frac1S\sum_s q(y_*\mid x_*,(X,y^{(s)}))$ hội tụ về **đúng** $p(y_*\mid x_*,X,W)$. Với $m\ge2$ điểm test cùng lúc, chuỗi hội tụ về $p(y_{*k}\mid X_*,X,W)$ (điều kiện hoá trên mọi feature test), và số hạng Rao–Blackwell là $q(y_{*k}\mid x_{*k},(X,y^{(s)})\cup(X_{*,-k},y^{(s)}_{*,-k}))$, tức các điểm test khác cũng nằm trong ngữ cảnh. Cài đặt ghi Rao–Blackwell tại lúc cập nhật $x_*$ trong vòng quét; cách này vẫn đúng vì mọi trạng thái trung gian có phân phối dừng $\pi$ **[bổ sung sau kiểm duyệt, F9]**.

*Chứng minh.* (1) Định lý ergodic cho chuỗi hữu hạn trạng thái bất khả quy (không cần không tuần hoàn), áp cho hàm bị chặn $y\mapsto p(y_*\mid x_*,(X,y))$, cùng (A1). (2) Luật xác suất toàn phần và (A0): $p(y_*\mid x_*,X,y,W)=p(y_*\mid x_*,X,y)$. Hiệu của hai kỳ vọng của cùng một hàm nhận giá trị trong $[0,1]$ bị chặn bởi TV của hai độ đo. Với prior tách được, $p(y\mid x_*,X,W)\propto\ell(y)\int p(y\mid X,\phi_{y\mid x})p(\phi_{y\mid x})\,d\phi_{y\mid x}\cdot\underbrace{\int p(X,x_*\mid\phi_x)p(\phi_x)\,d\phi_x}_{\text{không phụ thuộc }y}$, nên $x_*$ không làm đổi posterior của $y$ và $p(y\mid x_*,X,W)=p(y\mid X,W)$. (3) Áp 2(a) cho bài toán $n+1$ mẫu; Bổ đề 1 cho $\pi_*(y_*\mid y)=p(y_*\mid x_*,(X,y))$; định lý ergodic. $\square$

Kiểm chứng (E2): prior ô ⇒ $|\bar p-p_{\text{Bayes}}|<10^{-15}$; prior hỗn hợp (x mang thông tin) ⇒ chênh lệch tới 0,11 ở $n=8$; chế độ chuyển nạp khớp Bayes chính xác (`test_rao_blackwell_and_transductive`).

**Hệ quả cho bài.** TabPFN được huấn luyện trên prior SCM, trong đó $x$ mang thông tin về cơ chế sinh. Vì vậy chế độ chuyển nạp là phiên bản "Bayes chính xác", còn chế độ quy nạp là xấp xỉ có sai lệch bị chặn bởi $\mathrm{TV}\big(p(y\mid X,W),p(y\mid x_*,X,W)\big)$ (Định lý 1(c)(2)); độ lớn trên dữ liệu thật chưa đo. Cả hai được cài; thực nghiệm chọn theo chi phí (việc mở G4-2).

## 3. Mệnh đề 3 (Bayes-optimal hữu hạn mẫu, có điều kiện theo prior)

**Mệnh đề 3.** Gọi $S$ là một quy tắc chấm điểm proper nghiêm ngặt (log score, Brier). Trong mọi bộ dự đoán đo được $f:(X,W,x_*)\mapsto\Delta_C$, rủi ro Bayes $\mathbb E[S(f(X,W,x_*),y_*)]$ — kỳ vọng theo luật đồng thời của $(\phi,X,y,W,x_*,y_*)$ dưới prior — đạt cực tiểu duy nhất (h.c.c.) tại $f^\star=p(y_*\mid x_*,X,W)$.

*Chứng minh.* Rủi ro của $f^\star$ hữu hạn (với log score, $\le\log C$ vì là kỳ vọng của entropy có điều kiện), nên so sánh có nghĩa; bộ dự đoán có rủi ro vô hạn hiển nhiên thua. Theo tính chất tháp, $\mathbb E[S(f,y_*)]=\mathbb E\big[\mathbb E[S(f,y_*)\mid X,W,x_*]\big]$. Với mỗi giá trị của $(X,W,x_*)$, $f$ là một phân phối cố định và $y_*\sim p(\cdot\mid x_*,X,W)$; tính proper nghiêm ngặt cho cực tiểu duy nhất tại $f=p(\cdot\mid x_*,X,W)$. $\square$

**Mệnh đề 3′ (nhiều điểm test) [thêm sau kiểm duyệt, F2].** Với $m$ điểm test và bộ dự đoán $f:(X,W,X_*)\mapsto(\Delta_C)^m$, rủi ro $\mathbb E\big[\sum_kS(f_k,y_{*k})\big]$ đạt cực tiểu duy nhất tại $f_k=p(y_{*k}\mid X_*,X,W)$. Chứng minh y hệt Mệnh đề 3, áp cho từng $k$ với tập thông tin $(X,W,X_*)$.

**Hệ quả 3.1 (sửa sau kiểm duyệt, F2).** Dưới (A1): (a) bộ dự đoán chuyển nạp **một điểm** (chạy một chuỗi riêng cho mỗi $x_*$) hội tụ về $f^\star$; (b) bộ chuyển nạp **nhiều điểm** (`fit_transductive`, mọi điểm test trong cùng một chuỗi) hội tụ về $p(y_{*k}\mid X_*,X,W)$ — Bayes-optimal trong lớp lớn hơn $f(X,W,X_*)$ (Mệnh đề 3′), nhưng khác $f^\star$ khi feature mang thông tin về $\phi$ (phản ví dụ: hỗn hợp 3 thành phần, $n=6$, PU-SCAR, hai điểm test ở hai ô khác nhau — chênh tới 0,17); (c) bộ quy nạp hội tụ về $f^\star$ khi prior tách được. Bản trước viết (a) cho cả trường hợp nhiều điểm, và dùng điều kiện hẹp "$p(x\mid\phi)$ không phụ thuộc $\phi$" thay cho điều kiện tách được của Định lý 1(c)(2).
**Phạm vi phải nói rõ trong bài:** đây là tối ưu *trung bình theo prior*, không phải đảm bảo trên phân phối thật. Khi dữ liệu thật nằm ngoài prior của TFM, không có đảm bảo nào; đóng góp thực nghiệm (RQ2–RQ5) mới là bằng chứng.

## 4. Định lý 2 (độ bền khi TFM chỉ xấp xỉ) **[đổi so với G3: dạng Dobrushin, có trung bình theo π̃]**

Gọi $P$ là nhân của thuật toán khi dùng $p$ (chính xác, phân phối dừng $\pi$ theo Định lý 1) và $\tilde P$ là **cùng thuật toán** khi dùng $q$. Hệ số ergodic Dobrushin: $\delta(K)=\max_{y,y'}\mathrm{TV}(K(y,\cdot),K(y',\cdot))$. Đặt
$$\tau(P)=\sum_{t\ge0}\delta(P^t),\qquad \bar\varepsilon=\sum_y\tilde\pi(y)\,\mathrm{TV}\big(\tilde P(y,\cdot),P(y,\cdot)\big)\le\varepsilon_{\max}=\max_y\mathrm{TV}\big(\tilde P(y,\cdot),P(y,\cdot)\big).$$

**Định lý 2.** Giả sử $q>0$. Khi đó $\tilde P$ có phân phối dừng duy nhất $\tilde\pi$ — **kể cả khi các xác suất có điều kiện của $q$ không tương thích với bất kỳ phân phối đồng thời nào** — và
$$\mathrm{TV}(\tilde\pi,\pi)\;\le\;\tau(P)\,\bar\varepsilon\;\le\;\tau(P)\,\varepsilon_{\max}.$$
Hơn nữa:
- (i) *(dạng Mitrophanov)* nếu $2\max_y\mathrm{TV}(P^t(y,\cdot),\pi)\le C_{\rm erg}\rho^t$ với mọi $t$, thì $\tau(P)\le\hat t+C_{\rm erg}\rho^{\hat t}/(1-\rho)$, $\hat t=\max\{0,\lceil\log_\rho(1/C_{\rm erg})\rceil\}$. Kết hợp với $\bar\varepsilon\le\varepsilon_{\max}$, dạng này cho lại đúng chặn của Mitrophanov (2005, J. Appl. Probab. 42:1003–1014, Hệ quả 3.1, công thức (3.8)) — vốn viết theo chuẩn $\|\mu\|=|\mu|(\Omega)=2\,\mathrm{TV}$ và chuẩn toán tử $\|E\|=2\varepsilon_{\max}$ (người kiểm duyệt độc lập đã đối chiếu với bản gốc).
- (ii) *(Gibbs một mẫu, nhân quét ngẫu nhiên $P$)* $\varepsilon_{\max}\le\min\{1,\,2\kappa\,\varepsilon_{TV}\}$; với nhân một vòng quét $P_{sw}$ của cài đặt: $\varepsilon_{\max}\le\min\{1,\,2|F|\kappa\,\varepsilon_{TV}\}$ (TV của tích các nhân cộng dồn) và $\tau$ tính theo vòng quét với $\varepsilon_{TV}=\sup\mathrm{TV}(q(\cdot\mid x_i,D_{-i}),p(\cdot\mid x_i,D_{-i}))$ trên các ngữ cảnh mà thuật toán dùng, và $\kappa=\sup_{i\in F,y}\max_k\ell_i(k)\big/\sum_k p(k\mid x_i,D_{-i})\ell_i(k)$. Với PU-SCAR, $\kappa\le1/(1-c)$.
- (iii) *(MH hoán đổi)* $\varepsilon_{\max}\le\min\{1,\,4\varepsilon_\infty\}$ với $\varepsilon_\infty=\sup|\log q-\log p|$ trên các cặp (ngữ cảnh, truy vấn) mà thuật toán dùng.
- (iv) *(dự đoán)* với $\varepsilon'_{TV}=\sup_y\mathrm{TV}\big(q(\cdot\mid x_*,(X,y)),p(\cdot\mid x_*,(X,y))\big)$: $\big|\mathbb E_{\tilde\pi}q(y_*\mid x_*,\cdot)-\bar p(y_*\mid x_*)\big|\le\varepsilon'_{TV}+\mathrm{TV}(\tilde\pi,\pi)$.

**Định vị so với lý thuyết nhiễu loạn kinh điển [thêm sau kiểm duyệt độc lập, F3].** Chặn chính *không mới về dạng*. Với $E=\tilde P-P$ và ma trận cơ bản $Z=(I-P+\mathbf 1\pi)^{-1}-\mathbf 1\pi$, có đồng nhất thức chính xác $\tilde\pi-\pi=\tilde\pi EZ=\nu Z$ (Schweitzer 1968; Meyer 1980). Vì $\nu(\Omega)=0$, ta có $\mathrm{TV}(\tilde\pi,\pi)\le\tau_1(Z)\tfrac12\|\nu\|_1\le\tau_1(Z)\,\bar\varepsilon$ — dạng chặn theo hệ số ergodic của Seneta (1988), được so sánh với các chặn khác ở Cho & Meyer (2001) — và $\tau_1(Z)\le\tau(P)$. Chặn $\tau(P)\bar\varepsilon$ vì vậy là một trường hợp yếu hơn của chặn kinh điển, vốn cũng đã dùng $\nu=\tilde\pi E$ (lấy trung bình theo $\tilde\pi$). Nó chặt trong lớp tổng quát: dấu bằng đạt được trên chuỗi 2 trạng thái. Điều mới của Định lý 2 là: Bước 0 (tồn tại và duy nhất ngay cả khi các xác suất có điều kiện của $q$ không tương thích, nhờ cùng đồ thị chuyển); các ước lượng $\varepsilon$ cho nhân dựng từ TFM ở (ii)–(iii); và Mệnh đề 4(b). Kiểm số: đồng nhất thức đúng tới 3e-15 và $\tau_1(Z)\le\tau(P)$ ở 200/200 chuỗi ngẫu nhiên; RQ1 E3/E4 có thể báo thêm $\tau_1(Z)$, chặt hơn và tính được từ cùng ma trận. Trích dẫn cần đối chiếu số trang trước khi viết bài (G4-6).

*Chứng minh.*
**Bước 0 (tồn tại, duy nhất).** Không gian trạng thái hữu hạn nên $\tilde P$ có ít nhất một phân phối dừng. Với $q>0$, $\tilde P$ có cùng đồ thị chuyển (các cạnh dương) như $P$ — xác suất cập nhật một mẫu và xác suất chấp nhận MH đều dương — nên bất khả quy, do đó $\tilde\pi$ duy nhất. Lập luận này không dùng tới sự tương thích của $q$. (Đại lượng $\bar\varepsilon$ ở phát biểu chỉ xác định sau bước này, khi $\tilde\pi$ đã duy nhất. Nếu $\bar\varepsilon=0$ thì $\tilde P(y,\cdot)=P(y,\cdot)$ với mọi $y$ có $\tilde\pi(y)>0$, nên $\tilde\pi P=\tilde\pi\tilde P=\tilde\pi$ và $\tilde\pi=\pi$ vì $P$ bất khả quy; trường hợp $\tau=\infty$, $\bar\varepsilon=0$ vì vậy không cho dạng $\infty\cdot0$ — sửa sau kiểm duyệt, F13.)
**Bước 1 (đồng nhất thức nhiễu loạn).** Đặt $\Delta=\tilde\pi-\pi$ và $\nu=\tilde\pi(\tilde P-P)$; $\nu(\Omega)=0$. Từ $\tilde\pi=\tilde\pi\tilde P$, $\pi=\pi P$: $\Delta=\nu+\Delta P$. Lặp $T$ lần: $\Delta=\sum_{t<T}\nu P^t+\Delta P^T$.
**Bước 2 (co Dobrushin).** Với độ đo có dấu $\mu$, $\mu(\Omega)=0$: $\|\mu K\|_1\le\delta(K)\|\mu\|_1$. (Viết $\mu=a(\alpha-\beta)$ với $\alpha,\beta$ là xác suất, $a=\|\mu\|_1/2$; khi đó $\|\alpha K-\beta K\|_1\le2\max_{y,y'}\mathrm{TV}(K(y,\cdot),K(y',\cdot))$ theo tính lồi; xem Seneta 2006, §3.1.) Nếu $\tau(P)=\infty$ thì bất đẳng thức hiển nhiên đúng. Nếu $\tau(P)<\infty$ thì $\delta(P^T)\to0$, nên $\|\Delta P^T\|_1\le\delta(P^T)\|\Delta\|_1\to0$ và chuỗi $\Delta=\sum_{t\ge0}\nu P^t$ hội tụ tuyệt đối. (Định lý **không** cần giả định $P$ không tuần hoàn: ví dụ MH hoán đổi không lazy trên một túi $\{1,2\}$ với số đếm $(1,1)$ và $\pi$ đều là tuần hoàn, $\tau=\infty$, chặn trở thành hiển nhiên.)
**Bước 3.** $\|\nu\|_1\le\sum_y\tilde\pi(y)\|\tilde P(y,\cdot)-P(y,\cdot)\|_1=2\bar\varepsilon$. Do đó $\|\Delta\|_1\le2\bar\varepsilon\sum_t\delta(P^t)$, tức $\mathrm{TV}(\tilde\pi,\pi)\le\tau(P)\bar\varepsilon$.
**(i)** $\delta(P^t)\le1$ và $\delta(P^t)\le\max_{y,y'}[\mathrm{TV}(P^t(y),\pi)+\mathrm{TV}(\pi,P^t(y'))]\le C_{\rm erg}\rho^t$. Vậy $\tau\le\sum_t\min(1,C_{\rm erg}\rho^t)\le\hat t+\sum_{t\ge\hat t}C_{\rm erg}\rho^t$.
**(ii)** $\tilde P(y,\cdot)-P(y,\cdot)=\frac1{|F|}\sum_i[\tilde P_i(y,\cdot)-P_i(y,\cdot)]$, và $\mathrm{TV}(\tilde P_i(y,\cdot),P_i(y,\cdot))=\mathrm{TV}(q_\ell,p_\ell)$ với $q_\ell\propto q\ell_i$, $p_\ell\propto p\ell_i$. *Bổ đề tái trọng số:* với $Z_p=\sum p\ell$, $Z_q=\sum q\ell$,
$$2\,\mathrm{TV}(q_\ell,p_\ell)\le\sum\ell|q-p|/Z_p+|Z_q-Z_p|/Z_p\le2\max\ell\cdot\|q-p\|_1/Z_p,$$
nên $\mathrm{TV}(q_\ell,p_\ell)\le2(\max\ell/Z_p)\,\mathrm{TV}(q,p)\le2\kappa\varepsilon_{TV}$. Với PU-SCAR, $Z_p=p_0+(1-c)p_1\ge1-c$.
**(iii)** $x\mapsto\min(1,e^x)$ là 1-Lipschitz, $\log R_{ij}$ là tổng bốn số hạng $\pm\log q$, nên $|\tilde\alpha-\alpha|\le|\log\tilde R-\log R|\le4\varepsilon_\infty$. Hiệu hai hàng của nhân là $\sum_{o}Q_o(\tilde\alpha_o-\alpha_o)(\delta_{y'_o}-\delta_y)$ theo các đề xuất $o$, nên $\mathrm{TV}\le\sum_oQ_o|\tilde\alpha_o-\alpha_o|\le4\varepsilon_\infty$.
**(iv)** $|\mathbb E_{\tilde\pi}q-\mathbb E_\pi p|\le|\mathbb E_{\tilde\pi}(q-p)|+|\mathbb E_{\tilde\pi}p-\mathbb E_\pi p|$; số hạng đầu $\le\varepsilon'_{TV}$ (với một lớp $k$ cố định, $|q(k)-p(k)|\le\mathrm{TV}$), số hạng sau $\le\mathrm{TV}(\tilde\pi,\pi)$ vì $p(k\mid\cdot)\in[0,1]$. $\square$

**Nhận xét 4.1 (độ chặt và phụ thuộc vào $n$).** $\tau(P)$ là "thời gian trộn tích phân" của chuỗi chính xác, tính theo **bước**. Với Gibbs một mẫu quét ngẫu nhiên, khi mỗi mẫu tự do có ít nhất 2 giá trị, $\delta(P^t)=1$ với $t<|F|/2$ (hai trạng thái khác nhau ở mọi mẫu tự do cần tổng cộng $|F|$ bước để gặp nhau), nên $\tau\ge\lceil|F|/2\rceil$; chặn có dạng (thời gian trộn theo bước) × (sai số mỗi bước). (Bản trước viết "cho tới khi mọi mẫu được chạm tới", tức $|F|$ bước — sai; kiểm số với $|F|=2..6$: bước đầu tiên có $\delta<1$ là 1, 2, 2, 3, 3. Sửa sau kiểm duyệt, F6.) Hệ số $n$ trong bản nháp G3 ($\|\tilde P-P\|\le2n\kappa\varepsilon$ cho một vòng quét tuần tự) xuất hiện lại qua $\tau$ — không mất đi, nhưng giờ có hằng số tính được. Kiểm chứng (E3, E4): chặn luôn đúng; dạng trung bình $\tau\bar\varepsilon$ lỏng 2,2–12 lần so với sai lệch thật; dạng theo $\varepsilon$ của TFM lỏng thêm 15–44 lần (bảng E3).

**Nhận xét 4.2 (định vị so với Liu et al., Biometrika 2014; đã đọc bản arXiv 1012.2902 ngày 24/10/2026).** Liu, Gelman, Hill, Su & Kropko xét phân phối dừng của imputation lặp (chained equations) với dữ liệu liên tục. Định lý 1 của họ: khi các mô hình có điều kiện *tương thích*, TV giữa phân phối dừng của imputation lặp và posterior predictive Bayes → 0 theo xác suất khi $n\to\infty$ (cần các điều kiện về tập nhỏ, drift, tập trung của MLE). Định lý 2 của họ: nếu xác suất có điều kiện của imputation lệch khỏi mô hình đồng thời một lượng $\varepsilon$ trên một tập có khối lượng $\ge q$, thì TV giữa hai phân phối dừng $\ge q\varepsilon/4$ — tức chặn **dưới** tuyến tính theo $\varepsilon$. Định lý 3: với mô hình không tương thích nhưng "valid", ước lượng tham số nhất quán, nhưng phân phối imputation không nhất thiết khớp mô hình Bayes nào.
Khác biệt của Định lý 2 ở đây: không gian trạng thái hữu hạn; không tiệm cận theo $n$; hằng số tường minh $\tau(P)$ tính được; áp cho cả bước MH; sai số đo theo TV giữa các xác suất có điều kiện (là thứ đo được của một TFM) thay vì theo tham số. Về độ chặt, lập luận tự đủ: chặn $\tau(P)\bar\varepsilon$ đạt dấu bằng trên chuỗi 2 trạng thái, nên sự phụ thuộc tuyến tính vào $\bar\varepsilon$ không cải thiện được trong lớp tổng quát. (Bản trước suy "đúng bậc" từ chặn dưới $q\varepsilon/4$ của Liu et al.; nhưng $\varepsilon$ của họ là độ lệch của xác suất dự đoán có điều kiện trên một tập có khối lượng $>q$ với dữ liệu liên tục, không phải $\bar\varepsilon$, nên phép suy đó không trực tiếp — sửa sau kiểm duyệt, F14.)

**Nhận xét 4.3 (về tính "không tương thích").** Một quan sát phụ khi xây kiểm chứng: với nhãn nhị phân, mọi họ xác suất có điều kiện *chỉ phụ thuộc số đếm* trong một ô đều tương thích (đặt $f(k+1)/f(k)=q(1\mid k)/q(0\mid k)$). Vì vậy nhiễu dùng trong E3 phụ thuộc thêm số đếm toàn cục; `test_perturbed_is_incompatible` kiểm tra bằng hai đường Brook cho tỉ số khác nhau.

## 5. Mệnh đề 4 (cập nhật theo khối)

**Mệnh đề 4(a) — khối có sửa MH là chính xác.** Với khối $B$, đề xuất độc lập $Q_B(y'_B)=\prod_{i\in B}q_\ell(y'_i\mid x_i,D_{-B})$ (LLP: phân phối tích có điều kiện số đếm, lấy mẫu chính xác bằng DP) và chấp nhận với
$$\min\Big\{1,\frac{\pi(y')\,Q_B(y_B)}{\pi(y)\,Q_B(y'_B)}\Big\},$$
trong đó $\pi(y')/\pi(y)$ tính bằng bổ đề Brook dọc các mẫu bị đổi (một forward mỗi mẫu), thì dưới (A1) nhân thoả cân bằng chi tiết với $\pi$, và bất khả quy trên tập hỗ trợ.
*Chứng minh.* $Q_B$ chỉ phụ thuộc $y_{-B}$, không đổi trong bước này, nên đây là MH-trong-Gibbs với đề xuất độc lập; Brook + Bổ đề 1 cho đúng tỉ số $p(y'\mid X)/p(y\mid X)$, nhân thêm tỉ số $\ell$ (bằng 1 với LLP). Đề xuất dương trên tập hỗ trợ của khối và xác suất chấp nhận dương, nên bất khả quy. $\square$
Chi phí: 1 forward cho đề xuất + 1 forward cho mỗi nhãn bị đổi. Kiểm chứng: E1 (TV < $10^{-13}$), `test_sampler_instance_mc`, `test_sampler_llp_mc`.

**Mệnh đề 4(b) — sai số của khối không sửa.** Gọi $P_{bG}$ là Gibbs khối chính xác (chọn đều một khối) và $\hat P$ là cập nhật khối xấp xỉ. Theo Định lý 2 với $P=P_{bG}$: $\mathrm{TV}(\hat\pi,\pi)\le\tau(P_{bG})\,\bar\varepsilon_B$. Thêm giả định ổn định
> (S) thêm một điểm có nhãn, hoặc một feature không nhãn, vào ngữ cảnh làm PPD đổi tối đa $\beta$ theo TV,

thì với khối $B$ cỡ $m$ (theo D1 của METHOD_SPEC, khối gồm cả mẫu cố định; chỉ $m_f=|B\cap F|$ mẫu tự do được cập nhật, còn cả $B$ bị loại khỏi ngữ cảnh): $\mathrm{TV}\big(\prod_{i\in B\cap F}p(\cdot\mid x_i,D_{-B}),\,p(y_{B\cap F}\mid y_{-(B\cap F)},X)\big)\le m_f(m-1)\beta$, và sau tái trọng số bởi $\ell$: $\bar\varepsilon_B\le2\kappa_B\,m_f(m-1)\beta$ với $\kappa_B=\max\ell_{B\cap F}/\sum p(y_{B\cap F}\mid y_{-(B\cap F)},X)\,\ell_{B\cap F}(y_{B\cap F})$. (Khi khối chỉ gồm mẫu tự do, $m_f=m$.)
*Chứng minh.* Đánh số các mẫu tự do trong khối $b_1..b_{m_f}$. Quy tắc chuỗi: $p(y_{B\cap F}\mid y_{-(B\cap F)},X)=\prod_kA_k(y_{b_k}\mid y_{b_{<k}})$, trong đó $A_k=p(\cdot\mid x_{b_k},D_{-B}\cup D_{B\setminus F}\cup\{(x_{b_l},y_{b_l})\}_{l<k},\ \{x_{b_l}\}_{l>k})$ là PPD tổng quát (§0). Phép xấp xỉ dùng $B_k=p(\cdot\mid x_{b_k},D_{-B})$. Hai ngữ cảnh khác nhau $(m-m_f)+(k-1)+(m_f-k)=m-1$ điểm, nên theo (S) và bất đẳng thức tam giác dọc dãy ngữ cảnh thêm từng điểm, $\mathrm{TV}(A_k,B_k)\le(m-1)\beta$. Với hai phân phối dạng chuỗi, $\mathrm{TV}(\prod A_k,\prod B_k)\le\sum_k\mathbb E_A\mathrm{TV}(A_k(\cdot\mid y_{<k}),B_k)\le m_f(m-1)\beta$ (lập luận lai từng bước). Tái trọng số theo bổ đề ở 4(ii). $\square$

**Ghi chú về phạm vi của chứng minh [thêm sau kiểm duyệt, F11].** (1) Phân tích dùng nhân "chọn đều một khối"; cài đặt quét lần lượt $K$ khối của một phân hoạch ngẫu nhiên, nên sai số mỗi vòng quét có thêm hệ số $K$ (như $|F|$ ở Định lý 2(ii)) và $\tau$ phải tính theo vòng quét. (2) Việc đồng nhất $A_k$ với PPD tổng quát cần một Bổ đề 1 mở rộng cho feature không nhãn; chứng minh y hệt Bổ đề 1 (cùng lập luận trao đổi được, thêm các feature không nhãn vào phần được điều kiện hoá) nhưng chưa viết ra. (3) Với chuỗi có cập nhật tham số phiền nhiễu (Mệnh đề 1.4), Định lý 2 cần $\varepsilon(\bar P)\le\sup_\theta\varepsilon(K_\theta)$, đúng do tính lồi của TV.

**Giới hạn của 4(b) (ghi trung thực; viết lại sau kiểm duyệt, F1).** Với $\beta=O(1/n)$ và $K$ cố định ($m=n/K$), $m_f(m-1)\beta=O(n/K^2)$ — chặn **tăng** theo $n$, không chỉ "không tiến về 0". Ngoài ra $\kappa_B$ có thể rất lớn: với bag-block (mặc định của LLP), $\kappa_B=1/P(\text{mọi túi trong nhóm khớp số đếm}\mid\cdot)$; với $n\approx2.000$, $K=10$, $\log_{10}\kappa_B\approx15{,}8$ (túi 8), 10,3 (túi 16), 6,6 (túi 32). **Hệ quả: cấu hình mà thực nghiệm thật sự dùng (khối $K=10$; bag-block cho LLP) không có đảm bảo nào khác rỗng; Định lý 1, Định lý 1(c) và Mệnh đề 3 chỉ áp cho các sơ đồ một mẫu, hoán đổi và khối có sửa MH.** Bag-block xấp xỉ cũng chưa từng được đo: RQ1 E5 chỉ đo khối cấp mẫu, E1mc chỉ đo bag-block *có* sửa MH. Trên bài chính xác LLP 3 túi × 4, bag-block xấp xỉ cho TV(π̃, π) từ 0 đến khoảng 0,6 tuỳ prior và cách chia nhóm (người kiểm duyệt: 0,16–0,64; kiểm lại độc lập với dữ liệu ngẫu nhiên khác: 0–0,52, `code/experiments/g6_review_checks.py`); $n=12$ phóng đại độ lớn nhưng cho thấy sai lệch có thật. Việc mở G4-5.

**Quan sát 4(c) — số liệu (E5).** Xem [RQ1_REPORT.md](RQ1_REPORT.md) §5. Trên chuỗi gộp một ô (tính chính xác tới $n=2000$), tách được hai đại lượng: TV trên posterior của nhãn, và sai số của *dự đoán*. Với khối chia trên mọi mẫu: (1) TV $\approx0{,}1/K$, không giảm theo $n$ — sai ở độ phân tán của posterior do bỏ qua phụ thuộc trong khối; (2) sai số dự đoán giảm theo $n$ (PU, K = 10: 0,0017 ở $n=50$, 0,00017 ở $n=1000$; độ dốc log–log từ −0,5 đến −0,8 ở PU và không đơn điệu ở SSL, chỉ với 3 giá trị $n$ — bản trước viết "như $O(1/n)$", không được dữ liệu ủng hộ; sửa sau kiểm duyệt, F12). Khi chia chỉ trên mẫu tự do, có thêm lệch thành phần không giảm theo $n$ (PU, K = 10: 0,031). Kết luận ở dạng quan sát, không phải định lý; một chứng minh cho (2) là việc mở G4-4.

## 6. Việc mở sau G4

| # | Việc | Ảnh hưởng nếu không làm |
|---|---|---|
| G4-1 | Kiểm tra điều kiện hội tụ cho PU-SAR (MH trên $\eta$, không gian tổng quát): đóng bằng trích dẫn Tierney (1994) kèm điều kiện cụ thể (số định lý cần kiểm); thêm một đoạn về định danh yếu của $\eta$ | Định lý 1 chỉ phát biểu cho $\ell$ đã biết và các cập nhật liên hợp; PU-SAR nay là bối cảnh phụ |
| G4-2 | Chọn quy nạp hay chuyển nạp cho thực nghiệm chính (chi phí so với độ chính xác) — quyết ở G5, dựa trên tập phát triển | Bài phải nói rõ bộ dự đoán nào được dùng và nó hội tụ về đâu |
| G4-3 | Liu et al. (2014): đã đọc bản arXiv, cập nhật Nhận xét 4.2. Còn lại: đối chiếu bản Biometrika in chính thức (số định lý có thể khác bản arXiv) và đọc lại Hệ quả 3.1 của Mitrophanov (2005) ở bản PDF | Rủi ro trích sai số định lý khi viết |
| G4-4 | Chặn chặt hơn cho cập nhật khối, hoặc giữ dạng quan sát | Mệnh đề 4(b) tăng theo $n$ với $K$ cố định (xem Giới hạn của 4(b)) |
| G4-5 | **Bag-block xấp xỉ (mặc định của LLP): chưa có chặn khác rỗng, chưa đo** (kiểm duyệt độc lập, F1). Kế hoạch: (a) trên tập phát triển, LLP túi 8, so chuỗi bag-block xấp xỉ với bag-block có sửa MH (chính xác) — chênh AUC và tỉ lệ chấp nhận MH (`run_g7.py --mh-check`, G6 lượt 5); (b) trên tập đánh giá, ablation "khối có sửa MH" đã có trong giao thức Mục 7; (c) bài viết rõ Định lý 1 và Mệnh đề 3 không áp cho cấu hình mặc định | Khẳng định chính LLP túi 8 dựa trên một sơ đồ xấp xỉ chưa đo sai lệch |
| G4-6 | Đối chiếu trích dẫn với nguồn gốc: Schweitzer (1968), Meyer (1980), Seneta (1988), Cho & Meyer (2001) cho F3; Tierney (1994) cho G4-1; Seneta (2006) §3.1; Müller et al. (2022) — mệnh đề cụ thể về PPD; Elkan & Noto (2008) Lemma 1; Besag (1974)/Brook (1964) — trang | Rủi ro trích sai |
| G6-T7a | Trích dẫn cho chặn hối tiếc AUC dạng $\lambda=1$ | **Xong 26/10/2026:** Agarwal (2014), JMLR 15(49):1653–1674, Hệ quả 12 (đối chiếu trên bản arXiv 1207.0268; số hiệu trong bản JMLR nên xem lại khi viết bài) |
| G6-T7b | Kiểm duyệt độc lập §0–§7 | **Xong 26/10/2026** — 0 Critical, 4 Major, 12 Minor; phản hồi từng điểm ở [G6/G6_T7_REVIEW_RESPONSE.md](../G6/G6_T7_REVIEW_RESPONSE.md) |

## 7. Mệnh đề 5, Mệnh đề 6 và Quan sát 6(b): khi nào bộ ngây thơ đã đủ, và Gibbs lấy lại được gì **[mới, G6 T7, 26/10/2026]**

**Đính chính giả thuyết cũ.** Giao thức G5 (Mục 0, nhận xét cuối) và [PILOT_RQ4_REPORT](pilot/PILOT_RQ4_REPORT.md) §4 viết rằng Gibbs có chỗ giúp ở PLL và LLP vì ở đó "bộ ngây thơ không nhất quán". **Điều này sai với cơ chế sinh nhãn của giao thức.** Với ứng viên sai độc lập (PLL) và túi chia ngẫu nhiên (LLP), mục tiêu quần thể của bộ học từng điểm trên nhãn ngây thơ vẫn là biến đổi affine tăng ngặt của posterior thật, tức xếp hạng đúng ở mức quần thể, như ở PU-SCAR (Mệnh đề 5). Mục này không khẳng định TFM hội tụ tới mục tiêu đó (với LLP, các cặp trong một túi còn phụ thuộc nhau). Khác biệt thật nằm ở **lượng thông tin bộ ngây thơ bỏ đi** (Mệnh đề 6, Quan sát 6(b)). Trong các bối cảnh **tổng hợp** của giao thức (RQ2–RQ4), chỉ PU-SAR vi phạm (B1). Với nhãn yếu thật (RQ5), (B1) chưa được xác lập: tập ứng viên của PLL thật có thể phụ thuộc mẫu, còn túi của LLP-Bench tạo theo feature; giả định chuẩn $p(S\mid y=k,x)$ như nhau trên $S$ mà Gibbs dùng cũng chưa kiểm cho dữ liệu thật. **[Văn xuôi viết lại sau kiểm duyệt độc lập, F4.]**

**Ký hiệu riêng của mục này [F15].** Để tránh trùng với §0–§5: $\varpi$ là tỉ lệ lớp dương (không dùng $\pi$, vốn là phân phối mục tiêu, hay $p$); $N$ là biến nhị thức; $q$ là xác suất một lớp sai vào tập ứng viên của PLL (mức của PLL trong giao thức), không phải TFM.

**Thiết lập.** *TFM ngây thơ* là TFM điều kiện hoá trên các cặp $(x_i,\tilde w_i)$, với $\tilde w_i$ là *nhãn ngây thơ* sinh từ quan sát yếu, đúng như trong các script thí điểm:
- PU-SCAR: $\tilde w=s$. Nhãn nhiễu: $\tilde w=w$. SSL: chỉ dùng phần có nhãn, $\tilde w=y$ trên phần đó.
- PLL: $\tilde w_i$ chọn đều trong $S_i$. LLP: $\tilde w_{B}$ là một hoán vị ngẫu nhiên khớp số đếm của túi.

Đặt $\eta_k(x)=p(y=k\mid x)$ và $\tilde\eta_k(x)=P(\tilde w=k\mid x)$ — *mục tiêu ngây thơ*, tức hàm mà một bộ học từng điểm nhất quán, huấn luyện trên $(x,\tilde w)$, tiến tới. Mục này phát biểu ở mức quần thể; nó không khẳng định TFM hội tụ tới $\tilde\eta$.

**(B1) — cơ chế không phụ thuộc mẫu.** $P(\tilde w=k\mid x,y)=M_{y,k}$ không phụ thuộc $x$.
**(B2) — cột đều ngoài đường chéo.** Với mỗi lớp $k$, mọi phần tử ngoài đường chéo của cột $k$ của $M$ bằng nhau, $M_{y,k}=b_k$ ($y\ne k$), và $\lambda_k:=M_{k,k}-b_k>0$.

**Mệnh đề 5.** (a) Dưới (B1)–(B2), $\tilde\eta_k(x)=b_k+\lambda_k\,\eta_k(x)$ với mọi $k$.
(b) Do đó, với mỗi lớp $k$, thứ tự theo $\tilde\eta_k$ trùng thứ tự theo $\eta_k$: AUC một-chọi-phần-còn-lại của $\tilde\eta_k$ bằng của $\eta_k$, tức đạt tối ưu. Nếu thêm $b_k\equiv b$ và $\lambda_k\equiv\lambda$, thì $\arg\max_k\tilde\eta_k=\arg\max_k\eta_k$: bộ phân loại argmax của mục tiêu ngây thơ là bộ phân loại Bayes.
(c) *(nhị phân)* Với $\varpi=P(y=1)\in(0,1)$ và mọi hàm điểm $s$:
$$\mathrm{AUC}(\eta)-\mathrm{AUC}(s)\;\le\;\frac{\mathbb E\,|s(X)-\tilde\eta(X)|}{\lambda\,\varpi(1-\varpi)},\qquad \tilde\eta=b+\lambda\eta .$$
Tức là hối tiếc AUC của bộ ngây thơ bị chặn bởi sai số ước lượng mục tiêu của chính nó, khuếch đại $1/\lambda$ lần.

**Các cơ chế của giao thức** (kiểm bằng Monte Carlo theo đúng thủ tục sinh nhãn của thí điểm, `g6_t7_naive_theory.py`, kết quả ở [`pilot/g6_t7_naive_theory.txt`](pilot/g6_t7_naive_theory.txt)):

| Bối cảnh | $b$ | $\lambda$ | Giá trị ở các mức của giao thức |
|---|---|---|---|
| PU-SCAR | $0$ | $c$ | 0,1; 0,3 |
| Nhiễu phụ thuộc lớp (nhị phân; $\rho_{01}=P(w{=}1\mid y{=}0)$, $\rho_{10}=P(w{=}0\mid y{=}1)$) | $\rho_{01}$ | $1-\rho_{01}-\rho_{10}$ | thí điểm $T$: $\rho_{01}=r/2$, $\rho_{10}=r$ ⇒ $1-1{,}5r$ = 0,7; 0,4 |
| SSL, nhãn thiếu hoàn toàn ngẫu nhiên (MCAR) theo từng lớp với cùng tỉ lệ | $0$ | $1$ | luật của phần có nhãn trùng luật quần thể; nếu tỉ lệ khác nhau giữa các lớp (do làm tròn), $p(y{=}1\mid x,\text{có nhãn})$ vẫn là hàm tăng ngặt (không affine) của $\eta$, nên (b) vẫn đúng cho nhị phân |
| PLL, mỗi lớp sai vào tập độc lập với xác suất $q$ | $\frac{1-a}{C-1}$ | $a-b=\frac{(1-(1-q)^C)/q-1}{C-1}$, với $a=\frac{1-(1-q)^C}{Cq}$ | $C=4$: 0,51 ($q=0{,}3$); 0,21 ($q=0{,}6$). $C=7$, $q=0{,}6$: 0,11 |
| LLP, túi ngẫu nhiên cỡ $k$ | $\frac{(k-1)\varpi}{k}$ | $\frac1k$ (quần thể); $\frac1k-\frac{k-1}{k(n-1)}$ (quần thể hữu hạn $n$ điểm) | 0,125; 0,0625; (0,031 ở túi 32) |
| **PU-SAR** | — | **(B1) sai**: $P(s{=}1\mid x)=e(x)\,\eta(x)$ | thứ tự bị $e(x)$ bóp méo |

Với LLP, "không phụ thuộc mẫu" hiểu như sau: khi túi chia ngẫu nhiên, các thành viên khác của túi độc lập với $(x_i,y_i)$, nên $P(\tilde w_i=1\mid x_i,y_i)=\big(y_i+(k-1)\varpi\big)/k$. Nếu túi được tạo theo độ gần của feature (như LLP-Bench), (B1) sai và bộ ngây thơ không còn nhất quán.

*Chứng minh.* (a) $\tilde\eta_k(x)=\sum_yP(y\mid x)M_{y,k}=M_{k,k}\eta_k+b_k(1-\eta_k)=b_k+\lambda_k\eta_k$. Các hệ số trong bảng: PU-SCAR và nhiễu, đọc trực tiếp từ $M$. SSL: $P(\text{có nhãn}\mid x,y)$ là hằng số, nên $p(y\mid x,\text{có nhãn})=\eta(x)$. PLL: với $y$ cố định, $|S|=1+N$, $N\sim\mathrm{Bin}(C-1,q)$, và mỗi ứng viên được chọn với xác suất $1/|S|$; do đó $a=M_{y,y}=\mathbb E[1/(1+N)]=\frac{1-(1-q)^C}{Cq}$ (đẳng thức $\mathbb E[1/(1+N)]=\frac{1-(1-q)^{m+1}}{(m+1)q}$ với $N\sim\mathrm{Bin}(m,q)$). Theo tính đối xứng giữa các lớp sai, các phần tử ngoài đường chéo bằng nhau và bằng $(1-a)/(C-1)$. Vì $1+N\le C$, ta có $a\ge1/C$, dấu bằng chỉ khi $q=1$; do đó $\lambda=(Ca-1)/(C-1)>0$ khi $q<1$. LLP: với hoán vị ngẫu nhiên khớp số đếm, $P(\tilde w_i=1\mid\text{túi})=c/k$; lấy kỳ vọng theo các thành viên khác khi biết $y_i$: $\big(y_i+(k-1)\varpi\big)/k$, hoặc $\big(y_i+(k-1)\tfrac{N_+-y_i}{n-1}\big)/k$ khi túi lấy không hoàn lại từ $n$ điểm có $N_+$ dương.
(b) Hàm $t\mapsto b_k+\lambda_kt$ tăng ngặt, nên giữ thứ tự và hoà. Khi các hệ số chung cho mọi lớp, cùng một hàm tăng ngặt tác động lên mọi lớp, nên argmax được giữ.
(c) Lấy $(X,Y),(X',Y')$ độc lập cùng phân phối, và đặt $h_s(X,X')=\mathbb 1[s(X)>s(X')]+\tfrac12\mathbb 1[s(X)=s(X')]$. Vì $P(Y{=}1,Y'{=}0\mid X,X')=\eta(X)(1-\eta(X'))$, ta có $\varpi(1-\varpi)\,\mathrm{AUC}(s)=\mathbb E[\eta(X)(1-\eta(X'))\,h_s(X,X')]$. Đối xứng hoá theo $X\leftrightarrow X'$, dùng $h_s(X,X')+h_s(X',X)=1$ và $\eta(X)(1-\eta(X'))-\eta(X')(1-\eta(X))=\eta(X)-\eta(X')$:
$$\varpi(1-\varpi)\,\mathrm{AUC}(s)=\tfrac12\,\mathbb E\big[\eta(X')(1-\eta(X))+(\eta(X)-\eta(X'))\,h_s(X,X')\big].$$
Trừ hai vế cho $s=\eta$ và $s$ bất kỳ, với $\Delta=\eta(X)-\eta(X')$:
$$\varpi(1-\varpi)\,[\mathrm{AUC}(\eta)-\mathrm{AUC}(s)]=\tfrac12\,\mathbb E\Big[|\Delta|\big(\mathbb 1[s\text{ đảo thứ tự cặp}]+\tfrac12\mathbb 1[s\text{ hoà},\,\Delta\ne0]\big)\Big]\ \ge0 .$$
Đặt $t=(s-b)/\lambda$; $t$ có cùng thứ tự với $s$ nên $\mathrm{AUC}(t)=\mathrm{AUC}(s)$, và $t-\eta=(s-\tilde\eta)/\lambda$. Nếu $\Delta>0$ mà $t(X)\le t(X')$ thì $\Delta\le(\eta(X)-t(X))+(t(X')-\eta(X'))\le|\eta(X)-t(X)|+|\eta(X')-t(X')|$; trường hợp $\Delta<0$ tương tự. Vậy vế phải $\le\tfrac12\mathbb E\big[|\eta(X)-t(X)|+|\eta(X')-t(X')|\big]=\mathbb E|s(X)-\tilde\eta(X)|/\lambda$. $\square$

Bất đẳng thức (c) với $\lambda=1$ là Hệ quả 12 của Agarwal (2014) (bản arXiv 1207.0268), được ghi là theo Clémençon & Robbiano (2011) và dựa trên một đồng nhất thức của Clémençon, Lugosi & Vayatis (2008). Mệnh đề 5(c) mở rộng nó cho mục tiêu affine $b+\lambda\eta$, và chứng minh trên tự đủ. Hằng số không giảm được: ví dụ hai nguyên tử cho tỉ số tiến tới 1 (kiểm duyệt độc lập). Kiểm chứng: `test_auc_regret_bound_never_violated` (2.000 bài toán rời rạc ngẫu nhiên, tính chính xác, có hoà; chặn luôn đúng, khe nhỏ nhất 0,017).

**Mệnh đề 6 (thông tin Fisher; bất đẳng thức xử lý dữ liệu).** Cho mô hình tham số $\eta_\theta$ và quan sát $O$ có luật $p_\theta(O)$ trên tập hữu hạn (hoặc với các điều kiện chính quy cho phép đạo hàm dưới dấu tích phân). Nếu $\tilde w$ sinh từ $O$ qua một nhân $R(\tilde w\mid O)$ **không phụ thuộc** $\theta$, thì
$$I_{\tilde w}(\theta)\le I_O(\theta),$$
dấu bằng xảy ra khi và chỉ khi điểm số (score) $\partial_\theta\log p_\theta(O)$ đo được theo $\sigma(\tilde w)$ hầu chắc chắn.
*Chứng minh.* $p_\theta(\tilde w)=\sum_Op_\theta(O)R(\tilde w\mid O)$, nên $\partial_\theta p_\theta(\tilde w)=\sum_Op_\theta(O)R(\tilde w\mid O)\,\partial_\theta\log p_\theta(O)$. Chia cho $p_\theta(\tilde w)$: $\partial_\theta\log p_\theta(\tilde w)=\mathbb E_\theta[\partial_\theta\log p_\theta(O)\mid\tilde w]$. Khi đó $I_{\tilde w}=\mathbb E\big[(\mathbb E[\text{score}_O\mid\tilde w])^2\big]\le\mathbb E[\text{score}_O^2]=I_O$ theo bất đẳng thức Jensen cho kỳ vọng có điều kiện; dấu bằng khi và chỉ khi $\mathrm{Var}(\text{score}_O\mid\tilde w)=0$ h.c.c. $\square$ (Trường hợp vô hướng; với $\theta$ nhiều chiều, bất đẳng thức theo thứ tự Loewner, chứng minh y hệt.) Kiểm chứng: `test_fisher_data_processing_inequality_random_models`.

Áp dụng, với $\theta$ là tham số của $p(y\mid x)$ và luật của $x$ không phụ thuộc $\theta$:
- **PU-SCAR, nhãn nhiễu:** $\tilde w=O$, dấu bằng. Bộ ngây thơ dùng **toàn bộ** thông tin trong quan sát. Khác biệt giữa Gibbs và bộ ngây thơ chỉ có thể đến từ thiên hướng quy nạp của bộ học và từ hiệu chỉnh (mục tiêu ngây thơ $\tilde\eta\neq\eta$), không đến từ thông tin.
- **SSL (MCAR):** phần không nhãn chỉ góp $p(x)$, không phụ thuộc $\theta$ khi prior tách được (Định lý 1(c)), nên $I_{\text{có nhãn}}=I_{\text{đầy đủ}}$. Khi $x$ mang thông tin về cơ chế sinh (prior SCM của TabPFN), phần không nhãn có thể giúp, nhưng đó là thông tin qua prior, không qua $\ell$.
- **PLL:** $\tilde w$ là một lần chọn ngẫu nhiên từ $S$. Bất đẳng thức đúng, và ngặt trong mọi trường hợp đã tính (bảng dưới).

**Quan sát 6(b) — LLP (số liệu, không phải định lý).** Với LLP, $\sum_{i\in B}\tilde w_i=c_B$, nên dữ liệu ngây thơ *đồng thời* của một túi không mất thông tin (thứ tự trong túi là nhiễu thuần). Mất mát đến từ việc bộ ngây thơ coi các cặp $(x_i,\tilde w_i)$ là độc lập, tức tối đa hoá hợp lý từng điểm (composite likelihood). Ngay cả khi biết đúng kênh $\tilde\eta=(\eta_\theta+(k-1)\varpi)/k$, thông tin Godambe $H^2/J$ của hợp lý từng điểm chỉ bằng khoảng $0{,}27/k$ thông tin của số đếm túi (bảng dưới, $k\cdot$tỉ số $\approx$ 0,24–0,29 với $k=2..32$). Còn chính thông tin của số đếm túi bằng khoảng $(1{,}5\text{–}1{,}7)/k$ lần thông tin của nhãn sạch.

*Số liệu (mô hình logistic một tham số, $\eta=\sigma(\theta x)$, $\theta=2$, $x\sim\mathcal N(0,I)$; thông tin mỗi mẫu).* Nhãn sạch: 0,0607 (nhị phân); PLL 0,10–0,14.

| LLP, $k$ | đầy đủ (số đếm túi) / sạch | ngây thơ / đầy đủ | | PLL ($C$, $q$) | đầy đủ ($S$) / sạch | ngây thơ / đầy đủ |
|---|---|---|---|---|---|---|
| 2 | 0,734 | 0,134 | | (3; 0,3) | 0,66 | 0,258 |
| 4 | 0,425 | 0,060 | | (3; 0,6) | 0,35 | 0,100 |
| 8 | 0,202 | 0,033 | | (4; 0,3) | 0,62 | 0,236 |
| 16 | 0,098 | 0,017 | | (4; 0,6) | 0,31 | 0,083 |
| 32 | 0,047 | 0,009 | | (7; 0,6) | 0,18 | 0,057 |

PU-SCAR và nhãn nhiễu: tỉ số ngây thơ / đầy đủ = 1 ở mọi mức (Mệnh đề 6).

**Hệ quả cho bài (thay cho nhận xét cũ ở giao thức Mục 0).** Nhãn yếu chia làm ba loại:
1. **Bộ ngây thơ dùng chính quan sát** (PU-SCAR, nhiễu phụ thuộc lớp; SSL-MCAR với prior tách được): không còn thông tin để Gibbs lấy lại qua $\ell$. (Với SSL dưới prior không tách được, như prior SCM của TabPFN, phần không nhãn có thể giúp qua prior — xem phần áp dụng Mệnh đề 6.) Không kỳ vọng Gibbs cải thiện AUC, ngoài hiệu ứng mẫu hữu hạn và thiên hướng quy nạp; ở mức quần thể, khác biệt chỉ nằm ở hiệu chỉnh xác suất.
2. **Bộ ngây thơ làm thô quan sát** (PLL: chọn một ứng viên; LLP: bỏ cấu trúc túi): mục tiêu quần thể của bộ ngây thơ vẫn xếp hạng đúng, nhưng bộ ngây thơ kém hiệu quả về thông tin. Gibbs dùng hợp lý đầy đủ ($\ell_i(k)=\mathbb 1[k\in S_i]$, ràng buộc số đếm), nên có chỗ để giúp. Chỗ đó lớn dần khi việc làm thô nặng hơn ($q$ lớn, $k$ lớn), miễn là chuỗi còn trộn tốt.
3. **Cơ chế phụ thuộc mẫu** (PU-SAR; LLP với túi theo feature): mục tiêu quần thể của bộ ngây thơ không xếp hạng đúng. Khi biết cơ chế quan sát, chuỗi Gibbs có phân phối dừng là posterior dưới prior của TFM (Định lý 1, dưới (A1)) — đây là tính đúng của bộ lấy mẫu, không phải tính nhất quán thống kê đối với $\eta$ thật — nhưng phải học được cơ chế đó (PU-SAR: propensity đã biết 8/12 ô, tự học 3/12 ô, [G6_T1_SAREM_REPORT](../G6/G6_T1_SAREM_REPORT.md)).

**Đối chiếu với tập phát triển (hậu kiểm, không phải bằng chứng xác nhận).** Chênh lệch Gibbs ấm − TFM ngây thơ, trung bình trên 6 dataset (PLL: 3), 3 seed:

| Loại | Bối cảnh, metric | Mức nhẹ | Mức nặng |
|---|---|---|---|
| 1 | PU-SCAR, AUC | +0,000 ($c$ = 0,3) | +0,018 ($c$ = 0,1) |
| 1 | Nhiễu, AUC | +0,001 ($r$ = 0,2) | −0,021 ($r$ = 0,4) |
| 1 | SSL, AUC | −0,004 (20% nhãn) | −0,008 (5% nhãn) |
| 2 | PLL, accuracy | +0,042 ($q$ = 0,3) | +0,285 ($q$ = 0,6) |
| 2 | LLP, AUC | +0,135 ($k$ = 8) | +0,234 ($k$ = 16); +0,143 ($k$ = 32, kiểm tra áp lực) |

Mẫu hình khớp với cách phân loại: gần 0 ở loại 1, lớn và tăng theo mức làm thô ở loại 2 (theo dataset: PLL tăng từ $q$ = 0,3 lên 0,6 ở 3/3 dataset; LLP tăng từ túi 8 lên 16 ở 4/6 — blood-transfusion và credit-g giảm nhẹ). Riêng LLP túi 32, chênh lệch giảm lại; điều này khớp với rủi ro trộn chậm (METHOD_SPEC, rủi ro 3) chứ không mâu thuẫn với Quan sát 6(b). Dự đoán tương ứng được ghi trước như phân tích thăm dò cho tập đánh giá (giao thức Mục 11, deviation 6).

## Tài liệu
- Besag, J. (1974). Spatial interaction and the statistical analysis of lattice systems. *JRSS-B* 36(2):192–236. (bổ đề Brook)
- Brook, D. (1964). On the distinction between the conditional probability and the joint probability approaches in the specification of nearest-neighbour systems. *Biometrika* 51:481–483.
- Mitrophanov, A. Yu. (2005). Sensitivity and convergence of uniformly ergodic Markov chains. *J. Appl. Probab.* 42(4):1003–1014.
- Seneta, E. (2006). *Non-negative Matrices and Markov Chains*, 2nd ed. Springer. (hệ số Dobrushin)
- Tierney, L. (1994). Markov chains for exploring posterior distributions. *Ann. Statist.* 22(4):1701–1728.
- Roberts, G. O. & Smith, A. F. M. (1994). Simple conditions for the convergence of the Gibbs sampler and Metropolis–Hastings algorithms. *Stoch. Proc. Appl.* 49(2):207–216.
- Liu, J., Gelman, A., Hill, J., Su, Y.-S. & Kropko, J. (2014). On the stationary distribution of iterative imputations. *Biometrika* 101(1):155–173.
- Müller, S., Hollmann, N., Pineda Arango, S., Grabocka, J. & Hutter, F. (2022). Transformers can do Bayesian inference. *ICLR*.
- Elkan, C. & Noto, K. (2008). Learning classifiers from only positive and unlabeled data. *KDD*.
- Schweitzer, P. J. (1968). Perturbation theory and finite Markov chains. *J. Appl. Probab.* 5(2):401–413. *(cần kiểm trang)*
- Meyer, C. D. (1980). The condition of a finite Markov chain and perturbation bounds for the limiting probabilities. *SIAM J. Algebraic Discrete Methods* 1(3):273–283. *(cần kiểm trang)*
- Seneta, E. (1988). Perturbation of the stationary distribution measured by ergodicity coefficients. *Adv. Appl. Probab.* 20(1):228–230. *(cần kiểm trang)*
- Cho, G. E. & Meyer, C. D. (2001). Comparison of perturbation bounds for the stationary distribution of a Markov chain. *Linear Algebra Appl.* 335:137–150. *(cần kiểm trang)*
- Agarwal, S. (2014). Surrogate regret bounds for bipartite ranking via strongly proper losses. *JMLR* 15(49):1653–1674. (Hệ quả 12 trong bản arXiv 1207.0268; đã đối chiếu 26/10/2026.)
- Clémençon, S. & Robbiano, S. (2011). Minimax learning rates for bipartite ranking and plug-in rules. *ICML*.
- Clémençon, S., Lugosi, G. & Vayatis, N. (2008). Ranking and empirical minimization of U-statistics. *Ann. Statist.* 36(2):844–874.
- Godambe, V. P. (1960). An optimum property of regular maximum likelihood estimation. *Ann. Math. Statist.* 31(4):1208–1211. (thông tin Godambe của hàm ước lượng)
