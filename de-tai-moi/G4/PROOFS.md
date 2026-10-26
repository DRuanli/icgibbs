# G4 · Chứng minh đầy đủ (đề tài A: In-Context Gibbs)

> Trạng thái: **bản G4, đã qua một lượt tự rà soát phản biện ([SELF_REVIEW_MATH.md](SELF_REVIEW_MATH.md)), chờ kiểm duyệt độc lập**. Thay cho [THEORY_DRAFT.md](../G3/THEORY_DRAFT.md). Mỗi kết quả có: giả định, phát biểu, chứng minh, và trỏ tới kiểm chứng số trong [RQ1_REPORT.md](RQ1_REPORT.md). Những chỗ khác với bản nháp G3 được đánh dấu **[đổi so với G3]** và ghi lý do.
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

**(A1) — TFM chính xác.** $q(\cdot\mid x,D)=p(\cdot\mid x,D)$ với mọi ngữ cảnh $D$ mà thuật toán dùng.
**(A2) — Dương.** PPD của prior thoả $p(k\mid x,D)>0$ với mọi $k$ và mọi ngữ cảnh có xác suất dương. Dưới (A1), điều này suy ra từ tính dương của đầu ra softmax của TFM.
**(A3) — Hợp lý.** $\ell$ đã biết (hoặc tham số phiền nhiễu được lấy mẫu trong chuỗi, xem Nhận xét 1.3) và $\sum_y p(y\mid X)\ell(y)>0$.

## 1. Bổ đề 1 và hệ quả

**Bổ đề 1.** Với mọi $i$ và mọi $y$ sao cho $p(y_{-i}\mid X)>0$ (với feature liên tục: với $p(X)$-hầu hết $X$): $\;p(y_i\mid y_{-i},X)=p(y_i\mid x_i,D_{-i})$.

*Chứng minh.* Các bộ $(x_j,y_j)$ độc lập cùng phân phối khi biết $\phi$, nên trao đổi được. Do đó luật đồng thời của $(D_{-i},(x_i,y_i))$ trùng với luật của $(D_{n-1},(x_*,y_*))$, trong đó $D_{n-1}$ là ngữ cảnh $n-1$ mẫu và $(x_*,y_*)$ là truy vấn. Thông tin $(X,y_{-i})$ trùng với $(x_i,D_{-i})$. Vì vậy
$$p(y_i\mid y_{-i},X)=\frac{p\big((x_i,y_i),D_{-i}\big)}{\sum_k p\big((x_i,k),D_{-i}\big)}=p(y_i\mid x_i,D_{-i}).\qquad\square$$

*Ghi chú.* PPD ở đây điều kiện hoá cả trên $x_*$; feature của truy vấn có thể mang thông tin về $\phi$ qua $p(x\mid\phi)$. Đây đúng là đại lượng mà loss của PFN nhắm tới, vì trong lúc huấn luyện $x_*$ được sinh từ prior cùng với ngữ cảnh (Müller et al., ICLR 2022; khóa học, Chương 7). Kiểm chứng: `test_lemma1_full_conditional_equals_ppd` (sai lệch < $10^{-12}$ trên mọi $2^7$ cách gán, cả prior ô và prior hỗn hợp).

**Hệ quả 1.1 (dương).** Dưới (A2), $p(y\mid X)>0$ với mọi $y\in[C]^n$.
*Chứng minh.* Tồn tại $y^0$ với $p(y^0\mid X)>0$ vì tổng bằng 1. Theo bổ đề Brook (Besag 1974), với dãy đổi từng mẫu $y^0\to y^1\to\dots\to y$,
$p(y\mid X)/p(y^0\mid X)=\prod_k p(y^k_{s_k}\mid y^k_{-s_k},X)/p(y^{k-1}_{s_k}\mid y^{k-1}_{-s_k},X)$; mỗi thừa số dương theo Bổ đề 1 và (A2). $\square$

**Nhận xét 1.2 (cập nhật khối không chính xác).** Với $|B|\ge2$, nói chung $p(y_B\mid y_{-B},X)\ne\prod_{i\in B}p(y_i\mid x_i,D_{-B})$, vì hai lý do: (i) các nhãn trong $B$ phụ thuộc nhau khi biết $D_{-B}$; (ii) vế phải không điều kiện hoá trên $x_j$, $j\in B\setminus\{i\}$, trong khi $x_j$ mang thông tin về $\phi$ trừ khi $p(x\mid\phi)$ không phụ thuộc $\phi$. Kiểm chứng: E5a (prior ô chỉ có nguồn (i); prior hỗn hợp có cả hai).

**Nhận xét 1.3 (tham số phiền nhiễu).** Với $\theta_W$ độc lập tiên nghiệm, $\pi(y,\theta_W)\propto p(y\mid X)\,p(\theta_W)\,\ell(y;\theta_W)$, nên Bổ đề 1 vẫn cho cập nhật $y_i$ khi biết $\theta_W$, và $\theta_W\mid y$ không cần TFM. Cập nhật liên hợp: PU-SCAR $c\mid y\sim\mathrm{Beta}(a+L,\,b+P-L)$ với $L=\#\{s_i=1\}$, $P=\#\{y_i=1\}$; nhãn nhiễu $T_{k,\cdot}\mid y\sim\mathrm{Dir}(\beta_k+N_{k,\cdot})$. PU-SAR: random-walk MH trên $\eta$ của $e(x)=\sigma(\eta^\top\tilde x)$.

## 2. Định lý 1 (tính đúng)

### 2(a) Gibbs một mẫu cho quan sát cấp mẫu

Gọi $F=\{i:|\{k:\ell_i(k)>0\}|\ge2\}$ là các mẫu tự do. Nhân cập nhật mẫu $i$:
$$P_i(y,y')=\pi_i(y'_i\mid y_{-i})\,\mathbb 1[y'_{-i}=y_{-i}],\qquad \pi_i(k\mid y_{-i})=\frac{q(k\mid x_i,D_{-i})\,\ell_i(k)}{\sum_{k'}q(k'\mid x_i,D_{-i})\,\ell_i(k')}.$$
Quét ngẫu nhiên: $P=\frac1{|F|}\sum_{i\in F}P_i$. Cài đặt dùng **một vòng quét theo hoán vị ngẫu nhiên**: $P_{sw}=\frac1{|F|!}\sum_{\sigma}P_{\sigma(1)}\cdots P_{\sigma(|F|)}$. Mỗi $P_i$ giữ $\pi$ nên $P_{sw}$ giữ $\pi$; $P_{sw}$ bất khả quy vì trong một vòng quét mỗi mẫu có thể nhận bất kỳ giá trị nào trong tập hỗ trợ của nó với xác suất dương, và không tuần hoàn vì $P_{sw}(y,y)>0$. $P_{sw}$ nói chung không khả nghịch; kết quả dưới đây không cần khả nghịch.

**Định lý 1(a).** Dưới (A0)–(A3) với $\ell$ đã biết:
1. mỗi $P_i$ thoả cân bằng chi tiết với $\pi$;
2. $P$ bất khả quy và không tuần hoàn trên $\mathrm{supp}(\pi)=\prod_i\{k:\ell_i(k)>0\}$, nên $\max_y\mathrm{TV}(P^t(y,\cdot),\pi)\le C\rho^t$ với $C<\infty$, $\rho<1$.

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
PU-SAR với MH trên $\eta$: dãy $y$ không còn Markov riêng; hội tụ theo lý thuyết chuỗi trên không gian tổng quát (Tierney 1994; Roberts & Smith 1994) — **chỉ nêu, chưa kiểm tra chi tiết điều kiện** (việc mở G4-1).

### 2(d) Dự đoán: quy nạp và chuyển nạp **[đổi so với G3]**

Bản nháp G3 khẳng định ước lượng Rao–Blackwell hội tụ về $p(y_*\mid x_*,X,W)$. **Điều này chỉ đúng khi feature không mang thông tin về $\phi$.** Phát biểu đúng:

**Định lý 1(c).** Đặt $\bar p(y_*\mid x_*)=\sum_y\pi(y)\,p(y_*\mid x_*,(X,y))$. Dưới (A0)–(A3):
1. *(quy nạp)* Với chuỗi của 2(a)/(b), $\hat p_S=\frac1S\sum_{s\le S}q(y_*\mid x_*,(X,y^{(s)}))\to\bar p(y_*\mid x_*)$ hầu chắc chắn, với mọi trạng thái đầu.
2. $p(y_*\mid x_*,X,W)=\sum_y p(y\mid x_*,X,W)\,p(y_*\mid x_*,(X,y))$, nên $|\bar p-p(y_*\mid x_*,X,W)|\le\mathrm{TV}\big(p(y\mid X,W),\,p(y\mid x_*,X,W)\big)$; dấu bằng $\bar p=p(y_*\mid x_*,X,W)$ xảy ra khi prior **tách được**: $\phi=(\phi_x,\phi_{y\mid x})$ độc lập tiên nghiệm, $p(x\mid\phi)=p(x\mid\phi_x)$ và $p(y\mid x,\phi)=p(y\mid x,\phi_{y\mid x})$ (trường hợp riêng: $p(x\mid\phi)$ không phụ thuộc $\phi$).
3. *(chuyển nạp)* Thêm $x_*$ vào chuỗi như một mẫu tự do ($\ell_*\equiv1$). Chuỗi mới chính xác với mục tiêu $p(y,y_*\mid X,x_*,W)$ theo 2(a), và ước lượng Rao–Blackwell $\frac1S\sum_s\pi_*(y_*\mid y^{(s)})=\frac1S\sum_s q(y_*\mid x_*,(X,y^{(s)}))$ hội tụ về **đúng** $p(y_*\mid x_*,X,W)$. Với $m$ điểm test cùng lúc, chuỗi hội tụ về $p(y_{*k}\mid X_*,X,W)$ (điều kiện hoá trên mọi feature test).

*Chứng minh.* (1) Định lý ergodic cho chuỗi hữu hạn trạng thái bất khả quy (không cần không tuần hoàn), áp cho hàm bị chặn $y\mapsto p(y_*\mid x_*,(X,y))$, cùng (A1). (2) Luật xác suất toàn phần và (A0): $p(y_*\mid x_*,X,y,W)=p(y_*\mid x_*,X,y)$. Hiệu của hai kỳ vọng của cùng một hàm nhận giá trị trong $[0,1]$ bị chặn bởi TV của hai độ đo. Với prior tách được, $p(y\mid x_*,X,W)\propto\ell(y)\int p(y\mid X,\phi_{y\mid x})p(\phi_{y\mid x})\,d\phi_{y\mid x}\cdot\underbrace{\int p(X,x_*\mid\phi_x)p(\phi_x)\,d\phi_x}_{\text{không phụ thuộc }y}$, nên $x_*$ không làm đổi posterior của $y$ và $p(y\mid x_*,X,W)=p(y\mid X,W)$. (3) Áp 2(a) cho bài toán $n+1$ mẫu; Bổ đề 1 cho $\pi_*(y_*\mid y)=p(y_*\mid x_*,(X,y))$; định lý ergodic. $\square$

Kiểm chứng (E2): prior ô ⇒ $|\bar p-p_{\text{Bayes}}|<10^{-15}$; prior hỗn hợp (x mang thông tin) ⇒ chênh lệch tới 0,11 ở $n=8$; chế độ chuyển nạp khớp Bayes chính xác (`test_rao_blackwell_and_transductive`).

**Hệ quả cho bài.** TabPFN được huấn luyện trên prior SCM, trong đó $x$ mang thông tin về cơ chế sinh. Vì vậy chế độ chuyển nạp là phiên bản "Bayes chính xác", còn chế độ quy nạp là xấp xỉ có sai lệch $O(\cdot)$ theo ảnh hưởng của một điểm lên posterior. Cả hai được cài; thực nghiệm chọn theo chi phí (việc mở G4-2).

## 3. Mệnh đề 3 (Bayes-optimal hữu hạn mẫu, có điều kiện theo prior)

**Mệnh đề 3.** Gọi $S$ là một quy tắc chấm điểm proper nghiêm ngặt (log score, Brier). Trong mọi bộ dự đoán đo được $f:(X,W,x_*)\mapsto\Delta_C$, rủi ro Bayes $\mathbb E[S(f(X,W,x_*),y_*)]$ — kỳ vọng theo luật đồng thời của $(\phi,X,y,W,x_*,y_*)$ dưới prior — đạt cực tiểu duy nhất (h.c.c.) tại $f^\star=p(y_*\mid x_*,X,W)$.

*Chứng minh.* Rủi ro của $f^\star$ hữu hạn (với log score, $\le\log C$ vì là kỳ vọng của entropy có điều kiện), nên so sánh có nghĩa; bộ dự đoán có rủi ro vô hạn hiển nhiên thua. Theo tính chất tháp, $\mathbb E[S(f,y_*)]=\mathbb E\big[\mathbb E[S(f,y_*)\mid X,W,x_*]\big]$. Với mỗi giá trị của $(X,W,x_*)$, $f$ là một phân phối cố định và $y_*\sim p(\cdot\mid x_*,X,W)$; tính proper nghiêm ngặt cho cực tiểu duy nhất tại $f=p(\cdot\mid x_*,X,W)$. $\square$

**Hệ quả 3.1.** Dưới (A1), bộ dự đoán chuyển nạp của Định lý 1(c.3) hội tụ về $f^\star$; bộ quy nạp hội tụ về $f^\star$ khi $p(x\mid\phi)$ không phụ thuộc $\phi$.
**Phạm vi phải nói rõ trong bài:** đây là tối ưu *trung bình theo prior*, không phải đảm bảo trên phân phối thật. Khi dữ liệu thật nằm ngoài prior của TFM, không có đảm bảo nào; đóng góp thực nghiệm (RQ2–RQ5) mới là bằng chứng.

## 4. Định lý 2 (độ bền khi TFM chỉ xấp xỉ) **[đổi so với G3: dạng Dobrushin, có trung bình theo π̃]**

Gọi $P$ là nhân của thuật toán khi dùng $p$ (chính xác, phân phối dừng $\pi$ theo Định lý 1) và $\tilde P$ là **cùng thuật toán** khi dùng $q$. Hệ số ergodic Dobrushin: $\delta(K)=\max_{y,y'}\mathrm{TV}(K(y,\cdot),K(y',\cdot))$. Đặt
$$\tau(P)=\sum_{t\ge0}\delta(P^t),\qquad \bar\varepsilon=\sum_y\tilde\pi(y)\,\mathrm{TV}\big(\tilde P(y,\cdot),P(y,\cdot)\big)\le\varepsilon_{\max}=\max_y\mathrm{TV}\big(\tilde P(y,\cdot),P(y,\cdot)\big).$$

**Định lý 2.** Giả sử $q>0$. Khi đó $\tilde P$ có phân phối dừng duy nhất $\tilde\pi$ — **kể cả khi các xác suất có điều kiện của $q$ không tương thích với bất kỳ phân phối đồng thời nào** — và
$$\mathrm{TV}(\tilde\pi,\pi)\;\le\;\tau(P)\,\bar\varepsilon\;\le\;\tau(P)\,\varepsilon_{\max}.$$
Hơn nữa:
- (i) *(dạng Mitrophanov)* nếu $2\max_y\mathrm{TV}(P^t(y,\cdot),\pi)\le C\rho^t$ với mọi $t$, thì $\tau(P)\le\hat t+C\rho^{\hat t}/(1-\rho)$, $\hat t=\max\{0,\lceil\log_\rho(1/C)\rceil\}$. Kết hợp với $\bar\varepsilon\le\varepsilon_{\max}$, dạng này cho lại đúng chặn của Mitrophanov (2005, J. Appl. Probab. 42:1003–1014, Hệ quả 3.1) — vốn viết theo chuẩn $\|\mu\|=|\mu|(\Omega)=2\,\mathrm{TV}$ và chuẩn toán tử $\|E\|=2\varepsilon_{\max}$. Định lý 2 làm mịn chặn đó bằng cách thay $\varepsilon_{\max}$ bằng trung bình $\bar\varepsilon$ theo $\tilde\pi$.
- (ii) *(Gibbs một mẫu, nhân quét ngẫu nhiên $P$)* $\varepsilon_{\max}\le\min\{1,\,2\kappa\,\varepsilon_{TV}\}$; với nhân một vòng quét $P_{sw}$ của cài đặt: $\varepsilon_{\max}\le\min\{1,\,2|F|\kappa\,\varepsilon_{TV}\}$ (TV của tích các nhân cộng dồn) và $\tau$ tính theo vòng quét với $\varepsilon_{TV}=\sup\mathrm{TV}(q(\cdot\mid x_i,D_{-i}),p(\cdot\mid x_i,D_{-i}))$ trên các ngữ cảnh mà thuật toán dùng, và $\kappa=\sup_{i\in F,y}\max_k\ell_i(k)\big/\sum_k p(k\mid x_i,D_{-i})\ell_i(k)$. Với PU-SCAR, $\kappa\le1/(1-c)$.
- (iii) *(MH hoán đổi)* $\varepsilon_{\max}\le\min\{1,\,4\varepsilon_\infty\}$ với $\varepsilon_\infty=\sup|\log q-\log p|$ trên các cặp (ngữ cảnh, truy vấn) mà thuật toán dùng.
- (iv) *(dự đoán)* với $\varepsilon'_{TV}=\sup_y\mathrm{TV}\big(q(\cdot\mid x_*,(X,y)),p(\cdot\mid x_*,(X,y))\big)$: $\big|\mathbb E_{\tilde\pi}q(y_*\mid x_*,\cdot)-\bar p(y_*\mid x_*)\big|\le\varepsilon'_{TV}+\mathrm{TV}(\tilde\pi,\pi)$.

*Chứng minh.*
**Bước 0 (tồn tại, duy nhất).** Không gian trạng thái hữu hạn nên $\tilde P$ có ít nhất một phân phối dừng. Với $q>0$, $\tilde P$ có cùng đồ thị chuyển (các cạnh dương) như $P$ — xác suất cập nhật một mẫu và xác suất chấp nhận MH đều dương — nên bất khả quy, do đó $\tilde\pi$ duy nhất. Lập luận này không dùng tới sự tương thích của $q$.
**Bước 1 (đồng nhất thức nhiễu loạn).** Đặt $\Delta=\tilde\pi-\pi$ và $\nu=\tilde\pi(\tilde P-P)$; $\nu(\Omega)=0$. Từ $\tilde\pi=\tilde\pi\tilde P$, $\pi=\pi P$: $\Delta=\nu+\Delta P$. Lặp $T$ lần: $\Delta=\sum_{t<T}\nu P^t+\Delta P^T$.
**Bước 2 (co Dobrushin).** Với độ đo có dấu $\mu$, $\mu(\Omega)=0$: $\|\mu K\|_1\le\delta(K)\|\mu\|_1$. (Viết $\mu=a(\alpha-\beta)$ với $\alpha,\beta$ là xác suất, $a=\|\mu\|_1/2$; khi đó $\|\alpha K-\beta K\|_1\le2\max_{y,y'}\mathrm{TV}(K(y,\cdot),K(y',\cdot))$ theo tính lồi; xem Seneta 2006, §3.1.) Nếu $\tau(P)=\infty$ thì bất đẳng thức hiển nhiên đúng. Nếu $\tau(P)<\infty$ thì $\delta(P^T)\to0$, nên $\|\Delta P^T\|_1\le\delta(P^T)\|\Delta\|_1\to0$ và chuỗi $\Delta=\sum_{t\ge0}\nu P^t$ hội tụ tuyệt đối. (Định lý **không** cần giả định $P$ không tuần hoàn: ví dụ MH hoán đổi không lazy trên một túi $\{1,2\}$ với số đếm $(1,1)$ và $\pi$ đều là tuần hoàn, $\tau=\infty$, chặn trở thành hiển nhiên.)
**Bước 3.** $\|\nu\|_1\le\sum_y\tilde\pi(y)\|\tilde P(y,\cdot)-P(y,\cdot)\|_1=2\bar\varepsilon$. Do đó $\|\Delta\|_1\le2\bar\varepsilon\sum_t\delta(P^t)$, tức $\mathrm{TV}(\tilde\pi,\pi)\le\tau(P)\bar\varepsilon$.
**(i)** $\delta(P^t)\le1$ và $\delta(P^t)\le\max_{y,y'}[\mathrm{TV}(P^t(y),\pi)+\mathrm{TV}(\pi,P^t(y'))]\le C\rho^t$. Vậy $\tau\le\sum_t\min(1,C\rho^t)\le\hat t+\sum_{t\ge\hat t}C\rho^t$.
**(ii)** $\tilde P(y,\cdot)-P(y,\cdot)=\frac1{|F|}\sum_i[\tilde P_i(y,\cdot)-P_i(y,\cdot)]$, và $\mathrm{TV}(\tilde P_i(y,\cdot),P_i(y,\cdot))=\mathrm{TV}(q_\ell,p_\ell)$ với $q_\ell\propto q\ell_i$, $p_\ell\propto p\ell_i$. *Bổ đề tái trọng số:* với $Z_p=\sum p\ell$, $Z_q=\sum q\ell$,
$$2\,\mathrm{TV}(q_\ell,p_\ell)\le\sum\ell|q-p|/Z_p+|Z_q-Z_p|/Z_p\le2\max\ell\cdot\|q-p\|_1/Z_p,$$
nên $\mathrm{TV}(q_\ell,p_\ell)\le2(\max\ell/Z_p)\,\mathrm{TV}(q,p)\le2\kappa\varepsilon_{TV}$. Với PU-SCAR, $Z_p=p_0+(1-c)p_1\ge1-c$.
**(iii)** $x\mapsto\min(1,e^x)$ là 1-Lipschitz, $\log R_{ij}$ là tổng bốn số hạng $\pm\log q$, nên $|\tilde\alpha-\alpha|\le|\log\tilde R-\log R|\le4\varepsilon_\infty$. Hiệu hai hàng của nhân là $\sum_{o}Q_o(\tilde\alpha_o-\alpha_o)(\delta_{y'_o}-\delta_y)$ theo các đề xuất $o$, nên $\mathrm{TV}\le\sum_oQ_o|\tilde\alpha_o-\alpha_o|\le4\varepsilon_\infty$.
**(iv)** $|\mathbb E_{\tilde\pi}q-\mathbb E_\pi p|\le|\mathbb E_{\tilde\pi}(q-p)|+|\mathbb E_{\tilde\pi}p-\mathbb E_\pi p|$; số hạng đầu $\le\varepsilon'_{TV}$ (với một lớp $k$ cố định, $|q(k)-p(k)|\le\mathrm{TV}$), số hạng sau $\le\mathrm{TV}(\tilde\pi,\pi)$ vì $p(k\mid\cdot)\in[0,1]$. $\square$

**Nhận xét 4.1 (độ chặt và phụ thuộc vào $n$).** $\tau(P)$ là "thời gian trộn tích phân" của chuỗi chính xác, tính theo **bước**. Với Gibbs một mẫu quét ngẫu nhiên, $\delta(P^t)=1$ ít nhất cho tới khi mọi mẫu được chạm tới, nên $\tau\gtrsim|F|$; chặn có dạng (thời gian trộn theo bước) × (sai số mỗi bước). Hệ số $n$ trong bản nháp G3 ($\|\tilde P-P\|\le2n\kappa\varepsilon$ cho một vòng quét tuần tự) xuất hiện lại qua $\tau$ — không mất đi, nhưng giờ có hằng số tính được. Kiểm chứng (E3, E4): chặn luôn đúng; dạng trung bình $\tau\bar\varepsilon$ lỏng khoảng 2–15 lần; dạng theo $\varepsilon_{TV}$ lỏng thêm khoảng 10 lần.

**Nhận xét 4.2 (định vị so với Liu et al., Biometrika 2014; đã đọc bản arXiv 1012.2902 ngày 24/10/2026).** Liu, Gelman, Hill, Su & Kropko xét phân phối dừng của imputation lặp (chained equations) với dữ liệu liên tục. Định lý 1 của họ: khi các mô hình có điều kiện *tương thích*, TV giữa phân phối dừng của imputation lặp và posterior predictive Bayes → 0 theo xác suất khi $n\to\infty$ (cần các điều kiện về tập nhỏ, drift, tập trung của MLE). Định lý 2 của họ: nếu xác suất có điều kiện của imputation lệch khỏi mô hình đồng thời một lượng $\varepsilon$ trên một tập có khối lượng $\ge q$, thì TV giữa hai phân phối dừng $\ge q\varepsilon/4$ — tức chặn **dưới** tuyến tính theo $\varepsilon$. Định lý 3: với mô hình không tương thích nhưng "valid", ước lượng tham số nhất quán, nhưng phân phối imputation không nhất thiết khớp mô hình Bayes nào.
Khác biệt của Định lý 2 ở đây: không gian trạng thái hữu hạn; không tiệm cận theo $n$; hằng số tường minh $\tau(P)$ tính được; áp cho cả bước MH; sai số đo theo TV giữa các xác suất có điều kiện (là thứ đo được của một TFM) thay vì theo tham số. Kết hợp với chặn dưới $q\varepsilon/4$ của Liu et al., sự phụ thuộc **tuyến tính** vào sai số của xác suất có điều kiện là đúng bậc; phần có thể lỏng là hệ số $\tau(P)$.

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

**Giới hạn của 4(b) (ghi trung thực).** Với $\beta=O(1/n)$, sai số mỗi bước là $O(\kappa_Bm^2/n)$, nhưng $\tau(P_{bG})$ tăng theo số khối, và $\kappa_B$ có thể tăng theo cấp số nhân với số mẫu tự do trong khối. Vì vậy chặn **không** cho thấy sai số tiến về 0 với $K$ cố định. Đây là chặn đủ, không chặt.

**Quan sát 4(c) — số liệu (E5).** Xem [RQ1_REPORT.md](RQ1_REPORT.md) §5. Trên chuỗi gộp một ô (tính chính xác tới $n=2000$), tách được hai đại lượng: TV trên posterior của nhãn, và sai số của *dự đoán*. Với khối chia trên mọi mẫu: (1) TV $\approx0{,}1/K$, không giảm theo $n$ — sai ở độ phân tán của posterior do bỏ qua phụ thuộc trong khối; (2) sai số dự đoán giảm như $O(1/n)$ (PU, K = 10: 0,0017 ở $n=50$, 0,00017 ở $n=1000$). Khi chia chỉ trên mẫu tự do, có thêm lệch thành phần không giảm theo $n$ (PU, K = 10: 0,031). Kết luận ở dạng quan sát, không phải định lý; một chứng minh cho (2) là việc mở G4-4.

## 6. Việc mở sau G4

| # | Việc | Ảnh hưởng nếu không làm |
|---|---|---|
| G4-1 | Kiểm tra điều kiện hội tụ cho PU-SAR (MH trên $\eta$, không gian tổng quát) | Định lý 1 chỉ phát biểu cho $\ell$ đã biết và các cập nhật liên hợp |
| G4-2 | Chọn quy nạp hay chuyển nạp cho thực nghiệm chính (chi phí so với độ chính xác) — quyết ở G5, dựa trên tập phát triển | Bài phải nói rõ bộ dự đoán nào được dùng và nó hội tụ về đâu |
| G4-3 | Liu et al. (2014): đã đọc bản arXiv, cập nhật Nhận xét 4.2. Còn lại: đối chiếu bản Biometrika in chính thức (số định lý có thể khác bản arXiv) và đọc lại Hệ quả 3.1 của Mitrophanov (2005) ở bản PDF | Rủi ro trích sai số định lý khi viết |
| G4-4 | Chặn chặt hơn cho cập nhật khối, hoặc giữ dạng quan sát | Mệnh đề 4(b) chỉ là chặn đủ |

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
