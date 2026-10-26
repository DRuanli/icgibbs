# Phản biện độc lập phần toán: `G4/PROOFS.md` (commit c08e02b)

Phạm vi đã đọc: §0–§7 của PROOFS.md (đủ 263 dòng), METHOD_SPEC.md, RQ1_REPORT.md, `icgibbs/{exact,sampler,conditional,weak}.py`, `g6_t7_naive_theory.py`, `tests/test_exact.py`, `tests/test_t7_naive_theory.py`, các dòng sinh nhãn yếu trong `kaggle_g4_pilot.py` và `kaggle_g4_rq4_pilot.py`, các dòng liên quan trong `G5/EXPERIMENT_PROTOCOL.md`. Các tệp được review trùng với c08e02b (`git diff c08e02b` rỗng). Chưa đọc SELF_REVIEW_MATH.md, GATE_REVIEW_G4.md hay G6/*.md trước khi viết các mục 1–5. Mục 6 thêm sau, khi đã đọc lướt SELF_REVIEW_MATH.md.

## 1. Kết luận chung

Không có lỗi Critical. Mọi định lý và mệnh đề chính đều đúng như đã phát biểu (riêng Hệ quả 3.1 sai khi có nhiều điểm test), và tôi đã kiểm độc lập bằng liệt kê chính xác hoặc phản ví dụ ngẫu nhiên: Bổ đề 1, Định lý 1(a)(b)(c), Mệnh đề 1.4, Mệnh đề 3, Định lý 2 cùng (i)–(iv), Mệnh đề 4(a)(b), Mệnh đề 5(a)–(c), dạng đóng của PLL và LLP, Mệnh đề 6. Riêng chặn của Định lý 2 và chặn AUC 5(c) là **chặt**: dấu bằng đạt được, và hằng số không giảm được nữa. Quy ước chuẩn khi so với Mitrophanov (2005, Hệ quả 3.1) cũng đúng; tôi đã đối chiếu với bản gốc. Vấn đề nằm ở bốn chỗ (Major):
1. Các sơ đồ mà thực nghiệm **thật sự dùng** (khối K = 10; bag-block cho LLP) không có đảm bảo nào khác rỗng. Bag-block xấp xỉ thì không có cả chứng minh lẫn kiểm chứng số. Mục §6 nói nhẹ hơn thực tế.
2. Hệ quả 3.1 sai với chế độ chuyển nạp nhiều điểm test, tức đúng chế độ mã cài.
3. Chặn chính của Định lý 2 là hệ quả của lý thuyết nhiễu loạn kinh điển, cụ thể là đồng nhất thức $\tilde\pi-\pi=\tilde\pi EZ$ và hệ số $\tau_1(Z)$. Bài phải trích các nguồn này, và câu "làm mịn Mitrophanov" không nên trình bày như đóng góp mới.
4. Văn xuôi §7 dùng chữ "nhất quán" vượt quá điều đã chứng minh, cả cho bộ ngây thơ lẫn cho Gibbs. Câu "chỉ PU-SAR" mâu thuẫn với chính giao thức.

Còn lại là 12 lỗi Minor: một câu sai về tính khả nghịch của $P_{sw}$, một câu sai ở Nhận xét 4.1, một chứng minh viết vòng (Hệ quả 1.1), giả định ngầm của (A0) mà giao thức SSL vi phạm, chồng ký hiệu, và số liệu lệch với RQ1_REPORT.

## 2. Bảng kết luận theo từng kết quả

| Kết quả | Kết luận | Lý do một dòng |
|---|---|---|
| Bổ đề 1 | đúng | Hai σ-đại số $(X,y_{-i})$ và $(x_i,D_{-i})$ trùng nhau; tính trao đổi được cho cùng một hàm PPD. Liệt kê hỗn hợp 3 lớp: sai lệch 1,2e-15 |
| Hệ quả 1.1 | đúng nhưng cần sửa cách viết | Tích Brook cần $p(y^{k-1}_{-s_k}\mid X)>0$, chính là điều đang chứng minh; phải viết thành quy nạp (F7) |
| Nhận xét 1.2 | đúng | — |
| Nhận xét 1.3 | đúng | Cập nhật Beta và Dirichlet kiểm lại bằng tay |
| Định lý 1(a), kể cả đoạn về $P_{sw}$ | đúng nhưng cần sửa cách viết | Chứng minh đúng. Câu "$P_{sw}$ nói chung không khả nghịch" sai: dưới (A1), $P_{sw}$ khả nghịch (F5) |
| Định lý 1(b) | đúng | Cặp có thứ tự, $N_g(y')=N_g(y)$, xác suất đề xuất $2/(GN_g)$ đều đúng. Ma trận chuyển 3 lớp: TV 6,9e-16 |
| Mệnh đề 1.4 | đúng | Nhân biên $\bar P$ (cầu phương 60 nút) giữ $\pi$ tới 6e-15 |
| Định lý 1(c)(1) | đúng | Định lý ergodic cho chuỗi hữu hạn bất khả quy |
| Định lý 1(c)(2) | đúng nhưng cần sửa cách viết | Chặn TV và điều kiện đủ "prior tách được" đều đúng. Câu tiêu đề "chỉ đúng khi feature không mang thông tin" vừa sai vừa không khớp với điều kiện tách được (F10) |
| Định lý 1(c)(3) | đúng với điều kiện bổ sung | Với LLP, chuỗi là hoán đổi + Gibbs một mẫu, không phải "theo 2(a)". Với $m\ge2$, công thức Rao–Blackwell phải đưa các điểm test khác vào ngữ cảnh (F9) |
| Mệnh đề 3 | đúng | Tính chất tháp + tính proper nghiêm ngặt; rủi ro hữu hạn đã được xử lý |
| Hệ quả 3.1 | sai (khi $m\ge2$ điểm test) | Chuỗi hội tụ về $p(y_{*1}\mid X_*,X,W)\neq f^\star$; phản ví dụ chênh tới 0,123 (F2) |
| Định lý 2 (chặn chính, Bước 0–3) | đúng | Chặt: dấu bằng đạt được trên chuỗi 2 trạng thái. Là trường hợp yếu hơn của chặn kiểu Seneta $\tau_1(Z)\lVert\nu\rVert$ (F3) |
| Định lý 2(i) | đúng | Khớp chuẩn và hằng số của Mitrophanov (2005) (3.8), đã đối chiếu bản gốc |
| Định lý 2(ii) | đúng | Bổ đề tái trọng số đúng; hằng 2 hợp lệ nhưng có thể hạ xuống 1,5. Hệ số $\lvert F\rvert$ cho một vòng quét đúng |
| Định lý 2(iii) | đúng | $\min(1,e^x)$ là 1-Lipschitz; kiểm số đạt tối đa 0,28 lần chặn |
| Định lý 2(iv) | đúng | — |
| Nhận xét 4.1 | sai (một mệnh đề phụ) | $\delta(P^t)<1$ đã xảy ra từ $t=\lceil\lvert F\rvert/2\rceil$, không phải "cho tới khi mọi mẫu được chạm tới". Số liệu "2–15 lần", "thêm khoảng 10 lần" lệch RQ1 (F6) |
| Nhận xét 4.2 | chưa kiểm được (một phần) | Nội dung các định lý của Liu et al. khớp bản arXiv. Kết luận "đúng bậc" là suy diễn chéo bối cảnh (F14) |
| Nhận xét 4.3 | đúng | — |
| Mệnh đề 4(a) | đúng | MH-trong-Gibbs với đề xuất độc lập; tỉ số Brook đúng |
| Mệnh đề 4(b) | đúng với điều kiện bổ sung | Cần Bổ đề 1 mở rộng cho feature không nhãn; nhân trong phân tích là chọn khối ngẫu nhiên, không phải vòng quét; chặn rỗng với cấu hình mặc định (F1, F11) |
| Quan sát 4(c) | đúng nhưng cần sửa cách viết | Số liệu khớp RQ1, nhưng "giảm như $O(1/n)$" không được dữ liệu ủng hộ: độ dốc log–log từ −0,25 đến −0,79 (F12) |
| Mệnh đề 5(a) | đúng | — |
| Bảng hệ số §7 (PLL, LLP, nhiễu, PU-SCAR) | đúng | Dạng đóng PLL khớp liệt kê tới 1e-15; độ dốc LLP quần thể hữu hạn khớp liệt kê mọi cách chia túi |
| Mệnh đề 5(b) | đúng | Argmax chỉ được giữ khi $b_k,\lambda_k$ chung cho mọi lớp: đúng với PLL, không đúng với PU-SCAR và nhiễu (bài đã viết có điều kiện) |
| Mệnh đề 5(c) | đúng | Từng bước của chứng minh đều đúng, kể cả trường hợp hoà. Hằng số không cải thiện được: tỉ số → 0,999 |
| Mệnh đề 6 | đúng | Bất đẳng thức Jensen có điều kiện; điều kiện dấu bằng đúng (kiểm bằng ví dụ gộp kết cục) |
| Các áp dụng của Mệnh đề 6 | đúng nhưng cần sửa cách viết | Mục SSL trộn "prior tách được" (Bayes) với thông tin Fisher (tần suất) (F10) |
| Quan sát 6(b) | đúng (chỉ là số liệu) | Chạy lại script, ra **y hệt** tệp `pilot/g6_t7_naive_theory.txt` |
| Văn xuôi §7 ("Đính chính", "Hệ quả cho bài") | sai một phần | "nhất quán", "chỉ PU-SAR", "Gibbs nhất quán (Định lý 1)", loại 1 của SSL (F4) |

## 3. Các phát hiện (nặng trước)

### F1 — Major — Sơ đồ mặc định của thực nghiệm không có đảm bảo khác rỗng; bag-block xấp xỉ (mặc định của LLP) không có cả chứng minh lẫn kiểm chứng số
**Vị trí.** §5 "Giới hạn của 4(b)": "*chặn **không** cho thấy sai số tiến về 0 với K cố định*". §6 G4-4: "*Mệnh đề 4(b) chỉ là chặn đủ*". METHOD_SPEC §4: "block, K khối … **mặc định thực nghiệm**", "bag-block, K nhóm … **mặc định LLP**". METHOD_SPEC §6c: "*Chế độ được khẳng định: LLP với túi ≤ 16*".
**Sai ở đâu.** Định lý 1, Định lý 1(c), Hệ quả 3.1 và Mệnh đề 3 chỉ áp cho 'single', 'swap' và '+MH'. Với cấu hình mặc định, chặn duy nhất là Mệnh đề 4(b), và chặn này rỗng vì hai lý do:
- **Thứ nhất**, với $K$ cố định thì $m=n/K$, nên $m_f(m-1)\beta=O(n^2/K^2\cdot1/n)=O(n/K^2)$ *tăng* theo $n$. Bài chỉ nói "không tiến về 0", trong khi thực ra chặn tăng.
- **Thứ hai**, với bag-block, $\kappa_B=1/P(\text{mọi túi trong nhóm khớp số đếm}\mid\cdot)$. Với túi cỡ 8, khoảng 25 túi mỗi nhóm ($n\approx2000$, $K=10$), $\log_{10}\kappa_B\approx15{,}8$; túi 16: ≈ 10,3; túi 32: ≈ 6,6 (`chk_sec7.py`).

Kiểm chứng số cũng không phủ: RQ1 E5 chỉ đo khối cấp mẫu trên chuỗi gộp **một ô**, tức không có nguồn sai (ii). E1mc chỉ đo bag-block **có** sửa MH. `test_bag_block_exact_vs_approx` chỉ kiểm tổng hàng bằng 1.

**Bằng chứng.** Tôi đo trên bài chính xác LLP 3 túi × 4, dùng `exact.kernel_bag_block(kind="approx")` (`chk_bagblock.py`):

| prior | nhóm | TV(π̃, π) | sai số dự đoán quy nạp |
|---|---|---:|---:|
| ô | 1 túi | 0,157 | 0,041 |
| ô | 3 túi | 0,494 | 0,095 |
| hỗn hợp | 1 túi | 0,227 | 0,069 |
| hỗn hợp | 3 túi | 0,639 | 0,122 |

Lưu ý: với prior ô, khối cấp mẫu có sai số dự đoán ≈ 0 (RQ1 E5a), còn bag-block thì **không**. $n=12$ phóng đại độ lớn, nhưng chứng tỏ sai lệch này tồn tại và chưa từng được đo.

**Sửa.**
- (a) Thêm G4-5 vào §6: "bag-block xấp xỉ: chưa có chặn khác rỗng, chưa đo".
- (b) Đo sai số bag-block bằng chuỗi gộp một ô cho LLP với $n$ lớn (trạng thái là số dương trong mỗi túi), cùng kiểu E5b/E5c.
- (c) Chạy bag-block+MH như phân tích độ nhạy *bắt buộc* cho khẳng định LLP chính, hoặc ít nhất báo cáo tỉ lệ chấp nhận MH của bản sửa.
- (d) Viết rõ trong bài rằng Định lý 1 và Mệnh đề 3 không áp cho cấu hình mặc định.

### F2 — Major — Hệ quả 3.1 sai với chuyển nạp nhiều điểm test
**Vị trí.** §3: "*Hệ quả 3.1. Dưới (A1), bộ dự đoán chuyển nạp của Định lý 1(c.3) hội tụ về $f^\star$*". Câu này mâu thuẫn với chính Định lý 1(c.3): "*Với $m$ điểm test cùng lúc, chuỗi hội tụ về $p(y_{*k}\mid X_*,X,W)$*". `fit_transductive` đưa **toàn bộ** `X_test` vào chuỗi.
**Sai ở đâu.** Khi $x$ mang thông tin về $\phi$, $p(y_{*1}\mid x_{*1},x_{*2},X,W)\neq p(y_{*1}\mid x_{*1},X,W)=f^\star$.
**Phản ví dụ.** `DiscreteMixturePPD.random` (3 thành phần, 3 ô, nhị phân, seed 100–129), $n=6$, PU-SCAR với $c=0{,}5$, $x_{*1}$ ở ô 0 và $x_{*2}$ ở ô 2. Chênh lệch tối đa 0,123, trung vị 0,020 trên 30 prior (`chk_thm1.py`).
**Hậu quả.** Phần tối ưu không mất: giới hạn mới là bộ Bayes trong lớp lớn hơn $f:(X,W,X_*)\to\Delta_C$, và chứng minh của Mệnh đề 3 áp nguyên văn. Nhưng phát biểu hiện tại sai, và so sánh với các baseline dự đoán từng điểm là so trên hai tập thông tin khác nhau.
**Sửa.** Phát biểu lại Mệnh đề 3 cho $f(X,W,X_*)$ và Hệ quả 3.1 cho giới hạn $p(y_{*k}\mid X_*,X,W)$. Nếu muốn đúng $f^\star$ thì chạy chuyển nạp từng điểm. Thay "khi $p(x\mid\phi)$ không phụ thuộc $\phi$" bằng "khi prior tách được" cho khớp với 1(c)(2).

### F3 — Major (vị trí đóng góp; phần toán đúng) — Chặn chính của Định lý 2 là hệ quả của lý thuyết nhiễu loạn kinh điển, và có chặn chặt hơn từ cùng dữ liệu
**Vị trí.** §4 (i): "*Định lý 2 làm mịn chặn đó bằng cách thay $\varepsilon_{\max}$ bằng trung bình $\bar\varepsilon$ theo $\tilde\pi$*". Nhận xét 4.2: "*Khác biệt của Định lý 2 ở đây: … hằng số tường minh $\tau(P)$ tính được*".
**Lập luận.** Với $E=\tilde P-P$ và $Z=(I-P+\mathbf 1\pi)^{-1}-\mathbf 1\pi$, ta có đồng nhất thức chính xác $\tilde\pi-\pi=\tilde\pi EZ=\nu Z$ (kinh điển; nguồn ở mục 5). Vì $\nu(\Omega)=0$:
$$\mathrm{TV}(\tilde\pi,\pi)\le\tau_1(Z)\,\tfrac12\lVert\nu\rVert_1\le\tau_1(Z)\,\bar\varepsilon,\qquad \tau_1(Z)\le\sum_t\delta(P^t)=\tau(P).$$
Như vậy Định lý 2 là một trường hợp yếu hơn của chặn "số điều kiện" kiểu Seneta, vốn đã dùng $\nu=\tilde\pi E$, tức đã lấy trung bình theo $\tilde\pi$.
**Kiểm số.** Trên 400 chuỗi ngẫu nhiên: đồng nhất thức đúng tới 1e-9, và luôn có $\mathrm{TV}\le\tau_1(Z)\lVert\nu\rVert_{TV}\le\tau\bar\varepsilon$. Chặn $\tau\bar\varepsilon$ đạt dấu bằng (sai 3e-14) trên chuỗi 2 trạng thái, nên nó chặt trong lớp tổng quát. Câu "cho lại đúng chặn Mitrophanov" là **đúng**: (3.8) của Mitrophanov dùng $\lVert q\rVert=\lvert q\rvert(S)=2\,\mathrm{TV}$, $\lVert E\rVert=\sup_x\lVert E_x\rVert=2\varepsilon_{\max}$, điều kiện đặt trên chuỗi *không nhiễu*, và $\hat n=\lceil\log_\rho C^{-1}\rceil$, khớp với (i).
**Sửa.**
- Trích Schweitzer/Meyer (đồng nhất thức), Seneta và Cho & Meyer (chặn $\tau_1(Z)$).
- Trình bày chặn chính như một bổ đề đã biết. Đóng góp thật nằm ở Bước 0 (tồn tại và duy nhất khi $q$ không tương thích, nhờ cùng đồ thị chuyển), ở các ước lượng $\varepsilon$ theo TFM trong (ii)–(iii), và ở 4(b).
- Có thể báo $\tau_1(Z)$, chặt hơn và tính được từ cùng ma trận, trong RQ1 E3/E4.

### F4 — Major — Văn xuôi §7 nói quá điều đã chứng minh, và tự mâu thuẫn
**Vị trí và lỗi.**
- (a) "Đính chính": "*mục tiêu của bộ ngây thơ vẫn là hàm affine tăng của posterior thật, nên ở mức quần thể nó **nhất quán** về xếp hạng*". Mệnh đề 5 chỉ nói về *mục tiêu* $\tilde\eta$, và chính bài viết "*nó không khẳng định TFM hội tụ tới $\tilde\eta$*". "Nhất quán" là tính chất của bộ ước lượng. Với LLP, các cặp $(x_i,\tilde w_i)$ trong một túi phụ thuộc nhau (tổng cố định), trong khi PPD của TFM giả định i.i.d.
- (b) "Hệ quả cho bài" mục 3: "*Gibbs **nhất quán** khi biết cơ chế quan sát (Định lý 1)*". Định lý 1 chỉ nói chuỗi chính xác (phân phối dừng là posterior dưới prior của TFM, với (A1)), không nói về tính nhất quán thống kê đối với $\eta$ thật. §3 tự nói "*không phải đảm bảo trên phân phối thật*". Phép đối chiếu loại 3 ("ngây thơ không nhất quán, Gibbs nhất quán") vì vậy so hai khái niệm khác nhau.
- (c) "*Trong các bối cảnh của giao thức, chỉ PU-SAR có bộ ngây thơ không nhất quán*". Giao thức có RQ5: PLL thật (Lost, MSRCv2, BirdSong) và LLP-Bench (EXPERIMENT_PROTOCOL dòng 19 và 87). Ngay câu sau, PROOFS viết: "*Nếu túi được tạo theo độ gần của feature (như LLP-Bench), (B1) sai*". Với PLL thật, tập ứng viên phụ thuộc mẫu nên (B1) chưa được xác lập. Ngoài ra $\ell_i(k)=\mathbb 1[k\in S_i]$ của Gibbs cũng cần "$p(S\mid y=k,x)$ như nhau trên $S$", điều chưa kiểm cho dữ liệu thật.
- (d) Mục 1 đưa SSL-MCAR vào "*không còn thông tin để Gibbs lấy lại*", trong khi mục SSL ở phần áp dụng Mệnh đề 6 viết "*phần không nhãn có thể giúp*" dưới prior SCM, tức prior không tách được.

**Sửa.**
- (a) Viết "mục tiêu quần thể của bộ học từng điểm là biến đổi tăng ngặt của $\eta$".
- (b) Viết "chuỗi có phân phối dừng là posterior dưới prior của TFM (Định lý 1, dưới A1)".
- (c) Giới hạn thành "trong các bối cảnh *tổng hợp* (RQ2–RQ4)", và thêm câu về RQ5.
- (d) Tách SSL ra khỏi loại 1, hoặc ghi điều kiện "prior tách được".

### F5 — Minor — "$P_{sw}$ nói chung không khả nghịch" sai
**Vị trí.** §2(a): "*$P_{sw}$ nói chung không khả nghịch; kết quả dưới đây không cần khả nghịch.*"
**Lập luận.** Mỗi $P_i$ tự liên hợp trong $L^2(\pi)$, nên $(P_{\sigma(1)}\cdots P_{\sigma(n)})^*=P_{\sigma(n)}\cdots P_{\sigma(1)}$. Tập các hoán vị đóng dưới phép đảo thứ tự, nên $P_{sw}^*=P_{sw}$. Kiểm số (`chk_psw_reversible.py`): sai số cân bằng chi tiết 3,5e-17. Quét hệ thống cố định thì không khả nghịch (2,2e-2), và $\tilde P_{sw}$ với $q$ không tương thích cũng không (5,6e-2).
**Sửa.** "Dưới (A1), $P_{sw}$ khả nghịch theo $\pi$; với $q$ không tương thích thì nói chung không. Định lý 2 không cần tính khả nghịch."

### F6 — Minor — Nhận xét 4.1: một mệnh đề phụ sai, và số liệu lệch RQ1
**Vị trí.** "*$\delta(P^t)=1$ ít nhất cho tới khi mọi mẫu được chạm tới, nên $\tau\gtrsim|F|$*". "*lỏng khoảng 2–15 lần; dạng theo $\varepsilon_{TV}$ lỏng thêm khoảng 10 lần*".
**Phản ví dụ.** $|F|=2$, nhị phân: từ $(0,0)$ và $(1,1)$, sau 1 bước cả hai đều tới được $(1,0)$, nên $\delta(P)<1$. Tổng quát, $\delta(P^t)=1$ chỉ khi $t<|F|/2$. Kiểm số với $|F|=2..6$: $t$ đầu tiên có $\delta<1$ là 1, 2, 2, 3, 3. Kết luận về bậc, $\tau\ge\lceil|F|/2\rceil$, vẫn đúng.
Về số liệu, RQ1 cho độ lỏng 2,2–12×. Theo bảng E3, dạng theo $\varepsilon$ lỏng thêm 15–44× so với $\tau\bar\varepsilon$ (RQ1 tự ghi "10–40").
**Sửa.** Thay bằng "$\delta(P^t)=1$ với $t<|F|/2$ khi mỗi mẫu tự do có ít nhất 2 giá trị", và cập nhật số cho khớp RQ1.

### F7 — Minor — Hệ quả 1.1 viết vòng; câu suy (A2) từ softmax chưa đủ
**Vị trí.** "*mỗi thừa số dương theo Bổ đề 1 và (A2)*" và (A2): "*Dưới (A1), điều này suy ra từ tính dương của đầu ra softmax*".
**Lỗi.** Bổ đề 1 cần $p(y_{-i}\mid X)>0$. Áp nó dọc đường Brook cần $p(y^{k-1}\mid X)>0$, đúng là điều đang chứng minh. Cách viết đúng là quy nạp theo $k$: $p(y^{k-1}\mid X)>0$ suy ra ngữ cảnh $D^{k-1}_{-s_k}$ có xác suất dương, nên (A2) áp được và thừa số thứ $k$ dương. Ngoài ra, đường Brook đi qua các nhãn *ngoài* $\mathrm{supp}(\pi)$ mà thuật toán single-site không bao giờ truy vấn. (A1) chỉ phủ "ngữ cảnh mà thuật toán dùng", nên câu "suy ra từ softmax" không phủ được các điểm đó.
**Sửa.** Viết quy nạp ra. Giữ (A2) như một giả định về prior và bỏ hoặc thu hẹp câu về softmax.

### F8 — Minor — (A0) thiếu điều kiện để $\ell$ đúng; giao thức SSL vi phạm
**Vị trí.** §0: "*SSL: $\ell_i\equiv1$ nếu không nhãn*"; "*LLP: … $\ell(y)=\prod_g\mathbb 1[\cdot]$*".
**Lỗi.** $\ell_i\equiv1$ cần cơ chế chọn mẫu có nhãn MAR với tỉ lệ không phụ thuộc $y$. LLP cần việc tạo túi không phụ thuộc $y$ khi biết $X$. Cả hai điều kiện đều không được nêu. `kaggle_g4_rq4_pilot.py` (dòng 120–126) gán nhãn đúng $\max(2,\mathrm{round}(\text{level}\cdot n_k))$ mẫu mỗi lớp. Khi đó $p(W\mid y)$ thật chứa thừa số toàn cục phụ thuộc $n_k(y)$ (kiểu siêu bội), và tỉ lệ khác nhau giữa các lớp. Ví dụ ở level 0,05: $n_1=30$ cho 2/30 = 0,067, còn $n_0=270$ cho 14/270 = 0,052. Bảng §7 có nhắc chuyện làm tròn cho bộ ngây thơ, nhưng không nói rằng mục tiêu $\pi$ của Gibbs cũng lệch. Ảnh hưởng nhỏ, nhưng lý thuyết và thiết lập thực nghiệm mâu thuẫn nhau.
**Sửa.** Thêm vào (A0): "chọn mẫu có nhãn (SSL) và tạo túi (LLP) độc lập với $y$ khi biết $X$". Hoặc đổi bộ sinh SSL sang Bernoulli(level), hoặc ghi rõ đây là sai lệch nhỏ của mô hình.

### F9 — Minor — Định lý 1(c)(3) áp "theo 2(a)" cho cả LLP; công thức Rao–Blackwell khi $m\ge2$
**Vị trí.** "*Chuỗi mới chính xác với mục tiêu $p(y,y_*\mid X,x_*,W)$ theo 2(a)*".
**Lỗi.** Với LLP, chuỗi gồm bước hoán đổi (2(b)) cộng Gibbs một mẫu cho $x_*$. Kết quả vẫn đúng vì mỗi thành phần giữ $\pi$ và tích bất khả quy, nhưng câu này phải được nói ra. Với $m\ge2$, $\pi_*(y_{*k}\mid\cdot)=q(y_{*k}\mid x_{*k},(X,y)\cup(X_{*,-k},y_{*,-k}))$, không phải $q(\cdot\mid x_*,(X,y^{(s)}))$. Ngoài ra, mã ghi Rao–Blackwell tại thời điểm cập nhật $x_*$ trong vòng quét chứ không tại $y^{(s)}$ cuối vòng. Cách đó vẫn đúng (mọi trạng thái trung gian đều có phân phối dừng $\pi$), nhưng nên ghi chú.
**Sửa.** Một câu cho mỗi điểm trên.

### F10 — Minor — "chỉ đúng khi" và điều kiện tách được dùng không nhất quán
**Vị trí.**
- §2(d): "*Điều này chỉ đúng khi feature không mang thông tin về $\phi$*".
- Hệ quả 3.1: "*khi $p(x\mid\phi)$ không phụ thuộc $\phi$*".
- Mệnh đề 6, mục SSL: "*không phụ thuộc $\theta$ khi prior tách được (Định lý 1(c))*".
- §2(d) "Hệ quả cho bài": "*sai lệch $O(\cdot)$*".

**Lỗi.**
- Điều kiện cần chưa được chứng minh, và sai. Phản ví dụ: $W$ xác định hoàn toàn $y$, khi đó $\pi$ là khối lượng điểm và $\bar p=p(y_*\mid x_*,X,W)$ với mọi prior.
- Điều kiện tách được *cho phép* $x$ mang thông tin về $\phi_x$.
- Trong khung Fisher (tần suất), điều kiện cần là "$p(x)$ không phụ thuộc $\theta$", và điều này đã được giả định ở đầu phần áp dụng. Nhắc "prior tách được" ở đây là trộn hai khung.
- "$O(\cdot)$" không có nội dung (RQ1 E2: "chưa đo").

**Sửa.** Thống nhất thành "điều kiện đủ: prior tách được"; bỏ "chỉ"; bỏ "$O(\cdot)$" hoặc thay bằng chặn TV ở 1(c)(2).

### F11 — Minor — Mệnh đề 4(b) và (ii): nhân trong phân tích khác nhân trong cài đặt; chưa phủ cập nhật tham số phiền nhiễu
**Lỗi.**
- 4(b) phân tích nhân "chọn đều một khối". Cài đặt quét lần lượt $K$ khối của một phân hoạch ngẫu nhiên, nên $\varepsilon$ mỗi vòng có thêm hệ số $K$, như $|F|$ ở (ii), và $\tau$ phải tính theo vòng.
- Việc đồng nhất $A_k$ với PPD tổng quát cần một Bổ đề 1 mở rộng cho feature không nhãn. Bổ đề này dễ nhưng chưa được viết.
- Định lý 2 cho chuỗi có cập nhật $\theta_W$ (Mệnh đề 1.4) cần $\varepsilon(\bar P)\le\sup_\theta\varepsilon(K_\theta)$ (do tính lồi). Điều này chưa được nêu.

**Sửa.** Ba câu tương ứng.

### F12 — Minor — Quan sát 4(c): "sai số dự đoán giảm như $O(1/n)$" không được dữ liệu ủng hộ
**Vị trí.** "*(2) sai số dự đoán giảm như $O(1/n)$*".
**Bằng chứng.** Hồi quy log–log trên số RQ1 E5c ($n$ = 50, 200, 1000):
- PU: $K=20$ cho −0,55, $K=10$ cho −0,77, $K=5$ cho −0,79.
- SSL-asym: $K=20$ cho −0,25 và **không đơn điệu** (0,0004 → 0,0007 → 0,0002); $K=10$ cho −0,45.

**Sửa.** "giảm theo $n$ (độ dốc log–log −0,5 đến −0,8 ở PU, chỉ 3 giá trị $n$)".

### F13 — Minor — Định lý 2: thứ tự định nghĩa và trường hợp $\infty\cdot0$
**Vị trí.** $\bar\varepsilon=\sum_y\tilde\pi(y)\ldots$ được định nghĩa *trước* khi Bước 0 chứng minh $\tilde\pi$ duy nhất. "*Nếu $\tau(P)=\infty$ thì bất đẳng thức hiển nhiên đúng*".
**Lỗi.** Nếu $\bar\varepsilon=0$ và $\tau=\infty$ thì vế phải là $\infty\cdot0$.
**Sửa.** Định nghĩa $\bar\varepsilon$ sau Bước 0. Thêm: nếu $\bar\varepsilon=0$ thì $\tilde\pi P=\tilde\pi$, nên $\tilde\pi=\pi$.

### F14 — Minor — Nhận xét 4.2: "đúng bậc" là suy diễn chéo bối cảnh
**Vị trí.** "*Kết hợp với chặn dưới $q\varepsilon/4$ của Liu et al., sự phụ thuộc tuyến tính … là đúng bậc*".
**Lỗi.** $\varepsilon$ của Liu et al. là độ lệch của xác suất dự đoán có điều kiện trên một tập $C$ với $\nu_1(C)>q$ (dữ liệu liên tục; tôi đã đối chiếu bản arXiv). Nó không phải $\bar\varepsilon$, là TV theo hàng của nhân lấy trung bình theo $\tilde\pi$, và bài không chuyển đổi giữa hai đại lượng.
**Sửa.** Dùng lập luận tự đủ: chặn $\tau\bar\varepsilon$ đạt dấu bằng trên chuỗi 2 trạng thái (kiểm số của tôi), nên phụ thuộc tuyến tính không cải thiện được trong lớp tổng quát.

### F15 — Minor (ký hiệu) — Một ký hiệu mang nhiều nghĩa, có khi trong cùng một mục
- $\pi$: phân phối mục tiêu; tỉ lệ lớp ở bảng §7 và Quan sát 6(b).
- $p$: PPD hoặc mật độ; $P(y=1)$ ở 5(c).
- $q$: TFM; xác suất ứng viên PLL; khối lượng tập trong Nhận xét 4.2.
- $C$: số lớp; hằng số Mitrophanov ở 1(a)(2) và 2(i).
- $B$: khối; túi $B_g$; biến Binomial (5(a)); các phân phối $B_k$ (4(b)).
- $c$: tần suất SCAR; số đếm túi $c_B$.
- $\eta$: tham số propensity (Nhận xét 1.3); $\eta(x)$ posterior (§7).
- $\rho$: tốc độ hội tụ; tỉ lệ nhiễu $\rho_{01}$.
- $a$: tham số Beta; hệ số $a$ của PLL; hằng số ở Bước 2.

Riêng §7 dùng cùng lúc $\pi$, $p$, $q$, $B$ với nghĩa thứ hai. **Sửa.** Đổi tên (ví dụ tỉ lệ lớp $\varpi$, xác suất ứng viên $r_{\rm PLL}$, hằng số $C_{\rm erg}$).

### F16 — Minor — G4-1 (PU-SAR): khoảng trống đã được ghi nhận dễ đóng hơn bài nói; vấn đề thật là định danh
**Vị trí.** §2(c): "*chỉ nêu, chưa kiểm tra chi tiết điều kiện (việc mở G4-1)*".
**Nhận xét.** Chuỗi trên $[C]^n\times\mathbb R^d$ gồm Gibbs cho $y$ và random-walk MH Gauss cho $\eta$, với prior Gauss và mật độ đề xuất dương. Chuỗi này $\pi$-bất khả quy và không tuần hoàn, nên hội tụ TV với $\pi$-hầu hết điểm đầu theo Tierney (1994); số hiệu định lý cần kiểm. Vấn đề nặng hơn, chưa được ghi trong PROOFS: ở PU-SAR, $\eta$ định danh yếu từ $(x,s)$, nên posterior của $\eta$ do prior và TFM quyết định. Điều đó ảnh hưởng tốc độ trộn và ý nghĩa của "mục tiêu Bayes", trong khi PU-SAR thuộc "chế độ được khẳng định" (METHOD_SPEC §6c).
**Sửa.** Đóng G4-1 bằng trích dẫn kèm điều kiện cụ thể; thêm một câu về định danh.

## 4. Các kiểm chứng số đã chạy
Mọi script ở `/tmp/claude-0/-home-claude/57242191-601e-571f-bad3-e3849bd239dc/scratchpad/review/`, chạy bằng `python -I`, không ghi gì vào repo. Đây là kiểm chứng, không thay chứng minh.

1. **Bộ kiểm thử có sẵn** (`pytest tests`): 63 passed, 2 skipped (4 phút 9 giây), chạy trên cây làm việc hiện tại. Cây này có thêm tệp test chưa commit của phiên khác; các tệp được review trùng với c08e02b.
2. **`g6_t7_naive_theory.py`** chạy lại (đầy đủ): kết quả **trùng từng chữ số** với `pilot/g6_t7_naive_theory.txt`. Mọi số trong bảng Quan sát 6(b) và bảng hệ số §7 khớp.
3. **`chk_thm2.py`**, 400 chuỗi ngẫu nhiên 2–6 trạng thái:
   - $\max\mathrm{TV}/(\tau\bar\varepsilon)=1{,}0000$, không vượt 1; $\max\mathrm{TV}/(\tau\varepsilon_{\max})=0{,}996$.
   - Đồng nhất thức $\tilde\pi-\pi=\nu Z$ đúng tới 1e-9; luôn có $\tau_1(Z)\le\tau$ và $\mathrm{TV}\le\tau_1(Z)\lVert\nu\rVert_{TV}\le\tau\bar\varepsilon$.
   - (i): 300 chuỗi × 4 giá trị $\rho$, 0 vi phạm.
   - Co Dobrushin cho độ đo khối lượng 0: 2000 ca, 0 vi phạm.
4. **`chk_thm2_sharp.py`**: trên chuỗi 2 trạng thái, $\max(\mathrm{TV}/(\tau\bar\varepsilon))-1=2{,}9\times10^{-14}$, tức chặn đạt dấu bằng.
5. **`chk_thm2_parts.py`**:
   - Bổ đề tái trọng số: tỉ số lớn nhất quan sát được là 1,0, so với hằng 2 trong bài.
   - (ii): 150 ca, $\{0,1\}^4$, $q$ không tương thích. Quét ngẫu nhiên đạt tối đa 0,31 lần chặn; $P_{sw}$ đạt tối đa 0,36 lần chặn $2|F|\kappa\varepsilon_{TV}$. $\kappa\le1/(1-c)$ đúng.
   - (iii): 300 ca, đạt tối đa 0,28 lần chặn $4\varepsilon_\infty$.
   - Nhận xét 4.1: phản ví dụ, $t$ nhỏ nhất có $\delta<1$ là 1, 2, 2, 3, 3 với $|F|=2..6$.
   - Lập luận lai của 4(b): 3000 ca, 0 vi phạm.
6. **`chk_thm1.py`**:
   - Bổ đề 1, hỗn hợp 3 lớp, $n=5$, mọi $3^5$ cách gán: 1,2e-15.
   - Định lý 1(b), 3 lớp, 2 túi, 18 trạng thái: TV 6,9e-16, cân bằng chi tiết 1,2e-16.
   - 1(c)(2): tỉ số $|\bar p-\text{Bayes}|/\mathrm{TV}\le0{,}839$ trên 30 prior.
   - Phản ví dụ Hệ quả 3.1: tối đa 0,123, trung vị 0,020.
   - Mệnh đề 1.4 (PU-SCAR, $c\sim\mathrm{Beta}(2,3)$, cầu phương Gauss–Legendre 60 nút): $\lVert\pi\bar P-\pi\rVert_1=6{,}2\times10^{-15}$, $\min\bar P=0{,}046$.
7. **`chk_psw_reversible.py`**: sai số cân bằng chi tiết của $P_{sw}$ là 3,5e-17; quét hệ thống cố định 2,2e-2; $\tilde P_{sw}$ không tương thích 5,6e-2.
8. **`chk_sec7.py`** và **`chk_auc_tight.py`**:
   - $a$ và $\lambda$ của PLL so với liệt kê chính xác ($C\in\{2,3,4,7,10\}$, $q\in\{0{,}05;0{,}3;0{,}6;0{,}95\}$): 1,2e-15. Các giá trị 0,511 / 0,208 / 0,111 khớp bảng.
   - LLP quần thể hữu hạn, liệt kê mọi hoán vị ($n\le9$): khớp công thức chính xác.
   - AUC, 20 000 ca: 0 vi phạm, tỉ số lớn nhất 0,663. Ví dụ hai nguyên tử cho tỉ số tiến tới $\max(P_1,P_2)$ (đạt 0,999), nên hằng số chặt.
   - Mệnh đề 6: gộp hai kết cục có tỉ số không phụ thuộc $\theta$ cho $I$ bằng nhau (0,221816), đúng điều kiện dấu bằng; nhân nhiễu cho 0,053.
   - $\log_{10}\kappa_B$ của bag-block: 15,8 / 10,3 / 6,6.
   - Độ dốc trong Quan sát 4(c): xem F12.
9. **`chk_bagblock.py`**: bag-block xấp xỉ trên bài LLP 3×4: TV từ 0,157 đến 0,639, sai số dự đoán từ 0,041 đến 0,122 (F1).

## 5. Trích dẫn cần đối chiếu với nguồn gốc
- **Mitrophanov (2005), J. Appl. Probab. 42:1003–1014, Hệ quả 3.1, công thức (3.8).** Tôi đã đọc bản PDF của Cambridge qua công cụ tóm tắt: chuẩn $\lvert q\rvert(S)$, $\lVert E\rVert=\sup_x\lVert E_x\rVert$, ergodic đều đặt trên chuỗi không nhiễu, $\hat n=\lceil\log_\rho C^{-1}\rceil$. Cả ba khớp (i). Tác giả vẫn nên tự mở PDF để đóng G4-3.
- **Liu, Gelman, Hill, Su & Kropko (2014), Biometrika 101(1):155–173.** Bản arXiv 1012.2902 có Định lý 1 (tương thích, TV → 0 theo xác suất, cần A1–A2 và Mệnh đề 2), Định lý 2 ($>q\varepsilon/4$, với $\varepsilon\in(0,1/2)$ và $\nu_1(C)>q$), Định lý 3 (mô hình "valid"). Số định lý trong bản Biometrika: cần kiểm.
- **Chặn AUC với $\lambda=1$ (G6-T7a).** Agarwal (2014), JMLR 15, **Hệ quả 12**: $\mathrm{regret}^{\rm rank}[\hat\eta]\le\frac{1}{p(1-p)}\mathbb E|\hat\eta-\eta|$, được ghi là theo Clémençon & Robbiano (2011), dựa trên một đồng nhất thức của Clémençon, Lugosi & Vayatis (2008) (Định lý 11 trong Agarwal). Thông tin này lấy từ bản JMLR qua công cụ tóm tắt; cần mở PDF xác nhận trước khi đóng G6-T7a.
- **Tiền lệ cho Định lý 2 (F3), mọi mục đều cần kiểm:**
  - Schweitzer (1968, J. Appl. Probab.) và/hoặc Meyer (1980, SIAM J. Algebraic Discrete Methods) cho đồng nhất thức $\tilde\pi-\pi=\tilde\pi EZ$.
  - Seneta (1988), "Perturbation of the stationary distribution measured by ergodicity coefficients", Adv. Appl. Probab. 20.
  - Seneta (1993), Stat. Probab. Lett.
  - Cho & Meyer (2001), "Comparison of perturbation bounds for the stationary distribution of a Markov chain", Linear Algebra Appl. 335.
  - Về sai số nhân lấy trung bình theo $\tilde\pi$, có thể thêm Rudolf & Schweizer (2018, Bernoulli) và Johndrow & Mattingly (arXiv 2017).
- **Seneta (2006), "§3.1"** cho hệ số Dobrushin: cần kiểm số mục.
- **Tierney (1994)**: định lý hoặc hệ quả nào cho MH-trong-Gibbs trên không gian tổng quát (để đóng G4-1): cần kiểm.
- **Müller et al. (2022)**: mệnh đề cụ thể nói loss của PFN nhắm tới PPD: cần kiểm số hiệu. Thay trích dẫn "*khóa học, Chương 7*" bằng nguồn trích được.
- **Elkan & Noto (2008), "Lemma 1"** (METHOD_SPEC): cần kiểm.
- **Besag (1974) / Brook (1964)** cho bổ đề Brook: thông tin thư mục đúng; trang hoặc mục: cần kiểm.

## 6. Đối chiếu sau khi đọc lướt SELF_REVIEW_MATH.md
Đọc sau khi đã viết xong các mục 1–5. Những điểm mà tự rà soát đã nêu hoặc đã "xác nhận" nhưng PROOFS hiện tại vẫn sai:
- **Nhận xét 4.1.** Tự rà soát (dòng 86) ghi: "*kiểm tra $\delta(P^t)=1$ với $t<|F|$ bằng lập luận hai trạng thái khác nhau ở mọi mẫu tự do*". Đây chính là mệnh đề sai ở F6. Lập luận đó chỉ cho $t<|F|/2$.
- **MINOR-2 (đã "sửa").** Đoạn thêm vào §2(a) mang theo câu sai "*$P_{sw}$ nói chung không khả nghịch*" (F5).
- **MINOR-4 / bảng "Hệ quả 1.1 … quy nạp dọc đường Brook".** Văn bản PROOFS chưa viết quy nạp ra. **MINOR-3**: câu "dưới (A1) suy ra từ softmax" vẫn không phủ các ngữ cảnh ngoài $\mathrm{supp}(\pi)$ trên đường Brook (F7).
- **MINOR-1 (đã sửa).** Trường hợp $\tau=\infty$, $\bar\varepsilon=0$ vẫn chưa xử lý (F13).
- **COSMETIC-1.** Câu "Định lý 2 làm mịn thành trung bình $\bar\varepsilon$" được giữ như đóng góp, chưa đối chiếu với chặn kiểu Seneta (F3).
- **MAJOR-1.** Cách sửa đề xuất ("trùng Bayes khi feature không mang thông tin") vẫn để lại câu "*chỉ đúng khi*" ở §2(d), và Hệ quả 3.1 vẫn dùng điều kiện hẹp hơn điều kiện tách được (F10). Tự rà soát cũng không phát hiện Hệ quả 3.1 sai khi có $m\ge2$ điểm test (F2).
- **Bảng tự rà soát "lỏng 2–15 lần".** Con số này lặp lại trong PROOFS và lệch RQ1 (2,2–12×) (F6).
