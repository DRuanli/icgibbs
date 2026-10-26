# Tự rà soát phản biện phần toán — PROOFS.md (G4)

> Người rà soát: Claude, theo quy trình `dm-paper-math-review` (đối kháng, mặc định "chưa kiểm chứng"). **Đây là tự rà soát của chính người viết, không thay thế kiểm duyệt độc lập** của người hướng dẫn hoặc đồng tác giả ở GATE_REVIEW_G4. Cột "Trạng thái" ghi việc sửa đã làm trong PROOFS.md sau lượt rà soát này.

## Kết luận một dòng
**CÓ VẤN ĐỀ LỚN (MAJOR), KHÔNG CÓ VẤN ĐỀ CHẶN (BLOCKING)** — vấn đề nặng nhất: các tài liệu đã đóng băng (RESEARCH_PLAN §4, CLAIM_EVIDENCE_MATRIX #3, #10) vẫn khẳng định bộ dự đoán Rao–Blackwell hội tụ về $p(y_*\mid x_*,X,W)$ và Bayes-optimal, trong khi PROOFS.md đã chứng minh điều đó chỉ đúng ở chế độ chuyển nạp hoặc khi feature không mang thông tin về $\phi$.

## Phạm vi
Đã đọc toàn bộ PROOFS.md (§0–§6), METHOD_SPEC.md (để đối chiếu thuật toán được phân tích với thuật toán được cài), `icgibbs/sampler.py`, `icgibbs/exact.py`, và các CSV E1–E3, E5b. Chưa có E4 (PFN học được), E5a2, E5c lúc rà soát. Không kiểm được: bản in Biometrika của Liu et al. (chỉ có bản arXiv), bản PDF đầy đủ của Mitrophanov (2005).

## Sổ phát hiện

### [MAJOR-1] Khẳng định Bayes chính xác trong tài liệu đã đóng băng — RESEARCH_PLAN §4 dòng Định lý 1; CLAIM_EVIDENCE_MATRIX #3, #10
**Như đang viết.** "ước lượng Rao–Blackwell hội tụ về $p(y_*\mid x_*,X,W)$"; "cho ra posterior predictive Bayes dưới nhãn yếu"; "Bộ dự đoán là Bayes-optimal hữu hạn mẫu theo prior".
**Vì sao sai.** $p(y_*\mid x_*,X,W)=\sum_y p(y\mid x_*,X,W)\,p(y_*\mid x_*,X,y)$, còn chuỗi lấy mẫu từ $p(y\mid X,W)$. Hai phân phối này khác nhau khi $x_*$ mang thông tin về $\phi$.
**Bằng chứng.** Phản ví dụ số E2: prior hỗn hợp 3 thành phần, $n=8$, SSL: $\bar p=0{,}634$ còn Bayes chính xác $=0{,}699$ (ô 1); PU-SCAR: chênh 0,110. Prior ô (feature không mang thông tin): chênh $<10^{-15}$.
**Phạm vi ảnh hưởng.** Khẳng định #3 (abstract), #10 (Method), Hệ quả 3.1.
**Sửa tối thiểu.** Ghi thay đổi vào RESEARCH_PLAN §9; sửa #3, #10 thành: "ở chế độ chuyển nạp, hội tụ về đúng posterior predictive Bayes; ở chế độ quy nạp, hội tụ về $\bar p$, trùng Bayes khi feature không mang thông tin về cơ chế sinh nhãn".
**Trạng thái.** Đã ghi §9 và sửa ma trận (xem GATE_REVIEW_G4).

### [MAJOR-2] Mệnh đề 4(b) phân tích cách chia khối khác với cách được cài — PROOFS §5 vs METHOD_SPEC D1
**Như đang viết.** Khối $B$ gồm $m$ mẫu tự do; hai ngữ cảnh trong quy tắc chuỗi khác nhau $m-1$ điểm, nên mỗi thừa số lệch $\le(m-1)\beta$.
**Vì sao sai.** D1 chia khối trên **mọi** mẫu: mẫu cố định (có nhãn) rơi vào $B$ bị loại khỏi ngữ cảnh nhưng xác suất có điều kiện chính xác $\pi(y_{B\cap F}\mid y_{-(B\cap F)})$ vẫn điều kiện hoá trên chúng. Số điểm khác nhau giữa hai ngữ cảnh là $m-1$ với $m=|B|$ (gồm cả mẫu cố định), và số thừa số là $m_f=|B\cap F|$. Ngoài ra (S) dùng "PPD với feature không nhãn trong ngữ cảnh", đại lượng chưa được định nghĩa ở §0.
**Sửa tối thiểu.** Định nghĩa PPD tổng quát $p(y_*\mid x_*,D,X_u)$ ở §0; phát biểu lại chặn thành $m_f(m-1)\beta$.
**Trạng thái.** Đã sửa trong PROOFS §0 và §5.

### [MINOR-1] Bước 2 của Định lý 2 dùng "P ergodic đều" mà định lý không giả định — PROOFS §4
**Phản ví dụ.** MH hoán đổi không lazy, một túi 2 mẫu, số đếm (1,1), hai mẫu ở hai ô khác nhau của prior ô: $\pi(10)=\pi(01)=\tfrac12$, $R=1$, nhân là $\begin{psmallmatrix}0&1\\1&0\end{psmallmatrix}$ — tuần hoàn, $\delta(P^t)=1$ với mọi $t$.
**Hậu quả.** Bất đẳng thức vẫn đúng (vì $\tau=\infty$), nhưng lập luận "ΔP^T → 0" sai trong trường hợp này.
**Sửa.** Tách trường hợp: nếu $\tau(P)=\infty$ thì không có gì để chứng minh; nếu $\tau<\infty$ thì $\delta(P^t)\to0$, chuỗi hội tụ tuyệt đối.
**Trạng thái.** Đã sửa.

### [MINOR-2] Phân tích nhân quét ngẫu nhiên, cài đặt dùng quét theo hoán vị — PROOFS §2(a), §4(ii) vs `sampler.py`
Định lý 1(a) chỉ nêu tính bất biến của quét theo hoán vị; tính bất khả quy/không tuần hoàn của nhân một vòng quét chưa viết. Hằng số ở 4(ii) là cho một bước quét ngẫu nhiên; với nhân một vòng quét, $\varepsilon_{\max}\le\min\{1,2|F|\kappa\varepsilon_{TV}\}$ còn $\tau$ tính theo vòng quét.
**Trạng thái.** Đã bổ sung một đoạn ở §2(a) và một dòng ở §4(ii).

### [MINOR-3] (A2) phát biểu cho $p$ nhưng biện minh bằng softmax của $q$ — PROOFS §0
**Sửa.** (A2) là giả định về prior; dưới (A1) nó suy ra từ tính dương của softmax. **Trạng thái.** Đã sửa.

### [MINOR-4] Bổ đề 1 "với mọi $y$" trong khi xác suất có điều kiện chỉ xác định khi $p(y_{-i}\mid X)>0$ — PROOFS §1
Hệ quả 1.1 dùng Bổ đề 1 trước khi biết mọi trạng thái có xác suất dương; lập luận quy nạp vẫn đúng nếu Bổ đề 1 phát biểu "khi $p(y_{-i}\mid X)>0$". **Trạng thái.** Đã sửa phát biểu.

### [MINOR-5] Điều kiện dấu bằng ở Định lý 1(c.2) mạnh hơn cần thiết
Đủ khi $\phi=(\phi_x,\phi_{y\mid x})$ độc lập tiên nghiệm, $p(x\mid\phi)=p(x\mid\phi_x)$, $p(y\mid x,\phi)=p(y\mid x,\phi_{y\mid x})$. **Trạng thái.** Đã thêm.

### [MINOR-6] Mệnh đề 3 chưa nêu tính khả tích của quy tắc chấm điểm
Với log score, rủi ro của $f^\star$ $\le\log C<\infty$ nên cực tiểu duy nhất có nghĩa. **Trạng thái.** Đã thêm.

### [MINOR-7] Chế độ chuyển nạp chỉ chính xác với sơ đồ một mẫu — METHOD_SPEC §2 vs PROOFS §2(d)
Với sơ đồ khối, xác suất có điều kiện được ghi tại điểm test là xấp xỉ (ngữ cảnh bỏ cả khối). **Trạng thái.** Đã ghi rõ trong METHOD_SPEC.

### [COSMETIC-1] "Đây đúng là hằng số trong Mitrophanov" — PROOFS §4(i)
Nên nói: dạng (i) cùng hằng số với Mitrophanov (2005, Hệ quả 3.1), vốn dùng $\varepsilon_{\max}$; Định lý 2 làm mịn thành trung bình $\bar\varepsilon$ theo $\tilde\pi$. **Trạng thái.** Đã sửa.

## Kết luận theo từng kết quả

| Kết quả | Mức chứng minh | Kết luận | Ghi chú |
|---|---|---|---|
| Bổ đề 1 | L4 | PROVEN (sau MINOR-4) | kiểm số mọi $2^7$ trạng thái, hai prior |
| Hệ quả 1.1 | L4 | PROVEN | quy nạp dọc đường Brook |
| Định lý 1(a) | L4 | PROVEN | thử: $F$ rỗng, mẫu cố định, PLL 3 lớp; E1 TV $<10^{-13}$ |
| Định lý 1(b) | L4 | PROVEN (bản lazy cho không tuần hoàn) | thử phản ví dụ tuần hoàn (MINOR-1); dựng tường minh dãy hoán đổi cho bất khả quy |
| Mệnh đề 1.4 | L4 | PROVEN cho cập nhật liên hợp | PU-SAR: INSUFFICIENT INFORMATION (việc mở G4-1) |
| Định lý 1(c) | L4 | PROVEN | sau khi đã sửa phát biểu so với G3; E2 |
| Mệnh đề 3 | L4 | PROVEN (sau MINOR-6) | có điều kiện theo prior |
| Định lý 2 | L4 | PROVEN (sau MINOR-1) | E3: chặn đúng ở 30/30 cấu hình, lỏng 2–15 lần |
| Mệnh đề 4(a) | L4 | PROVEN | E1 TV $<10^{-13}$, MC |
| Mệnh đề 4(b) | L3 | PROVEN UNDER ADDITIONAL ASSUMPTIONS: (S) với PPD tổng quát | chặn đủ, không chặt; không cho thấy sai số → 0 với K cố định |

## Bảng giả định
| Giả định | Nêu ở | Dùng ở | Cần? | Không có thì |
|---|---|---|---|---|
| (A0) W ⊥ (x*,y*) \| (X,y) | §0 | Định lý 1(c), Mệnh đề 3 | Có | dự đoán phụ thuộc W qua đường khác |
| (A1) q = p | §0 | Định lý 1, Mệnh đề 4(a) | Có (Định lý 2 thay thế) | chỉ còn Định lý 2 |
| (A2) PPD dương | §0 | Hệ quả 1.1, bất khả quy | Có | có thể có trạng thái xác suất 0 làm đứt chuỗi |
| (A3) ℓ đã biết | §0 | Định lý 1 | Có, hoặc tham số phiền nhiễu với prior độc lập | mục tiêu sai |
| q > 0 | Định lý 2 | Bước 0 (duy nhất) | Có | có thể nhiều phân phối dừng |
| (S) ổn định | §5 | Mệnh đề 4(b) | Chỉ cho 4(b) | 4(b) chỉ còn dạng $\tau\bar\varepsilon_B$ tính số |
| Không giả định ergodic của P trong Định lý 2 | — | — | Không cần sau MINOR-1 | — |

## Những gì đã thử mà không phá được
- Bổ đề 1: liệt kê mọi trạng thái với prior ô và prior hỗn hợp (feature mang thông tin), sai lệch $<10^{-12}$.
- Định lý 1: 7 cấu hình nhãn yếu × 2 prior, phân phối dừng của ma trận chuyển chính xác lệch $\pi$ dưới $10^{-13}$; cân bằng chi tiết $<10^{-16}$; PLL 3 lớp.
- Định lý 1(b): dựng tường minh dãy hoán đổi (sắp lại từng vị trí) cho tính bất khả quy; thử túi 2 mẫu để tìm tuần hoàn (thấy, xem MINOR-1).
- Định lý 2: tính lại bổ đề tái trọng số từng bước; kiểm tra chặn Dobrushin qua phân tích $\mu=a(\alpha-\beta)$; đối chiếu hằng số Mitrophanov (chuẩn $2\,\mathrm{TV}$, chuẩn toán tử); 30 cấu hình nhiễu không tương thích (E3), chặn luôn đúng.
- Nhận xét 4.1: kiểm tra $\delta(P^t)=1$ với $t<|F|$ bằng lập luận hai trạng thái khác nhau ở mọi mẫu tự do.
- Nhận xét 4.3: chứng minh lại tính tương thích của xác suất có điều kiện chỉ phụ thuộc số đếm (dựng $f$ tường minh).

## Thứ tự sửa ưu tiên
1. MAJOR-1: ghi RESEARCH_PLAN §9, sửa CLAIM_EVIDENCE_MATRIX #3, #10.
2. MAJOR-2: sửa §0 và §5 của PROOFS.
3. MINOR-1 tới MINOR-7, COSMETIC-1.
