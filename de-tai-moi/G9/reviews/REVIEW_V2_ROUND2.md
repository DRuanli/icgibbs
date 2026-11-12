# REVIEW_V2_ROUND2 — Phản biện vòng 2 cho bản sửa (NeurIPS 2027)

- **Ngày:** 12/11/2026. **Người phản biện:** referee vòng 2, không tham gia vòng 1 và không tham gia sửa bài.
- **Bản được đọc:** `paper/main.pdf` (36 trang, dựng 04:19 UTC ngày 12/11), nguồn `paper/sections/*.tex`, `paper/tables/*.tex`, `paper/numbers*.tex`, `paper/scripts/make_v2.py`, `make_figures.py`. Ba báo cáo vòng 1: `REVIEW_V2_MATH_01.md`, `REVIEW_V2_AUDIT_01.md`, `REVIEW_V2_NEURIPS_01.md`.
- **Cách làm:** đọc lại từng dòng `theory.tex`, `appendix_proofs.tex`, `experiments.tex`, abstract, introduction, limitations, conclusion; dựng lại đại số của các chỗ đã sửa; tính lại số từ dữ liệu thô bằng script riêng (không import script của dự án), đọc npz bằng `python3 -I`, `allow_pickle=False`. Không sửa tệp nào của repo ngoài báo cáo này.
- **Script:** `/tmp/claude-0/-home-claude/643477cb-9e2c-5e3b-9dd1-1dba7a8f3cd0/scratchpad/review_r2/s3re.py`, `/tmp/claude-0/-home-claude/643477cb-9e2c-5e3b-9dd1-1dba7a8f3cd0/scratchpad/review_r2/rmap.py`, `/tmp/claude-0/-home-claude/643477cb-9e2c-5e3b-9dd1-1dba7a8f3cd0/scratchpad/review_r2/s1chains.py`.

---

## 1. Kết luận trong 5 dòng và điểm cập nhật

1. **Bản sửa xử lý thật phần lớn phát hiện vòng 1.** Cả 4 MAJOR toán và 5 MAJOR kiểm toán đều đã được sửa đúng hướng (7 sửa hẳn, 2 sửa một phần); câu viện dẫn Định lý 3(b), "+0.052 … strongest", "with no trained model", "14.1% behind a tuned learner", câu MH và bảng dự đoán 7/17 đều đã đúng. Mọi con số mới tôi tính lại đều khớp dữ liệu.
2. **Hai lỗi MAJOR mới, đều sửa được bằng câu chữ.** (N1) Câu "the naive labels carry less Fisher information than the observation exactly for PLL and LLP" và Mệnh đề 5(b) sai với LLP: nhãn ngây thơ khớp số đếm túi thì số đếm là hàm của chúng, nên theo chính điều kiện dấu bằng của mệnh đề, thông tin Fisher KHÔNG giảm; mất mát nằm ở hợp lý từng điểm (composite), đúng như `G4/PROOFS.md` Quan sát 6(b) đã ghi. (N2) Câu "for n_F between 756 to 3,934 it needs margins above about 0.17 to 0.30" gắn sai số: 0,17 và 0,30 là biên ở n_F = 3 000 và 1 000; ở 3 934 và 756 là 0,15 và 0,35.
3. **Sửa bài sinh ra một số lỗi nhỏ:** phần dư "(1+O(Cα/N))" của Mệnh đề 3 thiếu số hạng K/n; quy tắc hoà 0,001 nay được phát biểu chung nhưng Bảng 1 vẫn đếm theo so sánh chính xác (16/0/2, theo quy tắc là 15/1/2); "cell" được định nghĩa là một lượt chạy nhưng "38 cells" là cặp (dataset, mức); `\claim` bị ẩn để lại khoảng trắng trước dấu câu ở 22 chỗ trong chính văn; chú thích Hình 2 (phụ lục) nay mâu thuẫn với §6.3.
4. **Phần chưa sửa là phần cần thí nghiệm hoặc cần chỗ:** Algorithm 1 vẫn ở phụ lục, Bảng 2 vẫn ghép hai bảng cỡ ~6pt, không có đoạn khuyến nghị thực hành, δ của Giả thiết 1 vẫn chưa được ước lượng, baseline vẫn chỉ là learner nơ-ron, Study 3 không có naive/hard-EM/oracle.
5. **Dự đoán:** bài nay trung thực và nhất quán hơn rõ; từ "borderline reject" lên "borderline accept", nhưng trần vẫn bị chặn bởi baseline, bởi việc cấu hình được khuyến nghị chỉ có bằng chứng thăm dò, và bởi Định lý 3 "certifies nothing" ở cỡ dữ liệu của bài.

### Điểm NeurIPS cập nhật

| | Vòng 1 | Vòng 2 (bản hiện tại) | Sau khi sửa N1, N2 và các MINOR bằng câu chữ |
|---|---|---|---|
| Soundness (1–4) | 2 | **3** (thấp) | 3 |
| Presentation (1–4) | 2 | **2** (sát 3) | 3 |
| Contribution (1–4) | 2 | **3** (thấp) | 3 |
| Overall (1–10) | 4 | **5** (borderline accept; dao động 4–6) | 5–6 |
| Confidence (1–5) | 4 | **4** | 4 |

**Lý do.** *Soundness 3:* các khẳng định chính nay khớp với kiểm định và định lý; thất bại được nêu ngang hàng. Chưa lên cao hơn vì còn một mệnh đề phát biểu sai (N1), Giả thiết 1 chưa được kiểm ở dạng phát biểu, và phương pháp được khuyến nghị (multi-start, multi-TFM) không có kiểm định xác nhận so với baseline. *Presentation 2:* vẫn rất nén, thuật toán ở phụ lục, bảng chính của Study 3 khó đọc, còn khoảng chục thuật ngữ phải đoán (mục 6). *Contribution 3:* ý tưởng gọn (Bổ đề 1), một replication thật trên dữ liệu mới, kết quả âm được báo cáo đầy đủ; phần lý thuyết cho sampler mặc định là định tính.

**Cải thiện đơn lẻ có giá trị nhất không cần GPU.** Bổ sung baseline chạy bằng CPU trên đúng các lần chia đã lưu của Study 3 (và Study 2 nếu kịp): một phương pháp PLL cổ điển cho dữ liệu bảng (PL-kNN hoặc IPAL), cây tăng cường với loss PLL/LLP, và một ensemble của L qua seed/cấu hình; báo cáo là thăm dò. Đây là điểm mà chuyên gia PLL/LLP sẽ dùng để từ chối, và mọi dự đoán của chuỗi đã có sẵn nên không cần chạy lại TFM. Nếu chỉ được sửa câu chữ: thêm một bảng nhỏ "cấu hình | bằng chứng (xác nhận/thăm dò) | chi phí | khi nào dùng" lấy từ Bảng 2 và Bảng 15, kèm hai con số đã có sẵn bảo vệ nhãn "frozen" ở LLP (chuỗi cold-start của Study 3: 0,793 so với L 0,783; ablation cold start của Study 1: −0,005 AUC).

**Thí nghiệm mà người phản biện thật vẫn sẽ đòi.** (i) Naive TFM, hard-EM và oracle cho từng TFM ở Study 3 (chuỗi đơn có làm hại không?). (ii) Cold start cho LLP trên dataset Study 2 (kết quả đầu bảng có phụ thuộc DLLP không?). (iii) Chạy dài T ≥ 100 trên các ô kẹt: ở MiceProtein hai chuỗi TabPFN v2 vẫn đang tăng ở vòng 30 (0,50 và 0,53 ở vòng cuối, trung bình 5 vòng cuối 0,44), nên câu "More sweeps then do not repair a start in the wrong basin" chưa có kiểm chứng trực tiếp. (iv) Ánh xạ phản ứng trên dataset Study 3 và một ước lượng δ; một kiểm chứng không vòng tròn (map của chuỗi A dự đoán chuỗi B). (v) Baseline PLL/LLP phi nơ-ron, phương pháp belief-propagation cho LLP mà bài đã trích (`bp_llp`), fine-tuning TFM. (vi) Dataset > 10 000 dòng và context không cắt; LLP thật; thêm PLL thật; q = 0,3 và túi 16/32 ở Study 3; hơn 3 seed.

### Đếm theo trạng thái

| Nguồn | Số mục | Đã sửa | Sửa một phần | Chưa sửa | Sửa sinh lỗi mới |
|---|---|---|---|---|---|
| Toán: MAJOR M1–M4 | 4 | 2 (M1, M3) | 2 (M2, M4) | 0 | 0 |
| Toán: MINOR m1–m16 | 16 | 13 | 2 (m1, m14) | 0 | 1 (m15) |
| Toán: COSMETIC c1–c2 | 2 | 1 | 1 | 0 | 0 |
| Kiểm toán: BLOCKING B1 | 1 | 0 | 1 | 0 | 0 |
| Kiểm toán: MAJOR M1–M5 | 5 | 5 | 0 | 0 | 0 |
| Kiểm toán: MINOR m1–m21 | 21 | 7 | 5 | 8 | 1 (m6) |
| Kiểm toán: COSMETIC c1–c8 | 8 | 2 | 1 | 5 | 0 |
| NeurIPS: nhóm điểm yếu 3.1–3.8 | 8 | 0 | 6 | 2 (3.4, 3.7) | 0 |
| NeurIPS: danh sách [must] 1–10 | 10 | 4 (3, 8, 9, 10) | 5 (1, 2, 4, 5, 6) | 1 (7) | 0 |
| **Tổng** | **75** | **34** | **23** | **16** | **2** |

**Phát hiện mới: BLOCKING 0 · MAJOR 2 · MINOR 14 · COSMETIC 4.**

---

## 2. Bảng phát hiện vòng 1: trạng thái và bằng chứng

Trích dẫn là câu hiện tại của bài (tệp:dòng xấp xỉ).

### 2.1 Báo cáo toán (`REVIEW_V2_MATH_01.md`)

| ID | Mức | Trạng thái | Bằng chứng (câu hiện tại) và nhận xét |
|---|---|---|---|
| M1 | MAJOR | **Đã sửa** | Câu "by Theorem 3(b), a chain that starts there cannot climb" đã bỏ. `experiments.tex:79-84`: "It is not a faithful model of the chains, however: … For the uniform map only the lower half of \cref{ass:H} is therefore plausible; it can indicate that a chain will climb, not that it will stall." Tiêu đề §6.3 đổi thành "Stalling, and what response maps do and do not explain". Abstract không còn "which the response map explains". Câu thay thế có một chỗ chưa chính xác, xem N3. |
| M2 | MAJOR | **Sửa một phần** | Abstract (`abstract.tex:8-10`): "For the sampler as run, we show that under a homogeneity assumption on the TFM a one-dimensional response map bounds where a chain can go, so that with two attractors the start matters." Theory (`theory.tex:91-94`): "the expected time to leave is at least $1/p$ sweeps, which for a fixed fraction of unknown labels and fixed margins grows exponentially with $n_F$ … at the sizes of our datasets it certifies nothing". Còn lại: (i) Contributions (`introduction.tex:57-58`) vẫn viết "**invariant** floors and ceilings of the label accuracy" (định lý cho sàn/trần giữ với xác suất 1 − Hp, không bất biến); (ii) "for the sampler as run" ở abstract và intro không nói "instance-level", trong khi LLP là một nửa thực nghiệm và context bị cắt nằm ngoài phạm vi; (iii) "bounds where a chain can go" thiếu "with high probability". Xem N11. |
| M3 | MAJOR | **Đã sửa** | `theory.tex:38-43`: "In a conjugate model with unconstrained unknown labels and a fresh random block at every step, the stationary mean of the unknown-label counts equals their exact posterior mean under a prior concentration inflated by $K(n-1)/((K-1)n)$, and the resulting bias of the predictor is of order $1/((K-1)N)$ for $N$ labelled points … For the partition scan that the sampler uses this holds to leading order numerically". Cả ba điểm (counts chứ không phải predictor; bậc 1/((K−1)N); kiểu quét) đều đúng. |
| M4 | MAJOR | **Sửa một phần** | (ii) đã sửa: `theory.tex:57-58` "for every state $y$ and every block $B$ of at most $\lceil n/K\rceil$ points with $B\cap F\neq\emptyset$ and $F\not\subseteq B$". (iii) đã sửa: không còn "We checked Assumption 1"; `experiments.tex:86-88` và `limitations.tex:9-11` "rests on an assumption we could test only through its consequences along chains". (i) chưa sửa về thực chất: vẫn là sup trên mọi trạng thái; bài chỉ thừa nhận (`theory.tex:67-68` "This is a strong, uniform requirement, which we treat as a model of the TFM to be confronted with data") và **δ vẫn chưa được ước lượng ở đâu** (các khoá `th-delta-*` có trong `numbers_v2.tex` nhưng không dùng). Lượng từ "at most ⌈n/K⌉ points" còn rộng hơn cần thiết (N13). |
| m1 | MINOR | **Sửa một phần** | Intro (`introduction.tex:26-27`): "the accuracy of the TFM's relabelling as a function of the accuracy of the labels in its context"; chú thích Hình 1 đúng nghĩa cấp khối. Còn sót `appendix.tex:291`: "we measured the **one-sweep response** at a grid of accuracy levels". |
| m2 | MINOR | **Đã sửa** | `theory.tex:87-91`: "For a non-decreasing $\Phi$ … a chain that starts sufficiently far above the crossing stays up, one that starts sufficiently far below stays down". Câu "every start reaches the same band" đã bỏ. |
| m3 | MINOR | **Đã sửa** | `appendix_proofs.tex:282-284`: "$\ge\max\{0,1-p\}^H\ge1-Hp$ for every integer $H\ge1$. If $p<1$ … ($\tau$ may be infinite); if $p\ge1$ the claim is trivial, since $\tau-s\ge1$." Tôi kiểm lại: đúng. |
| m4 | MINOR | **Đã sửa** | `theory.tex:69`: "$\underline\Phi(x)=\inf\{\Phi(r):r\in[0,1],r\ge x\}$"; `appendix_proofs.tex:231-232`: quy ước inf ∅ = +∞, sup ∅ = −∞; Lemma (ii) có "$\mathcal F_s$-measurable". |
| m5 | MINOR | **Đã sửa** | `appendix_proofs.tex:269-272`: "each increment $\xi_i-c_i$ lies in the interval $[-c_i,1-c_i]$ of length 1, whose endpoints are determined by the past, so Hoeffding's lemma applied conditionally to each increment (the martingale form of Hoeffding's inequality with predictable ranges) gives $P(Z_k\le-t)\le\exp(-2t^2/n_F)$". Đúng. |
| m6 | MINOR | **Đã sửa** | Remark (i), `appendix_proofs.tex:294-296`: "The proofs use the assumption only at the pairs (state before a block update, block) of the sweeps in question, and for a floor only when the context accuracy $r_{-B}$ is at least $x-\nu$". |
| m7 | MINOR | **Đã sửa** | `appendix_proofs.tex:314-317`: "suppose that every redraw in block $k$ is correct, independently given the context, with probability exactly $c_0+b\,r_k$ … (a point-wise affine response, stronger than \cref{ass:H} …, which constrains block averages only)"; chứng minh: "The recursion for the means is affine in $S$ and can be evaluated at any real $S$". Câu "have a closed form" ở thân bài đã bỏ. Tôi dựng lại (a), (b): đúng. |
| m8 | MINOR | **Đã sửa** | `appendix_proofs.tex:162-163`: "With blocks of $M<n$ points over all points, … $\alpha'=\alpha(n-1)/(n-M)$, which is $\alpha\,K(n-1)/((K-1)n)$ for $M=n/K$". Tôi dựng lại hai ngoặc và điểm bất động: đúng. |
| m9 | MINOR | **Đã sửa** | `appendix_proofs.tex:198-201`: "agrees with the fixed point of the affine drift to within 1.5×10⁻¹¹ in 15 configurations, and the drift fixed points agree with the stationary means computed from full transition matrices to within 4×10⁻¹¹ in 9 further configurations with $n\le200$". Khớp `e7f_alpha_prime_check.csv` (15 dòng, 1,46·10⁻¹¹) và `e7f_affine_check.csv` (9 dòng, 3,9·10⁻¹¹). |
| m10 | MINOR | **Đã sửa** | `theory.tex:47-48`: "uncorrected blocks, a fresh random partition in every sweep, any deterministic TFM, fixed weak-label factors, context not subsampled"; định lý viết "in the setting above". |
| m11 | MINOR | **Đã sửa** | `theory.tex:91-92`: "which for a fixed fraction of unknown labels and fixed margins grows exponentially with $n_F$". |
| m12 | MINOR | **Đã sửa** | Remark (iii), `appendix_proofs.tex:302-303`: "the proven two-sided invariant band for $H=60$ sweeps at total failure probability 0.05 has half-width 0.22". Còn thiếu tên ánh xạ và việc cận trên bị cắt ở 1, xem kiểm toán m14 và N14. |
| m13 | MINOR | **Đã sửa** | `theory.tex:36-38`: "for it the bound above is useless: on exactly solvable problems it exceeds 1 in most cases, while the true $\TV(\tilde\pi,\pi)$ is small but does not shrink with $n$". |
| m14 | MINOR | **Sửa một phần** | Thân bài bỏ "symmetric channel", ghi "binary label noise" (`theory.tex:97`). Nhưng (i) tiền đề mới "generated independently of the features given the true label" không đủ để suy ra bảo toàn thứ hạng (cần thêm điều kiện cột và λ_k > 0); (ii) Mệnh đề 5(a) ở phụ lục vẫn ghi "class-dependent noise" không có "binary" và kết luận "λ_k > 0" không suy ra từ giả thiết. Xem N1 (cùng đoạn) và N10. |
| m15 | MINOR | **Sửa sinh lỗi mới** | `appendix_proofs.tex:167`: "$=\frac{\alpha(1-\lambda)(1-Cp_c)}{(K-1)\,N}\,\big(1+O(C\alpha/N)\big)$". Thừa số (n−K)/(n+Cα) đã bị bỏ, nên phần dư thiếu số hạng K/n. Xem N6. |
| m16 | MINOR | **Đã sửa** | Chú thích Hình 1: "On optdigits the context is subsampled, outside the scope of \cref{thm:basins}." Remark (iv) nói phép kiểm thực nghiệm gồm cả ô có context bị cắt. |
| c1 | COSMETIC | **Sửa một phần** | N_k của bổ đề đổi thành Ξ_k. Còn trùng: x (mức) với x_i (đặc trưng); δ (dung sai) với δ(P^t) Dobrushin và δ₁, δ₂; τ với τ(P); p(ν, λ) với p (PPD) và p-value; b (mức đích ở (c)) với b (hệ số góc, Mệnh đề 4); q, q_j, q₀–q₂; H (số vòng) với (H_δ) và H1″. |
| c2 | COSMETIC | **Đã sửa** | `appendix_proofs.tex:161`: "for all $n$, $u\le U$ and $c$". |

### 2.2 Báo cáo kiểm toán (`REVIEW_V2_AUDIT_01.md`)

| ID | Mức | Trạng thái | Bằng chứng và nhận xét |
|---|---|---|---|
| B1 | BLOCKING | **Sửa một phần** | Hai `\todo` đã bỏ; PDF không còn "TODO", "[C#]", "??". Còn: `[Pending: licences …]` ở checklist mục 12 (tác giả chủ ý giữ; `\finaltrue` sẽ dừng biên dịch), `\finalfalse` ở `main.tex:27`, style `neurips_placeholder.sty`. Việc ẩn `\claim` sinh lỗi khoảng trắng (N15). |
| M1 | MAJOR | **Đã sửa** | `experiments.tex:136-138`: "the average of single chains over TFMs, which does not use $L$, was above $L$ on all 12 LLP datasets but only 6/1/1 (W/T/L) on PLL". Tính lại từ npz: LLP 12/0/0, PLL 6/1/1 (nhỏ nhất −0,200). |
| M2 | MAJOR | **Đã sửa** | Intro (`introduction.tex:51-53`): "with one TFM the multi-start average was more than 0.02 behind the better of its single chain and the tuned learner in 14.1% of cases, against a registered limit of 10%"; Limitations: "more than 0.02 behind the tuned learner in 9 of 71 pairs"; abstract: "falls behind the better of its single chain and the tuned learner". Tính lại: 10/71; thua L 9; thua chuỗi đơn 2; cả hai 1. |
| M3 | MAJOR | **Đã sửa** | `appendix.tex:262-263`: "lists every prediction and its outcome, including the 7 of 17 that were wrong (the number P10 was never assigned)"; Bảng 16 có P1–P4. Đối chiếu `PHASE0_PREDICTIONS.md`, `V0_REPORT.md`, `PHASE1_PLAN.md §5`, `PHASE1B_PLAN.md`, `V1D_PLAN.md`: 17 dự đoán, sai P2, P4, P5, P11, P12, P14, P15 = 7. |
| M4 | MAJOR | **Đã sửa** | Abstract: "(+0.052 AUC over the registered comparator, +0.041 over the strongest baseline)". Tính lại từ `s2_table_llp8.csv`: Gibbs − DLLP-MLP tune = +0,0409, 16/0/2; khoảng bootstrap của tôi [+0,018; +0,073], bài in [+0,018; +0,072]. |
| M5 | MAJOR | **Đã sửa** | `experiments.tex:150-153`: "was not consistently better than the default at the same 30 sweeps (MH minus default with TabICL v2: +0.033 AUC on LLP, −0.021 accuracy on PLL at $q=0.6$; \cref{tab:s2e}) … the comparison confounds bias with slower mixing". |
| m1 | MINOR | **Sửa một phần** | Thân bài: "(mean gap 0.000)". **Bảng 2 vẫn in "gap +0.000"**: `make_v2.py:191` ghi `s3_tests.tex` trước khi dòng 570 ghi đè `s3-c5-gap`. Sửa: chuyển dòng 569–570 lên trước dòng 182. |
| m2 | MINOR | **Đã sửa** | Bảng 14 nay sinh từ `s3_pairs.csv`: "TabPFN-3 & micro-mass & 0.047". |
| m3 | MINOR | **Chưa sửa** | `appendix.tex:229-233` giữ nguyên: "The pairs … are of two kinds: single chains that stall so deeply … (cnae-9, micro-mass), and datasets on which $L$ is better than every chain of that TFM (coupon, seismic-bumps, climate-crashes)" và "i.e. it converged to the labelling with the two classes exchanged within bags' counts". TabPFN-3 × seismic-bumps (0,755; 0,731; L 0,719) không thuộc loại nào; gesture-phase và ozone-8hr không được xếp loại; bốn ô AUC 0,30–0,49 gồm cả ô gần ngẫu nhiên. Câu thay thế: "In 9 of the 10 pairs $L$ is better than every chain of that TFM; in 3 of them the single chain had stalled; the tenth (TabPFN-3, seismic-bumps) is behind its own single chain only. In 4 cells the single chain had an AUC below 0.5 on LLP (0.30–0.49)." |
| m4 | MINOR | **Đã sửa** | `appendix.tex:281-282`: "This small gain is why we expected a typical dataset of Study 3 to gain little, which held (median +0.003), and few pairs to regress, which did not." |
| m5 | MINOR | **Sửa một phần** | P7 ("not below the single chain by more than 0.002 in any of four groups"), P17 ("$\le 2$ regressions each"), P13 (thêm "$>0.02$ behind on $\le 2$ datasets") đã đúng. Còn thiếu: P13 chấm trên 13 dataset (ngưỡng ghi trước là 2/15); phạm vi ô của P12. |
| m6 | MINOR | **Sửa sinh lỗi mới** | `experiments.tex:15`: "differences below 0.001 count as ties" nay là quy tắc chung của §6.1, nhưng Bảng 1 đếm theo so sánh chính xác. Xem N7. |
| m7 | MINOR | **Sửa một phần** | H3″ đã định nghĩa (`experiments.tex:32-33`) và "66 failing cells of 213" đã có. "the five flagged datasets" (`experiments.tex:121`) vẫn chưa được giải thích trong chính văn. |
| m8 | MINOR | **Chưa sửa** | Chú thích Bảng 2 và abstract vẫn không nói +0,058 và +0,046 là trung bình gộp chênh AUC (LLP) và chênh accuracy (PLL). Thêm vào chú thích: "(AUC for LLP and accuracy for PLL, pooled as registered)". |
| m9 | MINOR | **Chưa sửa** | Chú thích Bảng 15 chỉ ghi "two-sided Wilcoxon signed-rank $p$, unadjusted", không nêu xấp xỉ chuẩn khi có số 0. |
| m10 | MINOR | **Đã sửa** | `experiments.tex:80-83`: "(both TFMs, LLP and PLL; exploratory, …), chains ended above the fixed point of the uniform-error map in 96% of cells". Tính lại: 224/234 = 95,7% (TabICL v2 95,2%; TabPFN v2 97,1%). |
| m11 | MINOR | **Đã sửa** | Limitations không còn câu "hard for every TFM"; `appendix.tex:321-322`: "run with TabICL v2 only". |
| m12 | MINOR | **Chưa sửa** | Bảng 4 vẫn "10 Nov 04:21"; không có `.git` để kiểm. |
| m13 | MINOR | **Chưa sửa** | `appendix_proofs.tex:209-210` giữ nguyên "(33 configurations) … up to 0.052 when $n\le160$ and by at most 0.0024 from $n=320$ on ($K=10$)". |
| m14 | MINOR | **Sửa một phần** | H và xác suất hỏng đã nêu; chưa nêu ánh xạ (Φ = 0,45 + 0,5r) và việc cận trên bị cắt ở 1 (dải chứng minh là [0,676; 1]). Xem N14. |
| m15 | MINOR | **Đã sửa** | `experiments.tex:158-159`: "the 9 or 12 chains of the multi-TFM average took a median of 37 minutes per dataset and seed, not counting the tuning of $L$". |
| m16 | MINOR | **Đã sửa** | `experiments.tex:68-70`: "it beat tuned CC on 7 of the 10 datasets with fewer than eight classes (+0.033 on average)". |
| m17 | MINOR | **Sửa một phần** | `conclusion.tex`: "averaging over starts and TFMs removed most failures of single chains". Bỏ "in a third pre-registered study" nhưng vẫn không nói đây là kết quả thăm dò; với một TFM, multi-start chỉ giảm số cặp thua L quá 0,02 từ 14 xuống 9. Câu thay thế: "averaging over starts and over TFMs improved on single chains in a third pre-registered study; in exploratory analysis the multi-TFM average was never more than 0.02 behind the tuned learner". |
| m18 | MINOR | **Chưa sửa** | `experiments.tex:141-143` vẫn "in 81% of cells" rồi cho hai số chỉ của PLL (tỉ lệ đồng thuận ở PLL là 61%, khoá `s3x-basin-same-pll` có sẵn). |
| m19 | MINOR | **Đã sửa** | Bảng 18 có chú thích $^{\ast}$: "The TFM is frozen; the LLP warm start trains a linear DLLP model and the learner start a tuned specialised learner". |
| m20 | MINOR | **Chưa sửa** | `appendix.tex:177-178` vẫn "at their most relaxed registered tier" không nêu ba ngưỡng. |
| m21 | MINOR | **Chưa sửa** | Còn số gõ tay: "about one accuracy point", "23 development and earlier datasets", các tỉ số thông tin Fisher, giờ tính toán, giờ commit; checklist mục 4 vẫn viết "every number is generated by scripts". |
| c1 | COSMETIC | Chưa sửa | Thân bài "from the same suites, OpenML-CC18 and TabArena" so với phụ lục "from the suites of Study 2 and OpenML-CC18". |
| c2 | COSMETIC | Chưa sửa | Cận "+0.110" vẫn 3 chữ số. |
| c3 | COSMETIC | Chưa sửa | `appendix.tex:250`: "the average of single chains is −0.028 below it". |
| c4 | COSMETIC | Đã sửa | Câu "4 of the four" không còn. |
| c5 | COSMETIC | Đã sửa | "2 behind the single chain, 9 behind $L$, 1 behind both". |
| c6 | COSMETIC | Chưa sửa | "+0.0032 … on these pairs" vẫn không ghi là trung bình của bốn trung bình nhóm. |
| c7 | COSMETIC | Chưa sửa | Chú thích Bảng 2 không mô tả hai dòng "Multi-TFM" và "Tuned learner L"; khoá `s3x-loo-other-max` vẫn tính bằng `min()`. |
| c8 | COSMETIC | Sửa một phần | Không còn chữ tiếng Việt trong `sections/`; chú thích LaTeX vẫn có đường dẫn nội bộ (`G4/PROOFS.md`, `V2/reviews/…`, `G9/reviews/REVIEW_V2_MATH_01.md`). |

### 2.3 Báo cáo kiểu NeurIPS (`REVIEW_V2_NEURIPS_01.md`)

Báo cáo này không gắn nhãn BLOCKING/MAJOR; tôi coi tám nhóm điểm yếu 3.1–3.8 và mười mục [must] là các phát hiện chính.

| Mục | Trạng thái | Bằng chứng và nhận xét |
|---|---|---|
| 3.1 Phương pháp cuối không "training-free"; bằng chứng xác nhận không phủ cấu hình khuyến nghị | **Sửa một phần** (đã khai, thực chất còn) | `method.tex:136-137`: "The TFMs stay frozen throughout, but the starts are not free of training: the LLP warm start fits a linear model, and the learner start a tuned one." Limitations: "The comparison with specialised baselines is confirmatory only for the single chain … its comparison with $L$ is exploratory … came from the learner start, which needs a trained learner." Chưa có so sánh công bằng (ensemble của L, naive-TFM ensemble). |
| 3.2 Định lý 3 và response map | **Sửa một phần** | Mâu thuẫn §6.3 đã hết; bài nói thẳng "it certifies nothing" và "not that \cref{thm:basins} forecasts anything". Còn: δ chưa ước lượng; phép kiểm chain-seeded vẫn được đọc là "shows that chains settle at a fixed point of a one-dimensional map" (N8); không có kiểm chứng không vòng tròn. |
| 3.3 Định lý 1–2 nói về sampler không chạy | **Sửa một phần** | Limitations: "The exactness results do not cover the uncorrected blocks used by default." Tuyên bố mới của Định lý 2 đã thu hẹp và có trích dẫn. Còn: `related.tex:15-16` "has guarantees for incompatible conditionals" và dấu ✓ ở cột "inference over latent labels with guarantees" của Bảng 18 vẫn áp cho biến thể không chạy. |
| 3.4 Baseline và phạm vi | **Chưa sửa** (chỉ khai ở Limitations: "the baselines are neural learners") | Cần thí nghiệm. `experiments.tex:148-149` vẫn "single chain with TabICL v2, registered in Study 1" cho dữ liệu PLL thật, trong khi Phụ lục D coi mọi kết quả TabICL v2 của Study 1 là thăm dò (N12). |
| 3.5 Thống kê | **Sửa một phần** | Đã có: "one-sided" ở §6.1 và chú thích Bảng 2; quy tắc hoà; định nghĩa "fails" của H3″; "at $q=0.6$ the test does not survive a second chain". Chưa có: phương sai theo seed; ghi rõ gộp AUC và accuracy; cơ chế đăng ký kiểm chứng được từ bên ngoài. |
| 3.6 Abstract/intro/kết luận nói quá | **Sửa một phần** (gần hết) | Đã sửa: "governs", "explains", "strongest … +0.052", "averaging chains from several starts" (nay có "one of them a tuned specialised learner" và "median +0.003"), PLL có "narrowly". Còn: tiêu đề §6.2 "One chain with a robust TFM **outperforms** specialised learners" và Conclusion "a single chain with a robust TFM outperformed specialised learners" (N9); "invariant" (N11). |
| 3.7 Người dùng không biết nên chạy gì | **Chưa sửa** | Không có đoạn khuyến nghị thực hành hay bảng chi phí–chất lượng. |
| 3.8 Tính mới | **Sửa một phần** | `theory.tex:21-22`: "Samplers built from possibly incompatible conditionals are classical \citep{heckerman2000,liu2014}, and their stationary distribution depends on the scan order \citep{chen2015gibbs}." Chưa nối với Hogwild/asynchronous Gibbs, stochastic EM, multiple imputation, EXIT chart. |
| [must] 1 Dấu vết nháp | Sửa một phần | Như B1. |
| [must] 2 Viết lại abstract | Sửa một phần | Như 3.6. Abstract chưa có "+0.018, p = 0.14 so với L với một TFM" và chưa ghi p là một phía. |
| [must] 3 Mâu thuẫn §6.3 | **Đã sửa** | Như M1. |
| [must] 4 Tầm với của Định lý 3 | Sửa một phần | Remark (iii) có bảng biên theo n_F; §6.3 có câu định lượng nhưng gắn sai số (N2); δ chưa ước lượng. |
| [must] 5 "Training-free" | Sửa một phần | Như 3.1. Kết quả cold start (Bảng 24: LLP −0,005 AUC, 10/0/20) chưa được đặt cạnh tuyên bố chính. |
| [must] 6 Định nghĩa thuật ngữ | Sửa một phần | Đã có: cell, pair, tie, hard-EM, naive TFM, development datasets, "fails". Chưa có: oracle, Stochastic EM (N16), H1/H5 (Bảng 4 vẫn "H1--H5", "H3 and H5 met"), flagged, "published limits", quy tắc chọn primary comparator, năm suite. |
| [must] 7 Algorithm 1 và hình vào chính văn; tách Bảng 2 | **Chưa sửa** | `method.tex:3`: "Pseudocode of one chain is given as \cref{alg:icg} in \cref{app:alg}." Định lý 3 vẫn "Run \cref{alg:icg}". Bảng 2 vẫn ghép. |
| [must] 8 Thu hẹp tuyên bố Định lý 2 | **Đã sửa** | Như 3.8. |
| [must] 9 "the gain comes from sampling the labels jointly" | **Đã sửa** | `experiments.tex:59-62`: "With the same TFM and warm start, sampling beats taking the arg max: … lower log loss than hard-EM in 78.9% of 38 cells (registered H3$'$) and higher mean accuracy at $q=0.6$ (0.851 against 0.803). Averaging over samples adds little to predicting from the last sample alone (0.843 …)". Khớp dữ liệu. Lưu ý: ở q = 0,3 hai phương pháp bằng nhau (0,862 và 0,862; Gibbs hơn hard-EM ở 4/10 dataset), câu không nói. |
| [must] 10 CI trùng nhau | **Đã sửa** | Chú thích Bảng 1 và `appendix.tex:122-124`: "+0.0125 [+0.0065, +0.0192] at $q=0.3$ and +0.0121 [+0.0056, +0.0193] at $q=0.6$". Khớp `s2_confirmatory.json`. |

---

## 3. Phát hiện MỚI

| ID | Mức | Vị trí | Vấn đề | Cách sửa cụ thể (câu thay thế) |
|---|---|---|---|---|
| N1 | **MAJOR** | `theory.tex:98-100`; `appendix_proofs.tex:360-363` (Mệnh đề 5(b)) | "the naive labels carry less Fisher information than the observation **exactly for PLL and LLP**" và "(b) … for PLL and LLP the naive labels coarsen the observation (for LLP with bags of 8, about 3% of the information in the bag counts …)". **Sai với LLP.** Nhãn ngây thơ của LLP khớp số đếm của túi (`baselines.py:36-41` dùng `weak.init_labels`; §6.1: "one label per point consistent with the weak labels"), nên số đếm là hàm của nhãn ngây thơ và điểm số của quan sát đo được theo σ(w̃): theo chính điều kiện dấu bằng của (b), thông tin Fisher **bằng nhau**. Con số 3% là thông tin Godambe của hợp lý từng điểm (đoạn "Information lost", `appendix_proofs.tex:387-388`, ghi đúng), không phải thông tin Fisher của nhãn ngây thơ. Nguồn gốc của bài (`G4/PROOFS.md`, Quan sát 6(b)) viết đúng: "dữ liệu ngây thơ đồng thời của một túi không mất thông tin … Mất mát đến từ việc bộ ngây thơ coi các cặp là độc lập". Với PLL, tính ngặt chỉ được tính số, không được chứng minh. Một referee lý thuyết bắt được trong một dòng. | **Thân bài:** "For these weak labels the naive target is an increasing affine function of the true posterior, class by class, so it ranks points as the posterior does. What differs is information. For PU and label noise the naive labels are the observation. For PLL they carry less Fisher information than the candidate sets (strictly, in every case we computed). For LLP the naive labels of a bag still determine its counts, but a learner that treats them as independent point labels uses a composite likelihood, whose information is about 3% of that of the counts for bags of 8 (\cref{prop:naive}). Only for PLL and LLP can the sampler recover information through $\lik$, …" **Mệnh đề 5(b):** "… with equality iff the observation's score is a function of the naive labels. Equality holds for PU-SCAR and label noise, where the naive labels are the observation, and for LLP when the naive labels match the bag counts. For PLL the inequality was strict in every case we computed. For LLP the loss arises at the learner: the point-wise (composite) likelihood on naive labels has a Godambe information of about 3% of the information in the counts for bags of 8 in a logistic model." |
| N2 | **MAJOR** (câu nói sai nghĩa con số; sửa một dòng) | `experiments.tex:90-92`; `make_v2.py:589-597, 633` | "for $n_F$ between 756 to 3,934 it needs margins above about 0.17 to 0.30, and the measured gains are at most 0.15". 0,17 và 0,30 là `th-margin-3k` và `th-margin-1k`, tức biên ở n_F = 3 000 và 1 000. Tính lại bằng đúng hàm `margin()`: n_F = 756 → **0,35**; 1 115 → 0,28; 1 395 → 0,25; 3 934 → **0,15**. Ở optdigits biên (0,152) bằng đúng gain lớn nhất đo được (0,151, TabICL v2 trên chính optdigits), nên so sánh "0,17 so với 0,15" chỉ đứng được nhờ dùng sai n_F; kết luận "silent" vẫn đúng cho ba dataset trong phạm vi (gain lớn nhất 0,137), còn optdigits nằm ngoài phạm vi vì context bị cắt. Câu cũng không nói biên này là cho 30 vòng quét ở mức hỏng 0,05 với δ = 0, và sai ngữ pháp ("between … to"). | Sinh hai khoá mới từ `margin(n_train)` của từng dataset, rồi viết: "Quantitatively the theorem is silent at these sizes. On the three datasets within its scope ($n_F$ from 756 to 1,395) a floor over 30 sweeps at failure probability 0.05 needs a gain margin of 0.25 to 0.35 even with $\delta=0$ (\cref{app:dynamics}), and the measured gains are at most 0.14; on optdigits ($n_F=3{,}934$, context subsampled) the margin would be 0.15, the size of the largest gain measured there." |
| N3 | MINOR | `experiments.tex:79-80` | "three of the four TabPFN v2 datasets climbed through the region of negative gain". Theo `s2x_tabpfn.csv` và monitor của Study 1: semeion đi từ 0,17 lên 0,34–0,39 và dừng **trong** vùng gain âm (âm từ r = 0,2 đến ít nhất 0,8; −0,11 tại r = 0,3); MiceProtein: vùng âm khoảng 0,21–0,56, hai seed kết thúc ở 0,50 và 0,53 (trong vùng), một seed 0,66 (vượt qua); optdigits: vùng âm khoảng 0,2–0,35, hai seed vượt qua (0,94; 0,95), một seed dừng ở 0,27. Chỉ optdigits (2/3) và MiceProtein (1/3) "climbed through". Tại điểm xuất phát (r ≈ 0,16–0,23) gain của ánh xạ đều ≈ 0 (−0,005 đến +0,010), không âm; vùng âm bắt đầu ngay phía trên. Ý chính (ánh xạ không mô tả đúng chuỗi) vẫn đứng. | "It is not a faithful model of the chains, however: on three of the four datasets TabPFN v2 chains rose where the map has negative gain (semeion from 0.17 to 0.34–0.37, inside a region that is negative up to at least $r=0.8$; MiceProtein to 0.44–0.64; optdigits to 0.89–0.93 in two seeds of three)." |
| N4 | MINOR | `appendix.tex:82-83` (chú thích Hình 2) | "With TabPFN v2 the chains stall where the gain is negative" mâu thuẫn với câu mới của §6.3 (chuỗi leo trong vùng gain âm; ánh xạ "can indicate that a chain will climb, not that it will stall"). Sót lại từ bản trước. | "With TabPFN v2 the chains start near the lower zero of the gain; on mfeat-factors they stay there, and on the other three datasets some rise although the uniform-error gain is negative. With TabICL v2 they label 93–99% of the training points correctly within 5–15 sweeps." |
| N5 | MINOR | `tables/app_s1_chains.tex`, `appendix.tex:89-91`; `make_v2.py:614` | Cột "start" và chú thích "mean starting value" là `monitor[0]`, tức độ đúng nhãn **sau vòng quét thứ nhất** (`sampler.py:158-159`); Study 1 không lưu r_start. Với TabPFN v2 sai khác nhỏ, nhưng với TabICL v2 bảng in 0,26–0,32 trong khi các lượt chạy lại cùng seed (`V2/raw/V0*`, có `r_start`) cho 0,17–0,25. Bảng vì thế gợi ý TabICL v2 xuất phát tốt hơn TabPFN v2 (0,16–0,22), trong khi hai điểm xuất phát gần như nhau; điều này làm yếu chính lập luận "khác biệt là ở TFM". Câu "stayed at its starting value on mfeat-factors (0.16–0.17)" đúng về thực chất (r_start 0,160–0,169). | Đổi tiêu đề cột thành "after sweep 1" và chú thích: "label accuracy after the first sweep (the starting value was not stored in Study 1; in reruns with the same seeds it was 0.16–0.23 for TabPFN v2 and 0.17–0.25 for TabICL v2), final value …". Tốt hơn: lấy r_start từ các lượt V0 cùng seed. |
| N6 | MINOR (sinh từ việc sửa m15) | `appendix_proofs.tex:166-167` | Phần dư "(1+O(Cα/N))" sai. Chính xác: bias = α(1−λ)(1−Cp_c)/((K−1)N) · (n−K)/(n+Cα) · N²/((N+Cα′)(N+Cα)). Thừa số thứ hai là 1 − O(K/n) và K/n không bị chặn bởi Cα/N. Ví dụ n = 1 000, N = 500, L_c = 400, C = 2, K = 10, α = 0,01: tỉ số chính xác/dẫn đầu = 0,9899, trong khi Cα/N = 4·10⁻⁵ và K/n = 0,01. | "$=\frac{\alpha(1-\lambda)(1-Cp_c)}{(K-1)\,N}\cdot\frac{n-K}{n+C\alpha}\,\big(1+O(C\alpha/N)\big)$" hoặc "$\big(1+O(C\alpha/N+K/n)\big)$". |
| N7 | MINOR (sinh từ việc sửa m6) | `experiments.tex:14-15` so với `tables/main_results.tex` | §6.1 nay viết "differences below 0.001 count as ties" như quy tắc chung, nhưng W/T/L của Bảng 1 đếm theo so sánh chính xác: LLP so với DLLP tuyến tính in 16/0/2, theo quy tắc 0,001 là **15/1/2** (mushroom +0,00007). Các dòng PLL không đổi. | Hoặc sinh lại Bảng 1 theo quy tắc 0,001, hoặc viết: "in Study 3 differences below 0.001 count as ties, as registered; Studies 1 and 2 count exact ties". |
| N8 | MINOR | `experiments.tex:86-88` | "this shows that chains settle at a fixed point of a one-dimensional map and that the placement of errors matters". Như báo cáo NeurIPS đã nêu, khi lỗi được gieo theo biên của chính chuỗi tại r gần độ đúng cuối, trạng thái gieo gần trạng thái dừng nên gain ≈ 0 theo tính dừng, bất kể mô tả một chiều có đúng không. Câu đã mềm đi nhưng vẫn giữ hàm ý cũ. | "Those marginals come from the chain whose end is predicted, and a state that resembles the chain's final state is nearly stationary by construction; the agreement therefore shows that the placement of errors matters, not that the one-dimensional description, or \cref{thm:basins}, forecasts anything." |
| N9 | MINOR | tiêu đề §6.2 (`experiments.tex:51`); `conclusion.tex` | "One chain with a robust TFM **outperforms** specialised learners" khái quát kết quả của Study 2 ở thì hiện tại, trong khi chính bài báo: PLL q = 0,6 không qua khi đổi seed lấy mẫu, và ở Study 3 chuỗi đơn TabICL v2 kẹt trên vài dataset (`appendix.tex:255` "here too single chains stalled on individual datasets"). | Tiêu đề: "Study 2: one chain with TabICL v2 outperformed specialised learners". Conclusion: "in Study 2 a single chain with TabICL v2 outperformed specialised learners on new datasets". |
| N10 | MINOR | `theory.tex:97-99`; `appendix_proofs.tex:357-359` | (i) Tiền đề "For weak labels generated independently of the features given the true label" không đủ: Mệnh đề 5(a) cần thêm "equal off-diagonal entries in each column" và M_kk > b_k. Phản ví dụ: nhiễu phụ thuộc lớp với C ≥ 3 và T bất đối xứng; PLL có distractor độc lập nhưng xác suất phụ thuộc cặp lớp. Các trường hợp liệt kê trong ngoặc thì đúng (tôi kiểm lại PLL: a = (1−(1−q)^C)/(Cq), λ = (Ca−1)/(C−1) > 0; LLP: λ = 1/k, đa lớp cũng thoả điều kiện cột). (ii) "PU" phải là PU-SCAR (PU-SAR phụ thuộc đặc trưng). (iii) Mệnh đề 5(a) vẫn ghi "class-dependent noise" không có "binary", và kết luận "λ_k > 0" không suy ra từ giả thiết. (iv) "ranks points as the true posterior does" là theo từng lớp; với accuracy đa lớp cần b_k, λ_k không phụ thuộc k (đúng với PLL đều). | Thân bài: "For weak labels whose naive label depends on the features only through the true label, through a channel with equal off-diagonal entries in each column (PU with labels selected completely at random, binary label noise, semi-supervised learning, PLL with uniform independent distractors, LLP with random bags), …". Mệnh đề 5(a): "… with equal off-diagonal entries in each column of the channel and $M_{k,k}>b_k$, then …; this covers PU-SCAR, binary class-dependent noise with $T_{00}+T_{11}>1$, SSL …". |
| N11 | MINOR (phần còn lại của M2) | `introduction.tex:57-58`; `abstract.tex:8-9` | "invariant floors and ceilings"; "For the sampler as run" không giới hạn ở nhãn cấp mẫu. | Contributions: "for the default sampler on instance-level weak labels and under a homogeneity assumption, floors and ceilings of the label accuracy that hold with high probability". Abstract: "For the default sampler on instance-level weak labels, we show that under a homogeneity assumption on the TFM a one-dimensional response map bounds, with high probability, where a chain can go, so that with two attractors the start matters." |
| N12 | MINOR | `theory.tex:14-16` (Định lý 1 dạng nén) | "The estimator \eqref{eq:predict} then converges almost surely to $\sum_y\pi(y)\,p(y_*\mid x_*,(X,y))$, which equals $p(y_*\mid x_*,X,\W)$ for the transductive chain, and for the inductive one when …". Tổng Σ_y π(y)p(·) không phụ thuộc chuỗi; với chuỗi transductive giới hạn là Σ_y π′(y)p(·) (π′ = p(y ∣ x_*, X, W)), một đại lượng khác. Phát biểu đầy đủ ở phụ lục thì đúng. Ngoài ra `experiments.tex:148-149` "registered in Study 1" cho kết quả TabICL v2 trên PLL thật vẫn mơ hồ (xem 3.4). | "The inductive estimator \eqref{eq:predict} then converges almost surely to $\sum_y\pi(y)\,p(y_*\mid x_*,(X,y))$, which equals $p(y_*\mid x_*,X,\W)$ when the feature distribution carries no information about $p(y\mid x)$ under the prior (in general up to a TV term, \cref{app:proofs}); the transductive chain, in which $x_*$ joins as a free point, converges to $p(y_*\mid x_*,X,\W)$ exactly." Và ở §6.5: "single chain with TabICL v2, the registered TFM ablation of Study 1 (exploratory, \cref{app:s1-tabicl})". |
| N13 | MINOR | `theory.tex:57-58` | Giả thiết 1 lượng hoá trên mọi khối "of at most ⌈n/K⌉ points", gồm cả khối một điểm; chứng minh chỉ dùng khối cỡ ⌊n/K⌋ hoặc ⌈n/K⌉. Giả thiết mạnh hơn cần thiết và càng khó tin. | "every block $B$ of $\lfloor n/K\rfloor$ or $\lceil n/K\rceil$ points with $B\cap F\neq\emptyset$" (điều kiện F ⊄ B khi đó suy ra từ n_F > ⌈n/K⌉). |
| N14 | MINOR | `appendix_proofs.tex:301-306` (Remark (iii)) | "the proven two-sided invariant band … has half-width 0.22": dải chứng minh là [0,676; 1,0] quanh r* = 0,9, phía trên bị cắt ở 1, và ánh xạ (0,45 + 0,5r) không được nêu; ánh xạ còn lại cho 0,32 so với 0,04. Các số biên 0,30; 0,17; 0,10; 0,03 tôi tính lại đúng (chia đều p cho hai số hạng; tối ưu hoá cách chia cho 0,299; 0,173; 0,095; 0,030), nhưng câu không nêu K = 10 và h_max ≤ 1/2. | "… for the map $\Phi(r)=0.45+0.5r$ ($r^*=0.9$), the proven band for $H=60$ sweeps at total failure probability 0.05 reaches 0.22 below $r^*$ (and is cut at 1 above), while the simulated 99% band has half-width 0.02. For a slope-one map with $\delta=0$, $K=10$, $h_{\max}\le\frac12$ and all labels unknown, …". |
| N15 | COSMETIC (lộ rõ, phải sửa) | `main.tex:36-39`; 22 chỗ ở trang 2–9 | `\claim{…}` bị ẩn nhưng khoảng trắng đứng trước còn lại, nên PDF in "weak labels , with averages over starts and over TFMs . (2) Theory: exactness with an exact TFM , stability with an approximate one , and", "(better on 16/18 datasets) . On PLL", "not counting the tuning of L ." | Trong cả hai nhánh: `\newcommand{\claim}[1]{\unskip\ifshowclaims\textsuperscript{…}\fi}`; với `\finaltrue`: `\newcommand{\claim}[1]{\unskip}`. |
| N16 | MINOR | `experiments.tex:60-62` so với Bảng 11 (`tables/s2_means.tex`) và §6.1 | (i) §6.1 định nghĩa "A cell is one run (TFM, dataset, setting, seed)", nhưng "78.9% of 38 cells" là 38 cặp (dataset, mức) (chú thích Bảng 23 nói vậy). (ii) Câu "predicting from the last sample alone (0.843; \cref{tab:s2means})" trỏ tới dòng có tên "Stochastic EM (same TFM)", không được định nghĩa ở đâu. (iii) "an oracle" vẫn chưa định nghĩa trong chính văn. | "in 78.9% of the 38 (dataset, level) pairs"; đổi tên dòng thành "Last sample only (same chain)"; "and an \emph{oracle} (the TFM given the true training labels)". |
| N17 | MINOR | `experiments.tex:71-73`; `make_v2.py:613` | "Final label accuracy" là trung bình 5 vòng cuối; với chuỗi còn đang tăng nó thấp hơn điểm cuối (MiceProtein: 0,437 và 0,442 so với 0,533 và 0,499 ở vòng 30). "reached only … 0.44–0.64 on MiceProtein" vì thế hơi thấp, và các chuỗi này chưa "stall". | Thêm vào câu hoặc chú thích Bảng 6: "(mean of the last five sweeps; two MiceProtein chains were still rising at sweep 30, at 0.50 and 0.53)". |
| N18 | COSMETIC | Bảng 2 (`tables/s3_tests.tex`) | "gap +0.000" (xem kiểm toán m1). | Chuyển phần ghi đè `s3-c5-gap` lên trước khi ghi bảng. |
| N19 | COSMETIC | `experiments.tex:73-74` | "every chain on these datasets ended at 0.93 or above": giá trị nhỏ nhất là 0,925. | "ended above 0.92". |
| N20 | COSMETIC | `experiments.tex:136` | "by the measure of C4$''$ it regressed in 0 of 71 pairs" cho một đại lượng theo dataset vẫn khó hiểu. | "it was never more than 0.02 below the better of $L$ and any TFM's single chain (0 of 71 pairs)". |

**Hai trích dẫn mới.** `heckerman2000` (Heckerman, Chickering, Meek, Rounthwaite, Kadie, "Dependency Networks for Inference, Collaborative Filtering, and Data Visualization", JMLR 1:49–75, 2000): theo hiểu biết của tôi đúng cả thông tin thư mục lẫn nội dung (bài đưa ra "pseudo-Gibbs sampler" trên các conditional học riêng rẽ, có thể không tương thích). `chen2015gibbs` (Chen & Ip, "Behaviour of the Gibbs sampler when conditional distributions are potentially incompatible", J. Stat. Comput. Simul. 85(16):3266–3275, 2015): nội dung bài nói đúng điều bài trích (phân phối dừng của Gibbs quét cố định phụ thuộc thứ tự quét khi các conditional không tương thích). Tôi không có mạng nên **không chắc** về số trang và DOI; cần đối chiếu trước khi nộp. Câu "their stationary distribution depends on the scan order" là đúng; nên thêm rằng Định lý 2 nói về một kernel cố định (kể cả cách quét).

---

## 4. Những gì đã tính lại (giá trị)

### 4.1 Study 3, từ 213 npz (`s3re.py`)

| Đại lượng (khoá) | Trong bài | Tính lại |
|---|---|---|
| Ô / cặp, đủ 3 seed | 213 / 71 | 213 / 71, đủ |
| H1″ trung bình, trung vị, W/T/L | +0,058; +0,003; 13/6/1 | +0,05848; +0,00251; 13/6/1 |
| H1″ theo setting | PLL +0,136; LLP +0,007 | +0,1360; +0,0068 |
| H2″ trung bình, W/T/L | +0,046; 18/2/0 | +0,04575; 18/2/0 |
| `s3-h3-npos`, AUC | 66 trên 213; 0,766 | 66; 0,76613 |
| `s3-c4-n`, tỉ lệ | 10 trên 71 (14,1%) | 10; 14,08% |
| `s3-c4-vs-v1`, `s3-c4-vs-L`, `s3-c4-both` | 2; 9; 1 | 2 (TabICL v2 và TabPFN-3 trên seismic-bumps); 9; 1 (TabICL v2 × seismic-bumps) |
| Chuỗi đơn thua L quá 0,02 | 14 | 14 |
| C5″ gap; số dataset thua quá 0,02; tệ nhất | 0,000; 1; micro-mass 0,645 so với 0,671 | +0,000024; 1; 0,6453 so với 0,6705 |
| Multi-TFM − L | +0,064; 20/0/0 | +0,0637; 20/0/0; nhỏ nhất +0,0034 |
| `s3x-Xwarm-L-pll-wtl` (và LLP) | 6/1/1; LLP "all 12" | PLL 6/1/1 (trung bình +0,030; nhỏ nhất −0,200); LLP 12/0/0 (+0,040) |
| Multi-TFM thụt lùi theo phép đo C4″ | 0 trên 71 | 0 |
| Bảng 2, bảy dòng phải | 0,802 0,803 0,749 0,805; 0,817 0,817 0,371 0,650; 0,786 0,799 0,648 0,720; 0,784 0,798 0,488 0,494 | trùng cả 16 ô |
| Multi-start − L (một TFM) | +0,018 | +0,01796 |

### 4.2 Study 2 (`s2_table_*.csv`, `s2_confirmatory.json`)

| Đại lượng | Trong bài | Tính lại |
|---|---|---|
| `s2-llp-strongest-diff`, CI, thắng | +0,041 [+0,018; +0,072]; 16/18 | +0,04088; json: [0,01807; 0,07156]; bootstrap riêng [0,018; 0,073]; 16/0/2. Baseline có trung bình cao nhất: DLLP-MLP tune 0,7992 (LLP-VAT tune 0,7980) |
| So với DLLP tuyến tính mặc định | +0,052; 16/0/2 | +0,0522; chính xác 16/0/2; theo ngưỡng 0,001: **15/1/2** (N7) |
| `s2-mean-gibbs/hardem/sem-pll6` | 0,851; 0,803; 0,843 | 0,8512; 0,8032; 0,8431 |
| Cùng các trung bình ở q = 0,3 và LLP | 0,862; 0,862; 0,858 và 0,840; 0,827; 0,835 | 0,8621; 0,8615; 0,8579 và 0,8400; 0,8270; 0,8351 |
| Dòng "Stochastic EM" | "last sample alone" | đúng: `methods.py:473-474` gọi `predict_proba(…, last_only=True)` |
| PLL so với CC tune, 4 chữ số | +0,0125 [+0,0065; +0,0192]; +0,0121 [+0,0056; +0,0193] | 0,01245; 0,01209 [0,00561; 0,01934] |

### 4.3 Chuỗi Study 1 và ánh xạ phản ứng (`s1chains.py`, `rmap.py`)

| Đại lượng | Trong bài | Tính lại |
|---|---|---|
| `s1-fin-pfn-mfeat` | 0,16–0,17 | 0,161; 0,169; 0,172 (sau vòng 1: 0,148; 0,163; 0,160; r_start ở lượt chạy lại cùng seed: 0,160–0,169) |
| `s1-fin-pfn-semeion` | 0,34–0,37 | 0,342; 0,346; 0,370 (vòng cuối 0,341–0,387) |
| `s1-fin-pfn-mice` | 0,44–0,64 | 0,437; 0,442; 0,636 (vòng cuối 0,533; 0,499; 0,663) |
| `s1-fin-pfn-opt` | 0,27; 0,89; 0,93 | 0,265; 0,887; 0,934 |
| `s1-fin-icl-min` | 0,93 | 0,925 |
| `th-gain-absmax` | 0,15 | 0,1506 (TabICL v2, optdigits, r = 0,6). TabPFN v2: lớn nhất \|gain\| = 0,111 (semeion, r = 0,3). Ba dataset trong phạm vi: 0,137 |
| `th-nF-four` | 756 đến 3 934 | 756 (MiceProtein), 1 115, 1 395, 3 934 (optdigits) |
| Gain TabPFN v2 âm ở r trung gian, cả 4 dataset | có | mfeat 0,17–0,31; semeion 0,2–0,8; MiceProtein 0,30–0,46; optdigits 0,2–0,3 |
| Gain TabICL v2 dương khắp nơi | có | nhỏ nhất +0,019 |
| Gain TabPFN v2 tại điểm xuất phát (r ≈ 0,17–0,20) | (ngầm định là âm) | −0,005; +0,010; +0,004; +0,001: xấp xỉ 0 |

### 4.4 Phát triển và lý thuyết

| Đại lượng | Trong bài | Tính lại |
|---|---|---|
| `v2-uni-above-all` | 96% | 224/234 = 95,7% (10 ô dưới, 0 ô bằng) |
| Sai số trung vị uniform / chain | 0,073 / 0,016 | 0,07265 / 0,01564 |
| `v2-p1-rmin`, `v2-p1-rmax` | 0,95–0,99 | 0,953–0,989 (9 lượt, 3 dataset) |
| Dự đoán sai / tổng | 7 / 17 | 7 / 17 |
| `th-margin-1k/3k/10k/100k` | 0,30; 0,17; 0,10; 0,03 | 0,3010; 0,1738; 0,0952; 0,0301 (chia đều p; tối ưu 0,2991; 0,1727; 0,0946; 0,0299) |
| Biên ở n_F thật của bốn dataset | "0,17 đến 0,30" | 0,346 (756); 0,285 (1 115); 0,255 (1 395); 0,152 (3 934) |
| Mệnh đề 3: số hạng dẫn đầu | α(1−λ)(1−Cp_c)/((K−1)N) | đúng; tỉ số chính xác/dẫn đầu = (n−K)/(n+Cα) · N²/((N+Cα′)(N+Cα)): 0,603 (n = 100, N = 10, α = 1, K = 10); 0,9899 (n = 1 000, N = 500, α = 0,01) |
| `th-e7f`, `th-e7m` | 15 cấu hình, 1,5·10⁻¹¹; 9 cấu hình, 4·10⁻¹¹ | 15, 1,46·10⁻¹¹; 9, 3,94·10⁻¹¹ |
| `th-band-bound-half`, `th-band-obs-half` | 0,22; 0,02 | 0,2235 (0,9 − 0,676; phía trên cắt ở 1); 0,0195 |

### 4.5 Toán đã dựng lại bằng tay

- **Định lý 1** (nén so với đầy đủ): tính thuận nghịch, bất khả quy, tỉ số Brook của phép đổi chỗ đều đúng; chỉ có chỗ giới hạn của chuỗi transductive (N12).
- **Định lý 2:** Δ = ν + ΔP, co theo hệ số Dobrushin, ‖ν‖₁ ≤ 2ε̄: đúng.
- **Mệnh đề 3:** (a) một dòng, đúng; (b) đúng với u ≤ U; (c) dựng lại E u, E[ul], E u², hai ngoặc, điểm bất động U(L_c+α′)/(N+Cα′) với α′ = α(n−1)/(n−M): đúng. Câu "does not vanish as n grows with N fixed" và "The predictor itself is not the posterior mean under α′" đúng. Phần dư sai (N6).
- **Bổ đề "One sweep" với Ξ_k:** đẳng thức (n_F − f_k)r_k, quy nạp, ba cận đuôi và hằng số c: đúng. Hạn chế mới của Giả thiết 1 (khối ≤ ⌈n/K⌉, F ⊄ B) tương thích với mọi cặp (trạng thái, khối) mà chứng minh dùng, vì n_F > ⌈n/K⌉.
- **Định lý 3:** (a) với max{0, 1−p}, (b), (c) với q_j: đúng. Remark (ii): g(x) ≥ δ + ℓν + λ là đủ; p ≤ (1+3K)exp(…): đúng.
- **Mệnh đề 4:** điểm bất động r* = c₀/(1−b) (cần K ∣ n), α₁ ≠ 1, tự tương quan, đẳng thức phương sai: đúng.
- **Mệnh đề 5:** (a) đúng dưới điều kiện cột và M_kk > b_k; (b) bất đẳng thức đúng, câu áp dụng cho LLP sai (N1).

---

## 5. Đọc lại trang 1–9 như người đọc lần đầu

**Giới hạn trang.** Chính văn kết thúc ở cuối trang 9 (Conclusion, dòng 343), tài liệu tham khảo bắt đầu trang 10: đạt, nhưng không còn dòng trống nào và bản dựng dùng style giả lập. Sửa N15 sẽ trả lại vài dòng.

**Dấu vết nháp.** Không còn TODO, `[C#]`, "??". Còn một `[Pending: licences …]` ở checklist mục 12 (tác giả chủ ý giữ). Số dòng ở lề và "Anonymous Author(s)" là bình thường.

**Bảng và hình (đã dựng ảnh trang 7, 8 ở 110 dpi).** Bảng 1 đọc tốt. Bảng 2 đọc được nhưng chữ khoảng 6pt, hai bảng dính nhau, vẫn in "gap +0.000". Hình 1 rõ; trục r dừng ở 0,9. Đoạn §6.3 bị Hình 1 và Bảng 2 cắt giữa câu (dòng 277 sang 278).

**Có hiểu được mà không cần phụ lục?** Hiểu được ý chính và ba study. Không hiểu được thuật toán một cách chính xác (Algorithm 1 ở phụ lục, trong khi Định lý 3 viết "Run Algorithm 1"), và phải tin các Mệnh đề 1–5 chỉ phát biểu ở phụ lục.

**Thuật ngữ chưa định nghĩa hoặc phải đoán:**

1. "uninformative labelling" (intro).
2. "registered comparator" (abstract) và "A rule applied to development datasets": quy tắc không được mô tả.
3. "CC" xuất hiện ở intro trước khi có trích dẫn.
4. "free" ($y_i$) ở §4: chỉ định nghĩa ở phụ lục; §5 gọi là "points whose label is unknown".
5. "cold start drawn from the normalised likelihood": với LLP nghĩa là gì?
6. "PU data": không giải nghĩa viết tắt; "PU" ở §5 là SCAR hay SAR?
7. H2, H4, H3 có; H1 và H5 không có (Bảng 4 nhắc tới).
8. "oracle"; "Stochastic EM" (Bảng 11).
9. "five curated suites": không nêu tên.
10. "within its published limits" (TabPFN v2): không nêu giới hạn.
11. "the five flagged datasets" (§6.4).
12. "38 cells" trái với định nghĩa "cell" (N16).
13. "the fixed point of the uniform-error map … (median error 0.073)": sai số của cái gì so với cái gì, và điểm bất động được xác định từ đâu (quy tắc ở Phụ lục G).
14. "Chains do not mix at the level of individual labels": không có số nào trong chính văn đỡ cho câu này.
15. "registered ablation" so với "exploratory" cho TabICL v2 ở Study 1.
16. Ký hiệu trùng trong cùng 9 trang: δ (Dobrushin, dung sai, δ₁, δ₂); p (PPD, p(ν, λ), p-value); q (TFM, xác suất distractor); x (mức) và x_i; τ(P) và τ; H (số vòng), (H_δ), H1″; b ở Định lý 3(c) không được giới thiệu.

---

## 6. Rủi ro còn lại

1. **Chưa biên dịch lại.** Tôi không dựng lại PDF; chỉ đối chiếu văn bản PDF với nguồn ở những chỗ đã đọc (khớp) và thời gian sửa tệp (PDF mới hơn mọi tệp nguồn).
2. **Style chính thức.** Với `neurips_2027.sty` thật, 9 trang có thể tràn; `\finaltrue` hiện dừng biên dịch vì `\pending`.
3. **Không có `.git`.** Không kiểm được thứ tự commit, giờ đóng băng (Bảng 4), hay việc script chấm không đổi sau khi có số.
4. **Không chạy lại chuỗi nào.** Mọi npz và CSV thô được coi là đúng. Chỉ tính lại các số được nêu trong đề bài và các số liên quan; chưa rà lại toàn bộ `numbers.tex` của Study 1–2.
5. **Hai trích dẫn mới** chưa đối chiếu nguồn (không dùng mạng); `chen2015gibbs` cần kiểm số trang và DOI.
6. **Giả thiết 1 vẫn chưa có δ.** Người phản biện sẽ hỏi "δ bằng bao nhiêu"; dữ liệu `V2/results/w1_delta.csv` có sẵn để trả lời nhưng bài không dùng.
7. **Câu "More sweeps then do not repair a start in the wrong basin"** chưa có kiểm chứng: không có lượt chạy dài trên ô kẹt, và hai chuỗi MiceProtein còn tăng ở vòng 30.
8. **Tính mới của N1.** Tôi không biết câu về thông tin Fisher có từ bản nháp đầu hay sinh ra khi nén; dù thế nào thì bản hiện tại đang phát biểu sai trong một mệnh đề.
9. **Chú thích LaTeX** còn đường dẫn nội bộ và tên các tệp phản biện; phải lọc nếu nộp kèm nguồn.
10. **Thứ tự sửa đề nghị:** N1, N2 (nội dung); N15, N18 (lộ ngay khi nhìn PDF); N3, N4, N5, N6, N7, N16 (nhất quán); rồi m3, m8, m18 của kiểm toán; cuối cùng là Algorithm 1 và bảng khuyến nghị nếu còn chỗ.
