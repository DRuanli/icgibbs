# REVIEW_V2_TMLR_R3 — Phản biện độc lập theo tiêu chí TMLR (vòng 3)

- **Ngày:** 13/11/2026. **Người phản biện:** độc lập; không tham gia viết bài, không tham gia vòng 1 và vòng 2.
- **Bản được đọc:** `paper/main.pdf` (36 trang, dựng 13/11/2026 14:03; commit `b1e0d69`, gói nguồn `9532538`), nguồn `paper/sections/*.tex`, `paper/tables/*.tex`, `paper/numbers*.tex`.
- **Cách làm:**
  - Đọc toàn bộ chính văn (tr. 1–11) và các phụ lục A, B, C, E, F, G, H, J, K, M, N.
  - Dựng ảnh trang 3, 7–10, 34–35 để kiểm bảng và hình.
  - Giải nén PDF (`qpdf --qdf`) và quét chuỗi để kiểm ẩn danh.
  - Tính lại mọi số mới của S3B, và tách H1″/H2″ theo bối cảnh, từ `S3/results/*.csv` bằng script riêng chạy `python3 -I`: `/tmp/claude-0/-home-claude/643477cb-9e2c-5e3b-9dd1-1dba7a8f3cd0/scratchpad/rev/scripts/s3b_check.py`, `/tmp/claude-0/-home-claude/643477cb-9e2c-5e3b-9dd1-1dba7a8f3cd0/scratchpad/rev/scripts/s3_split.py`, `/tmp/claude-0/-home-claude/643477cb-9e2c-5e3b-9dd1-1dba7a8f3cd0/scratchpad/rev/scripts/s3_h2split.py`.
  - Đối chiếu giờ đóng băng với `git log`. Kho là bản clone nông, lịch sử chỉ có từ 05/11.
  - Đọc mã baseline CPU (`G4/code/icgibbs/baselines_cpu.py`), `S3/S3B_PLAN.md`, `S3/S3B_REPORT.md`, `S3/STUDY3_PLAN.md`, `G5/EXPERIMENT_PROTOCOL.md`.
  - Không sửa tệp nào của kho ngoài báo cáo này.
- **Mức độ:**
  - **Nghiêm trọng:** làm đổ một kết luận chính.
  - **Lớn:** phải sửa thì bài mới đạt tiêu chí 1 của TMLR.
  - **Nhỏ:** sai hoặc thiếu cục bộ.
  - **Trình bày:** chỉ ảnh hưởng tới việc đọc.

---

## (a) Tóm tắt bài trong 5 dòng

1. **Ý tưởng.** Dưới tính hoán đổi được, phân phối của một nhãn ẩn khi biết mọi nhãn khác là PPD, với các điểm còn lại làm ngữ cảnh (Bổ đề 1). Vì vậy một TFM đông cứng đóng vai full conditional của một Gibbs sampler trên nhãn ẩn, và nhãn yếu (PLL, LLP) đi vào qua hàm hợp lý.
2. **Phương pháp.** Gibbs theo khối, không hiệu chỉnh (K = 10, T = 30), dự đoán Rao–Blackwell. Có thêm bản *multi-start* (ba điểm xuất phát: warm, learner L đã tune, cold) và bản *multi-TFM* (trung bình đều 3–4 TFM).
3. **Lý thuyết.**
   - Đúng tuyệt đối khi TFM đúng (Định lý 1); cận nhiễu loạn (Định lý 2); độ lệch dự đoán của khối không hiệu chỉnh trong mô hình liên hợp (Mệnh đề 3).
   - Sàn và trần của độ đúng nhãn dưới giả thiết đáp ứng đồng nhất (Định lý 3). Chính bài thừa nhận định lý này không cho con số dùng được ở cỡ dữ liệu của bài.
4. **Ba nghiên cứu ghi trước.**
   - Study 1 (TabPFN v2): trượt tiêu chí.
   - Study 2 (TabICL v2, 28 dataset mới): qua mọi kiểm định so với các learner nơ-ron chuyên biệt.
   - Study 3 (20 dataset, 4 TFM): xác nhận multi-start hơn chuỗi đơn và multi-TFM hơn một TFM chọn ngẫu nhiên; trượt hai tiêu chí (H3″, C4″).
5. **Phân tích bổ sung S3B**, ghi trước nhưng sau khi Study 3 đã chấm:
   - multi-TFM hơn GBDT tự huấn luyện và PL-kNN, cả hai không tune;
   - chuỗi đơn không hơn rõ GBDT;
   - chạy 100 vòng quét hiếm khi sửa được chuỗi kẹt, nhưng độ chính xác vẫn tăng (+0.053), nên dự đoán P22 sai.

---

## (b) Hai tiêu chí của TMLR

### Tiêu chí 1. Các khẳng định có được bằng chứng chính xác, thuyết phục và rõ ràng ủng hộ không? → **Đạt một phần**

**Phần đứng vững.**
- Bổ đề 1, Định lý 1–2 và Mệnh đề 3–4 đúng như phát biểu ở Phụ lục A. Hai vòng trước đã dựng lại đại số; tôi không thấy lỗi mới trong chứng minh.
- Các kết quả xác nhận đứng được:
  - Study 2: Bảng 1, mọi kiểm định ghi trước đều qua.
  - Study 3: H1″, H2″, C5″ (Bảng 2).
- Thất bại được báo ngang hàng với thành công: H3″, C4″, P22, Study 1.
- Mọi con số S3B tôi tính lại đều khớp (mục 6).

**Phần chưa đứng.** Một số câu ở abstract, introduction, related work, theory và conclusion rộng hơn bằng chứng:
- Định lý 3 được gán cho "the sampler as run", trong khi nó không phủ LLP và không phủ context bị cắt. Ở Study 3 nó chỉ áp được cho 4/20 dataset.
- "A stalled chain is therefore a mixture of a wrong basin and slow mixing" không có dữ liệu nào tách được hai thành phần đó.
- Câu "this shows that chains settle at a fixed point of a one-dimensional map" dựa trên một phép kiểm vòng tròn.
- "Averaging over TFMs improved on a single TFM" thực ra là so với một TFM chọn ngẫu nhiên.
- "With a robust TFM" là nhãn chỉ biết được sau khi có kết quả.
- Baseline GBDT cho PLL suy biến, nhưng vẫn đứng trong abstract.
- Ở PU, nhiễu nhãn và SSL, phương pháp kém naive TFM ở mọi mức, nhưng bài viết là "did not improve".

Phần lớn các điểm trên sửa được bằng câu chữ và phân tích lại dữ liệu đã lưu. Không cần thêm GPU.

### Tiêu chí 2. Có người trong độc giả của TMLR quan tâm tới kết quả không? → **Đạt**

- Kết nối "PPD = full conditional" gọn và dùng được ngoài bài toán nhãn yếu. Độc giả làm TFM, weak supervision và MCMC với conditional học được sẽ quan tâm.
- Cách làm có nhiều điểm hiếm và có giá trị cho cộng đồng:
  - ghi trước giả thuyết và script chấm;
  - một replication thật trên dataset mới;
  - báo kết quả âm ngang hàng với kết quả dương: Study 1, H3″, C4″, P22.
- TMLR không đòi kết quả tốt nhất. Điều này đủ cho tiêu chí 2.

---

## (c) Khuyến nghị: **Major revision** (sửa lớn, có đường tới chấp nhận)

1. **Không có lỗi Nghiêm trọng.** Không phát hiện nào làm đổ kết luận xác nhận, và không việc nào cần GPU.
2. **Có 10 phát hiện mức Lớn.**
   - 6 trong số đó là khẳng định ở abstract, introduction hoặc conclusion vượt quá bằng chứng. Đây đúng là thứ tiêu chí 1 của TMLR dùng để quyết định.
   - 5 mục đã được nêu ở vòng 2 mà vẫn chưa sửa: N8, N9, N11, N12, NeurIPS 3.3. Lý do "không đủ chỗ" không còn từ khi chuyển sang TMLR.
3. **Hai việc phải chạy lại bằng CPU trên các lần chia đã lưu, hoặc phải rút khỏi abstract:**
   - baseline GBDT cho PLL, vốn suy biến và được thiết kế khi đã biết điểm của Study 3;
   - một bộ so sánh cùng thông tin cho multi-TFM: ensemble của L, hoặc chuỗi đơn trung bình với dự đoán của L.
4. **Việc ghi trước phải kiểm chứng được.** Đây là luận điểm phương pháp trung tâm của bài, nên cần một kho ẩn danh truy cập được (protocol, script chấm, log commit, mã băm kết quả).
5. **Nếu sửa đúng các mục Lớn**, tôi sẽ chuyển sang **Accept**, có thể kèm minor revision.

---

## (d) Các phát hiện

**Đếm theo mức:** Nghiêm trọng **0** · Lớn **10** (F1–F10) · Nhỏ **12** (F11–F22) · Trình bày **4** (F23–F26).

### F1 [Lớn] Định lý 3 được viện dẫn ngoài phạm vi giả thiết của nó

**Vị trí:** `abstract.tex:7–8`; `introduction.tex:26–30, 41–43`; `related.tex:22–23`; `method.tex:52–53`; `limitations.tex:12`.

**Phạm vi thật của định lý.** Định lý 3 chỉ phủ nhãn yếu cấp mẫu, khối không hiệu chỉnh, và context không bị cắt (`theory.tex:46–49`; Remark (iv), (v) ở Phụ lục A.6).

**Những câu dùng nó rộng hơn:**
- Abstract: "For the sampler as run, a homogeneity assumption on the TFM gives floors and ceilings for the label accuracy of a chain, so that where a chain starts can matter."
- Introduction: "it bounds where the sampler as run can go (Theorem 3); with two attractors the start matters. We therefore average chains from several starts".
- Contributions: "for the default sampler under a homogeneity assumption, invariant floors and ceilings of the label accuracy".
- Related work: "our floors and ceilings (Theorem 3) are a counterpart for the sampler as run".
- Method: "Section 5 shows that where a chain ends can depend on where it starts". Câu này được dùng để biện minh multi-start cho cả LLP.
- Limitations: "Theorem 3 covers the default sampler but not LLP or subsampled contexts". Câu này tự mâu thuẫn, vì sampler mặc định *chính là* sampler cắt context trên 2,000 điểm và dùng bag-block cho LLP.

**Lượng hoá trên Study 3.**
- 12/20 dataset là LLP.
- Trong 8 dataset PLL, 4 dataset (har, gesture-phase, texture, hiva-agnostic; n_train ≥ 2,678) có context 0.9·n_train > 2,000, tức bị cắt.
- Vậy Định lý 3 chỉ phủ 4/20 dataset của Study 3, và ở cả 4 nó "certifies nothing".

**Chữ "invariant" sai.** Kết luận của định lý chỉ giữ với xác suất ≥ 1 − Hp trong H vòng quét.

**Sửa:**
- Abstract: "For the default sampler on instance-level weak labels with unsubsampled contexts, a homogeneity assumption on the TFM gives floors and ceilings that hold with high probability; they show qualitatively that where a chain starts can matter, but are vacuous at our sample sizes."
- Contributions: "floors and ceilings of the label accuracy that hold with high probability, for instance-level weak labels and unsubsampled contexts".
- Related work: "a counterpart, in a model of the TFM, for the sampler as run on instance-level weak labels".
- Method: "Under the model of Section 5.2 (instance-level labels), where a chain ends can depend on where it starts; for LLP we observed this only empirically (four Study 3 cells converged to the exchanged labelling)."
- Limitations: "Theorem 3 covers the default sampler only for instance-level weak labels and contexts of at most 2,000 points (4 of the 20 Study 3 datasets)."

### F2 [Lớn] Bảo đảm của Định lý 1–2 và Bổ đề 1 bị chuyển sang sampler mặc định; phát biểu nén của Định lý 1 sai

**Vị trí:** `related.tex:20–21`; Bảng 25 (Phụ lục H); `introduction.tex:12–16`; `conclusion.tex:2`; `experiments.tex:226–227`; `theory.tex:14–16`.

**(i) Related work và Bảng 25.**
- Câu: "Our sampler conditions every latent label on all others, uses the weak-label likelihood, and has guarantees for incompatible conditionals".
- Sampler mặc định cập nhật theo khối: mỗi nhãn chỉ được điều kiện trên các khối khác, và context bị cắt còn 2,000 điểm.
- Bảo đảm của Định lý 2 chỉ áp cho các biến thể chính xác (single-site, swap-MH, block-MH). Với khối không hiệu chỉnh, chính bài viết "for it the bound above is useless" (`theory.tex:36–37`).
- Vì vậy dấu ✓ ở cột "inference over latent labels with guarantees" của Bảng 25 không đúng cho phương pháp đã chạy ở mọi Study. Đây là mục 3.3 của vòng 1, chưa sửa.

**(ii) Introduction và Conclusion.**
- Introduction: "A TFM is already a Gibbs sampler for unknown labels" và "A forward pass of a frozen TFM is therefore the full conditional".
- Conclusion: "a frozen TFM's predictive is the full conditional".
- Cả hai chỉ đúng khi q = p (giả thiết A1). Ngay câu trước, bài tự nói TFM chỉ xấp xỉ PPD.

**(iii) §6.7.** "The MH-corrected block sampler, exact when the TFM is" còn cần thêm điều kiện context không bị cắt. Dòng cuối Bảng 28 dùng optdigits, nơi context bị cắt.

**(iv) Phát biểu nén của Định lý 1** (`theory.tex:14–16`): "The estimator (3) then converges almost surely to $\sum_y\pi(y)p(y_*\mid x_*,(X,y))$, which equals $p(y_*\mid x_*,X,W)$ for the transductive chain".
- Câu này sai như đang viết: chuỗi transductive không dùng (3), và giới hạn của nó là $\sum_y\pi'(y)p(\cdot)$ với $\pi'=p(y\mid x_*,X,W)$.
- Phát biểu đầy đủ ở Phụ lục A, phần (c), thì đúng.
- Đây là N12 của vòng 2, chưa sửa.

**Sửa:**
- Related work: "…and its exact variants (single-site, swap-MH, block-MH) have guarantees for incompatible conditionals; the default uncorrected-block sampler has only the prediction-level analysis of Proposition 3."
- Bảng 25: đổi ✓ thành "✓ (exact variants only)".
- Introduction và Conclusion: "is, for an exact TFM, the full conditional … and approximately so for a real one".
- §6.7: "exact when the TFM is and the context is not subsampled".
- Định lý 1: dùng câu thay thế N12 của vòng 2, tách hai giới hạn inductive và transductive.

### F3 [Lớn] "A stalled chain is therefore a mixture of a wrong basin and slow mixing" không có bằng chứng

**Vị trí:** `theory.tex:93–97` (tr. 5–6).

**Vấn đề.**
- B3 chỉ cho thấy trộn chậm: 19/72 ô còn đang tăng ở vòng 100, và TabPFN-3 đi từ 0.40 lên 0.49 (macro `s3b-b3-rising`, tôi tính lại được 19).
- Không phép đo nào trên Study 3 cho thấy một ô nằm trong "wrong basin" theo nghĩa của Định lý 3:
  - không đo response map trên Study 3 (`S3B_PLAN.md` §6 loại việc này);
  - δ chưa được ước lượng;
  - 4/8 dataset PLL nằm ngoài phạm vi của định lý.
- 30 ô "không sửa được" có thể chỉ là những ô trộn còn chậm hơn.
- `S3B_PLAN.md` §5 tự cam kết: nếu P22 sai thì "kết luận của bài về 'lưu vực' phải viết yếu đi". Bài lại viết mạnh hơn bằng chữ "therefore".

**Sửa:** "Real chains also mix slowly: … 19 of 72 cells were still rising after 100 sweeps. Our data cannot separate a wrong basin from slow mixing; Theorem 3 describes only the former, and only in a model of the TFM." Câu này khớp với `S3B_REPORT.md` §3.

### F4 [Lớn] Phép kiểm "chain-seeded" vòng tròn mà vẫn được đọc là bằng chứng (N8 vòng 2, chưa sửa)

**Vị trí:** `experiments.tex:86–90`; Bảng 23 (P8, P9); Hình 5.

**Câu trong bài:** "Those marginals come from the chain whose end is predicted, so this shows that chains settle at a fixed point of a one-dimensional map and that the placement of errors matters".

**Vì sao vòng tròn.**
- Lỗi được gieo theo biên của chính chuỗi, nên trạng thái gieo gần trạng thái dừng của chuỗi.
- Gain ≈ 0 gần độ đúng cuối vì thế là hệ quả của tính dừng, bất kể mô tả một chiều có đúng hay không.
- Chữ "so" rút ra một kết luận ngược hướng với tiền đề vừa nêu.

**Kiểm chứng còn thiếu.** Một phép kiểm không vòng tròn đã được đề xuất ở cả vòng 1 và vòng 2, và chỉ cần dữ liệu đã lưu. Ví dụ:
- dùng map của chuỗi A để dự đoán chuỗi B trong cùng ô; hoặc
- dùng map gieo đều để dự đoán "kẹt / không kẹt" từ độ đúng ban đầu, rồi báo AUC.

**Sửa:** "Those marginals come from the chain whose end is predicted, and a state that resembles the chain's final state is nearly stationary by construction; the agreement therefore shows that the placement of errors matters, not that the one-dimensional description, or Theorem 3, forecasts anything." Nếu muốn giữ khẳng định cũ, thêm phép kiểm chéo giữa các chuỗi và ghi là exploratory.

### F5 [Lớn] "with a robust TFM" và "specialised learners" khái quát quá kết quả xác nhận (N9 vòng 2, chưa sửa)

**Vị trí:**
- Tiêu đề §6.2 (`experiments.tex:53`): "One chain with a robust TFM outperforms specialised learners".
- `conclusion.tex:3–4`: "In pre-registered tests a single chain with a robust TFM outperformed specialised learners on new datasets".
- `abstract.tex:9–11`.

**(i) "Robust" chỉ biết sau sự kiện.** TabICL v2 được chọn sau khi đã thấy kết quả Study 1. Ở Study 3, chuỗi đơn với các TFM mới hơn lại thua L:
- TabPFN-3: −0.118 (11/0/9);
- TabDPT: −0.026 (Bảng 16).

Ngay TabICL v2 cũng kẹt trên vài dataset (Bảng 15):
- soybean: chuỗi đơn 0.589, multi-start 0.798;
- micro-mass: chuỗi đơn 0.548, multi-start 0.671.

**(ii) "Specialised learners" của Study 2 đều là learner nơ-ron** (MLP hoặc tuyến tính). Trên dataset Study 3, S3B cho thấy:
- GBDT tự huấn luyện, không tune, hơn L đã tune (DLLP-MLP, "the strongest LLP baseline" của Study 2) ở 9/12 dataset LLP. Tôi tính từ `s3b_cpu_datasets.csv`: L − GBDT trên LLP là 2/1/9, trung bình −0.013.
- GBDT ngang chuỗi đơn: 0.801 so với 0.796, 6/1/5.

Bộ so sánh của Study 2 cho LLP vì thế là yếu. Kết quả xác nhận vẫn đúng, nhưng tầm của nó hẹp hơn chữ "specialised learners".

**Sửa:**
- Tiêu đề §6.2: "Study 2: one chain with TabICL v2 outperformed specialised neural learners".
- Conclusion: "In Study 2 a single chain with TabICL v2 outperformed the specialised neural learners we ran; on the Study 3 datasets it was level with untuned tree-based self-training on LLP."
- Abstract: "specialised neural learners".
- Bỏ chữ "robust", hoặc định nghĩa nó bằng một tiêu chí đo được trước khi chạy và nói rõ tiêu chí đó cần nhãn thật.

### F6 [Lớn] "averaging over TFMs improved on a single TFM": H2″ chỉ so với một TFM chọn ngẫu nhiên

**Vị trí:** `abstract.tex:13–14`; `introduction.tex:29–31`; Bảng 2 ("multi-TFM > one TFM"); Bảng 19.

**Vấn đề.**
- δ₂ so multi-TFM với *trung bình trên các TFM* của multi-start (`experiments.tex:30–31`: "the expected score of a TFM picked at random").
- Trung bình đều của vài bộ dự đoán có độ chính xác tương đương gần như luôn hơn thành viên trung bình. H2″ vì thế là một giả thuyết yếu: nó không cho biết multi-TFM có hơn việc dùng một TFM tốt cố định hay không.
- So sánh duy nhất với một TFM cố định là thăm dò, và không có ý nghĩa thống kê: multi-TFM − multi-start của TabICL v2 = +0.016, p = 0.08 (`experiments.tex:155–157`).
- Abstract lại đọc như multi-TFM hơn bất kỳ TFM đơn nào.

**Sửa:**
- Abstract: "…averaging over TFMs improved on the expected score of a randomly chosen TFM (+0.046, one-sided p < 0.001) and matched the best TFM of each dataset; it was not significantly better than TabICL v2 alone (exploratory, +0.016, p = 0.08)."
- Bảng 2: "multi-TFM > average single TFM".

### F7 [Lớn] Baseline phi nơ-ron: GBDT cho PLL suy biến, được thiết kế khi đã biết kết quả, không tune, nhưng đứng ở abstract

**Vị trí:** `abstract.tex:15–18`; `experiments.tex:175–186`; Bảng 4, Bảng 22; `G4/code/icgibbs/baselines_cpu.py:60–75`; Phụ lục F.

**(i) GBDT-PLL suy biến.**
- Accuracy của nó so với PL-kNN, cũng không tune (Bảng 22):

  | Dataset | Số lớp | GBDT-PLL | PL-kNN |
  |---|---|---|---|
  | soybean | 19 | 0.130 | 0.679 |
  | micro-mass | 20 | 0.122 | 0.209 |
  | vowel | 11 | 0.270 | 0.365 |
  | cnae-9 | 9 | 0.373 | 0.522 |

- Cách cài dễ sụp về các lớp đông:
  - nhãn giả ban đầu là một ứng viên ngẫu nhiên; ở q = 0.6 với C = 19, tập ứng viên trung bình có khoảng 11.8 lớp, nên chỉ khoảng 8% nhãn giả đúng;
  - HistGradientBoosting mặc định được huấn luyện trên 2 fold, rồi lấy argmax trong tập ứng viên, lặp 3 vòng;
  - với n_train khoảng 400–700 và `min_samples_leaf` mặc định là 20, cây hầu như không học được gì.
- Bài không báo độ chính xác của lớp đông nhất, nên người đọc không tự kiểm được.
- Chênh gộp +0.135 của multi-TFM − GBDT phần lớn đến từ chỗ này. Tôi tính lại: trên PLL +0.289; trên LLP chỉ +0.032 (trung vị +0.017).

**(ii) Thời điểm thiết kế.**
- S3B được ghi trước lúc 07:21 UTC ngày 12/11 (commit `d9e6062`).
- Study 3 đã được chấm bốn giờ trước đó (commit `aa0775d`, 03:18 UTC).
- Vậy người thiết kế baseline đã biết điểm của multi-TFM trên từng dataset. Abstract chỉ ghi "registered after Study 3 was scored".

**(iii) Không cân xứng về tuning và độ phủ.**
- L được tune trên 20 cấu hình; hai baseline CPU chỉ chạy một cấu hình.
- Không có phương pháp LLP phi nơ-ron đã công bố (ví dụ InvCal, ∝SVM), cũng không có baseline PLL phi nơ-ron mạnh như IPAL.
- PL-kNN không có trích dẫn trong bài, dù kế hoạch S3B có trích Hüllermeier & Beringer (2006).

**(iv) Kiểu kiểm định không được nói.** Chính văn không nói kiểm định B1 là Wilcoxon *hai phía*, Holm trên *ba* phép so (`S3B_PLAN.md` §3). Người đọc §6.1 sẽ tưởng đó là kiểm định một phía như của Study 3.

**Sửa:**
- (a) Chạy lại GBDT-PLL bằng CPU trên các lần chia đã lưu:
  - dùng nhãn mềm: nhân bản mỗi điểm theo từng ứng viên với trọng số 1/|S_i|, rồi cập nhật trọng số kiểu PRODEN;
  - tune 20 cấu hình như L;
  - báo kèm độ chính xác của lớp đông nhất;
  - thêm IPAL nếu có mã;
  - ghi là "post hoc, registered after Study 3 was scored, designed with knowledge of its results".
- (b) Nếu không chạy lại: bỏ "beat untuned tree-based self-training on 19 of 20 datasets" khỏi abstract, hoặc viết tách: "on LLP 12/12 (median +0.017 AUC); on PLL the tree-based baseline was near-degenerate".
- (c) Ghi "two-sided, Holm over three comparisons" và thêm trích dẫn cho PL-kNN.

### F8 [Lớn] "What to run" và Bảng 5: sai nhãn bằng chứng, thiếu các cấu hình rẻ hơn mà phân tích của chính bài ủng hộ, thiếu bộ so sánh cùng thông tin

**Vị trí:** `experiments.tex:205–221`; `tables/s3b_recommend.tex:8`; Bảng 18; Bảng 19.

**(i) Nhãn bằng chứng của hàng "Single chain" sai.**
- Hàng này ghi "confirmatory (Studies 1–2)".
- Nhưng số trên hàng là trung bình của 4 TFM trên dataset Study 3 (PLL 0.589, trong đó TabPFN-3 chỉ 0.371).
- Study 1 trượt tiêu chí; Study 2 chỉ xác nhận với TabICL v2.
- Bảng 19 ở phụ lục ghi đúng: "confirmatory with TabICL v2 (Study 2); criteria missed with TabPFN v2".

**(ii) Bảng 5 bỏ các cấu hình rẻ hơn.** Theo Bảng 18 và §6.4:

| Cấu hình | So với multi-start | Ghi chú |
|---|---|---|
| Chuỗi khởi từ L ("chain started from L") | +0.001 | rẻ hơn 3 lần |
| Chuỗi đơn trung bình với dự đoán của L | +0.008, p = 0.26 | rẻ hơn 3 lần |
| Multi-TFM so với multi-start của riêng TabICL v2 | +0.016, p = 0.08 | multi-TFM đắt hơn 3–4 lần |

Không cấu hình nào trong số này có trong Bảng 5. Vì vậy khuyến nghị "start chains from a trained learner and average over TFMs" (9–12 chuỗi cộng với tune L) không phải lựa chọn hiệu quả nhất theo bằng chứng của chính bài.

**(iii) Thiếu bộ so sánh cùng thông tin.**
- Lợi thế của multi-TFM so với L (+0.064, 20/0/0) là so một ensemble *có chứa* chuỗi khởi từ L với chính L.
- Bộ so sánh cùng thông tin chưa có: ensemble của L qua seed hoặc cấu hình; hoặc trung bình chuỗi đơn trên các TFM cộng với dự đoán của L.
- Việc này đã được đề xuất ở vòng 1 và vòng 2.

**(iv) Cột "Trained parts" thiếu.** Hàng "Multi-start" chỉ ghi "tuned L", trong khi với LLP còn có DLLP tuyến tính của warm start.

**Sửa:**
- Đổi ô Evidence của hàng Single chain thành "confirmatory with TabICL v2 only (Study 2)".
- Thêm ba hàng, ghi là exploratory: "chain from L", "single chain + L's prediction", "TabICL v2 multi-start", mỗi hàng có điểm và chi phí.
- Thêm hàng "ensemble of L (5 seeds)": chạy bằng CPU trên dữ liệu đã lưu.
- Viết lại khuyến nghị theo thứ tự chi phí: "If a tuned learner is available, a single chain started from it, or a single chain averaged with its prediction, gave the same mean score as three starts (exploratory); averaging over TFMs added +0.016 over TabICL v2 alone (not significant)."

### F9 [Lớn] Tác hại ở PU, nhiễu nhãn và SSL được viết thành "did not improve"

**Vị trí:** `limitations.tex:16–18`; `theory.tex:102–103` ("nothing is lost"); Phụ lục M (`appendix.tex:~548–552`); Bảng 35.

**Số trong Bảng 35** (In-Context Gibbs − naive TFM):

| Bối cảnh | Mức | Chênh | W/T/L |
|---|---|---|---|
| Nhiễu phụ thuộc lớp | r = 0.2 | −0.013 AUC | 1/0/9 |
| Nhiễu phụ thuộc lớp | r = 0.4 | −0.054 AUC | 1/0/9 |
| SSL | 5% có nhãn | −0.041 AUC | 0/0/10 |
| SSL | 20% có nhãn | −0.016 AUC | 0/0/10 |
| PU-SCAR | c = 0.1 | −0.033 (log loss đã hiệu chỉnh) | 5/0/5 |
| PU-SCAR | c = 0.3 | −0.028 (log loss đã hiệu chỉnh) | 2/0/8 |

**Vấn đề.**
- Phương pháp *kém* naive TFM ở mọi mức đã đăng ký, chứ không chỉ "không cải thiện".
- Phụ lục M đọc D3 ("stays below +0.02 AUC") như một dự đoán đúng, trong khi một phương pháp kém hơn cũng thoả ngưỡng đó.
- Theo Mệnh đề 3, khối không hiệu chỉnh co số đếm nhãn về prior. Đây là một lời giải thích khả dĩ cho tác hại ở SSL, đáng nêu trong bài.

**Sửa:**
- Limitations: "for PU, label noise and semi-supervised learning, where the naive labels lose no information, the sampler was worse than the naive TFM at every registered level (e.g. −0.054 AUC under class-dependent noise at r = 0.4, 1/0/9; −0.041 with 5% labelled, 0/0/10)".
- Theory: "nothing is lost by the naive labels, and the sampler can only add the bias of uncorrected blocks".
- Phụ lục M: nói rõ D3 cũng thoả khi phương pháp kém hơn.

### F10 [Lớn] Việc ghi trước, luận điểm phương pháp trung tâm, không kiểm chứng được từ bản nộp

**Vị trí:** Phụ lục B (`appendix.tex:9`: "In the anonymised repository each freeze is a commit"); Phụ lục N ("available in an anonymised repository and will be released"); `experiments.tex:160`.

**Vấn đề.**
- Chữ "pre-registered" có mặt ở abstract, introduction, contributions và mọi bảng kết quả. Nhưng bản nộp không có đường dẫn tới kho ẩn danh, protocol đóng băng hay script chấm.
- Giờ commit do tác giả kiểm soát.
- Phần tôi kiểm được trong bản clone nông:
  - Study 3 đóng băng tại `c9009e8`, 10/11 04:21:22 UTC: khớp Bảng 7.
  - S3B đóng băng tại `d9e6062`, 12/11 07:21:22 UTC, code `b59ef14` lúc 07:27:00: khớp Phụ lục F ("six minutes later").
- Phần không kiểm được: các mốc 26/10 và 1/11 của Study 1–2, vì nằm trước lịch sử của bản clone.
- Câu "Reviewers of an earlier draft asked for …" gợi ý bài đã từng được nộp ở đâu đó. Thực ra đó là các vòng phản biện nội bộ.

**Sửa:**
- Nộp kèm (supplementary, hoặc anonymous.4open.science) bản ẩn danh của protocol, script chấm, log commit có chữ ký và mã băm kết quả.
- Trước khi chia sẻ, lọc khỏi lịch sử git metadata tác giả và tên chủ kho.
- Ghi rõ giới hạn: "commit times are author-controlled; signed commits and result hashes are provided".
- Đổi câu ở §6.5 thành "Internal reviews of an earlier draft asked for …".

### F11 [Nhỏ] Abstract còn thiếu cân bằng ở bốn chỗ

**Vị trí:** `abstract.tex:9–18`.

1. **Đơn vị.** "+0.058" và "+0.046" không có đơn vị; đó là chênh gộp AUC (LLP) và accuracy (PLL). Đây là m8 vòng 2: đã sửa ở chú thích Bảng 2 nhưng chưa sửa ở abstract.
2. **Comparator.** Comparator của "+0.052 AUC over the registered comparator" là DLLP tuyến tính không tune. Abstract của vòng 2 có thêm "+0.041 over the strongest baseline", nay bị bỏ. Đây là một bước **thoái lui**.
3. **PLL mong manh.** PLL được viết "narrowly", nhưng không nói rằng kiểm định ở q = 0.6 không qua khi dùng chuỗi thứ hai (p_Holm = 0.074, Bảng 29).
4. **P22 mới nêu một vế.** "rarely repaired a stalled chain" chỉ nêu vế đúng. Vế sai (lợi trung bình +0.053, trái với dự đoán ghi trước là dưới 0.03) không có trong abstract, trái với chủ trương báo thất bại ngang thành công.

**Sửa:**
- "(+0.058 in pooled AUC/accuracy …)";
- "(+0.052 AUC over the registered comparator, linear DLLP; +0.041 over the strongest baseline)";
- "…and, narrowly, for partial labels (one test did not survive a second chain)";
- "…rarely repaired a stalled chain, although accuracy kept rising (+0.053 on average, against our registered prediction)".

### F12 [Nhỏ] Báo cáo thống kê còn thiếu

**Vị trí:** §6.1, §6.4, §6.5; Bảng 2; Bảng 18.

1. **Quy tắc hoà.** Ngưỡng hoà 0.001 chỉ dùng để đếm W/T/L; kiểm định Wilcoxon dùng mọi chênh khác 0 (`S3/s3_report.py:46–51`). Bài cần nói điều này.
2. **Một phía hay hai phía.** Chính văn không nói kiểm định B1 là hai phía, Holm trên 3 phép so (xem F7). Phép so mô tả multi-start − L (+0.018, p = 0.14) cũng là hai phía (`STUDY3_PLAN.md` §4), và cũng không được nói.
3. **Độ phân tán theo seed** không được báo ở đâu (mục [should] 15 của vòng 1).
4. **Tách theo bối cảnh.** Việc gộp AUC với accuracy đã được ghi trước và nay đã được nói rõ. Nên thêm bảng tách theo bối cảnh; bảng này *củng cố* kết luận. Tôi tính từ `s3_pairs.csv` và `s3b_cpu_datasets.csv` (mô tả, chưa hiệu chỉnh, một phía):

   | Giả thuyết | Bối cảnh | Trung bình | Trung vị | W/T/L | p |
   |---|---|---|---|---|---|
   | H1″ | LLP | +0.007 | +0.002 | 7/4/1 | 0.021 |
   | H1″ | PLL | +0.136 | +0.098 | 6/2/0 | 0.016 |
   | H2″ | LLP | +0.021 | +0.016 | 12/0/0 | < 0.001 |
   | H2″ | PLL | +0.083 | +0.025 | 6/2/0 | 0.004 |

5. **Xem tạm.** Study 3 và S3B mỗi nghiên cứu có một lần xem tạm, và đều đã được khai. Không có vấn đề.

**Sửa:** thêm một câu về quy tắc hoà và kiểu kiểm định ở §6.1; thêm bảng tách theo bối cảnh (ghi là mô tả); thêm độ lệch chuẩn theo seed vào Bảng 15 và Bảng 22, hoặc dùng bootstrap phân cấp.

### F13 [Nhỏ] Mệnh đề 5 và đoạn "When can the sampler help?" (N10 vòng 2, mới sửa một phần)

**Vị trí:** `theory.tex:100–104`; `appendix_proofs.tex:357–364`; `introduction.tex:9–10`.

1. **Tiền đề không đủ.** "When weak labels are generated independently of the features given the true label, the naive target ranks points as the true posterior does". Để kết luận đúng cần thêm hai điều kiện: các phần tử ngoài đường chéo của mỗi cột kênh bằng nhau, và M_kk > b_k.
2. **Mệnh đề 5(a) còn hở.**
   - Đã thêm điều kiện về cột, nhưng kết luận "λ_k > 0" vẫn không suy ra được từ giả thiết.
   - Danh sách áp dụng vẫn ghi "class-dependent noise", trong khi chỉ đúng với nhãn nhị phân và T00 + T11 > 1.
   - "PLL with independent distractors" chỉ được chứng minh khi mọi lớp sai có cùng xác suất q.
3. **Thân bài nói mạnh hơn phụ lục.** Thân bài viết "For PLL they carry less Fisher information" như một kết luận đã chứng minh, trong khi Mệnh đề 5(b) chỉ có "the inequality was strict in every case we computed".
4. **Introduction dùng Mệnh đề 5 cho TFM.** Câu: "A TFM given one plausible label per point uses only part of what candidate sets and label proportions say". Nhưng con số 3% là của likelihood composite trong mô hình logistic, không phải của TFM.

**Sửa:** dùng câu thay thế N10 của vòng 2; thêm "(numerically, in every case we computed)"; ở introduction viết "A learner given one plausible label per point … (Proposition 5, computed for a logistic model)".

### F14 [Nhỏ] Giả thiết 1: lượng từ quá rộng, và δ chưa từng được ước lượng

**Vị trí:** `theory.tex:57`; §6.3; Phụ lục G.

1. **Lượng từ.** "every block B of at most ⌈n/K⌉ points" bao gồm cả khối chỉ có một điểm. Chứng minh chỉ dùng khối cỡ ⌊n/K⌋ hoặc ⌈n/K⌉. Đây là N13 vòng 2, chưa sửa.
2. **δ chưa được báo ở đâu.**
   - `REVIEW_V2_RESPONSE.md` §4 ghi dữ liệu đã lưu cho được một cận dưới của δ.
   - Một cận dưới đã đủ cung cấp thông tin: nếu nó lớn hơn gain đo được (≤ 0.14), thì kể cả khi có hằng số tốt hơn, Định lý 3 vẫn không áp được cho TFM thật ở các dataset này.
   - Lý do "không đủ chỗ" không còn với TMLR.

**Sửa:** sửa lượng từ; báo cận dưới của δ ở Phụ lục G (tính từ chênh giữa map gieo đều, chain-seeded và structured tại cùng một r), kèm một câu ở §6.3.

### F15 [Nhỏ] Độ chính xác của hai chỗ ở Phụ lục A (N14, m13, m14 vòng 2, chưa sửa)

1. **Remark (iii)** (`appendix_proofs.tex:301–306`):
   - không nêu ánh xạ đã dùng (Φ = 0.45 + 0.5r);
   - không nói dải chứng minh bị cắt ở 1 phía trên (dải thật là [0.676, 1]);
   - với "slope-one map", không nêu K = 10 và h_max.
2. **A.5 (iii)** (`appendix_proofs.tex:208–210`): câu "(33 configurations) … up to 0.052 when n ≤ 160 and by at most 0.0024 from n = 320 on (K = 10)" bỏ qua trường hợp K = 5, nơi sai số lên tới 0.128.

**Sửa:** dùng câu thay thế N14 của vòng 2; viết "(33 configurations; at K = 10: …; at K = 5 up to 0.13)".

### F16 [Nhỏ] §6.3 chọn bằng chứng ngoài phạm vi, bỏ bằng chứng trong phạm vi

**Vị trí:** `experiments.tex:81–86`.

1. **Phản ví dụ nằm ngoài phạm vi.** "It is not a faithful model of the chains, however: with TabPFN v2, chains on optdigits (2 of 3 seeds) climbed through the region of negative gain." Nhưng optdigits có context bị cắt, nên nằm ngoài phạm vi của Định lý 3 (chính chú thích Hình 1 nói vậy).
2. **Bằng chứng trong phạm vi bị bỏ.** Khi sửa N3, bài bỏ semeion: ở đó chuỗi đi từ 0.17 lên 0.34–0.37 trong vùng gain âm kéo dài tới ít nhất r = 0.8. Đây là bằng chứng trong phạm vi, mạnh hơn optdigits.
3. **Con số 96% gộp cả LLP.** 234 ô gồm 126 ô LLP, nơi Giả thiết 1 không áp dụng. Con số này lại được dùng để kết luận về "the lower half of Assumption 1".

**Sửa:** dẫn semeion cạnh optdigits; tính lại tỉ lệ "above the fixed point" riêng cho 108 ô PLL và chỉ dùng con số đó cho câu về Giả thiết 1.

### F17 [Nhỏ] Phụ lục E và Bảng 23 còn câu không khớp số (m3, m5 vòng 2)

1. **Phụ lục E** (`appendix.tex:214–218`):
   - "Most pairs … are of two kinds" chưa khớp với Bảng 17;
   - "i.e. it converged to the labelling with the two classes exchanged" áp cả cho ô có AUC 0.49, tức gần ngẫu nhiên.
2. **Bảng 23:** P13 thực tế được chấm trên 13/15 dataset; P12 trên 145/234 ô. Cả hai không được ghi.

**Sửa:** dùng câu thay thế m3 của vòng 2; thêm "(13 of 15 datasets available)" cho P13 và "(145 of 234 cells)" cho P12.

### F18 [Nhỏ] PLL thật: cần giải thích trạng thái đăng ký và nêu biên độ

**Vị trí:** `experiments.tex:224–226`; Phụ lục C, K.

1. **Trạng thái đăng ký.** "single chain with TabICL v2, registered in Study 1" là đúng: protocol G5 (mục F6) chọn TabICL v2 cho các dataset trên 10 lớp trước khi chạy. Nhưng Phụ lục C lại nói mọi kết quả TabICL v2 của Study 1 là thăm dò, nên người đọc thấy mâu thuẫn.
2. **Biên độ rất nhỏ.** "highest mean accuracy on all three" chỉ hơn 0.002 trên Lost (0.750 so với PRODEN tuned 0.748), và 0.001 trên MSRCv2 so với hard-EM.

**Sửa:** thêm "(TabICL v2 was registered for these data because they exceed TabPFN v2's 10-class limit)"; nêu biên độ.

### F19 [Nhỏ] "sampling beats taking the arg max" chỉ đúng cho log loss, chưa đúng cho chỉ số chính

**Vị trí:** `experiments.tex:61–64`; §6.5.

- Study 2, PLL q = 0.3: hai bên bằng nhau (0.862 và 0.862).
- Study 3: +0.020 tổng, +0.001 trên LLP (11/3/6, khoảng tin cậy chạm 0 theo `S3B_REPORT.md`).

**Sửa:** "sampling beats taking the arg max in log loss (registered H3′), and in accuracy only at q = 0.6".

### F20 [Nhỏ] Broader Impact: đo được nhưng chưa đo

**Vị trí:** `impact.tex:5–6`: "We have not measured how much individual-level information the sampler recovers from bag counts."

Độ đúng nhãn huấn luyện của các chuỗi LLP đã được lưu (dữ liệu monitor và response map). Báo con số đó, ví dụ trung vị theo dataset, thay cho câu phủ định.

### F21 [Nhỏ] Sai lệch protocol của Study 1 chưa được nêu cụ thể

**Vị trí:** Bảng 7 (`appendix.tex:25`): "fixing the reading of an ambiguous rule to its weakest form".

Cần ghi: đó là quy tắc nào, và cách đọc đó có làm đổi phán quyết nào của H2/H4 không.

### F22 [Nhỏ] Thiếu văn liệu liên quan trực tiếp (NeurIPS 3.8, chưa sửa)

- **Gibbs song song/bất đồng bộ.** Khối không hiệu chỉnh, trong đó mỗi điểm trong khối được rút độc lập với ngữ cảnh "cũ", chính là Gibbs song song kiểu Hogwild. Văn liệu về độ lệch của nó liên quan trực tiếp tới Mệnh đề 3: Johnson, Saunderson & Willsky (NeurIPS 2013); De Sa, Olukotun & Ré (ICML 2016).
- **Gốc của hard-EM và "stochastic EM":** SEM (Celeux & Diebolt, 1985) và data augmentation (Tanner & Wong, 1987).
- **Baseline phi nơ-ron kinh điển:**
  - PLL: PL-kNN (Hüllermeier & Beringer, 2006), IPAL (Zhang & Yu, 2015);
  - LLP: InvCal (Rüping, 2010), ∝SVM (Yu et al., 2013).

Tôi trích các tài liệu trên theo trí nhớ; cần mở và đối chiếu trước khi dùng.

### F23 [Trình bày] Thuật ngữ chưa được định nghĩa trong chính văn

- **"free"** (Algorithm 1 dòng 6; §4 "each free y_i"): chỉ được định nghĩa ở Phụ lục A.
- **H1 và H5 của Study 1** (Bảng 7: "H1–H5", "H3 and H5 met"): không giải thích.
- **"within its published limits"** (TabPFN v2): không nêu; theo `STUDY3_PLAN.md` là d ≤ 500, C ≤ 10.
- **"five curated suites"**: không nêu tên.
- **Quy tắc chọn primary comparator** ("A rule applied to development datasets"): không mô tả.
- **"uninformative labelling"** (introduction): không giải thích.
- **"cold start drawn from the normalised likelihood"**: với LLP nghĩa là phân phối đều trên các phép gán khớp số đếm của túi; nên viết ra.
- **"TabPFN-3 was preferred to its successor"**: successor là mô hình nào?
- **"PU data"**: chưa viết đầy đủ ở lần xuất hiện đầu.

### F24 [Trình bày] Ký hiệu trùng nhau (math c1 vòng 2, chưa sửa)

**Trong chính văn:**
- **δ:** dung sai của Giả thiết 1; hệ số Dobrushin δ(P^t); δ₁, δ₂ ở §6.1.
- **p:** PPD; p(ν, λ); p-value.
- **q:** TFM; xác suất distractor của PLL.
- **H:** số vòng quét; giả thiết (H_δ); H1″.
- **τ:** τ(P); thời điểm rời khỏi mức.
- **b:** dùng ở Định lý 3(c) mà chưa giới thiệu.

**Trong phụ lục:**
- **λ:** biên; N/n ở Mệnh đề 3; λ_k ở Mệnh đề 5.
- **ℓ:** hàm hợp lý; "ℓ-Lipschitz" ở Remark (ii).
- **a:** 1 − ⌈n/K⌉/n_F; hệ số trôi ở Mệnh đề 3(a); M_yy ở chứng minh Mệnh đề 5.
- **S:** số mẫu; scoring rule; S_i; S = nR_s.

**Sửa:** thêm bảng ký hiệu; δ₁, δ₂ → Δ₁, Δ₂; xác suất distractor q → ϱ; số vòng H → h; "ℓ-Lipschitz" → "L_Φ-Lipschitz".

### F25 [Trình bày] Bảng và hình

- **Bảng 33** (tr. 35): bị thu nhỏ tới mức không đọc được.
- **Bảng 34:** cắt tên dataset ("hazelnut-spread-contamin", "students_dropout_and_").
- **Bảng 4 và Bảng 5:** trùng gần hết nội dung; nên gộp.
- **Hình 1:** không có dải bất định, và không nói map được đo trên bao nhiêu seed.

**Sửa:** đặt Bảng 33 theo chiều dọc hoặc landscape; ghi tên dataset đầy đủ; gộp Bảng 4 vào Bảng 5; thêm dải bất định theo seed cho Hình 1.

### F26 [Trình bày] Câu chữ còn sót

- `appendix.tex:236` "the average of single chains is −0.028 below it": phủ định kép (c3).
- `appendix.tex:336` "+0.0032 … on average": đây là trung bình của bốn trung bình nhóm (c6).
- `experiments.tex:148` "by the measure of C4″ it regressed in 0 of 71 pairs": khó hiểu (N20).
- `experiments.tex:75` "ended at 0.93 or above": giá trị nhỏ nhất là 0.925 (N19).
- `appendix.tex:348` "one-sweep response": nên là response của một lần vẽ lại mọi khối (math m1).
- `experiments.tex:202` "still rising in many cells": macro `s3b-b3-rising` = 19 đã có sẵn.
- `theory.tex:94` "reached the multi-start average or the tuned learner": định nghĩa thật là "within 0.02 of the better of the two".
- `method.tex:49–50`: phương trình (3) kết thúc bằng dấu phẩy rồi sang câu mới.
- Trích dẫn:
  - "In arXiv preprint" ở Fan et al., Feitelberg et al., Hu & Ghelichi, Liu et al.;
  - Feitelberg et al. có mã arXiv:2510.02625 nhưng ghi năm 2026;
  - Balakrishnan et al. nên trích bản đăng ở Annals of Statistics.
- Cận "+0.110" vẫn ghi 3 chữ số (c2).

---

## (e) Trạng thái các mục chưa xong của vòng 2

**Nguồn:** `REVIEW_V2_ROUND2.md` §2 (các mục "Chưa sửa" / "Sửa một phần"), §3 (N1–N20), và `REVIEW_V2_RESPONSE.md` §4–5.

**Quy ước:** Đ = đã sửa; SP = sửa một phần; C = chưa sửa. Trích dẫn là văn bản của bản hiện tại.

### e.1 Báo cáo toán

| ID | Trạng thái | Bằng chứng (văn bản mới) |
|---|---|---|
| M2 | SP (không đổi) | Abstract: "For the sampler as run, a homogeneity assumption on the TFM gives floors and ceilings…". Contributions vẫn "invariant floors and ceilings". Theory có thêm "In this model …". → F1 |
| M4 | SP (không đổi) | Giả thiết 1 vẫn "at most ⌈n/K⌉ points"; δ chưa được ước lượng. → F14 |
| m1 | C | `appendix.tex:348` "we measured the one-sweep response". → F26 |
| m14 | SP | MĐ 5(a) thêm "with equal off-diagonal entries in each column of the channel"; còn "λ_k > 0" và "class-dependent noise". → F13 |
| m15 | Đ | Xem N6. |
| c1 | C | → F24 |

### e.2 Báo cáo kiểm toán

| ID | Trạng thái | Bằng chứng |
|---|---|---|
| B1 | Đ | PDF không còn TODO, [Pending] hay [C#] (tôi đã quét bản giải nén). Dùng `tmlr.sty`, header "Under review as submission to TMLR". Không có checklist. Giấy phép ở Phụ lục N. |
| m1 | Đ | Bảng 2: "gap 0.000". |
| m3 | SP | "Most pairs … are of two kinds …; the table lists every pair"; câu "i.e. … exchanged" vẫn còn. → F17 |
| m5 | C | Bảng 23: P12 không ghi 145/234 ô; P13 không ghi 13/15 dataset. → F17 |
| m6 | Đ | Xem N7. |
| m7 | Đ | "without the 5 datasets flagged before the study for group or sequence structure". |
| m8 | SP | Chú thích Bảng 2 và §6.4: "Differences pool AUC … and accuracy"; abstract chưa có. → F11 |
| m9 | Đ | Chú thích Bảng 18: "(SciPy defaults: zero differences dropped; exact distribution without zeros, normal approximation otherwise)". |
| m12 | Đ (kiểm được) | Bảng 7 ghi "10 Nov 04:21"; `git log`: `c9009e8` lúc 04:21:22 UTC. |
| m13 | C | → F15 |
| m14 | C | → F15 |
| m17 | Đ | Conclusion không còn "removed most failures". Chữ "robust TFM" vẫn còn (→ F5). |
| m18 | Đ | "in 81% of cells (91% on LLP, 61% on PLL)". |
| m20 | Đ | Phụ lục E: "(500 to 20,000 rows, at most 2,000 features, 3 to 20 classes, a minority class of at least 5% for binary tasks)". |
| m21 | SP | "about one accuracy point" đã bỏ. Còn gõ tay: "23 development and earlier datasets", "about 3%", các tỉ số Fisher 0.134…0.009, số giờ tính toán. Checklist đã bỏ, nên PDF không còn câu tự mâu thuẫn. |
| c1 | Đ | §6.1 và Phụ lục E cùng viết "suites of Study 2 … and OpenML-CC18". |
| c2 | C | "[+0.030, +0.110]". |
| c3 | C | → F26 |
| c6 | C | → F26 |
| c7 | Đ (chú thích) | Bảng 3: "of the multi-TFM average, and of the tuned learner L used in the learner start". Tôi không kiểm khoá `s3x-loo-other-max`. |
| c8 | Đ cho bản nộp | Commit `9532538`: packager "strips comments, prunes bib, anonymity check". Nguồn trong kho vẫn còn chú thích nội bộ (`theory.tex:2–3`). |

### e.3 Báo cáo kiểu NeurIPS

| Mục | Trạng thái | Bằng chứng |
|---|---|---|
| 3.1 | SP | `method.tex:58–61`: "What is fitted, by configuration: …". Vẫn thiếu bộ so sánh cùng thông tin. → F8 |
| 3.2 | SP (không đổi) | δ chưa ước lượng; N8 chưa sửa; chưa có kiểm chứng không vòng tròn. → F4, F14 |
| 3.3 | C | `related.tex:20–21` "has guarantees for incompatible conditionals"; dấu ✓ ở Bảng 25. → F2 |
| 3.4 | SP | §6.5 thêm GBDT tự huấn luyện và PL-kNN, đều không tune, chỉ trên Study 3; GBDT-PLL suy biến; không có bp_llp, IPAL hay fine-tuning. → F7 |
| 3.5 | SP | Kiểm định một phía của Study 3 đã được nói; việc gộp chỉ số đã được nói. Còn thiếu: độ lệch chuẩn theo seed, kiểu kiểm định của S3B, cách kiểm chứng việc ghi trước. → F10, F12 |
| 3.6 | SP | Mới tốt hơn: abstract có "with one TFM the multi-start average was not significantly better than the tuned learner" và "one-sided". Còn N9, N11; phát sinh thêm "improved on a single TFM" (→ F6) và việc bỏ "+0.041" (→ F11). |
| 3.7 | SP | Có §6.6 và Bảng 5 mới; nhãn bằng chứng sai và thiếu cấu hình rẻ. → F8 |
| 3.8 | C | → F22 |
| must 1 | Đ | Như B1. |
| must 2 | SP | → F6, F11 |
| must 4 | SP | §6.3: "on the 3 datasets within its scope (n_F from 756 to 1,395) a floor over 30 sweeps needs a gain margin of 0.25 to 0.35 even with δ = 0". δ vẫn chưa ước lượng. |
| must 5 | SP | `method.tex:58–61` đã nói phần nào có huấn luyện. Kết quả cold start (Bảng 31: LLP −0.005; Bảng 18: chuỗi cold 0.793) chưa được đặt cạnh khẳng định chính. |
| must 6 | SP | Đã định nghĩa: oracle, stochastic EM, flagged, cell, pair. Chưa định nghĩa: H1/H5, published limits, five suites, quy tắc chọn comparator. → F23 |
| must 7 | Đ | Algorithm 1 ở tr. 3; bảng cũ được tách thành Bảng 2 và Bảng 3. Hình quỹ đạo và Hình 4 vẫn ở phụ lục; chấp nhận được. |

### e.4 Phát hiện mới của vòng 2 (N1–N20)

| ID | Trạng thái | Bằng chứng |
|---|---|---|
| N1 | Đ | `theory.tex:103–104`: "for LLP a learner that treats them as independent point labels uses about 3%…". MĐ 5(b): "Equality holds … for LLP when the naive labels match the bag counts". Còn một câu thừa nhỏ về PLL. → F13(3) |
| N2 | Đ | "on the 3 datasets within its scope (n_F from 756 to 1,395) … 0.25 to 0.35 … at most 0.14". |
| N3 | Đ | "chains on optdigits (2 of 3 seeds) climbed through the region of negative gain". Nhưng bằng chứng được chọn nằm ngoài phạm vi. → F16 |
| N4 | Đ | Chú thích Hình 2: "though not all stay within the region of negative gain". |
| N5 | Đ | Bảng 9: cột "after sweep 1". |
| N6 | Đ | "the last two factors are 1 − O(K/n) − O(Cα/N)". |
| N7 | Đ | "(Studies 1 and 2 count exact ties only)". |
| N8 | C | → F4 |
| N9 | C | → F5 |
| N10 | SP | → F13 |
| N11 | C | → F1 |
| N12 | C | Phát biểu nén của Định lý 1 không đổi (→ F2(iv)). Câu "registered in Study 1" đúng nhưng cần giải thích (→ F18). |
| N13 | C | → F14 |
| N14 | C | → F15 |
| N15 | Đ | PDF: "group-level weak labels, with averages over starts and over TFMs. (2) Theory: exactness with an exact TFM, stability…" — không còn khoảng trắng thừa. |
| N16 | Đ | "(dataset, setting) comparisons"; "(“stochastic EM”: 0.843 …)"; "an oracle (the TFM given the true labels)". |
| N17 | C | Bảng 9 không ghi rằng hai chuỗi MiceProtein còn đang tăng ở vòng 30. B3 nay càng cho thấy điều này quan trọng. |
| N18 | Đ | Bảng 2: "gap 0.000". |
| N19 | C | → F26 |
| N20 | C | → F26 |

### e.5 `REVIEW_V2_RESPONSE.md` §4–5 (các mục từng hoãn vì thiếu chỗ hoặc cần chạy)

| Mục | Trạng thái | Ghi chú |
|---|---|---|
| Mã giả và hình vào thân bài | SP | Algorithm 1 đã vào thân bài; các hình vẫn ở phụ lục. Đủ dùng. |
| Baseline cây/kNN; belief propagation | SP | → F7 |
| Study 3 với q khác 0.6, túi khác 8 | C | Limitations nói phạm vi: "LLP with random bags of 8 and PLL with uniform distractors at q ≤ 0.6". Chấp nhận được với TMLR nếu mọi khẳng định giữ đúng phạm vi đó. |
| Ước lượng δ | C | Lý do "không đủ chỗ" không còn. → F14 |
| Mục "Pending" về giấy phép | Đ | Phụ lục N "Licences". |
| Style chính thức | Đ | `tmlr.sty`, chế độ ẩn danh. |
| GPU: naive TFM, hard-EM, oracle ở Study 3 | Đ | B2: Bảng 20, §6.5. |
| GPU: chạy dài T ≥ 100 trên ô kẹt | Đ | B3: Bảng 21; P22 sai và đã được báo. → F3, F11 |
| Response map trên Study 3 | C | `S3B_PLAN.md` §6 tự loại; ảnh hưởng tới F3. |

**Tổng 68 mục** (không đếm hai lần m15 = N6 và m6 = N7): **Đ 27 · SP 19 · C 22**.

Các mục **C** có tác động tới tiêu chí 1 là N8, N9, N11, N12 và NeurIPS 3.3; tất cả đều đã thành phát hiện mức Lớn ở trên (F1, F2, F4, F5). Phần còn lại là mục nhỏ hoặc trình bày.

---

## 6. Những gì tôi đã kiểm và tính lại

**Ẩn danh.**
- `pdfinfo`: Title và Author trống.
- Bản giải nén (`qpdf --qdf`) không chứa "/home", tên kho, tên người, "github" hay "kaggle.com". Chuỗi "Elio" khớp duy nhất một chỗ, là "Eliott Kalfon" trong tài liệu tham khảo.
- `PTEX.FileName` chỉ có đường dẫn tương đối `./figures/*.pdf`.
- Header: "Under review as submission to TMLR"; tác giả: "Anonymous authors".
- Có mục Broader Impact Statement.
- Chính văn dài 11 trang, dưới ngưỡng 12 trang của bài thường ở TMLR.

**S3B** (`s3b_check.py`, đọc CSV bằng `python3 -I`). Mọi số khớp với bài:

| Đại lượng | Trong bài | Tôi tính lại |
|---|---|---|
| Multi-TFM − GBDT | +0.135, trung vị +0.031, 19/0/1, p_Holm < 0.001 | +0.1350, +0.0306, 19/0/1, p_Holm = 0.00025 (hai phía, Holm trên 3) |
| Multi-TFM − PL-kNN | +0.188, 8/0/0, p_Holm = 0.016 | +0.1882, 8/0/0, p_Holm = 0.0156 |
| L − baseline CPU tốt hơn | +0.022, trung vị −0.003, 7/1/12, p_Holm = 0.841 | +0.0215, −0.0035, 7/1/12, p_Holm = 0.841 |
| Chuỗi đơn − GBDT | +0.031, trung vị +0.003, 11/1/8 | +0.0308, +0.0029, 11/1/8 |
| Chuỗi đơn − naive | +0.128, 18/1/1 | +0.1283, 18/1/1 |
| Chuỗi đơn − hard-EM (chỉ số chính) | +0.020 | +0.0204, 11/3/6 |
| B3 | 72 ô, kẹt 35, sửa 5, lợi +0.053 | 72, 35, 5, +0.0525; còn đang tăng 19 |
| B3 theo TFM: kẹt / sửa / lợi / độ đúng nhãn 30 → 100 | — | TabICL v2: 9 / 3 / +0.031 / 0.764 → 0.784; TabPFN-3: 16 / 1 / +0.111 / 0.403 → 0.488; TabDPT: 10 / 1 / +0.016 / 0.628 → 0.648 |

**Số mới tôi tính, chưa có trong bài:**
- Multi-TFM − GBDT: LLP 12/0/0, +0.032 (trung vị +0.017); PLL 7/0/1, +0.289.
- L − GBDT: LLP 2/1/9 (−0.013); PLL 5/0/3.
- Tách H1″ và H2″ theo bối cảnh: xem F12.

**Đối chiếu git** (bản clone nông):
- `c9009e8` 2026-11-10T04:21:22Z: Study 3 đóng băng.
- `7cc9679` 04:57:48: code của Study 3.
- `aa0775d` 2026-11-12T03:18:42Z: Study 3 chấm xong.
- `d9e6062` 07:21:22: S3B ghi trước.
- `b59ef14` 07:27:00: code của S3B.
- Thứ tự này khớp với Bảng 7 và Phụ lục F.

**Mã nguồn:**
- `baselines_cpu.py`: xác nhận cách cài GBDT-PLL mô tả ở F7.
- `S3/s3_report.py:46–51`: xác nhận ngưỡng hoà chỉ dùng cho W/T/L (F12).
- `G5/EXPERIMENT_PROTOCOL.md` mục F6: xác nhận TabICL v2 được đăng ký trước cho PLL thật (F18).

---

## 7. Giới hạn của báo cáo này

1. Tôi không tính lại các số của Study 1–2 và Study 3 ngoài phần đã nêu; vòng 2 đã làm việc đó và mọi số đều khớp.
2. Tôi không biên dịch lại PDF. Văn bản PDF khớp với nguồn ở mọi chỗ tôi đã đối chiếu.
3. Tôi không đối chiếu tài liệu tham khảo với nguồn gốc. Các trích dẫn đề xuất ở F22 lấy theo trí nhớ và cần được kiểm.
4. Kho là bản clone nông, nên các mốc đóng băng của Study 1–2 không kiểm được.
5. Nhận định "suy biến" ở F7 dựa trên so sánh với PL-kNN và trên việc đọc mã. Tôi không có tỉ lệ lớp đông nhất của từng dataset để so trực tiếp.
