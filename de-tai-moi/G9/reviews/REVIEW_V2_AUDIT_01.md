# Kiểm toán độc lập phần thực nghiệm của bản thảo v2 (12/11/2026)

- **Phạm vi:** mọi con số và khẳng định thực nghiệm trong `paper/` (abstract, introduction, experiments, limitations, appendix, appendix_proofs, checklist), đối chiếu với dữ liệu thô và protocol. Trọng tâm là Study 3, phần thăm dò của Study 3, hồ sơ phát triển V2 và các số kiểm lý thuyết.
- **Cách làm:** tự viết script đọc 213 tệp npz (`python3 -I`, `allow_pickle=False`), không import script phân tích nào của dự án; AUC tự cài bằng hạng trung bình. Script và đầu ra ở `/tmp/claude-0/-home-claude/643477cb-9e2c-5e3b-9dd1-1dba7a8f3cd0/scratchpad/review_audit/` (`recompute.py`, `recompute.json`, `recompute_pairs.json`, `recompute_datasets.json`).
- **Không sửa tệp nào của repo** ngoài báo cáo này. Không đọc `REVIEW_V2_NEURIPS_01.md` (xuất hiện trong lúc kiểm) để giữ tính độc lập.

## 1. Kết luận (5 dòng)

1. **Mọi con số xác nhận của Study 3 tái lập đúng tới sai số máy** từ 213 npz: H1″, H2″, H3″ (cả theo TFM và khoảng bootstrap), C4″, C5″, độ nhạy S1, Bảng 2, Bảng 11 (200/200 ô), A1 so với L, 4 ô nhãn đảo. `numbers_v2.tex` và 9 bảng sinh lại từ `make_v2.py` trùng từng byte với bản trong repo.
2. **Chỉ một số in lệch ở chữ số thứ ba (Bảng 13: 0,046 thay vì 0,047), nhưng có 5 chỗ câu chữ nói sai nghĩa của con số (MAJOR):** "with no trained model", "14.1% behind a tuned learner", "+0.052 over DLLP" gắn với "strongest", câu về bộ lấy mẫu MH, và bảng dự đoán phát triển bỏ P1–P4.
3. **Gắn nhãn xác nhận/thăm dò nhìn chung trung thực:** H3″ sai, C4″ không đạt, A1 so với L không có ý nghĩa, trung vị +0,003, lần xem tạm 60/71 cặp và các kỳ vọng ghi trước bị sai đều có mặt. Chỗ yếu duy nhất là hồ sơ phát triển (thiếu 4 dự đoán, 2 trong đó sai).
4. **Một mục chặn nộp bài, không thuộc số liệu:** PDF còn 2 `\todo`, 1 `\pending`, nhãn `[C#]`, `\finalfalse` và style giả lập.
5. **Ẩn danh sạch** trong PDF và metadata; rủi ro còn lại là chú thích tiếng Việt trong mã nguồn LaTeX nếu nộp kèm nguồn.

**Đếm theo mức:** BLOCKING 1 · MAJOR 5 · MINOR 21 · COSMETIC 8.

## 2. Bảng phát hiện

| ID | Mức | Vị trí | Vấn đề | Cách sửa cụ thể |
|---|---|---|---|---|
| B1 | BLOCKING | `main.tex:27` (`\finalfalse`); `appendix.tex:510, 515`; `checklist.tex:26`; PDF tr. 33–35 | PDF còn in "[TODO: quantify, or keep this sentence (G9/REVIEW_RESPONSE_PLAN R8)]", "[TODO: align with the venue's LLM policy; docs/AI_USAGE.md]", "[Pending: licences …]", 23 dòng có nhãn `[C#]`, và dùng `neurips_placeholder.sty`. Bật `\finaltrue` thì `\pending` dừng biên dịch | Chốt giấy phép từng tài sản, xoá hai `\todo` (chúng còn lộ đường dẫn nội bộ), bật `\finaltrue`, thay style chính thức |
| M1 | MAJOR | `experiments.tex:125–126`, đoạn Exploratory (i) | "even the average of single chains over TFMs, **with no trained model**, was above $L$ on 18 datasets". Sai với LLP: điểm xuất phát `warm` của LLP là **DLLP tuyến tính huấn luyện 200 epoch** (`G4/code/icgibbs/g7/methods.py:101–105`; chính `method.tex` cũng ghi "for LLP, linear DLLP"). 12 trong 18 lần thắng là dataset LLP (12/0/0). Ở PLL, nơi thật sự không có mô hình huấn luyện, kết quả là **6/1/1**, thua 0,200 ở soybean; khoảng bootstrap của cả 20 dataset là [−0,000; +0,073] | Viết: "even the average of single chains over TFMs, without the tuned learner (for LLP the warm start is still an untuned linear DLLP), was above $L$ on 18 datasets (LLP 12/0/0; PLL 6/1/1, 0.20 below $L$ on soybean)" |
| M2 | MAJOR | `introduction.tex:32–33`; `limitations.tex:7–8`; `abstract.tex:15–16` | "fell behind a tuned specialised learner in 14.1% of cases". 14,1% = 10/71 cặp có multi-start < max(**chuỗi đơn**, L) − 0,02. Thua riêng L quá 0,02 chỉ có **9/71 = 12,7%**; cặp TabPFN-3 × seismic-bumps (0,731 so với L 0,719) **hơn L** và chỉ bị tính vì thua chuỗi đơn của chính nó. Câu cũng bỏ ngưỡng 0,02 | "was more than 0.02 behind the better of its single chain and a tuned learner in 14.1% of (TFM, dataset) pairs (behind the learner in 9 of 71)". Abstract: "a bound on how often one TFM with several starts falls behind its single chain or a tuned specialised learner" |
| M3 | MAJOR | `appendix.tex:246–249` và chú thích Bảng 15 (`appendix.tex:253–255`); `make_v2.py:439–473` | Văn bản nói bảng "lists **every** prediction and its outcome, including the 5 of 13 that were wrong". Thực tế có **17** dự đoán đánh số (P1–P9, P11–P18). P1–P4 thuộc giai đoạn 0 (`V2/PHASE0_PREDICTIONS.md`, `V2/V0_REPORT.md`, 04–05/11, nằm trong khoảng "4–10 November" mà bài nêu) và bị bỏ: **P2 sai, P4 sai**, P1 và P3 đúng. Tổng đúng ra là **7/17 sai**. Chú thích "(P1–P4, P10) belong to an earlier phase with a different aggregation rule, which was abandoned" không chính xác: chỉ P4 nói về quy tắc chọn theo WPLL; P1–P3 nói về điểm xuất phát và lưu vực (P1 là bằng chứng cho câu "a learner start usually lies above the crossing" ở §5.2); **P10 chưa từng được gán** (không xuất hiện trong bất kỳ tệp `V2/*.md` nào) | Thêm 4 dòng P1–P4 kèm kết quả (P1 đúng: r cuối 0,953–0,989 ở 9/9 lần chạy; P2 sai; P3 đúng; P4 sai: 19/24 = 79,2% < 80%). Sửa thành "7 of 17". Chú thích: "P10 was never assigned" |
| M4 | MAJOR | `abstract.tex:11–12`; `introduction.tex:24–25` | "outperformed the **strongest** specialised learners for label proportions (+0.052 AUC over DLLP)". +0,052 là so với mốc chính đã đăng ký, **DLLP tuyến tính cấu hình mặc định** (trung bình 0,788). Baseline LLP mạnh nhất của Study 2 là DLLP-MLP đã tune (0,799); chênh so với nó là **+0,041** (16/0/2), tự tính từ `S2/results/s2_table_llp8.csv`. §6.2 viết đúng, abstract và intro thì gộp hai ý | "(+0.052 AUC over the registered comparator, linear DLLP; ahead of the strongest baseline, tuned DLLP-MLP, on 16 of 18 datasets)" |
| M5 | MAJOR | `experiments.tex:139–142` (§6.5) | "The MH-corrected block sampler … **did not predict better** than the default … (with TabICL v2: +0.033 AUC on LLP, −0.021 accuracy on PLL at q = 0.6)". Hai số là **MH − mặc định** (tiêu đề cột Bảng 20; `S2/results/s2e_mh_summary.csv`: `mh_minus_approx = 0.033`, `mh_better = 6` trên 9 ô). Vậy ở LLP bản MH **tốt hơn** 0,033 AUC (TabPFN v2: +0,010), ngược với câu văn | "did not predict consistently better (MH − default with TabICL v2: +0.033 AUC on LLP, 6 of 9 cells; −0.021 accuracy on PLL at q = 0.6; three smallest datasets), had worse log loss in both, and was a median 7 times slower" |
| m1 | MINOR | Bảng 2 (`tables/s3_tests.tex`), `experiments.tex:109`; `make_v2.py:137` | "gap +0.000": số 0 có dấu. Giá trị thật +0,00002 | In "0.000" (bỏ dấu khi \|x\| < 0,0005) hoặc "< 0.001 in absolute value" |
| m2 | MINOR | Bảng 13 (`tables/app_s3_regress.tex`) so với Bảng 11; `make_v2.py:238–240` | TabPFN-3 × micro-mass, chuỗi đơn: **0,046** ở Bảng 13, **0,047** ở Bảng 11. Giá trị thật 0,04651. Lỗi làm tròn hai lần: Bảng 13 đọc số đã làm tròn 4 chữ số (0,0465) trong `s3_summary.json` | Sinh Bảng 13 từ `s3_pairs.csv` (đủ độ chính xác) |
| m3 | MINOR | `appendix.tex:215–219`, đoạn Regressions | Phân loại "two kinds" không khớp dữ liệu: (a) TabPFN-3 × seismic-bumps bị xếp vào "L is better than every chain of that TFM", nhưng cả ba chuỗi (0,755; 0,723; 0,722) đều ≥ L (0,719); đây là cặp duy nhất thụt lùi so với chuỗi đơn mà không thua L. (b) gesture-phase (TabPFN-3) và ozone-8hr (TabPFN v2) không được xếp loại nào. (c) L hơn mọi chuỗi ở 9/10 cặp, kể cả cnae-9 và micro-mass. Câu "i.e. it converged to the labelling with the two classes exchanged" áp cho cả ô AUC 0,492, tức mức ngẫu nhiên (bốn ô: 0,30; 0,40; 0,43; 0,49) | Viết lại theo số: "in 9 of the 10 pairs L is better than every chain of that TFM; in 3 of them the single chain had stalled (accuracy ≤ 0.28); the tenth (TabPFN-3, seismic-bumps) is behind its own single chain only". Với ô nhãn đảo: "below 0.5 (0.30–0.49)" |
| m4 | MINOR | `appendix.tex:266–267` | "made us expect, wrongly, that a typical dataset of Study 3 would gain little and few would regress". Vế đầu **đúng** (trung vị +0,003; 6 dataset hoà). Cái sai là tỉ lệ thụt lùi (14,1% so với ≤ 10%) | "…made us expect that a typical dataset would gain little, which held (median +0.003), and that few pairs would regress, which did not" |
| m5 | MINOR | Bảng 15 (`make_v2.py:439–468`) so với `V2/PHASE1_PLAN.md §5`, `PHASE1B_PLAN.md`, `V1D_PLAN.md` | Ngưỡng ghi trước bị chép lệch: **P17** là "≤ **2** thụt lùi" (bảng ghi ≤ 3; kết quả 2/2 nằm đúng ở ngưỡng). **P7** là "G ≥ −0,002 ở mọi nhóm" (bảng ghi "gains … in all four groups"). **P13** còn điều kiện "≤ 2/15 dataset thua quá 0,02" và chỉ có 13/15 dataset đủ 4 TFM (kết quả 2, đúng ngưỡng). **P12** chấm trên 145/234 ô (phần TabICL bị cắt ở giới hạn phiên) mà bảng không nói | Chép đúng ngưỡng; thêm "(145 of 234 cells)" cho P12 và "threshold ≤ 2 of 15; 13 available" cho P13 |
| m6 | MINOR | Chú thích Bảng 2, Bảng 14; `experiments.tex` | Ngưỡng hoà \|chênh\| < 0,001 (ghi trước ở STUDY3_PLAN §4) không được nêu ở đâu trong bài. Study 2 dùng quy tắc khác: LLP so với DLLP tuyến tính in 16/0/2, còn theo ngưỡng 0,001 là 15/1/2 | Thêm "(tie: absolute difference < 0.001, as registered)" vào chú thích Bảng 2 và Bảng 14; ghi quy tắc hoà của Bảng 1 |
| m7 | MINOR | `experiments.tex:31–32, 110, 113–116` | (a) Đích của H3″ ("the single chain fails") không được định nghĩa: chuỗi đơn thua max(multi-start, L) quá 0,02 trên ô (TFM, dataset, seed). (b) Không nêu số ô dương tính: **66/213** (TabICL 18/60, TabPFN-3 20/60, TabDPT 17/60, TabPFN v2 11/33); macro `s3-h3-npos` có sẵn nhưng không dùng. (c) "the five flagged datasets" xuất hiện ở thân bài mà chưa định nghĩa | Thêm một mệnh đề định nghĩa và "(66 of 213 cells)"; thêm "(flagged in advance for group or sequence structure, Appendix F)" |
| m8 | MINOR | `abstract.tex:13–14`; chú thích Bảng 2 | +0,058 và +0,046 là trung bình **gộp** chênh AUC (12 dataset LLP) và chênh accuracy (8 dataset PLL). Bài không nói rõ, và abstract không ghi đơn vị | Thêm "(AUC for LLP and accuracy for PLL, pooled as registered)" vào chú thích Bảng 2 |
| m9 | MINOR | Bảng 14, dòng đầu; `S3/s3_explore.py:50–51` | "single chain − multi-start: p = 0.002" là Wilcoxon hai phía bằng **xấp xỉ chuẩn** (scipy tự chuyển vì có 2 chênh bằng 0). Cùng phép so đó, H1″ dùng p chính xác một phía 0,0004 (hai phía chính xác: 0,0008). Các dòng khác có số 0 cũng lệch nhẹ (base − A1: 0,398 so với 0,418; X − A1 của TabICL: 0,084 so với 0,087) | Ghi phương pháp trong chú thích, hoặc bỏ số 0 rồi dùng `method="exact"` cho thống nhất với script xác nhận |
| m10 | MINOR | `experiments.tex:77–79` (§6.3) | "With TabPFN v2 the error is one-sided" tách riêng TabPFN v2. Trên chính 234 ô đó, ánh xạ gieo đều lệch một phía ở **cả hai** TFM: chuỗi kết thúc cao hơn dự đoán ở 95% ô TabICL (trung vị có dấu +0,074) và 97% ô TabPFN v2 (+0,070); Hình 5(b) cũng cho thấy vậy. Ngoài ra 126/234 ô là LLP, nơi Định lý 3 không áp dụng, nhưng thân bài không nói | "With both TFMs the error is one-sided (chains end above the uniform-map prediction in 95–97% of cells)…"; thêm "(126 of the 234 cells are LLP, which the theorem does not cover)" |
| m11 | MINOR | `limitations.tex:16–17` | "bags of 32 and instance-dependent candidate sets were hard for **every TFM**". PLL phụ thuộc mẫu chỉ chạy với **TabICL v2** trên 3 dataset (V1C phần G); lượt 4 TFM bị bỏ (`V2/V1D_REPORT.md §4`: "chưa chạy") | "bags of 32 were hard for every TFM, and instance-dependent candidate sets, tried with TabICL v2 only, were much harder than uniform ones" |
| m12 | MINOR | `appendix.tex:55` (Bảng 4) | "10 Nov 04:21" gõ tay. `V2/EXPERIMENT_LOG.md:21` ghi commit `c9009e8` lúc 11:18 giờ Việt Nam, tức **04:18 UTC**. Repo không có `.git` nên không kiểm được | Lấy giờ từ `git log -1 --format=%cI c9009e8` |
| m13 | MINOR | `appendix_proofs.tex:194–196`; `make_v2.py:524–530` | "(33 configurations) … off by up to 0.052 when n ≤ 160 and by at most 0.0024 from n = 320 on (K = 10)". Hai giá trị lớn nhất chỉ lấy trên **8** cấu hình bất đối xứng có K = 10. Trong 33 cấu hình có 9 cấu hình đối xứng (sai số ≈ 0), và ở K = 5 sai số lên **0,128** (n = 160, túi 8) | "(33 configurations; at the default K = 10: …; at K = 5 up to 0.13)" |
| m14 | MINOR | `appendix_proofs.tex:282–284`; `make_v2.py:498–501` | "half-width 0.22 … simulated 99% band 0.02" chỉ lấy ánh xạ Φ = 0,45 + 0,5r, và phía trên của dải bị cắt ở 1. Ánh xạ còn lại (0,3 + 0,5r) cho 0,32 so với 0,04. Bài không nói đang dùng ánh xạ nào | Nêu tên ánh xạ, hoặc báo cả hai |
| m15 | MINOR | `experiments.tex:146–148` | "all chains of the four-TFM average took a median of 37 minutes" gộp 33 ô có 4 TFM (12 chuỗi, trung vị 28 phút) với 27 ô chỉ có 3 TFM (9 chuỗi, trung vị 49 phút). Con số **không gồm** thời gian tune L (LLP: trung vị 12 phút CPU mỗi ô, tối đa 24 phút; `s3_lprefetch_llp.csv`), dù điểm xuất phát learner cần nó | "all chains of the multi-TFM average (9 or 12) took a median of 37 minutes per dataset and seed, excluding the tuning of L (median 12 CPU-minutes on LLP)" |
| m16 | MINOR | `experiments.tex:64–65` (§6.3) | "At q = 0.6 it was ahead on ten datasets by +0.033 on average": trên 10 dataset dưới 8 lớp, nó chỉ thắng **7/10** (`s1-pll6-rest-w = 7`; Bảng 1: 7/0/7) | "on the ten datasets with fewer than eight classes it was ahead by +0.033 on average (7 of 10)" |
| m17 | MINOR | `conclusion.tex:4–5` | "averaging over starts and over TFMs **removed most of the resulting failures** in a third pre-registered study". Đây không phải đại lượng ghi trước; tiêu chí ghi trước về thụt lùi (C4″) không đạt. Chỗ dựa là phần thăm dò (multi-TFM thụt lùi 0/71) | "…improved on the single chain and matched the best TFM of each dataset in a third pre-registered study; in exploratory analysis the multi-TFM average was never more than 0.02 behind the tuned learner" |
| m18 | MINOR | `experiments.tex:130–132`, mục (iii) | "81% of cells" tính trên cả 213 ô, nhưng hai số lợi ngay sau đó chỉ là PLL, nơi tỉ lệ đồng thuận là **61%** (46/75 ô; 29 ô lệch). Ngưỡng 0,03 được chọn sau khi xem số V1 | "…in 81% of cells (61% on PLL); on PLL the gain is +0.328 in the 29 cells where they differ and +0.006 in the 46 where they agree" |
| m19 | MINOR | Bảng 17 (`appendix.tex`, Positioning) | Dòng In-Context Gibbs đánh dấu ✓ ở cột "no training (frozen model)". Điểm xuất phát `warm` của LLP huấn luyện DLLP tuyến tính, và điểm xuất phát learner huấn luyện L đã tune. Đây là phần sót lại từ bản hai study | Thêm chú thích: "the TFM is frozen; the LLP warm start and the learner start train small models" |
| m20 | MINOR | `appendix.tex:161–165` so với `appendix.tex:425–427` | Study 3 lấy "the most relaxed registered tier" nhưng không nói tier đó là gì (d ≤ 2.000, 3–20 lớp, lớp nhỏ nhất ≥ 5%; `S2/S2_PLAN.md:72`). Phụ lục L vẫn ghi tiêu chí Study 1 (≤ 500 feature, ≥ 10%, 3–10 lớp), trong khi Bảng 11 có d tới 1.776, C tới 20, lớp nhỏ 6–8% | Thêm một mệnh đề nêu ba ngưỡng của tier 3 |
| m21 | MINOR | Rải rác | Số gõ tay không phải hằng số protocol: "about one accuracy point" (`introduction.tex:25`; có sẵn `s2-pll3-cmp-diff`), "23 development and earlier datasets" (`appendix.tex:274`; có `v2-p16-n`), số giờ tính toán ở Statements, giờ commit ở Bảng 4, và các tỉ số thông tin Fisher ở `appendix_proofs.tex:340, 365–366` và `theory.tex:105` (0,134; 0,060; 0,033; 0,017; 0,009; 0,24–0,26; 0,06–0,10; "about 3%"). Checklist mục 4 lại khẳng định "every number is generated by scripts" | Chuyển sang macro, hoặc sửa câu ở checklist |
| c1 | COSMETIC | `experiments.tex:24–26` so với `appendix.tex:161` | Thân bài: "from the same suites, OpenML-CC18 and TabArena". Phụ lục: "from the suites of Study 2 and OpenML-CC18". TabArena vốn đã nằm trong các suite của Study 2 (`S2_PLAN.md:59`) | Thống nhất: "the suites of Study 2 (which include TabArena) and OpenML-CC18" |
| c2 | COSMETIC | `experiments.tex:124–125` | Cận trên "+0.110" của khoảng bootstrap X − L dao động ở chữ số thứ ba theo thứ tự dataset và seed (0,108–0,110; với 200.000 lần lấy mẫu là 0,109) | Báo 2 chữ số, hoặc tăng số lần lấy mẫu |
| c3 | COSMETIC | `appendix.tex:236` | "the average of single chains is −0.028 below it": phủ định kép | "is 0.028 below it" |
| c4 | COSMETIC | `experiments.tex:81–82` | "on 4 of the four datasets" | "on all four datasets" |
| c5 | COSMETIC | `experiments.tex:118–119` | "in 2 of them behind the single chain and in 9 behind L": 2 + 9 = 11 > 10; một cặp (TabICL v2 × seismic-bumps) thua cả hai | Thêm "(one pair counts in both)" |
| c6 | COSMETIC | `appendix.tex:264–265`; `make_v2.py:383` | "+0.0032 … on these pairs": 0,0032 là trung bình của bốn trung bình nhóm (đại lượng G ghi trước). Trung bình trên 78 cặp là +0,0033 | Ghi "(mean of the four group means)" |
| c7 | COSMETIC | `make_v2.py:289`; chú thích Bảng 2 | Khoá `s3x-loo-other-max` tính bằng `min()` (không dùng trong bài). Chú thích Bảng 2 không mô tả hai dòng "Multi-TFM" và "Tuned learner L" | Đổi tên hoặc xoá khoá; thêm nửa câu vào chú thích |
| c8 | COSMETIC | `sections/theory.tex:3–4`; chú thích đầu các tệp `.tex`; `paper/README.md` | Chú thích LaTeX có chữ tiếng Việt ("Bổ đề", "Định lý", "Mệnh đề") và đường dẫn nội bộ (`G4/PROOFS.md`, `V2/reviews/…`). Không vào PDF, nhưng lộ nếu nộp kèm mã nguồn | Lọc chú thích trước khi đóng gói nguồn (ví dụ `latexpand --empty-comments`) |

## 3. Bảng tính lại độc lập

Mọi giá trị "của em" lấy từ `recompute.py` trên 213 npz. W/T/L dùng ngưỡng hoà 0,001.

### 3.1 Xác nhận (STUDY3_PLAN)

| Đại lượng | Trong bài | Tính lại | Khớp |
|---|---|---|---|
| Số tệp / cặp / dataset (LLP + PLL) | 213 / 71 / 20 (12 + 8) | 213 / 71 / 20 (12 + 8) | có |
| Cặp theo TFM (ICL / PFN-3 / DPT / PFN v2) | 20 / 20 / 20 / 11 (10 LLP + 1 PLL) | 20 / 20 / 20 / 11 (10 + 1) | có |
| Cùng lần chia, cùng `yte`, cùng `P_L` giữa các TFM | (ngầm định) | 60/60 ô giống nhau | có |
| H1″ trung bình δ₁ | +0,058 | +0,05848 | có |
| H1″ khoảng bootstrap | [+0,016; +0,114] | [+0,0163; +0,1138] | có |
| H1″ trung vị | +0,003 | +0,00251 | có |
| H1″ W/T/L | 13/6/1 | 13/6/1 | có |
| H1″ p một phía, Holm | < 0,001 (0,00042) | 0,00042 khi bỏ \|δ\| ≤ 10⁻¹²; 0,00039 nếu giữ δ = −3,7·10⁻¹⁷ của har | có (cùng kết luận) |
| H1″ theo setting | PLL +0,136; LLP +0,007 | +0,1360; +0,0068 | có |
| H2″ trung bình δ₂ | +0,046 | +0,04575 | có |
| H2″ khoảng bootstrap | [+0,019; +0,079] | [+0,0187; +0,0794] | có |
| H2″ W/T/L; p Holm | 18/2/0; < 0,001 | 18/2/0; 1,9·10⁻⁶ | có |
| Độ nhạy S1 (15 dataset) p Holm | 0,003 và < 0,001 | 0,00336 và 6,1·10⁻⁵ | có |
| Độ nhạy S1 trung bình, W/T/L | +0,062, 10/4/1; +0,044, 13/2/0 | +0,0622, 10/4/1; +0,0442, 13/2/0 | có |
| H3″ AUC(c3 → T2), 213 ô | 0,766 | 0,76613 (66 ô dương tính) | có |
| H3″ khoảng bootstrap theo dataset | [0,59; 0,92] | [0,588; 0,924] | có |
| H3″ theo TFM (PFN-3 / ICL / DPT / PFN v2) | 0,93 / 0,85 / 0,59 / 0,64 | 0,931 / 0,847 / 0,585 / 0,636 | có |
| C4″ số cặp thụt lùi | 10/71 (14,1%) | 10/71 (14,08%) | có |
| C4″ tách: thua chuỗi đơn / thua L | 2 / 9 | 2 / 9 (1 cặp thuộc cả hai) | có |
| Chuỗi đơn thua L quá 0,02 | 14/71 | 14/71 | có |
| C4″ theo TFM (ICL / PFN-3 / DPT / PFN v2) | 1 / 4 / 3 / 2 | 1 / 4 / 3 / 2 | có |
| C5″ trung bình (X − A1 tốt nhất) | "+0.000" | +0,000024 | có (xem m1) |
| C5″ số dataset thua quá 0,02 | 1/20 | 1/20 | có |
| C5″ dataset tệ nhất | micro-mass: 0,645 so với 0,671 | 1515: 0,6453 so với 0,6705 | có |
| A1 so với L (mô tả, hai phía) | +0,018; p = 0,14 | +0,01796; 14/0/6; p = 0,1429 | có |
| Ô chuỗi đơn có AUC < 0,5 (LLP) | 4 | 4 (TabDPT × 1467 seed 0, 1; TabICL × 46956 seed 0, 2) | có |
| G theo TFM (ICL / PFN-3 / DPT / PFN v2) | +0,024 / +0,112 / +0,037 / +0,013 | +0,0235 / +0,1118 / +0,0369 / +0,0132 | có |
| Bảng 2: TabICL v2 (LLP đơn, đa; PLL đơn, đa) | 0,802 0,803 0,749 0,805 | 0,8016 0,8032 0,7486 0,8050 | có |
| Bảng 2: TabPFN-3 | 0,817 0,817 0,371 0,650 | 0,8170 0,8173 0,3708 0,6498 | có |
| Bảng 2: TabDPT | 0,786 0,799 0,648 0,720 | 0,7857 0,7992 0,6478 0,7198 | có |
| Bảng 2: TabPFN v2 | 0,784 0,798 0,488 0,494 | 0,7840 0,7980 0,4882 0,4939 | có |
| Bảng 2: Multi-TFM; L | 0,828 0,808; 0,783 0,717 | 0,8282 0,8083; 0,7830 0,7168 | có |
| Bảng 11 (theo dataset) | 200 ô số | 200/200 ô trùng tới 3 chữ số | có |
| Bảng 13, TabPFN-3 × micro-mass, chuỗi đơn | 0,046 | 0,04651 → 0,047 | **không** (m2) |
| `s3_pairs.csv`, `s3_datasets.csv` | — | lệch tối đa 2·10⁻¹⁶ | có |

### 3.2 Thăm dò (đoạn "Exploratory analysis" và Phụ lục F)

| Đại lượng | Trong bài | Tính lại | Khớp |
|---|---|---|---|
| Multi-TFM − L: trung bình, W/T/L | +0,064; 20/0/0 | +0,0637; 20/0/0; nhỏ nhất +0,0034 | có |
| Multi-TFM − L: khoảng bootstrap | [+0,030; +0,110] | [+0,030; +0,109] (thứ tự theo ID); [+0,030; +0,110] (thứ tự của script) | có (xem c2) |
| Multi-TFM − L theo setting | LLP +0,045; PLL +0,091 | +0,0452; +0,0915 | có |
| Multi-TFM thụt lùi theo phép đo C4″ | 0/71 | 0/71 (chênh nhỏ nhất −0,0149) | có |
| Trung bình chuỗi đơn qua TFM − L | thắng 18 dataset | +0,0360; 18/1/1; **LLP 12/0/0, PLL 6/1/1, nhỏ nhất −0,200** | số khớp; câu chữ sai (M1) |
| Chuỗi xuất phát từ L − multi-start | +0,001 | +0,00084; 5/6/9; p = 0,398 | có |
| (Chuỗi đơn + dự đoán của L)/2 − multi-start | +0,008; p = 0,26 | +0,00788; 12/1/7; p = 0,2611 | có |
| warm + L starts − multi-start (bỏ cold) | +0,001 | +0,00104; 5/13/2 | có |
| Multi-start + L − multi-start | +0,021; p = 0,005 | +0,0214; 14/2/4; p = 0,0049 | có |
| Ba chuỗi cách nhau < 0,03 về độ đúng nhãn | 81% | 80,75% (LLP 91,3%; PLL 61,3%) | có |
| Lợi PLL khi chuỗi lệch / đồng thuận | +0,328 / +0,006 | +0,3281 (29 ô) / +0,0061 (46 ô) | có |
| Multi-TFM − multi-start của TabICL v2 | +0,016; p = 0,08 | +0,0163; 10/5/5; p = 0,0836 | có |
| Bỏ TabICL v2 khỏi multi-TFM | (Bảng 12, không trích số) | −0,0284; 2/5/13; các TFM khác −0,0011 đến −0,0013 | có (so với `s3_explore.json`) |
| Trung bình chuỗi đơn qua TFM − multi-TFM | −0,028 | −0,0278; 3/8/9 | có |
| Trung bình chuỗi base qua TFM − multi-TFM | −0,001 | −0,0008 | có |
| Bỏ TabPFN v2 (X3 − X) | −0,001; 3/13/4 | −0,0006; 3/13/4; p = 0,286 | có |
| TabICL v2 chuỗi đơn − L (LLP; PLL) | +0,019 (9/0/3); +0,032 (7/0/1) | +0,0186 (9/0/3); +0,0318 (7/0/1) | có |
| TabICL v2 multi-start − L (LLP; PLL) | +0,020 (10/0/2); +0,088 (8/0/0) | +0,0202 (10/0/2); +0,0882 (8/0/0) | có |
| Bảng 12, ba cột cuối | 12 ô số | trùng cả 12 | có |
| Bảng 14, mọi dòng (trung bình, chênh, W/T/L, p) | 14 dòng | trùng cả 14 | có |
| Chi phí mỗi chuỗi (ICL / PFN v2), trung vị | 92 s / 208 s | 91,9 s / 208,4 s (DPT 119,7; PFN-3 172,4) | có |
| Chi phí multi-TFM mỗi ô, trung vị | 37 phút | 36,9 phút (ô 4 TFM: 28,1; ô 3 TFM: 48,9) | có (xem m15) |

### 3.3 Hồ sơ phát triển và số kiểm lý thuyết (đọc thẳng CSV/JSON)

| Đại lượng | Trong bài | Tính lại | Khớp |
|---|---|---|---|
| V1: số cặp / ô | 78 / 234 | 78 (32 + 23 + 10 + 13) / 234 | có |
| V1: thụt lùi A1 / A2 / A3 | 2 / 3 / 3 | 2 / 3 / 3 | có |
| V1: A1 so với chuỗi đơn (thắng / thua / hoà) | 39 / 11 / 28; không lần nào thua quá 0,02 | 39 / 11 / 28; 0 | có |
| V1: lợi trung bình | +0,0032 | +0,00318 (trung bình nhóm); +0,00327 (trung bình cặp) | có (xem c6) |
| P8, P9: sai số trung vị chain / uniform | 0,016 / 0,073 (234 ô) | 0,01564 / 0,07265 (234 ô) | có |
| Bảng 16 (theo TFM × setting) | 96, 69, 30, 39 ô và 10 trung vị | trùng hết | có |
| Phân vị 90 của sai số chain-seeded | 0,051 | 0,0514 | có |
| P11 (số dataset không có điểm hút thấp) | PFN-3: 0/4; "TabDPT low on 2" | PFN-3 0, DPT 2, ICL 4, PFN v2 0 | có |
| P12, P13, P14, P15, P16, P17, P18 | 6; −0,008 và 2; 1/4; 0,45; −0,004 và 1; +0,0115 / +0,0002, 2 / 2; 0,86 (138 ô) | trùng với `v1c_summary.json`, `v1d_summary.json` | có |
| Số dự đoán sai / tổng | 5 / 13 | 5 / 13 trong bảng; **7 / 17** nếu tính P1–P4 | **không** (M3) |
| Tỉ số tốc độ TabPFN-3.5 / TabPFN-3 | 3,1 | 3,085 | có |
| Mô-men: tỉ số s.d. | 0,985–1,011 | 0,9848–1,0114 | có |
| Jacobi thấp hơn | 18%–21% | 18,4%–21,0% | có |
| Dải bất biến: nửa rộng chứng minh / mô phỏng | 0,22 / 0,02 | 0,2235 / 0,0195 (ánh xạ 0,45 + 0,5r; ánh xạ kia 0,32 / 0,04) | có (xem m14) |
| E7b: số cấu hình; vượt 1; TV; độ lỏng; sai số dự đoán | 24; 22; 0,0012–0,0248; 60–629; 0,0056 | 24; 22; 0,00122–0,02484; 60,1–629,4; 0,0056 | có |
| E7b: SSL, K = 10, n = 50…500 | 0,005; 0,009; 0,010; 0,011 | 0,0051; 0,0093; 0,0104; 0,0112 | có |
| E5c: PU khối tự do / mọi điểm | 0,070 / 0,004 | 0,0698 / 0,0041 | có |
| E7f: số cấu hình; lệch tối đa | 15; 1,5·10⁻¹¹ | 15; 1,46·10⁻¹¹ | có |
| E7e: LLP, K = 10 | 0,052 (n ≤ 160); 0,0024 (n ≥ 320) | 0,0521; 0,0024 (trên 8 cấu hình; xem m13) | có |
| Rà dataset Study 3 | 43 ứng viên; giữ 20; 8 nhân tạo, 14 cùng nguồn, 1 trùng | 43; 20; 8; 14; 1 | có |
| ID bị loại; ID huấn luyện TabDPT; lấy mẫu con; gắn cờ | 176; 123; 6; 5 | 176; 123; 6; 5 | có |

## 4. Những gì đã kiểm và thấy đúng

**Số liệu và bộ sinh**
- Sinh lại `make_v2.py` trong thư mục nháp (đầu vào là liên kết tới `S3/`, `V2/`, `G4/`): `numbers_v2.tex` và cả 9 bảng trùng từng byte với bản trong `paper/`. Bài không dùng số cũ.
- 164 khoá `\val{…}` được dùng đều đã định nghĩa; không khoá nào trùng giữa `numbers.tex`, `numbers_appendix.tex`, `numbers_v2.tex`; PDF không có "??".
- `make_v2.py`: đúng cột, đúng bộ lọc, đúng dấu, đúng mẫu số ở mọi khoá được dùng. Ngưỡng thắng/hoà/thua đúng protocol. p trong Bảng 2 là p đã hiệu chỉnh Holm.
- Kiểm chọn lọc số cũ từ dữ liệu Study 2: +0,052 (so với DLLP tuyến tính mặc định), +0,116 (naive), +0,012 / +0,012 (CC tune), 10/10 và 9/10, 0,8 và 1,9 điểm so với oracle, 16/18 so với DLLP-MLP tune đều đúng.

**Mô tả Study 3 so với STUDY3_PLAN**
- Định nghĩa δ₁, δ₂, phép thử (Wilcoxon một phía, Holm trên hai giả thuyết), ba tiêu chí không kèm phép thử cùng các ngưỡng (0,80; 10%; 0,01 và 3/20) được chép đúng.
- Danh sách TFM và giới hạn của TabPFN v2 (11 dataset: 10 LLP + 1 PLL), cấu hình chuỗi (K = 10, 30 vòng quét, burn-in 10, 10 mẫu, ngữ cảnh 2.000), 3 seed, chỉ số theo setting: đúng với protocol và với `methods.py`.
- L đúng là DLLP-MLP tune (LLP) và CC tune (PLL) (`V0_BASE`). Khẳng định "the strongest LLP baseline and the primary PLL comparator of Study 2" **đúng**: `s2-llp-strongest-name` = "tuned DLLP (MLP)" (trung bình 0,799, cao nhất Bảng 10); CC tune là mốc chính PLL.
- Các báo cáo kèm ghi trước đều có: theo TFM, theo setting, A1 so với L (hai phía), độ nhạy S1, ô nhãn đảo, thời gian chạy.
- Kết quả bất lợi không bị giấu: H3″ sai (kèm số theo TFM), C4″ không đạt, A1 so với L không có ý nghĩa (+0,018; p = 0,14), trung vị +0,003 (thân bài và Limitations), 4 ô nhãn đảo (Phụ lục F), kỳ vọng ghi trước sai ở hai mục ("Two did not").
- Lần xem tạm 60/71 cặp có nêu ở Bảng 4 và Phụ lục F, khớp `S3/results/interim_s2/` và nhật ký. Thay đổi script chấm trước khi chấm và việc đổi thứ tự hàng đợi GPU có nêu, khớp `EXPERIMENT_LOG.md:22`.
- Đoạn thăm dò được gắn nhãn rõ ("not registered … computed after the confirmatory scoring"); so sánh multi-TFM với L được gọi là thăm dò ở cả thân bài, Limitations và chú thích Bảng 14.

**Nhất quán giữa các phần**
- Abstract, Introduction, §6.4, Limitations, Phụ lục F dùng cùng số: +0,058; +0,046; 14,1%; +0,003; 20 dataset; 8 đa lớp; 71 cặp; 213 ô.
- 24 + 28 + 20 = 72; Study 1 gồm 10 + 14; Study 2 gồm 18 + 10; Study 3 gồm 12 + 8.
- Tên gọi thống nhất (single chain / multi-start / multi-TFM / tuned learner L; H1″–H3″, C4″, C5″). Không còn ký hiệu nội bộ (v1, A1, X) trong văn bản.
- Không còn câu "both studies" hay "two studies", không còn khẳng định "chưa có lý thuyết cho bộ lấy mẫu mặc định"; mọi tham chiếu chéo đều giải được.
- Chữ cái phụ lục và số bảng, hình trong thân bài khớp PDF (F = Study 3, G = Development, C = Pre-registration).

**Ẩn danh và trình bày**
- Không tìm thấy tên, cơ quan, e-mail, URL kho mã, tên người dùng hay mã commit trong `sections/`, `tables/`, `numbers*.tex`, `refs.bib` và văn bản PDF. Metadata PDF trống (Author, Title); hình chỉ ghi Matplotlib.
- `main.log`: 0 Overfull, 0 tham chiếu hay trích dẫn chưa định nghĩa. Đã dựng ảnh trang 8, 25–29: bảng không tràn lề, chú thích khớp nội dung (trừ c7).

## 5. Rủi ro còn lại, chưa kiểm được

1. **Không có `.git`:** không xác minh được thứ tự commit (protocol trước code, trước script chấm, trước dữ liệu), giờ đóng băng, hay việc `s3_report.py` không đổi sau khi có số ngoài thay đổi đã khai. Chỉ đối chiếu được với nhật ký.
2. **Không chạy lại chuỗi nào:** npz được coi là đúng. Chưa kiểm mã băm code (85/85 tệp) và tính "không đổi một byte" giữa các phiên.
3. **Số của Study 1–2** (`numbers.tex`, `numbers_appendix.tex`) chỉ kiểm chọn lọc; chưa tính lại p Holm, phép Bayes dấu hạng, thứ hạng trên dữ liệu PLL thật, Bảng 18–27. Hai lỗi câu chữ ở phần cũ (M5, m16) cho thấy nên rà lại toàn bộ §6.2, §6.3, §6.5 theo cùng cách.
4. **Giờ tính toán ở Statements** ("42 hours", "six further sessions"…) là số gõ tay. Nhật ký V2 có 8–9 phiên GPU trong 04–10/11 nếu tính cả V0A, V0B và lượt V0B chạy nhầm.
5. **Tài liệu tham khảo mới** (TabPFN-3, TabDPT, TabArena) có trong `G9/REFERENCE_CHECK.md` nhưng chưa đối chiếu lại nguồn (không dùng mạng).
6. **Các chứng minh** không thuộc phạm vi; chỉ kiểm số minh hoạ đi kèm.
7. **`monitor_*`:** coi là độ đúng nhãn của trạng thái chuỗi theo từng vòng quét (lệch trung vị 0,017 so với độ đúng của argmax Q̄); chưa lần lại trong `sampler.py`.
8. **Kích thước chữ Bảng 2** sau `\resizebox` chỉ xem trên ảnh 70 dpi; cần kiểm lại với style chính thức.
9. **Chuỗi tìm kiếm ẩn danh** (họ tên, tên trường) lấy từ hồ sơ người dùng; chưa dò các định danh khác như tên tài khoản Kaggle hay OpenML.
