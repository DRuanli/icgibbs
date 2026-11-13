# Kiểm toán số liệu vòng 3: phần bổ sung S3B của bản thảo v2 (13/11/2026)

- **Phạm vi:** §6.5 "Supplementary analysis registered after Study 3", §6.6 "What to run" (Bảng 5 = `tab:recommend`), abstract, introduction, limitations, conclusion, đoạn chuỗi dài ở `theory.tex`, Phụ lục F; các bảng `tab:s3b` (Bảng 4), `tab:recommend` (Bảng 5), `tab:s3b-tfm` (Bảng 20), `tab:s3b-long` (Bảng 21), `tab:s3b-cpu` (Bảng 22). PDF: `paper/main.pdf` (commit `b1e0d69`).
- **Cách làm:** em tự viết script, không import code nào của dự án và không đọc `S3/results/s3b_*` khi tính. Chỉ sau khi có số mới mở `s3b_summary.json` và `s3b_report.py` để giải thích chỗ lệch. Các script nằm ở `/tmp/claude-0/-home-claude/643477cb-9e2c-5e3b-9dd1-1dba7a8f3cd0/scratchpad/audit_r3/`: `load.py`, `step1_s3.py`, `step2_s3b.py`, `step3_extra.py`, `step4_compare.py`, đầu ra là `step*_out.txt`. Mọi tệp đọc bằng `python3 -I`, npz đọc với `allow_pickle=False`.
  - v1, A1, L và X tính lại từ 213 npz của Study 3. Đem đối chiếu với `s3_cells.csv` và `s3_datasets.csv` thì lệch bằng 0 (X lệch tối đa 8·10⁻⁸).
  - naive, hard-EM, oracle lấy từ `s3b_tfm_*.csv`; GBDT, PL-kNN từ `s3b_cpu_*.csv`; chuỗi 100 vòng từ `s3b_long_*.csv` và 72 npz.
  - Bootstrap phân vị 20.000 lần. Wilcoxon hai phía của scipy 1.18 (chính xác). Holm áp trên 3 phép so của B1. Hoà khi |d| < 0,001.
- **Quy tắc gộp:** đúng như bài nói: với mỗi dataset, lấy trung bình trên 3 seed và trên các TFM chạy được, rồi lấy trung bình trên dataset. Có hai ngoại lệ, xem phát hiện N4 và N1:
  - "better CPU baseline" được chọn theo từng dataset, không theo bối cảnh;
  - độ đúng nhãn ở B3 là trung bình cửa sổ 5 vòng, không phải giá trị tại vòng 30 và vòng 100.
- Em không sửa tệp nào của repo, ngoài việc tạo báo cáo này.

## (a) Tóm tắt (5 dòng)

1. **Số khớp.** Em tính lại từ dữ liệu thô và so với bản in:
   - 130/130 macro của `numbers_s3b.tex` khớp tới làm tròn khi dùng định nghĩa của script (46 macro được in, cả 46 có nguyên văn trong PDF). Một macro lệch nếu dùng định nghĩa của kế hoạch, xem dòng 3.
   - 120/120 ô Bảng 22, 40/40 ô Bảng 20, 12/12 ô Bảng 4, 16/16 điểm số của Bảng 5 và mọi ô đếm của Bảng 21 cũng khớp. Sáu khoảng bootstrap lệch ≤ 0,003, chỉ là nhiễu Monte Carlo.
   - Mã băm khớp hết: 867 dòng và 72 npz trùng lần chia và nhãn yếu của Study 3; 87/87 file code trùng `b59ef14` ở cả 3 lượt.
   - Phiên 2 chỉ thêm đúng 10 dòng, tức 5 ô.
2. **Một chuỗi gõ tay sai trong bảng khuyến nghị.** Hard-EM ghi "≈310 passes" (`make_s3b.py:171`). Dữ liệu là 101 lượt (LLP) và 106 lượt (PLL), khoảng 1/3 một chuỗi. Vì thế §6.6 bỏ mất ý chính: ở LLP, hard-EM ngang chuỗi đơn (0,800 so với 0,801) với 1/3 số lượt gọi TFM.
3. **Một số lệch do định nghĩa.** Bảng 21 ghi "label acc. sweep 30 → 100" của TabICL v2 là 0,76 → 0,78. Theo kế hoạch (§5, giá trị ở vòng 100) thì phải là 0,79. Script lấy trung bình vòng 96–100, mà bài không nói.
4. **Câu chữ vượt số.**
   - Abstract và intro chỉ nêu vế đúng của P22, trái với chính câu "wrong predictions reported as prominently as the confirmations".
   - `theory.tex:94` viết "reached the multi-start average or the tuned learner"; đọc theo chữ thì là 14/35 ô, không phải 5/35.
   - "L … not the strongest baseline" chỉ đúng ở LLP: ở PLL, L hơn PL-kNN 7/0/1, trung bình +0,097.
   - "still rising" không có con số nào trong bài đỡ.
5. **Thủ tục và tài liệu.**
   - Phụ lục gọi "cần hai phiên" là deviation, nhưng kế hoạch đã ước 2 phiên. Sai lệch thật là chấm B3 hai lần.
   - Câu "would fail whatever the missing cells gave" sai về logic.
   - "Default hyperparameters" không đúng: code tắt early stopping.
   - TabDPT không tất định (12/24 chuỗi 100 vòng đã khác chuỗi 30 vòng ngay trong 30 vòng đầu), nhưng bài không nói.

**Đếm theo mức:** Nghiêm trọng 0 · Lớn 4 · Nhỏ 17.

## (b) Bảng các số đã kiểm

Ký hiệu: v1 là chuỗi đơn, A1 là multi-start, X là multi-TFM, L là learner đã tune. "Tính lại" ghi giá trị trước khi làm tròn khi cần.

| # | Vị trí | Số in | Tính lại | Kết luận |
|---|---|---|---|---|
| 1 | abstract.tex:16–17 | X thắng GBDT 19 of 20 | 19/0/1 (thua 4538: 0,499 so với 0,534) | Khớp |
| 2 | abstract.tex:17 | X thắng PL-kNN "all 8" | 8/0/0 | Khớp |
| 3 | experiments.tex:178 | X − GBDT +0,135 | +0,1350 | Khớp |
| 4 | experiments.tex:178 | KTC [+0,051, +0,234] | [+0,051, +0,234] | Khớp |
| 5 | experiments.tex:178–179 | trung vị +0,031; 19/0/1 | +0,0306; 19/0/1 | Khớp |
| 6 | experiments.tex:179 | p_Holm < 0,001 | 0,00025 (thô 8,2·10⁻⁵, Holm trên 3) | Khớp |
| 7 | experiments.tex:179–180 | X − PL-kNN +0,188; 8/0/0; p_Holm = 0,016 | +0,1882; 8/0/0; 0,0156 (thô 2/2⁸) | Khớp |
| 8 | experiments.tex:181 | v1 − GBDT +0,031 [−0,003, +0,071] | +0,0308 [−0,003, +0,071] (lần chạy thứ hai: +0,072) | Khớp (nhiễu Monte Carlo) |
| 9 | experiments.tex:181 | trung vị +0,003; 11/1/8 | +0,0029; 11/1/8 | Khớp |
| 10 | experiments.tex:182 | LLP: v1 0,801 so với GBDT 0,796 | 0,8006 / 0,7958 (6/1/5) | Khớp |
| 11 | experiments.tex:182–183 | PLL: v1 0,589 so với PL-kNN 0,620 | 0,5891 / 0,6201 | Khớp; xem N7 |
| 12 | experiments.tex:184–185 | L − baseline CPU tốt hơn +0,022; trung vị −0,003; 7/1/12; p_Holm = 0,841 | +0,0215; −0,0035; 7/1/12; 0,841, với "tốt hơn" chọn theo **từng dataset**. Nếu chọn theo bối cảnh (PL-kNN cho PLL): +0,031; −0,002; 9/1/10; p 0,522 | Khớp với script; định nghĩa lệch kế hoạch, xem N4 và L4 |
| 13 | experiments.tex:189 | v1 − naive +0,128 (18/1/1) | +0,1283; 18/1/1 | Khớp |
| 14 | experiments.tex:190 | log loss: 17/0/3, p < 0,001 hai phía | 17/0/3; p = 8,2·10⁻⁵; trung bình hard-EM − v1 = +0,906 | Khớp |
| 15 | experiments.tex:191 | v1 − hard-EM +0,020 tổng; +0,001 LLP | +0,0204; +0,0007; (PLL +0,050, 5/1/2, không in) | Khớp; xem N6 |
| 16 | experiments.tex:191–193 | X dưới oracle tốt nhất 0,085; LLP 0,052; PLL 0,135 | −0,0853; −0,0518; −0,1355 (thấp hơn ở 20/20 dataset, quá 0,02 ở 12) | Khớp; xem N8 |
| 17 | experiments.tex:193–195 | TabPFN-3 PLL: v1 0,371; hard-EM 0,367; naive 0,357; A1 0,650 | 0,3708; 0,3665; 0,3573; 0,6498 | Khớp; xem N9 |
| 18 | experiments.tex:197 | 72 ô B3 | 72 (3 TFM × 8 × 3) | Khớp |
| 19 | experiments.tex:198 | 35 ô kẹt | 35 (v1 < max(A1, L) − 0,02, từ npz S3) | Khớp |
| 20 | experiments.tex:200 | 5 ô được sửa, 14% | 5/35 = 14,29% | Khớp |
| 21 | experiments.tex:201 | lợi trung bình +0,053 | +0,05251 | Khớp |
| 22 | experiments.tex:201 | TabPFN-3 +0,111 | +0,1107 (70% tổng lợi) | Khớp |
| 23 | experiments.tex:202 | độ đúng nhãn TabPFN-3 0,40 → 0,49 | điểm vòng 30/100: 0,4034 → 0,4942; cửa sổ 28–32/96–100: 0,4026 → 0,4884 | Khớp theo cả hai định nghĩa |
| 24 | experiments.tex:208 | "9 or 12" chuỗi | 3 hoặc 4 TFM × 3 chuỗi | Khớp |
| 25 | experiments.tex:209 | 0,801; 0,800; 0,796 AUC | 0,8006; 0,7998; 0,7958 | Khớp |
| 26 | experiments.tex:216 (chú thích Bảng 5) | "a chain costs 310 passes"; trung vị 92 s đến 208 s | 310 lượt (LLP) và 315 lượt (PLL, cộng 5 lượt khởi động ấm); 91,9 s (TabICL) và 208,4 s (TabPFN v2), trung vị trên 3 chuỗi | Khớp gần đúng; số 310 gõ tay, xem N14 |
| 27 | theory.tex:94–95 | 5 of 35; +0,053 | 5/35; +0,0525 | Số khớp; câu chữ sai định nghĩa, xem L3 |
| 28 | appendix.tex:272–273 | plan commit, code commit "six minutes later" | `d9e6062` 07:21:22 UTC, `b59ef14` 07:27:00 UTC (5 phút 38 giây) | Khớp |
| 29 | appendix.tex:273 | 87 file code khớp ở mọi lượt | `sha256sum -c` trên cây `b59ef14`: 87/87 OK. Ba tệp mã băm (S3B, S3BC, S3B phiên 2) giống hệt nhau | Khớp |
| 30 | appendix.tex:276–277 | "none differed" | 0 lệch trên 867 dòng CSV và 72 npz | Khớp |
| 31 | appendix.tex:278, 280 | 5/72 ô thiếu; phiên 2 thêm đúng 5 ô, không đổi dòng cũ | `git diff 392b8de ce0fe93` trên CSV: +10 dòng, 0 dòng xoá hay sửa; thêm 5 npz | Khớp |
| 32 | appendix.tex:281–282 | lần xem tạm cho thấy vế 2 của P22 sai "whatever the missing cells gave" | trung bình trên 67 ô = +0,0564. Vế 2 chỉ đúng nếu 5 ô thiếu mất trung bình hơn 0,324 accuracy (trường hợp xấu nhất cho trung bình 72 ô = −0,014) | **Sai về logic**, xem N11 |
| 33 | Bảng 4 (`s3b_main`) | 6 hàng × 2 cột: 0,828/0,808; 0,807/0,725; 0,801/0,589; 0,783/0,717; 0,796/0,519; –/0,620 | 0,8282/0,8083; 0,8074/0,7251; 0,8006/0,5891; 0,7830/0,7168; 0,7958/0,5193; –/0,6201 | 12/12 khớp |
| 34 | Bảng 5, điểm số | 16 ô (8 hàng × 2) | naive 0,7176/0,3929; hard-EM 0,7998/0,5392; các hàng khác như ở dòng 33 | 16/16 khớp |
| 35 | Bảng 5, cột chi phí | naive "1 pass" | `n_tfm_calls` = 1 ở mọi dòng | Khớp |
| 36 | Bảng 5, cột chi phí | hard-EM "≈310 passes" | `n_tfm_calls` = **101** (LLP) và **106** (PLL) ở mọi dòng; code: 10 vòng EM × 10 khối + 1 | **Sai**, xem L1 |
| 37 | Bảng 5, cột chi phí | chuỗi đơn "310 passes" | 310 (LLP), 315 (PLL) | Gần đúng, xem N14 |
| 38 | Bảng 5, cột chi phí | multi-start "3 chains + tuned L" | `gibbs_ms` 930/935 lượt | Khớp |
| 39 | Bảng 5, cột chi phí | L "20 tuning runs" | `n_configs=20` (`baselines_ext.py:250, 388`) | Khớp |
| 40 | Bảng 20 (`app_s3b_tfm`) | 8 hàng × 5 cột | tính lại từng TFM, số dataset 12/8/12/8/12/8/10/1 | 40/40 khớp |
| 41 | Bảng 21: cells, stuck, repaired, gain | 24/9/3/+0,031; 24/16/1/+0,111; 24/10/1/+0,016; 72/35/5/+0,053 | giống hệt (gain 0,0307; 0,1107; 0,0161; 0,0525) | Khớp |
| 42 | Bảng 21: độ đúng nhãn | TabICL v2 0,76 → **0,78**; TabPFN-3 0,40 → 0,49; TabDPT 0,63 → 0,65 | điểm: 0,7636 → **0,7852**; 0,4034 → 0,4942; 0,6275 → 0,6501. Cửa sổ: 0,7636 → 0,7839; 0,4026 → 0,4884; 0,6276 → 0,6475 | TabICL r100 lệch theo định nghĩa của kế hoạch, xem N1 |
| 43 | Bảng 22 (`app_s3b_cpu`) | 20 hàng × 6 cột | tính lại từng ô | 120/120 khớp |
| 44 | `numbers_s3b.tex`, 84 macro không in | ví dụ `a1-gbdt`, `x-cpu`, `oracle-gap-*`, `b3-rising` = 19, `p22` = false | khớp, trừ 5 KTC không in lệch ≤ 0,003 (Monte Carlo) và `tabicl_v2-r100` (dòng 42) | Khớp |

Ghi chú về tính tất định. Ở TabICL v2 và TabPFN-3, 30 vòng đầu của chuỗi 100 vòng trùng hệt chuỗi 30 vòng của Study 3 (24/24 ô mỗi TFM). Ở **TabDPT chỉ 12/24 ô trùng**; chênh độ đúng nhãn tối đa là 0,135. Xem N3.

## (c) Phát hiện

### Lớn

**L1. Chi phí hard-EM ghi sai 3 lần trong bảng khuyến nghị, nên §6.6 bỏ mất ý chính.**
- **Vị trí:** `paper/scripts/make_s3b.py:171`, sinh ra `paper/tables/s3b_recommend.tex:6`; câu văn ở `experiments.tex:208–210`.
- **Vấn đề:** Bảng 5 ghi hard-EM "≈310 passes". Đây là chuỗi gõ tay trong script, không đọc từ dữ liệu.
  - Trong `s3b_tfm_*.csv`, `n_tfm_calls` của `hard_em_warm` là 101 ở mọi dòng LLP và 106 ở mọi dòng PLL.
  - Code ở `b59ef14` cho cùng con số: `GIBBS["em_iters"]=10`, `n_blocks=10` (`G4/code/icgibbs/g7/methods.py:17, 483–485`; `icgibbs/baselines.py:47–53`), tức 10 × 10 + 1 lượt.
  - Chuỗi đơn tốn 310 hoặc 315 lượt. Vậy hard-EM tốn khoảng 1/3 một chuỗi.
  - Thời gian thực, trung vị, có tính cả khởi động ấm (TabICL v2): LLP 89 s so với 111 s; PLL 58 s so với 174 s.
- **Hệ quả:** bảng làm chuỗi đơn trông không đắt hơn hard-EM. Câu "For label proportions a single chain, hard-EM … and untuned tree-based self-training are level" vì thế không rút ra được điều hiển nhiên từ dữ liệu: ở LLP, chạy chuỗi đơn thay cho hard-EM (1/3 số lượt) hay GBDT (CPU) đều không mua thêm được gì.
- **Sửa:**
  1. Ở `make_s3b.py:171–172`, đọc `n_tfm_calls` từ CSV thô: hard-EM "101 passes (LLP), 106 (PLL)"; chuỗi đơn "310 (LLP), 315 (PLL) passes".
  2. Thêm vào §6.6: "hard-EM reaches the same AUC with about a third of the forward passes, and GBDT self-training needs no GPU".

**L2. Abstract và introduction chỉ nêu vế đúng của P22.**
- **Vị trí:** `abstract.tex:18`, "running chains for 100 instead of 30 sweeps rarely repaired a stalled chain"; `introduction.tex:37–38`, "longer chains rarely repaired a stalled one".
- **Vấn đề:** P22 sai vì lợi trung bình là +0,053, vượt ngưỡng ghi trước 0,03. Lợi đến chủ yếu từ TabPFN-3: +0,111, độ đúng nhãn 0,40 → 0,49. Abstract và intro không nhắc gì, trong khi `introduction.tex:44–46` viết "the failures … and the wrong predictions reported as prominently as the confirmations". Limitations có nhắc (`limitations.tex:10`), abstract thì không.
- **Sửa (abstract):** "…and running chains for 100 instead of 30 sweeps repaired only 5 of 35 stalled chains, although, against our registered prediction, it raised accuracy on average (+0.053, mostly with TabPFN-3)". Sửa intro tương tự, dùng macro.

**L3. Định nghĩa "repaired" ở theory.tex sai; đọc theo chữ thì là 14/35, không phải 5/35.**
- **Vị trí:** `theory.tex:94`, "\val{s3b-b3-rep} of \val{s3b-b3-stuck} stalled chains reached the multi-start average or the tuned learner".
- **Vấn đề:**
  - Định nghĩa đã đăng ký (`S3B_PLAN.md:52`; chú thích Bảng 21) là đạt trong vòng 0,02 của max(A1, L), tức của cái **tốt hơn**.
  - Câu "reached A or L" nghĩa là đạt ít nhất một trong hai. Trên dữ liệu, số ô thoả "trong vòng 0,02 của A1 hoặc của L" là 14/35; số ô thoả "≥ A1 hoặc ≥ L" cũng là 14/35. Câu còn bỏ dung sai 0,02.
  - `experiments.tex:199` có chỗ mơ hồ tương tự: "would come within 0.02 of it", trong đó "it" đi sau "the multi-start average or $L$".
- **Sửa:**
  - theory: "…came within 0.02 of the better of the multi-start average and the tuned learner".
  - experiments.tex:198–199: "…more than 0.02 behind the better of the multi-start average and $L$ would come within 0.02 of that better score".

**L4. Câu "L … a reference, not the strongest baseline" chỉ đúng ở LLP; số gộp che mất chiều ngược lại ở PLL.**
- **Vị trí:** `experiments.tex:184–186`.
- **Vấn đề:** phép so đã đăng ký, gộp 20 dataset, cho +0,022 và 7/1/12. Tách theo bối cảnh thì hai chiều ngược nhau:
  - **LLP:** L − GBDT = −0,013, 2/1/9. DLLP-MLP đã tune thua GBDT tự huấn luyện không tune ở 9/12 dataset.
  - **PLL:** L − max(GBDT, PL-kNN) theo từng dataset = +0,073, 5/0/3. L − PL-kNN = +0,097, 7/0/1. L − GBDT = +0,198.
  - Trên PLL, L là baseline mạnh nhất hiện có. Câu văn khiến người đọc nghĩ Study 3 dùng một mốc yếu ở cả hai bối cảnh. Hướng sai lệch là bất lợi cho chính bài, nhưng nó vẫn là khẳng định sai về một nửa dữ liệu.
- **Sửa:** "On LLP the tuned DLLP-MLP was below untuned GBDT self-training on 9 of 12 datasets (−0.013); on PLL tuned CC was above both untuned baselines (+0.097 over PL-kNN, 7/0/1). On these LLP datasets $L$ is therefore a reference, not the strongest baseline." Cần thêm macro cho các số theo từng bối cảnh.

### Nhỏ

**N1. Độ đúng nhãn ở Bảng 21 không phải "sweep 30 → 100".**
- **Vị trí:** `appendix.tex:300–302`, tiêu đề cột `make_s3b.py:150`; `experiments.tex:202`.
- **Vấn đề:**
  - Kế hoạch (`S3B_PLAN.md:56`) định nghĩa là "độ đúng nhãn cuối ở vòng 30 và vòng 100 của cùng chuỗi".
  - Script (`S3/s3b_report.py:151–152`) lấy `mean(mon[27:32])` và `mean(mon[-5:])`, tức trung bình các vòng 28–32 và 96–100.
  - Theo định nghĩa của kế hoạch, TabICL v2 r100 = 0,7852, in ra phải là **0,79**; bảng in 0,78. Hai TFM còn lại không đổi khi làm tròn.
- **Sửa:** hoặc sửa chú thích thành "label accuracy averaged over sweeps 28–32 and 96–100 (as in the scoring script)", hoặc in giá trị tại vòng 30 và vòng 100 đúng như kế hoạch.

**N2. "Still rising" không có số nào đỡ, và chỉ số đã đăng ký không đo điều đó.**
- **Vị trí:** `experiments.tex:202–203` ("was still rising in many cells (\cref{tab:s3b-long})"); `theory.tex:95–96` ("whose label accuracy was still rising after 100 sweeps"; "A stalled chain is therefore a mixture of a wrong basin and slow mixing").
- **Vấn đề:**
  - Bảng 21 không có cột nào cho ý này. Macro `s3b-b3-rising` = 19 không được in.
  - Chỉ số ghi trước đo mức tăng từ quanh vòng 30 đến cuối chuỗi, không đo việc độ đúng nhãn còn đang tăng ở vòng 100. Theo chỉ số này, TabPFN-3 có 10/24 ô.
  - Độ dốc 20 vòng cuối > 0,01 trên mỗi 10 vòng chỉ có ở 7/24 ô TabPFN-3 (tính thăm dò của kiểm toán viên).
  - "Therefore" mạnh hơn bằng chứng: 30/35 ô kẹt không sửa được sau 100 vòng. Điều đó hợp với cả giả thuyết lưu vực sai lẫn giả thuyết trộn rất chậm.
- **Sửa:**
  - Thêm cột "rose > 0.03 after sweep 30" vào Bảng 21 (TabICL v2 5, TabPFN-3 10, TabDPT 4, tổng 19; định nghĩa như kế hoạch).
  - Viết lại: "rose by more than 0.03 after sweep 30 in 10 of 24 cells".
  - Thay "therefore" bằng "consistent with".

**N3. "Gain" so hai thứ khác nhau, ngoài số vòng quét.**
- **Vị trí:** chú thích Bảng 21 (`appendix.tex:301–302`); `experiments.tex:197–203`.
- **Vấn đề:**
  - Chuỗi 100 vòng dùng burn-in 33, còn chuỗi 30 vòng dùng burn-in 10 (`S3B_PLAN.md:49`). Bài không nói điều này.
  - Với TabDPT, 12/24 chuỗi 100 vòng đã khác chuỗi 30 vòng của Study 3 ngay trong 30 vòng đầu, chênh độ đúng nhãn tới 0,135. Độ đúng nhãn ở vòng 30 là 0,6275 so với 0,6307. Vậy "gain" của TabDPT (+0,016) và việc 1/10 ô "được sửa" có lẫn nhiễu giữa hai lần chạy.
- **Sửa:** thêm vào chú thích: "The 100-sweep chain uses burn-in 33; with TabDPT, runs are not bit-reproducible (12 of 24 chains differ from the Study 3 chain within the first 30 sweeps), so its gain includes run-to-run variation."

**N4. "The better untuned non-neural baseline" được chọn theo từng dataset, tức có lợi thế nhìn sau.**
- **Vị trí:** `experiments.tex:184`; `appendix.tex:274`.
- **Vấn đề:**
  - Kế hoạch (`S3B_PLAN.md:37`) viết "baseline CPU tốt hơn của từng bối cảnh". Script (`s3b_report.py:85`) lại lấy `max` theo dataset.
  - Chọn theo bối cảnh (PL-kNN cho PLL, vì trung bình 0,620 > 0,519) thì kết quả là +0,031, trung vị −0,002, 9/1/10, p_Holm = 0,522.
  - Kết luận "không hơn" không đổi, nhưng định nghĩa phải được nêu.
- **Sửa:** "the better of the two untuned baselines on each dataset (chosen with hindsight)". Ghi lệch kế hoạch–script thành một mục deviation có ngày trong `S3B_REPORT.md`.

**N5. Phụ lục mô tả số phép kiểm định khác với cách tính và cách in.**
- **Vị trí:** `appendix.tex:273–274`, "two tests (… Holm-adjusted) and one comparison of $L$".
- **Vấn đề:**
  - Holm được áp trên **3** phép so, kể cả L, đúng như kế hoạch §3. Chính p = 0,016 của PL-kNN chỉ ra được khi Holm trên 3; nếu Holm trên 2 thì p = 0,008.
  - Thân bài in $p_{\mathrm{Holm}}$ cho cả phép so L.
  - Kế hoạch gọi các kiểm định B1 là "mô tả", nhưng `experiments.tex:178–180` in p mà không có nhãn đó, trong khi phép so v1 − GBDT thì có nhãn "descriptive".
- **Sửa:** "three Wilcoxon comparisons, Holm-adjusted together (descriptive, as registered)"; thêm "descriptive" vào câu ở dòng 178.

**N6. "A similar score" so với hard-EM che mất phần PLL và sự khác nhau giữa các TFM.**
- **Vị trí:** `experiments.tex:188–191`.
- **Vấn đề:**
  - Ở PLL, v1 − hard-EM = +0,050 (5/1/2) nhưng không được in. Theo từng TFM: TabICL v2 PLL +0,071, TabDPT LLP −0,028 (hard-EM tốt hơn), TabPFN v2 PLL −0,038.
  - "With each of the four TFMs … improved on the naive TFM by +0.128" là số gộp trên các TFM, không phải số của từng TFM.
- **Sửa:** "…a similar score on LLP (+0.001) and a higher one on PLL (+0.050, 5/1/2), with TabDPT on LLP the exception (hard-EM 0.814 against 0.786)". Câu đầu đổi thành "Averaged over the TFMs of each dataset, …".

**N7. Giải thích "because of the stalled chains on datasets with many classes" là thăm dò, và trung bình gây hiểu lầm.**
- **Vị trí:** `experiments.tex:182–183`.
- **Vấn đề:**
  - v1 chỉ dưới PL-kNN ở 3/8 dataset: 42 (C = 19), 307 (C = 11), 40499 (C = 11).
  - Trung vị v1 − PL-kNN = +0,029, thắng/hoà/thua 5/0/3.
  - Ở 1515 (C = 20, 8/9 ô kẹt), v1 lại hơn PL-kNN (0,291 so với 0,209).
  - Cũng ở câu đó, v1 hơn GBDT trên PLL (0,589 so với 0,519) mà bài không nói.
- **Sửa:** "on PLL the single chain's mean was below PL-kNN (0.589 against 0.620), because of three datasets on which it stalled (exploratory; median difference +0.029, 5/0/3)". Dùng macro `s3b-v1-below-plknn-n` đã có sẵn.

**N8. "Stayed 0.085 below the best oracle of each dataset" thiếu chữ "on average".**
- **Vị trí:** `experiments.tex:191–193`.
- **Vấn đề:** 0,085 là trung bình. X thấp hơn oracle tốt nhất ở 20/20 dataset và thấp hơn quá 0,02 ở 12 dataset.
- **Sửa:** "stayed on average 0.085 below the best oracle of each dataset (more than 0.02 below on 12 of 20)". Các macro `x-minus-best-oracle-nworse`, `-wtl` đã có.

**N9. "Only the learner start moved it (0.650)": con số đi kèm là trung bình ba chuỗi.**
- **Vị trí:** `experiments.tex:194–195`.
- **Vấn đề:**
  - 0,650 là A1, tức trung bình 3 chuỗi. Riêng chuỗi xuất phát từ L đạt 0,692, chuỗi cold đạt 0,373 (tính thăm dò từ npz S3).
  - Bản thân L đạt 0,717, hơn cả A1 lẫn chuỗi xuất phát từ L. Với TabPFN-3 trên PLL, TFM không thêm được gì so với L.
- **Sửa:** "only the learner start moved it (multi-start 0.650; the learner itself 0.717)".

**N10. Phụ lục ghi sai đâu là deviation.**
- **Vị trí:** `appendix.tex:277–281`.
- **Vấn đề:**
  - "Deviation from the plan's schedule: the GPU part needed two Kaggle sessions". Nhưng kế hoạch §7 (`S3B_PLAN.md:69`) đã viết "ước 2 phiên", nên đây không phải lệch lịch.
  - Lệch thật so với kế hoạch là hai việc. Thứ nhất, kế hoạch §7 yêu cầu chấm "đúng một lần trên dữ liệu đủ", nhưng B3 được chấm tạm theo §5 rồi chấm lại sau khi xem tạm 67 ô và chạy thêm 5 ô. Thứ hai, `S3B_PLAN.md` không có mục deviation nào có ngày.
  - "The remaining cells were looked at once" mơ hồ: thực tế là 43 ô đã có của TabICL v2 và TabPFN-3, không phải 5 ô còn thiếu.
- **Sửa:** "Deviation: the plan asked for one scoring on complete data; B3 was first scored on the complete TFM (as §5 prescribes for incomplete runs), the other 43 available cells were looked at once, the 5 missing cells were then run and B3 was scored again on all 72." Ghi mục deviation có ngày vào `S3B_REPORT.md`.

**N11. "Would fail whatever the missing cells gave" sai về logic.**
- **Vị trí:** `appendix.tex:281–282`.
- **Vấn đề:** trung bình trên 67 ô là +0,0564. Vế 2 của P22 vẫn có thể đúng nếu 5 ô thiếu mất trung bình hơn 0,324 accuracy; trường hợp xấu nhất cho trung bình 72 ô là −0,014. Kết quả cuối không đổi, nhưng câu văn sai.
- **Sửa:** "…would fail unless the 5 missing cells lost more than 0.32 accuracy each on average (their 30-sweep accuracy was 0.96)".

**N12. "Default hyperparameters" và "default configuration" không đúng với code.**
- **Vị trí:** `appendix.tex:284`; Bảng 4 ("default configuration"); Bảng 5 ("default").
- **Vấn đề:** code `G4/code/icgibbs/baselines_cpu.py:20–21` đặt `early_stopping=False`. Mặc định của sklearn là `'auto'`, tức bật early stopping khi n > 10.000. Dự đoán ngoài fold dùng 2 fold, điều mà bài không nói.
- **Sửa:** "a fixed configuration (100 iterations, learning rate 0.1, no early stopping; 2-fold out-of-fold relabelling)".

**N13. Kết luận viết mạnh hơn bằng chứng.**
- **Vị trí:** `conclusion.tex:5–6`, "while the single chain alone did not".
- **Vấn đề:** v1 − GBDT = +0,031 [−0,003; +0,071], 11/1/8. Ở chỗ khác bài viết đúng là "not clearly".
- **Sửa:** "while the single chain alone did not clearly do so".

**N14. Số gõ tay ngoài macro.**
- **Vị trí và vấn đề:**
  - `limitations.tex:19` "twelve chains": macro `s3x-cost-nchains` đã có.
  - `appendix.tex:278, 280` "5 of the", "exactly the 5": đây là số kết quả.
  - `experiments.tex:216` "310 passes": PLL thực ra là 315.
  - `appendix.tex:272` "12 November 2026", "six minutes".
  - `make_s3b.py:170–177`: các chuỗi chi phí "≈310", "310", "3 chains", "9--12", "20 tuning runs" là hằng số gõ tay; chuỗi "≈310" sai (L1).
- **Sửa:** sinh mọi số trên từ dữ liệu: `n_tfm_calls`, số dòng thiếu, giờ commit lấy từ `git log`.

**N15. Hai bảng mô tả khác nhau phần huấn luyện của cùng một cấu hình.**
- **Vị trí:** Bảng 4 (`make_s3b.py:131–132`) và Bảng 5 (`make_s3b.py:173–174`).
- **Vấn đề:** với multi-start và multi-TFM, Bảng 4 ghi "starts use $L$ and linear DLLP", còn Bảng 5 ghi "tuned $L$". Thực tế chuỗi `warm` của LLP vẫn dùng DLLP tuyến tính.
- **Sửa:** Bảng 5 ghi "tuned $L$; LLP: also linear DLLP".

**N16. Đánh số dự đoán có lỗ hổng không được giải thích.**
- **Vị trí:** `appendix.tex:320`.
- **Vấn đề:** bài nhảy từ P18 (Bảng 23) sang P22. P19–P21 thuộc kế hoạch V1E đã huỷ (`V2/V1E_PLAN.md:24–32`). Câu ở dòng 320 chỉ giải thích P10.
- **Sửa:** "(the number P10 was never assigned; P19–P21 belonged to a cancelled run; P22 is reported in Appendix F)".

**N17. Tài liệu nội bộ không khớp nhau (không in trong bài).**
- **Vị trí và vấn đề:**
  - `S3/S3B_REPORT.md:4` ghi "Kaggle, một phiên", trong khi cùng tệp, dòng 10, nói về phiên 2.
  - `V2/EXPERIMENT_LOG.md:79` ghi phiên 2 lúc "06:28 – 10:58". `s2_S3B_phien2_master_log.txt` cho thấy lượt bắt đầu 06:06:22 UTC, tức **trước** lần xem tạm (06:12).
  - Câu "211 dòng cũ không đổi" (`S3B_REPORT.md:10`, `EXPERIMENT_LOG.md:79`) không khớp số đếm em kiểm được: trước phiên 2, B3 có 134 dòng CSV và 67 npz; git diff cho 0 dòng sửa trên mọi CSV.
- **Sửa:** sửa ba chỗ theo log gốc và ghi rõ đã đếm cái gì.

## Thông tin cho phiên chính

Không có nội dung nào trong báo cáo này lấy từ bộ nhớ của người dùng. Các con số "thăm dò" ở N2 (độ dốc cuối chuỗi) và N9 (độ chính xác theo từng điểm xuất phát) là phép tính của kiểm toán viên sau khi thấy dữ liệu, không thuộc protocol.
