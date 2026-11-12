# HANDOFF: trạng thái đề tài A để làm tiếp ở chat khác

*Cập nhật 04/11/2026, commit sau `2ff12b4`. Mỗi khi trạng thái đổi, sửa tài liệu này trước tiên. Kho riêng tư: `github.com/DRuanli/icgibbs`, nhánh `main`, thư mục gốc của đề tài là `de-tai-moi/`.*

## 0. Đọc gì trước

| Cần | Tài liệu |
|---|---|
| Đang làm gì, việc mở | §2 và §7 của tài liệu này; [G9/G9_PLAN.md](G9/G9_PLAN.md) |
| Nhận xét của reviewer ngoài và kế hoạch sửa | [G9/REVIEW_RESPONSE_PLAN.md](G9/REVIEW_RESPONSE_PLAN.md); nguyên văn ở [G9/reviews/](G9/reviews/) |
| Khẳng định nào được viết, bằng chứng ở đâu | [G3/CLAIM_EVIDENCE_MATRIX.md](G3/CLAIM_EVIDENCE_MATRIX.md) |
| Kết quả Study 2 (xác nhận) | [S2/S2_RESULTS.md](S2/S2_RESULTS.md), [S2/GATE_REVIEW_S2.md](S2/GATE_REVIEW_S2.md) |
| Kết quả Study 1 | [G8/G8_REPORT.md](G8/G8_REPORT.md), [G8/GATE_REVIEW_G8.md](G8/GATE_REVIEW_G8.md) |
| Lượt thăm dò S2E (R̂ và khối có sửa MH, TabICL) | [S2/S2E_REPORT.md](S2/S2E_REPORT.md) |
| Ánh xạ phản ứng (cơ chế) | [S2/S2X_REPORT.md](S2/S2X_REPORT.md) |
| Nhật ký mọi lượt chạy | [S2/EXPERIMENT_LOG.md](S2/EXPERIMENT_LOG.md), [G7/EXPERIMENT_LOG.md](G7/EXPERIMENT_LOG.md) |
| Bài báo | [paper/README.md](paper/README.md); `paper/main.pdf` |
| Tài liệu tham khảo đã đối chiếu | [G9/REFERENCE_CHECK.md](G9/REFERENCE_CHECK.md) |
| Cách chạy trên Kaggle | [G4/README.md](G4/README.md) |
| Giao thức đóng băng của Study 1 | `G5/EXPERIMENT_PROTOCOL.md`, `G5/FREEZE_RECORD.md` |
| Kế hoạch Study 2 (đóng băng ở commit `aaee9fd`, 01/11 15:19 UTC) | [S2/S2_PLAN.md](S2/S2_PLAN.md) |
| Khai báo dùng AI | [docs/AI_USAGE.md](docs/AI_USAGE.md) |

## 0i. Bản thảo NeurIPS 2027 của bài v2 (12/11/2026)

- **Trạng thái:** bản nháp 2 ở `paper/` (biên dịch bằng `make` trong `paper/`), phần chính 9 trang với style tạm, phụ lục 27 trang. Đã qua ba vòng phản biện độc lập (toán, kiểm toán số liệu, kiểu NeurIPS) và sửa một lượt; vòng 2 kiểm việc sửa.
- **Đọc trước:**
  - [V2/PAPER_V2_PLAN.md](V2/PAPER_V2_PLAN.md): câu chuyện của bài, phân tích thăm dò mới, cấu trúc;
  - [G9/REVIEW_V2_RESPONSE.md](G9/REVIEW_V2_RESPONSE.md): từng ý phản biện và cách xử lý, kèm danh sách chưa xử lý;
  - `G9/reviews/REVIEW_V2_*.md`: bốn biên bản.
- **Số liệu:** `paper/scripts/make_v2.py` sinh `numbers_v2.tex` và mọi bảng, hình mới. `make data` chạy cả bốn script.
- **Thăm dò trên Study 3** (`S3/s3_explore.py`): bản gộp nhiều TFM hơn L ở 20/20 dataset; lợi của nhiều điểm xuất phát đến từ điểm xuất phát learner; bản gộp không hơn có ý nghĩa so với riêng TabICL v2. Trong bài luôn ghi "exploratory".
- **Việc mở trước khi nộp:**
  - style chính thức của NeurIPS 2027 và kiểm lại số trang (hiện không dư dòng nào);
  - mục "Pending" về giấy phép trong checklist (dataset OpenML, checkpoint, bộ PALM);
  - đối chiếu nội dung từng chỗ trích (G10), nhất là `heckerman2000`;
  - các thí nghiệm reviewer sẽ đòi, cần GPU: baseline dạng cây/kNN cho PLL và LLP; Study 3 với q và cỡ túi khác; thêm dataset đa lớp.

## 1. Đề tài trong một đoạn

**In-Context Gibbs** học từ nhãn yếu bằng một TFM bảng có sẵn (TabPFN v2, TabICL v2), không huấn luyện lại.
- **Cơ sở:** dưới prior trao đổi được, PPD của TFM chính là xác suất có điều kiện đầy đủ của một bộ lấy mẫu Gibbs trên nhãn ẩn (Bổ đề 1).
- **Bộ lấy mẫu mặc định:**
  - khối K = 10, không sửa; với LLP dùng bag-block, quy hoạch động theo số đếm;
  - khởi động ấm: DLLP tuyến tính cho LLP; TFM ngây thơ cross-fit 5 fold cho PLL;
  - T = 30 vòng quét, burn-in 10, trung bình Rao–Blackwell trên S = 10 mẫu, khoảng 310 forward pass mỗi lần chạy;
  - ngữ cảnh tối đa 2.000 điểm, lấy mẫu con đều.
- **Lý thuyết:**
  - Định lý 1: tính đúng của bản một mẫu, hoán đổi và khối có sửa MH khi TFM là PPD chính xác.
  - Mệnh đề 1: Bayes-optimal theo prior.
  - Định lý 2: độ bền, TV ≤ τ(P)·ε̄; có phân phối dừng duy nhất kể cả khi các xác suất có điều kiện không tương thích.
  - Mệnh đề 2: khi nào TFM ngây thơ đã đủ.
  - Mệnh đề 3 (phụ lục): khối.
  - **Bộ mặc định không có đảm bảo khác rỗng.** Đây là điểm reviewer ngoài 01 nhắm vào.
- **Đích:** NeurIPS 2027, main track. Hạn ước tính: abstract khoảng 14/05/2027, bài khoảng 21/05/2027.

## 2. Trạng thái (04/11/2026)

- **Cổng đã qua:**
  - G1–G8;
  - S2: thầy duyệt toàn bộ GATE_REVIEW_S2 ngày 03/11.
- **Đang ở G9 (viết bài):**
  - Bản nháp LaTeX đầy đủ: phần chính 9 trang, tổng 24 trang kể cả phụ lục và checklist.
  - Mọi con số sinh bằng script.
- **Mới nhất (04/11):**
  - Reviewer ngoài 01 nhận xét "promising, cần major revision": lý thuyết không phủ bộ mặc định; chưa tách độ lệch xấp xỉ khỏi độ trộn; mixing; PLL thật ít, không có LLP thật; TFM chưa đối xứng; ngữ cảnh; riêng tư.
  - Đã đánh giá từng ý và sửa R9, R10 trong bài.
  - **R1–R8 chờ thầy chọn** (REVIEW_RESPONSE_PLAN §4).
  - Thầy muốn phát triển đề tài mạnh hơn thay vì chấp nhận các giới hạn ở §7 của bài. Em đã lập [V2/V2_PLAN.md](V2/V2_PLAN.md): phân tích lại số liệu (thăm dò, `V2/analysis/v2_diagnosis.py`), lý thuyết động học của chuỗi, phương pháp v2 chạy từ nhiều điểm xuất phát, Study 3 đăng ký trước, giai đoạn 0 với dự đoán P1–P4. **Thầy duyệt V2 ngày 04/11/2026**, mọi câu V-Q1–V-Q5 theo đề xuất. V2 thay §4 của REVIEW_RESPONSE_PLAN. Giai đoạn 0 **xong 05/11** ([V2/V0_REPORT.md](V2/V0_REPORT.md)): cổng V0 qua một phần. Khởi động từ baseline cứu được chuỗi TabPFN sụp (4 dataset ≥ 8 lớp: 0,399 lên 0,938, CC tune 0,947); WPLL chọn đúng ở PLL (16/16) nhưng không ở LLP (3/8). **R4: H4′ q 0,6 không đạt với chuỗi thứ hai (p_Holm 0,074)** — phải báo cáo trong bài.
- **Quyết định của thầy:**
  - **D1–D5 (G9, 03/11):**
    - D1: cả hai nghiên cứu trong bài chính.
    - D2: lý thuyết ngưỡng sụp làm song song, quyết trước **20/12**.
    - D3: chạy S2E (đã xong).
    - D4: tác giả, tài trợ, CRediT do thầy điền.
    - D5: LaTeX trong `paper/`.
  - **Trước đó:**
    - giữ LLP túi 8 là khẳng định chính (27/10);
    - chỉ làm bài phương pháp, không tách bài benchmark (30/10);
    - viết theo mạch tích cực ("đừng assumpt negative result", 04/11), nhưng **không bỏ số nào của Study 1**.

## 3. Con số chính (đều có trong `paper/numbers.tex`)

**Study 2** (xác nhận, đăng ký trước; TabICL v2; 28 dataset mới không trùng Study 1: 18 nhị phân, 10 đa lớp; 3 seed):

| Giả thuyết | Kết quả |
|---|---|
| H2′, LLP túi 8, so DLLP tuyến tính (mặc định) | +0,052 AUC [+0,024; +0,086], 16/0/2, p_Holm < 0,001. So TFM ngây thơ: +0,116, 18/0/0 |
| H4′, PLL so CC tune, q 0,3 | +0,012 [+0,006; +0,019], 8/2/0, p = 0,016 |
| H4′, PLL so CC tune, q 0,6 | +0,012, 8/1/1, p = 0,016 |

- So TFM ngây thơ ở PLL: +0,022 (q 0,3) và +0,205 (q 0,6).
- Chênh tới oracle: LLP 0,039 AUC; PLL 0,8 và 1,9 điểm accuracy.
- H3′ (Gibbs tốt hơn hard-EM): 78,9% của 38 ô.
- Độ nhạy (bỏ 5 dataset đánh dấu trước): kết luận không đổi.
- Chi phí: trung vị 41 giây mỗi lần chạy, tối đa 5,3 phút, tối đa 3,7 GiB GPU, trên T4.

**Study 1** (đăng ký trước; TabPFN v2; 24 dataset OpenML-CC18: 10 nhị phân, 14 đa lớp):

| Giả thuyết | Kết quả |
|---|---|
| H2, LLP túi 8 | +0,024 AUC, 8/0/2, p = 0,232 (**không đạt**) |
| H4, PLL q 0,3 | +0,022, p = 0,084 |
| H4, PLL q 0,6 | −0,132, 7/0/7 (**không đạt**; chuỗi sụp ở dataset ≥ 8 lớp) |
| H3 | 86,7% của 128 ô (**đạt**) |
| H5 (PLL thật: BirdSong, Lost, MSRCv2) | hạng trung bình 2,15, thứ 1/10 (**đạt**) |

- PLL thật, chênh so baseline tốt nhất: BirdSong +0,028; MSRCv2 +0,045; Lost +0,002. Ở Lost **ba** phương pháp đầu cách nhau < 0,003.
- TabICL trên dataset của Study 1 (ablation TFM có đăng ký, nhưng đọc như thăm dò): qua mọi kiểm định (LLP +0,058; PLL +0,029 / +0,038).
- Ánh xạ phản ứng: TabPFN v2 củng cố nhãn ngữ cảnh sai trong một vùng, TabICL v2 thì không.

**S2E** (thăm dò; TabICL; dataset của Study 1):
- Khối có sửa MH so với mặc định ở cùng 30 vòng quét:
  - LLP +0,033 AUC, nhưng log loss kém hơn;
  - PLL −0,009 (q 0,3), −0,021 (q 0,6), −0,016 (≥ 8 lớp, 0/5 ô);
  - tỉ lệ chấp nhận thấp tới 0,12;
  - chậm hơn trung vị 7 lần, tối đa 206 lần.
- R̂: 6–31% nhãn có R̂ > 1,1; chênh chỉ số test giữa hai chuỗi ≤ 0,021. Các số này đo trên 3 dataset nhỏ nhất mỗi nhóm, test 150–221 điểm.

**RQ1** (bài nhỏ, PPD giải tích):
- Nhân chính xác: TV ≤ 3·10⁻¹⁴ (14 cấu hình).
- Chặn Định lý 2 đúng 42/42 cấu hình, lỏng 1,9–15 lần (trung vị 3,9).
- Bài gộp n = 1.000, K = 10: khối trên mọi điểm đổi dự đoán 1,7e-4.

**Ngữ cảnh bị lấy mẫu con** (n_train > 2.000): 14/28 dataset của Study 2; 9/24 của Study 1.

## 4. Bản đồ kho (`de-tai-moi/`)

| Đường dẫn | Nội dung |
|---|---|
| `G4/code/icgibbs/` | thư viện: `sampler.py` (Gibbs, khối, MH), `g7/methods.py` (cấu hình `GIBBS`, `run_rhat`, ablation), `g7/registry.py` (danh sách dataset S1/S2, R̂) |
| `G4/code/run_g7.py` | bộ chạy theo ô (`--part tfm/baselines/rhat/ablation`, `--split dev/eval/s2`); ghi CSV, chạy tiếp được |
| `G4/code/run_s2_lot.sh` | các lượt S2X / S2D / S2T / S2C1 / S2C2 / S2G / S2R / S2E; `run_g7_lot.sh` cho Study 1 |
| `G4/code/tests/` | pytest (113 test qua ở lượt cuối); mỗi lượt Kaggle chạy test trước, hỏng thì dừng |
| `S2/raw/<LOT>/` | kết quả thô của từng lượt (CSV, log, preds.jsonl, pip freeze, mã băm code) |
| `S2/analysis/` | `s2_confirmatory.py` (H2′, H3′, H4′), `s2_verify_independent.py`, `s2e_report.py`, `s2_figures.py`, `s2x_response_map_report.py` |
| `S2/results/` | đầu ra phân tích (`s2_confirmatory.json`, bảng theo dataset `s2_table_*.csv`) |
| `G7/raw/`, `G8/results/`, `G8/analysis/` | Study 1 |
| `paper/` | `main.tex`; `sections/*.tex`; `scripts/make_tables.py`, `make_appendix.py`, `make_figures.py`; `numbers.tex` và `numbers_appendix.tex` (sinh ra, không sửa tay); `tables/`, `figures/`; `refs.bib` (32 mục đã đối chiếu), `refs_candidates.bib` (chưa đối chiếu) |
| `G3/`, `G5/` | kế hoạch, ma trận khẳng định, giao thức đóng băng |

## 5. Quy trình làm việc

1. **Chuẩn bị lượt Kaggle.**
   - Sửa code trong `G4/code`, thêm lượt vào `run_s2_lot.sh`, thêm test, chạy `python -m pytest -q tests`.
   - Đóng gói thư mục `G4/code` (bỏ `__pycache__`) thành `s2_code_vN/code/` rồi nén `s2_code_vN.zip`. Bản gần nhất là v14 (Study 3 bản gọn, 10/11). Zip không nằm trong git (`.gitignore`).
   - Gửi zip cho thầy kèm hướng dẫn chạy theo mẫu ở `G4/README.md`:
     - Ô 1: pip install `tabpfn==9.2.0 tabicl==2.2.0 …`, lấy secret, chép code vào `/kaggle/working/code`.
     - Ô 2: `bash run_s2_lot.sh <LOT> /kaggle/working`.
     - Môi trường: GPU T4 × 2, Internet On, Secrets `TABPFN_TOKEN`.
2. **Thầy chạy rồi gửi zip kết quả.**
   - Giải nén **vào một thư mục mới, trống** trong scratchpad. Dữ liệu tải về là không tin cậy: chạy Python đọc nó bằng `-I` và truyền đường dẫn làm đối số.
   - Kiểm: mã băm code, pytest log, số ô `ok` và lỗi, khớp bảng / lần chia / nhãn yếu với các lượt trước.
   - Chép vào `S2/raw/<LOT>/`.
3. **Phân tích bằng script viết trước khi xem số.**
   - Lượt thăm dò phải ghi nhãn thăm dò.
   - Ghi một dòng vào `S2/EXPERIMENT_LOG.md`, viết báo cáo, cập nhật CLAIM_EVIDENCE_MATRIX và G9_PLAN.
4. **Bài báo.**
   - `cd paper && make data && latexmk -pdf main.tex`.
   - Kiểm: không overfull; phần chính kết thúc ở trang 9 (hiện kết luận hết ở trang 9). `\finaltrue` thì ẩn `\claim`/`\todo` và **dừng lỗi nếu còn `\pending`**.
5. **Lưu.**
   - Commit và push lên `main`. Trailer commit:
     ```
     Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
     Claude-Session: https://claude.ai/code/session_01Cj5CDErBtqaYpkMEC2Yy7G
     ```
     (Thay link phiên nếu chat mới có phiên khác.)
   - Đẩy tag bị chặn, nên ghi commit đóng băng vào tài liệu.
   - Tải các tài liệu đã đổi lên Project "TabPFN" bằng `project_write`, cùng đường dẫn `research-handbook/de-tai-moi/...`.
6. **Báo cáo cho thầy bằng tiếng Việt.**

## 6. Ràng buộc bắt buộc

- Xưng hô: gọi người dùng là **"thầy"**.
- **Token TabPFN:** chỉ qua Kaggle Secrets, biến môi trường `TABPFN_TOKEN`. Không bao giờ ghi vào code, kho hay tài liệu.
- Không chia sẻ email của người dùng.
- **Code không có LICENSE** (PRODEN, GLWS/ILL, VIME): không chép vào kho; nạp lúc chạy (git clone ở commit cố định).
- **Tính trung thực:**
  - Không giấu Study 1 hay kết quả bất lợi.
  - Phân tích sau sự kiện phải ghi "thăm dò".
  - Không đổi giao thức đã đóng băng mà không ghi deviation.
  - Study 2 đã xong, không chạy lại theo nghĩa xác nhận.
- **Mọi con số trong bài** qua macro `\val{key}`, sinh bằng script; không gõ tay. Còn vài hằng số cấu hình gõ tay, như 2.000, 10.000, 0,015 ở §6.6.
- **Tài liệu tham khảo:** chỉ trích mục đã đối chiếu (REFERENCE_CHECK). WebFetch chỉ dùng được với URL do WebSearch trả về.
- Không tự chạy tái lập trên clone sạch khi thầy chưa yêu cầu (việc của G10).
- **Ngôn ngữ:** tài liệu đề tài viết tiếng Việt; bài báo, hình và bảng viết tiếng Anh.
- **Memory** (nếu dùng) chỉ ghi dưới `/projects/01a16ca8-b9fd-7428-badd-d20cdb3f4439/`.

## 7. Việc mở và bước tiếp

| # | Việc | Hạn hoặc điều kiện | Trạng thái |
|---|---|---|---|
| 0 | Giai đoạn 0 của V2 **xong 05/11** ([V2/V0_REPORT.md](V2/V0_REPORT.md)): P1 đúng, P2 sai, P3 đúng, P4 sai; cổng V0 qua một phần. R4 xong: H4′ q 0,6 nhạy với nhiễu Monte Carlo (chuỗi b p_Holm 0,074; trung bình hai chuỗi đạt) | — | xong |
| 0b | Giai đoạn 1 (thầy duyệt 05/11): chọn bộ gộp chuỗi theo [V2/PHASE1_PLAN.md](V2/PHASE1_PLAN.md) (ghi trước, `7ec273b`). V1A, V1B chạy xong 05/11; [V2/V1_REPORT.md](V2/V1_REPORT.md): chọn **A1** (trung bình đều 3 chuỗi), P5 sai, P6 đúng, P7 đúng, không nhóm nào ≥ 3 thụt lùi. Thiếu 4 dataset LLP của TabPFN (OpenML 503); thầy quyết **không chạy bù** (D-V1-1). **Cổng V1-A chốt 05/11** | 05/11 | xong |
| 0c | ~~Đưa R4 vào bài~~: xong 05/11 (`eda65ed`): §6.2, §6.6, Limitations, Bảng tab:r4; phần chính vẫn 9 trang. Abstract chưa sửa (bài sẽ viết lại cho v2) | — | xong |
| 0d | V2 W3, W4 (05/11 tối, CPU, chưa chạy GPU): [V2/W4_TFM_REPORT.md](V2/W4_TFM_REPORT.md). Đề xuất TFM mới: **TabPFN-3** (hoặc 3.5 theo quy tắc tốc độ ghi trước) và **TabDPT**. TabDPT học trên 123 dataset OpenML thật, trùng **13** dataset Study 2 của ta: đề xuất loại khỏi Study 3 mọi dataset trong tập huấn luyện của TFM có công bố danh sách. Code: backend `tabpfn_v3`, `tabpfn_v3_5`, `tabdpt` (`TabDPTAdapter`); cơ chế nhãn yếu `--weak_variant feat` (LLP túi gom theo feature) và `inst` (PLL phụ thuộc mẫu); test với mô hình giả. **Thầy cần:** chấp nhận giấy phép TabPFN-3 và 3.5 trên ux.priorlabs.ai; duyệt lượt thử GPU (W4_TFM_REPORT §4) và quy tắc loại dataset | trước lượt GPU kế tiếp | chờ thầy |
| 0e | Lượt **V1C xong** (08/11; [V2/V1C_REPORT.md](V2/V1C_REPORT.md)): chọn TabPFN-3; **P8 đúng** (Φ_chuỗi, sai số 0,016); P9 đúng; **P13 đúng** (gộp 4 TFM gần bằng tiên tri); P11, P12, P14, P15 sai. v2 giữ A1 3 × 30 vòng quét. H8 cần tín hiệu mới. Cần thầy quyết cơ chế nhãn yếu cho Study 3 (§8) | 08/11 | xong |
| 0f | Quyết định sau V1C (thầy duyệt 08/11; [V2/V1D_PLAN.md](V2/V1D_PLAN.md)). Lượt **V1D mở rộng** (§3, `4203d9b`; code `s2_code_v12.zip`): đếm dataset Study 3, hiệu chỉnh q, bộ nhớ đệm L (`--l_cache`, `--l_only`), TabPFN-3 và TabDPT trên 69 ô chung (P16 v2-đa-TFM, P17, P18 H8′), bối cảnh khó với 4 TFM. 2 phiên Kaggle; **thầy chạy** theo G4/README mục V1D; chấm bằng `V2/analysis/v1d_report.py`. Sau đó: protocol Study 3 và đóng băng ở cổng V2 | sau V1D | chờ thầy chạy |
| 0g | **V1D xong** ([V2/V1D_REPORT.md](V2/V1D_REPORT.md)): P16, P17, P18 đúng. Thầy quyết 10/11: **dừng phát triển ở V1D, huỷ V1E, chạy Study 3 bản gọn**. Protocol [S3/STUDY3_PLAN.md](S3/STUDY3_PLAN.md) (`c9009e8`, ghi trước): 20 dataset mới, 4 TFM, H1 (v2 hơn v1), H2 (đa-TFM hơn một TFM), H3 (H8′). Code `s2_code_v14.zip`, chấm `S3/s3_report.py` (viết trước). **Thầy duyệt và chạy** theo G4/README mục Study 3. Sau đó: chấm, rồi viết bài | sau lượt S3 | chờ thầy duyệt và chạy |
| 0h | **Study 3 bản gọn xong, 12/11** ([S3/S3_REPORT.md](S3/S3_REPORT.md)): 20 dataset mới, 71/71 cặp. **H1 đúng** (v2 hơn v1, p 0,0004), **H2 đúng** (đa-TFM hơn một TFM, p 0,000002), C5 đạt; **H3 sai** (H8′ AUC 0,766), **C4 không đạt** (thụt lùi 14,1%, hầu hết so với learner L). **Thực nghiệm kết thúc.** Việc tiếp theo: viết lại bài (v2 + Study 3 + Mệnh đề 4′, 5, 6), đề xuất nhắm ICML 2027, dự phòng NeurIPS 2027 | 12/11 | xong; chuyển sang viết bài |
| 1 | ~~Thầy chọn R1–R8, Q1–Q5~~: thay bằng V2 W6 (thầy duyệt 04/11) | — | thay thế |
| 2 | Viết code cho các lượt được chọn (`--part equil`, gỡ chốt rhat cho s2, lưu vết, `--max_context`, backend TabPFN cho S2), bản zip `s2_code_v6` | sau 1 | — |
| 3 | D2: lý thuyết ngưỡng sụp (S2_PLAN §5.1); R2: chặn khối không sửa. **05/11: bản nháp W1 và R1a** ([V2/W1_R1A_REPORT.md](V2/W1_R1A_REPORT.md)): Mệnh đề 4 (drift) có chứng minh cho lịch quét ngẫu nhiên; tự rà lần 1 đã sửa 3 lỗi MAJOR ([V2/reviews/W1_SELF_REVIEW_01.md](V2/reviews/W1_SELF_REVIEW_01.md)). R1a: TC khối có giới hạn đóng; Định lý 2 rỗng với khối; TV/ε̄ ∈ [0,49; 1,17]. 05/11 chiều: Bổ đề 3 và Mệnh đề 4′ (lịch quét thật) chứng minh xong; rà độc lập ([V2/reviews/W1_REVIEW_INDEPENDENT_01.md](V2/reviews/W1_REVIEW_INDEPENDENT_01.md)) có 1 BLOCKING, 2 MAJOR, 8 MINOR, đã sửa hết; R1a cho LLP xong (§5.5). Cũng 05/11 chiều (§2.7, chưa rà độc lập): Bổ đề 3* phủ max_context và giả định tập điểm xấu; hằng số tinh chỉnh. 05/11 tối: Mệnh đề 6 (mô-men dừng chính xác, khớp mô phỏng ±1,5%) và Mệnh đề 5 (R2 ở mức dự đoán: khối free đúng tuyệt đối; khối all lệch α(1−λ)(1−Cp_c)/(λ(K−1)n)), rà độc lập lần 2 ([V2/reviews/W1_REVIEW_INDEPENDENT_02.md](V2/reviews/W1_REVIEW_INDEPENDENT_02.md)) có 1 BLOCKING (câu đề xuất thay 3(b) thiếu chữ "SSL"), 2 MAJOR, đã sửa hết; dạng đóng α′ = αK(n−1)/((K−1)n). Còn: Φ_chuỗi trên GPU (cần thầy duyệt, sau V1); viết bản tiếng Anh vào bài sau cổng V1 | 20/12/2026 | đang làm |
| 4 | Xác nhận giấy phép: checkpoint TabICL v2, từng dataset OpenML, dữ liệu PALM. Rồi bỏ `\pending` ở checklist mục 12 | trước nộp | mở |
| 5 | Chính sách LLM của NeurIPS 2027; style NeurIPS 2027 (thay `neurips_placeholder.sty`) | khi có call (thường khoảng tháng 3) | mở |
| 6 | Cập nhật tìm kiếm tính mới (G2) ≤ 30 ngày trước khi nộp; cập nhật nơi đăng của các bản arXiv | 04/2027 | mở |
| 7 | G10: một người hoặc agent chưa tham gia đọc như reviewer; tái lập một bảng chính từ clone sạch; đối chiếu nội dung từng trích dẫn | 02–03/2027 | — |
| 8 | Ẩn danh kho code khi nộp; tác giả, tài trợ, CRediT (D4) ở bản camera-ready | khi nộp | — |
