# G4 · Phát triển phương pháp và lý thuyết (đề tài A: In-Context Gibbs)

Theo [02_QUY_TRINH_CONG_KIEM_DUYET.md](../../research-handbook/02_QUY_TRINH_CONG_KIEM_DUYET.md) §G4: phương pháp và lý thuyết phải đúng *trước khi* tối ưu hiệu năng.

| Tệp | Nội dung |
|---|---|
| [METHOD_SPEC.md](METHOD_SPEC.md) | Đặc tả: vào/ra, thuật toán, mô hình quan sát, chi phí, 9 quyết định thiết kế có căn cứ, baseline, rủi ro |
| [PROOFS.md](PROOFS.md) | Chứng minh đầy đủ: Bổ đề 1, Định lý 1 (a, b, c), Mệnh đề 1.4, Mệnh đề 3, Định lý 2, Mệnh đề 4 |
| [SELF_REVIEW_MATH.md](SELF_REVIEW_MATH.md) | Tự rà soát phản biện phần toán (không thay kiểm duyệt độc lập) |
| [RQ1_REPORT.md](RQ1_REPORT.md) | Kết quả kiểm chứng số RQ1 (E1–E5) |
| [GATE_REVIEW_G4.md](GATE_REVIEW_G4.md) | Biên bản kiểm duyệt cổng G4 |
| `code/icgibbs/` | Thư viện (OOP) |
| `code/tests/` | 23+ kiểm thử (`cd code && python -m pytest -q tests`) |
| `code/experiments/` | Kiểm chứng RQ1 (chạy CPU): `rq1_exact.py`, `cell_pfn.py`, `rq1_figures.py` |
| `code/kaggle_g4_pilot.py` | Thí điểm trên 6 dataset phát triển (GPU Kaggle) — **thầy chạy** |
| [pilot/PILOT_V1_REPORT.md](pilot/PILOT_V1_REPORT.md) | Kết quả thí điểm lượt 1 và thiết kế lượt 2 |
| [pilot/PILOT_V2_REPORT.md](pilot/PILOT_V2_REPORT.md) | Kết quả thí điểm lượt 2: khởi động ấm, chẩn đoán, phạm vi |
| `code/kaggle_g4_rq4_pilot.py` | Thí điểm RQ4 (SSL, nhãn nhiễu, PLL) trên tập phát triển — **thầy chạy** |
| `code/g5_verify_datasets.py` | Kiểm tra danh sách dataset đánh giá dự kiến của G5 (chỉ đọc dữ liệu) — **thầy chạy** |
| `code/kaggle_g4_manyclass_pilot.py` | Thí điểm PLL > 10 lớp trên tập phát triển (TabPFN v2 bọc ECOC so với TabICL v2), chọn backend cho RQ5 — **thầy chạy** |
| `code/g6_reproduce_sarem.py` | G6 (T1): tái hiện số công bố của SAR-EM bằng code chính thức — **thầy chạy** |
| `code/icgibbs/baselines_sar.py`, `code/icgibbs/tuning.py` | G6: SAR-EM (bọc code chính thức) và random search 20 cấu hình trên nhãn yếu |
| `code/icgibbs/baselines_ext.py`, `code/run_g6_round2.sh`, `code/run_g6_round3.sh`, `code/run_g6_round4.sh`, `code/run_g6_round5.sh`, `code/icgibbs/baselines_secondary.py`, `code/run_g7.py`, `code/icgibbs/g7/`, `code/compare_g7_dev.py`, `code/g6_select_comparators.py` | G6 (T3–T4): các baseline peer-review còn lại, các lượt chạy trên tập phát triển, chọn đối thủ chính |
| `code/g5_verify_pll_real.py` | Xác minh chỉ đọc bộ PLL thật cho RQ5 (Lost, MSRCv2, BirdSong, …) — **thầy chạy** |
| [pilot/PILOT_RQ4_REPORT.md](pilot/PILOT_RQ4_REPORT.md) | Kết quả thí điểm RQ4 (cuối, lượt v4 + v5) và quyết định phạm vi |
| [pilot/PILOT_MANYCLASS_REPORT.md](pilot/PILOT_MANYCLASS_REPORT.md) | Thí điểm PLL nhiều lớp (lượt v5): chọn TabICL v2 cho RQ5 |
| `results/` | CSV, log, hình |

## G6 lượt 7 trên Kaggle — dùng `g6_code_v7.zip` (GPU T4 × 2, khoảng 2,5 giờ; 27/10/2026)

Mục đích: **tổng duyệt phần ablation** của bộ chạy G7 (khởi tạo lạnh, số khối K, số vòng quét, chuyển nạp, prior η của PU-SAR), **ablation cỡ dữ liệu** và **backend TabICL v2** (đã sửa cho đúng cấu hình F6) trên **tập phát triển**, theo đúng quy tắc chọn dataset sẽ dùng ở G7. Chạy hai luồng song song, mỗi luồng một GPU, để thử cách chạy cho G7. Không chạm tập đánh giá.

1. Tải `g6_code_v7.zip` lên dataset `icgibbs-g4` dưới dạng *New Version*. Notebook mới: *Accelerator: **GPU T4 × 2**, Internet: On, Add data `icgibbs-g4`, Secrets: tích `TABPFN_TOKEN`*. Nên chạy bằng *Save Version → Save & Run All (Commit)*.
2. Ô 1:
```python
!pip install -q tabpfn==9.2.0 tabicl==2.2.0 cleanlab dill "git+https://github.com/ML-KULeuven/SAR-PU.git@6e4fc3d8c84ac3512669a4e36ffcb5086f5b42a7#subdirectory=sarpu"
import os, glob, shutil
from kaggle_secrets import UserSecretsClient
os.environ["TABPFN_TOKEN"] = UserSecretsClient().get_secret("TABPFN_TOKEN")
src = glob.glob("/kaggle/input/**/g6_code_v7/code/run_g6_round7.sh", recursive=True)
print(src); assert len(src) == 1, "cập nhật dataset icgibbs-g4 lên phiên bản mới nhất"
shutil.rmtree("/kaggle/working/run_g6g", ignore_errors=True)
shutil.copytree(os.path.dirname(src[0]), "/kaggle/working/run_g6g", ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
```
3. Ô 2:
```python
%cd /kaggle/working/run_g6g
!bash run_g6_round7.sh /kaggle/working 2>&1 | tee /kaggle/working/g6r7_master_log.txt
```
4. Gửi lại các file `g6r7_*` trong /kaggle/working (csv, log, preds, `g6r7_gpu.txt`, `g6r7_pip_freeze.txt`).

## G6 lượt 6 trên Kaggle — dùng `g6_code_v6.zip` (GPU khoảng 2 giờ; 27/10/2026)

Mục đích: **đo thời gian và bộ nhớ ở cỡ của tập đánh giá** trước khi lên lịch G7. Ước tính hiện tại (G6_ROUND45_REPORT Mục 5) phải ngoại suy từ n_train ≤ 1.460 lên tới 4.500. Lượt này chạy bộ chạy G7 trên **dữ liệu tổng hợp** cùng cỡ với 4 dataset lớn nhất của tập đánh giá (phoneme, spambase, satimage, semeion), mỗi bối cảnh 1 mức, 1 seed; baseline chạy CPU song song. **Không tải, không chạm tập đánh giá**; metric trên dữ liệu tổng hợp không dùng cho kết luận nào. Code cũng có các sửa sau lượt 5 (quy tắc dự phòng khi fit lại baseline tune hỏng; `--mh-check` chỉ LLP; R̂ ghi metric từng chuỗi).

1. Tải `g6_code_v6.zip` lên dataset `icgibbs-g4` dưới dạng *New Version*. Notebook mới: *GPU T4, Internet: On, Add data `icgibbs-g4`, Secrets: tích `TABPFN_TOKEN`*. Nên chạy bằng *Save Version → Save & Run All (Commit)*.
2. Ô 1:
```python
!pip install -q tabpfn==9.2.0 cleanlab dill "git+https://github.com/ML-KULeuven/SAR-PU.git@6e4fc3d8c84ac3512669a4e36ffcb5086f5b42a7#subdirectory=sarpu"
import os, glob, shutil
from kaggle_secrets import UserSecretsClient
os.environ["TABPFN_TOKEN"] = UserSecretsClient().get_secret("TABPFN_TOKEN")
src = glob.glob("/kaggle/input/**/g6_code_v6/code/run_g6_round6.sh", recursive=True)
print(src); assert len(src) == 1, "cập nhật dataset icgibbs-g4 lên phiên bản mới nhất"
shutil.rmtree("/kaggle/working/run_g6f", ignore_errors=True)
shutil.copytree(os.path.dirname(src[0]), "/kaggle/working/run_g6f", ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
```
3. Ô 2:
```python
%cd /kaggle/working/run_g6f
!bash run_g6_round6.sh /kaggle/working 2>&1 | tee /kaggle/working/g6r6_master_log.txt
```
4. Gửi lại các file `g6r6_*` trong /kaggle/working (csv, log, preds, `g6r6_gpu.txt`, `g6r6_pip_freeze.txt`).

## G6 lượt 5 trên Kaggle — dùng `g6_code_v5.zip` (GPU; chạy độc lập với lượt 4)

Mục đích: (1) **tổng duyệt bộ chạy G7** (`run_g7.py`) trên **tập phát triển** cho hai bối cảnh chính — LLP túi 8, PLL q = 0,3 và 0,6 — rồi so với kết quả thí điểm cùng ô; (2) **PROOFS G4-5**: Gibbs bag-block có sửa MH (chính xác) so với bản xấp xỉ đang dùng, ở LLP túi 8, kèm tỉ lệ chấp nhận MH; (3) chẩn đoán R̂ từ 2 chuỗi. Không chạm tập đánh giá. GPU khoảng 2,5–3 giờ; phần baseline chạy CPU song song.

1. Tải `g6_code_v5.zip` lên dataset `icgibbs-g4` dưới dạng *New Version*. Notebook mới: *GPU T4, Internet: On, Add data `icgibbs-g4`, Secrets: tích `TABPFN_TOKEN`*. Nên chạy bằng *Save Version → Save & Run All (Commit)*.
2. Ô 1:
```python
!pip install -q tabpfn==9.2.0 dill "git+https://github.com/ML-KULeuven/SAR-PU.git@6e4fc3d8c84ac3512669a4e36ffcb5086f5b42a7#subdirectory=sarpu"
import os, glob, shutil
from kaggle_secrets import UserSecretsClient
os.environ["TABPFN_TOKEN"] = UserSecretsClient().get_secret("TABPFN_TOKEN")
src = glob.glob("/kaggle/input/**/g6_code_v5/code/run_g6_round5.sh", recursive=True)
print(src); assert len(src) == 1, "cập nhật dataset icgibbs-g4 lên phiên bản mới nhất"
shutil.rmtree("/kaggle/working/run_g6e", ignore_errors=True)
shutil.copytree(os.path.dirname(src[0]), "/kaggle/working/run_g6e", ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
```
3. Ô 2:
```python
%cd /kaggle/working/run_g6e
!bash run_g6_round5.sh /kaggle/working 2>&1 | tee /kaggle/working/g6r5_master_log.txt
```
4. Gửi lại các file `g6r5_*` trong /kaggle/working (csv, log, preds).

## G6 lượt 4 trên Kaggle — dùng `g6_code_v4.zip` (26/10/2026)

Mục đích: các baseline còn thiếu của **bối cảnh phụ** — GLWS cho PU và VPU (PU-SCAR, PU-SAR); ILL cho nhãn nhiễu; ILL và VIME cho SSL — bản mặc định và bản tune, trên tập phát triển. Sau lượt này chốt đối thủ chính của các bối cảnh phụ. **Chỉ cần CPU**, khoảng 2,5–3,5 giờ.

1. Tải `g6_code_v4.zip` lên dataset `icgibbs-g4` dưới dạng *New Version*. Notebook mới: *Accelerator: None (CPU), Internet: On, Add data `icgibbs-g4`*. Nên chạy bằng *Save Version → Save & Run All (Commit)*.
2. Ô 1:
```python
import os, glob, shutil
src = glob.glob("/kaggle/input/**/g6_code_v4/code/run_g6_round4.sh", recursive=True)
print(src); assert len(src) == 1, "cập nhật dataset icgibbs-g4 lên phiên bản mới nhất"
shutil.rmtree("/kaggle/working/run_g6d", ignore_errors=True)
shutil.copytree(os.path.dirname(src[0]), "/kaggle/working/run_g6d", ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
```
3. Ô 2:
```python
%cd /kaggle/working/run_g6d
!bash run_g6_round4.sh /kaggle/working 2>&1 | tee /kaggle/working/g6r4_master_log.txt
```
4. Gửi lại các file `g6r4_*` trong /kaggle/working (csv, log, preds).

## G6 lượt 3 trên Kaggle — dùng `g6_code_v3.zip` (26/10/2026)

Mục đích: (1) chạy lại PRODEN tune với không gian có thêm Adam; (2) bản tune của label spreading, self-training, forward correction cho hai bối cảnh phụ (SSL, nhãn nhiễu). Sau lượt này chốt đối thủ chính của PLL, nhãn nhiễu, SSL. **Chỉ cần CPU**, khoảng 1 giờ.

1. Tải `g6_code_v3.zip` lên dataset `icgibbs-g4` dưới dạng *New Version*. Notebook mới: *Accelerator: None (CPU), Internet: On, Add data `icgibbs-g4`*. Nên chạy bằng *Save Version → Save & Run All (Commit)*.
2. Ô 1:
```python
import os, glob, shutil
src = glob.glob("/kaggle/input/**/g6_code_v3/code/run_g6_round3.sh", recursive=True)
print(src); assert len(src) == 1, "cập nhật dataset icgibbs-g4 lên phiên bản mới nhất"
shutil.rmtree("/kaggle/working/run_g6c", ignore_errors=True)
shutil.copytree(os.path.dirname(src[0]), "/kaggle/working/run_g6c", ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
```
3. Ô 2:
```python
%cd /kaggle/working/run_g6c
!bash run_g6_round3.sh /kaggle/working 2>&1 | tee /kaggle/working/g6r3_master_log.txt
```
4. Gửi lại các file `g6r3_*` trong /kaggle/working (csv, log, preds); không cần thư mục `PRODEN/`.

## G6 lượt 2 trên Kaggle — dùng `g6_code_v2.zip` (26/10/2026)

Mục đích: G6 việc T3–T4. Chạy mọi baseline peer-review trên **tập phát triển**, mỗi baseline có bản mặc định và bản tune 20 cấu hình, để chọn đối thủ chính và kiểm lại phạm vi khẳng định trước G7. Các baseline:
- LLP túi 8, 16: DLLP (MLP và tuyến tính), LLP-VAT, EasyLLP, GLWS;
- PLL: PRODEN, GLWS/ILL, RC, CC;
- PU (bối cảnh phụ): nnPU và uPU có tune, SAR-E với propensity đã biết.

Kèm theo: kiểm giả thuyết về chênh lệch khi tái hiện SAR-EM. **Chỉ cần CPU** (không tốn quota GPU), khoảng 3,5–4,5 giờ.

1. Tải `g6_code_v2.zip` lên dataset `icgibbs-g4` dưới dạng *New Version*. Notebook mới: *Accelerator: None (CPU), Internet: On, Add data `icgibbs-g4`* (không cần token). Nên chạy bằng *Save Version → Save & Run All (Commit)*.
2. Ô 1:
```python
!pip install -q dill "git+https://github.com/ML-KULeuven/SAR-PU.git@6e4fc3d8c84ac3512669a4e36ffcb5086f5b42a7#subdirectory=sarpu"
import os, glob, shutil
src = glob.glob("/kaggle/input/**/g6_code_v2/code/run_g6_round2.sh", recursive=True)
print(src); assert len(src) == 1, "cập nhật dataset icgibbs-g4 lên phiên bản mới nhất"
shutil.rmtree("/kaggle/working/run_g6b", ignore_errors=True)
shutil.copytree(os.path.dirname(src[0]), "/kaggle/working/run_g6b", ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
```
3. Ô 2:
```python
%cd /kaggle/working/run_g6b
!bash run_g6_round2.sh /kaggle/working 2>&1 | tee /kaggle/working/g6r2_master_log.txt
```
4. Gửi lại các file `g6r2_*` trong /kaggle/working (csv, log, preds); không cần các thư mục `PRODEN/`, `SAR-PU/`.

## G6 lượt 1 trên Kaggle — dùng `g6_code_v1.zip` (26/10/2026)

Mục đích: G6 việc T1 (SAR-EM, GATE_REVIEW_G5 Phát hiện 4). Gồm ba phần, tổng khoảng 1,5 giờ:
(a) tái hiện số công bố của SAR-EM bằng code chính thức (CPU, tải dữ liệu UCI);
(b) SAR-EM, bản mặc định và bản tune, trên 6 dataset phát triển, PU-SAR ở mức 0,1 và 0,3 (CPU);
(c) In-Context Gibbs và các baseline cũ ở PU-SAR mức 0,1 trên tập phát triển — trước đây mới có mức 0,3 (GPU, khoảng 40 phút).

1. Tải `g6_code_v1.zip` lên dataset `icgibbs-g4` dưới dạng *New Version*. Notebook mới: *GPU T4, Internet: On, Add data `icgibbs-g4`, Secrets: tích `TABPFN_TOKEN`*.
2. Ô 1:
```python
!pip install -q tabpfn==9.2.0 dill "git+https://github.com/ML-KULeuven/SAR-PU.git@6e4fc3d8c84ac3512669a4e36ffcb5086f5b42a7#subdirectory=sarpu"
import os, glob, shutil
from kaggle_secrets import UserSecretsClient
os.environ["TABPFN_TOKEN"] = UserSecretsClient().get_secret("TABPFN_TOKEN")
src = glob.glob("/kaggle/input/**/g6_code_v1/code/g6_reproduce_sarem.py", recursive=True)
print(src); assert len(src) == 1, "cập nhật dataset icgibbs-g4 lên phiên bản mới nhất"
shutil.rmtree("/kaggle/working/run_g6", ignore_errors=True)
shutil.copytree(os.path.dirname(src[0]), "/kaggle/working/run_g6", ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
```
3. Ô 2 — (a) và (b), chạy trên CPU:
```python
%cd /kaggle/working/run_g6
!python g6_reproduce_sarem.py --repo /kaggle/working/SAR-PU --out /kaggle/working/g6_sarem_reproduction.csv 2>&1 | grep -v "saved to" | tee /kaggle/working/g6_sarem_reproduction_log.txt
!python kaggle_g4_pilot.py --backend tabpfn_v2 --settings pu_sar --levels "pu_sar=0.1,0.3" --only sar_em,sar_em_tuned --out /kaggle/working/g6_sarem_dev.csv 2>&1 | tee /kaggle/working/g6_sarem_dev_log.txt
```
4. Ô 3 — (c), dùng GPU:
```python
!python kaggle_g4_pilot.py --backend tabpfn_v2 --settings pu_sar --levels "pu_sar=0.1" --only naive,oracle,nnpu,upu,gibbs_warm,sem_warm,hard_em_warm,gibbs_known_warm --out /kaggle/working/g6_pusar01_dev.csv 2>&1 | tee /kaggle/working/g6_pusar01_dev_log.txt
```
5. Gửi lại các file `g6_*.csv`, `g6_*_log.txt`, `g6_*_preds.jsonl` trong /kaggle/working (không cần thư mục `SAR-PU/`).

## Việc F7 — xác minh bộ PLL thật (notebook CPU, khoảng 5 phút)

Lượt v5 chưa chạy ô này. Không cần GPU, không cần TabPFN token. Dùng lại dataset `icgibbs-g4` (bản có `g4_code_v5`).
1. Notebook mới: *Accelerator: None; Internet: On; Add data: `icgibbs-g4`*.
2. Ô 1:
```python
import glob, os, shutil
src = glob.glob("/kaggle/input/**/g4_code_v5/code/g5_verify_pll_real.py", recursive=True)
print(src); assert len(src) == 1
shutil.rmtree("/kaggle/working/run_f7", ignore_errors=True)
shutil.copytree(os.path.dirname(src[0]), "/kaggle/working/run_f7", ignore=shutil.ignore_patterns("prev_v4", "__pycache__"))
```
3. Ô 2:
```python
%cd /kaggle/working/run_f7
!(apt-get -qq update && apt-get -qq install -y unrar) > /dev/null 2>&1 || apt-get -qq install -y p7zip-full libarchive-tools > /dev/null 2>&1; which unrar 7z bsdtar
!python g5_verify_pll_real.py --out /kaggle/working/g5_pll_real_check.csv --download_dir /kaggle/working/pll_downloads --save_dir /kaggle/working/pll_real 2>&1 | tee /kaggle/working/g5_pll_real_log.txt
```
4. Gửi lại `g5_pll_real_check.csv` và `g5_pll_real_log.txt`. Nếu log báo không tải được hoặc không giải nén được: thầy tải tay `lost.rar`, `MSRCv2.rar`, `BirdSong.rar` ở palm.seu.edu.cn/zhangml/Resources.htm, đưa vào một Kaggle dataset, thêm dataset đó vào notebook rồi chạy lại ô 2.

## Lượt v5 trên Kaggle — dùng `g4_code_v5.zip` (đã chạy 26/10/2026: phần RQ4 và many-class xong; ô xác minh PLL thật chưa chạy — xem F7 ở trên)

Ba việc trong một notebook, tổng khoảng 3,5–5,5 giờ T4 (dưới giới hạn 12 giờ):
(a) chạy tiếp thí điểm RQ4 vào đúng file kết quả lượt v4 — chỉ chạy phần còn thiếu: PLL segment 0,6, cmc, các baseline SSL bị lỗi NaN (khoảng 1 giờ);
(b) xác minh chỉ đọc bộ PLL thật cho RQ5 (vài phút, không cần GPU);
(c) thí điểm PLL nhiều lớp trên tập phát triển để chọn backend cho RQ5 (khoảng 2–4 giờ).

1. Tải `g4_code_v5.zip` lên dataset `icgibbs-g4` dưới dạng *New Version*; trong notebook cập nhật dataset lên bản mới nhất. *GPU T4 x1, Internet: On.* Nên chạy bằng *Save Version → Save & Run All (Commit)* để máy chạy nền.
2. Ô 1 — cài đặt (ghim đúng phiên bản đã dùng/đã kiểm), chép mã và kết quả lượt v4:
```python
!pip install -q tabpfn==9.2.0 tabicl==2.2.0 cleanlab "tabpfn-extensions[many-class]==0.6.3"
import os, glob, shutil
from kaggle_secrets import UserSecretsClient
os.environ["TABPFN_TOKEN"] = UserSecretsClient().get_secret("TABPFN_TOKEN")
src = glob.glob("/kaggle/input/**/g4_code_v5/code/kaggle_g4_rq4_pilot.py", recursive=True)
print(src)   # thường là /kaggle/input/icgibbs-g4/g4_code_v5/code/kaggle_g4_rq4_pilot.py
assert len(src) == 1, "không thấy g4_code_v5: cập nhật dataset icgibbs-g4 lên phiên bản mới nhất"
code = os.path.dirname(src[0])
shutil.rmtree("/kaggle/working/run_v5", ignore_errors=True)
shutil.copytree(code, "/kaggle/working/run_v5", ignore=shutil.ignore_patterns(".pytest_cache", "__pycache__"))
for f in ("g4_rq4_pilot.csv", "g4_rq4_pilot_preds.jsonl", "g4_rq4_pilot_log.txt"):
    if not os.path.exists(f"/kaggle/working/{f}"):          # không ghi đè nếu chạy lại ô này
        shutil.copy(f"{code}/prev_v4/{f}", f"/kaggle/working/{f}")
print(os.listdir("/kaggle/working"))
```
3. Ô 2 — (a) chạy tiếp RQ4 (PLL trước, rồi SSL; phần đã xong tự bỏ qua):
```python
%cd /kaggle/working/run_v5
!python kaggle_g4_rq4_pilot.py --backend tabpfn_v2 --settings pll,ssl --out /kaggle/working/g4_rq4_pilot.csv 2>&1 | tee -a /kaggle/working/g4_rq4_pilot_log.txt
```
4. Ô 3 — (b) xác minh bộ PLL thật (chỉ đọc; tải từ palm.seu.edu.cn, cần công cụ giải nén RAR):
```python
!(apt-get -qq update && apt-get -qq install -y unrar) > /dev/null 2>&1 || apt-get -qq install -y p7zip-full libarchive-tools > /dev/null 2>&1; which unrar 7z bsdtar
!python g5_verify_pll_real.py --out /kaggle/working/g5_pll_real_check.csv --download_dir /kaggle/working/pll_downloads --save_dir /kaggle/working/pll_real 2>&1 | tee /kaggle/working/g5_pll_real_log.txt
```
   Nếu Kaggle không tải được từ máy chủ nguồn: thầy tải tay các tệp `lost.rar`, `MSRCv2.rar`, `BirdSong.rar` (và nếu được cả `FG-NET.rar`, `Soccer Player.rar`, `Yahoo! News.rar`) ở palm.seu.edu.cn/zhangml/Resources.htm, đưa vào một Kaggle dataset rồi chạy lại ô này — script tự tìm trong /kaggle/input.
5. Ô 4 — (c) thí điểm PLL nhiều lớp (vowel 11 lớp, letter 26 lớp; 2 mức × 2 seed × 2 backend):
```python
!python kaggle_g4_manyclass_pilot.py --out /kaggle/working/g4_manyclass_pilot.csv 2>&1 | tee -a /kaggle/working/g4_manyclass_pilot_log.txt
```
   Lần đầu TabICL tự tải checkpoint từ Hugging Face. Nếu một backend lỗi, backend kia vẫn chạy tiếp; chạy lại đúng lệnh thì script chỉ làm phần còn thiếu.
6. Gửi lại toàn bộ tab *Output* (nén như lần trước): `g4_rq4_pilot.csv`, `g4_rq4_pilot_preds.jsonl`, `g4_rq4_pilot_log.txt`, `g5_pll_real_check.csv`, `g5_pll_real_log.txt`, `g4_manyclass_pilot.csv`, `g4_manyclass_pilot_preds.jsonl`, `g4_manyclass_pilot_log.txt`. Thư mục `pll_real/` (bản .npz) **không cần gửi** — giữ trên Kaggle cho G6.

## Thí điểm RQ4 trên Kaggle (SSL, nhãn nhiễu, PLL) — dùng `g4_code_v4.zip` (đã chạy 25–26/10/2026; phần còn thiếu chạy ở v5)

Mục đích: GATE_REVIEW_G4 §3d, R5. Khoảng 2–2,5 giờ T4.
1. Tải `g4_code_v4.zip` lên dataset `icgibbs-g4` dưới dạng *New Version*; trong notebook cập nhật dataset lên bản mới nhất. *GPU T4, Internet: On.*
2. Ô 1:
```python
!pip install -q tabpfn tabicl cleanlab
import os, glob, shutil
from kaggle_secrets import UserSecretsClient
os.environ["TABPFN_TOKEN"] = UserSecretsClient().get_secret("TABPFN_TOKEN")
src = glob.glob("/kaggle/input/**/g4_code_v4/code/kaggle_g4_rq4_pilot.py", recursive=True)
print(src)   # thường là /kaggle/input/icgibbs-g4/g4_code_v4/code/kaggle_g4_rq4_pilot.py
assert len(src) == 1, "không thấy g4_code_v4: cập nhật dataset icgibbs-g4 lên phiên bản mới nhất"
shutil.rmtree("/kaggle/working/run_rq4", ignore_errors=True)
shutil.copytree(os.path.dirname(src[0]), "/kaggle/working/run_rq4",
                ignore=shutil.ignore_patterns(".pytest_cache", "__pycache__"))
print(os.listdir("/kaggle/working/run_rq4"))
```
3. Ô 2:
```python
%cd /kaggle/working/run_rq4
!python g5_verify_datasets.py --out /kaggle/working/g5_dataset_check.csv 2>&1 | tee /kaggle/working/g5_dataset_check_log.txt
!python kaggle_g4_rq4_pilot.py --backend tabpfn_v2 --out /kaggle/working/g4_rq4_pilot.csv 2>&1 | tee -a /kaggle/working/g4_rq4_pilot_log.txt
```
   Lệnh đầu chỉ đọc metadata và dữ liệu của các dataset đánh giá dự kiến để kiểm tra tiêu chí chọn (G5, việc F2); không chạy phương pháp nào trên chúng. Mất vài phút, không cần GPU.
4. Gửi lại cả thư mục output (nén như lần trước): `g4_rq4_pilot.csv`, `g4_rq4_pilot_preds.jsonl`, `g4_rq4_pilot_log.txt`, `g5_dataset_check.csv`.

## Thí điểm lượt 2 (v2) trên Kaggle — dùng `g4_code_v3.zip` (đã xong 25/10/2026)

`g4_code_v3.zip` thay cho `g4_code_v2.zip`: thêm metric khớp tiên nghiệm (quyết định Q1), `hard_em_warm` (Q2), lưu dự đoán test, và chỉ dùng lại các chuỗi tốn GPU của lượt 1.

1. Kaggle → dataset `icgibbs-g4` → *New Version* → tải `g4_code_v3.zip` lên (giữ các file cũ cũng được). Trong notebook, cập nhật dataset lên phiên bản mới nhất.
2. *Settings → Accelerator: GPU T4 x1; Internet: On.*
3. Ô 1 — cài đặt và chép mã (tự tìm đúng đường dẫn, không phụ thuộc cách Kaggle đặt tên thư mục):
```python
!pip install -q tabpfn tabicl
import os, glob, shutil
from kaggle_secrets import UserSecretsClient
os.environ["TABPFN_TOKEN"] = UserSecretsClient().get_secret("TABPFN_TOKEN")
src = glob.glob("/kaggle/input/**/g4_code_v3/code/kaggle_g4_pilot.py", recursive=True)
print(src)   # thường là /kaggle/input/icgibbs-g4/g4_code_v3/code/kaggle_g4_pilot.py
assert len(src) == 1, "không thấy g4_code_v3: kiểm tra đã cập nhật dataset icgibbs-g4 lên phiên bản mới nhất chưa"
shutil.rmtree("/kaggle/working/run_v2", ignore_errors=True)
shutil.copytree(os.path.dirname(src[0]), "/kaggle/working/run_v2",
                ignore=shutil.ignore_patterns(".pytest_cache", "__pycache__"))
print(os.listdir("/kaggle/working/run_v2"))   # phải có kaggle_g4_pilot.py, g4_pilot_v1.csv, icgibbs
```
4. Ô 2 — chạy (phần chính khoảng 4 giờ T4, phần chẩn đoán thêm khoảng 1,2 giờ; tổng khoảng 5 giờ, dưới giới hạn 12 giờ của một phiên):
```python
%cd /kaggle/working/run_v2
!python kaggle_g4_pilot.py --backend tabpfn_v2 --prev g4_pilot_v1.csv --out /kaggle/working/g4_pilot_v2.csv 2>&1 | tee -a /kaggle/working/g4_pilot_v2_log.txt
!python kaggle_g4_pilot.py --backend tabpfn_v2 --prev g4_pilot_v1.csv --diag --out /kaggle/working/g4_pilot_v2.csv 2>&1 | tee -a /kaggle/working/g4_pilot_v2_log.txt
```
5. Nên chạy bằng *Save Version → Save & Run All (Commit)* để máy chạy nền khi thầy đóng trình duyệt; kết quả nằm ở tab *Output*. Nếu bị ngắt giữa chừng: chạy lại đúng hai ô trên, script tự bỏ qua phần đã xong (với chế độ Commit thì phải tải `g4_pilot_v2.csv` của lượt dở dang lên lại; đơn giản hơn là chạy tương tác).
6. Gửi lại: `g4_pilot_v2.csv`, `g4_pilot_v2_preds.jsonl`, `g4_pilot_v2_log.txt`.

## Chạy thí điểm trên Kaggle (lượt 1, đã xong 24/10/2026)

1. Tải `g4_code.zip` lên Kaggle: *Datasets → New Dataset → Upload*, đặt tên ví dụ `icgibbs-g4`.
2. Tạo Notebook mới: *Settings → Accelerator: GPU T4 x1; Internet: On*. *Add data* → chọn dataset `icgibbs-g4`.
3. Các ô lệnh:
```python
!pip install -q tabpfn tabicl
from kaggle_secrets import UserSecretsClient; import os
os.environ["TABPFN_TOKEN"] = UserSecretsClient().get_secret("TABPFN_TOKEN")
!cp -r /kaggle/input/icgibbs-g4/code /kaggle/working/code
!cd /kaggle/working/code && python kaggle_g4_pilot.py --backend tabpfn_v2 --out /kaggle/working/g4_pilot.csv
```
4. Script in bảng tóm tắt và kết luận theo tiêu chí dừng ở cuối. Gửi lại file `/kaggle/working/g4_pilot.csv` (và nếu được, toàn bộ log của ô cuối).
5. Nếu phiên bị ngắt: chạy lại đúng lệnh ở bước 3, script tự bỏ qua các dòng đã xong.

Ước tính: 2–3 giờ GPU T4 (6 dataset × 4 mức nhãn yếu × 3 seed × 8 phương pháp). Muốn kiểm tra nhanh trước: thêm `--datasets 37 --seeds 0` (khoảng 5 phút).
