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

## Lượt bổ sung S3B và S3BC dùng `s2_code_v16.zip` (12/11/2026) — BỔ SUNG, ngoài protocol Study 3

Kế hoạch ghi trước ở [S3/S3B_PLAN](../S3/S3B_PLAN.md), commit `d9e6062`. Hai lượt độc lập, **chạy song song ở hai notebook**:

| Lượt | Notebook | Nội dung | Ước lượng |
|---|---|---|---|
| **S3BC** | CPU (không bật GPU) | B1: baseline không nơ-ron (PL-kNN, GBDT tự huấn luyện) | 1 phiên, vài giờ |
| **S3B** | GPU T4 × 2 | B2: naive, hard-EM, oracle với 4 TFM; rồi B3: chuỗi warm 100 vòng quét trên PLL | 2 phiên |

**Cả hai notebook:** Internet On; *Add data:* `icgibbs-g4` bản mới có `s2_code_v16.zip`. Notebook GPU cần thêm Secret `TABPFN_TOKEN`. Không cần output của lượt S3.

- **Ô 1 (giống nhau ở hai notebook):**
```python
!pip install -q tabpfn==9.2.0 tabicl==2.2.0 tabdpt==1.3.1 cleanlab==2.9.0 dill "git+https://github.com/ML-KULeuven/SAR-PU.git@6e4fc3d8c84ac3512669a4e36ffcb5086f5b42a7#subdirectory=sarpu"
import os, glob, shutil
from kaggle_secrets import UserSecretsClient
try:
    os.environ["TABPFN_TOKEN"] = UserSecretsClient().get_secret("TABPFN_TOKEN")
except Exception:
    pass
src = glob.glob("/kaggle/input/**/s2_code_v16/code/run_s2_lot.sh", recursive=True)
print(src); assert len(src) == 1
shutil.rmtree("/kaggle/working/code", ignore_errors=True)
shutil.copytree(os.path.dirname(src[0]), "/kaggle/working/code", ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
os.makedirs("/kaggle/working/s3b_npz", exist_ok=True)              # từ phiên 2 của S3B: chạy tiếp
for f in glob.glob("/kaggle/input/**/s3b_npz/*.npz", recursive=True):
    shutil.copy(f, "/kaggle/working/s3b_npz/")
for f in glob.glob("/kaggle/input/**/s3b_*.csv", recursive=True):
    shutil.copy(f, "/kaggle/working/")
print(len(glob.glob("/kaggle/working/s3b_*.csv")), "tệp csv cũ;", len(glob.glob("/kaggle/working/s3b_npz/*.npz")), "npz cũ")
```
- **Ô 2:** notebook CPU đặt `LOT = "S3BC"`, notebook GPU đặt `LOT = "S3B"`.
```python
LOT = "S3B"        # notebook CPU: "S3BC"
import subprocess, os, signal
os.chdir("/kaggle/working/code")
log_path = f"/kaggle/working/s2_{LOT}_master_log.txt"
with open(log_path, "a") as log:
    p = subprocess.Popen(["bash", "run_s2_lot.sh", LOT, "/kaggle/working"], stdout=log,
                         stderr=subprocess.STDOUT, start_new_session=True)
    try:
        p.wait(timeout=11 * 3600)
        print("Lượt chạy xong hết.")
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGTERM); p.wait()
        print("Đã dừng sau 11 giờ để lưu output; cần chạy phiên 2.")
print(open(log_path).read()[-4000:])
```
- **Kiểm nhanh sau khoảng 20 phút:** dòng đầu của `s2_S3B_master_log.txt` (hoặc `s2_S3BC_…`) là "lượt S3B bắt đầu"; các tệp `s3b_*.csv` bắt đầu có dòng `ok`.
- **Phiên 2 của S3B:** notebook mới, thêm output phiên 1, dán y hệt hai ô.
- **Gửi lại:** mọi tệp `s2_S3B*`, `s3b_*.csv`, `s3b_*.log` của cả hai notebook, và thư mục `s3b_npz/` của notebook GPU.
- **Chấm:** `S3/s3b_report.py`, đã viết và commit trước khi có số (đặt dữ liệu vào `S3/raw/S3B/`).

## Study 3 bản gọn — lượt S3 dùng `s2_code_v14.zip` (phiên 1) và `s2_code_v15.zip` (từ phiên 2) — XÁC NHẬN

> **11/11:** từ phiên 2 dùng `s2_code_v15.zip`. Bản này chỉ khác v14 ở `run_s2_lot.sh`: việc được chia lại cho cân giữa hai GPU, còn ô, seed và cấu hình giữ nguyên. Ô 1 dưới đây đã đổi sang tìm `s2_code_v15`.

Protocol ghi trước ở [S3/STUDY3_PLAN](../S3/STUDY3_PLAN.md), commit `c9009e8`. **Thầy chạy lượt này tức là duyệt và đóng băng protocol.**
- **Quy mô:** 20 dataset mới, 4 TFM, mỗi ô 3 chuỗi. Ước 2–3 phiên.
- **V1E đã huỷ:** thầy chọn bản gọn ngày 10/11, nên không chạy lượt V1E nữa.

**Notebook mới:** GPU T4 × 2, Internet On, Secret `TABPFN_TOKEN`.
- **Add data:** `icgibbs-g4` bản mới, có `s2_code_v14.zip`. Từ phiên 2 trở đi, thêm output của phiên trước.
- **Ô 1:**
```python
!pip install -q tabpfn==9.2.0 tabicl==2.2.0 tabdpt==1.3.1 cleanlab==2.9.0 dill "git+https://github.com/ML-KULeuven/SAR-PU.git@6e4fc3d8c84ac3512669a4e36ffcb5086f5b42a7#subdirectory=sarpu"
import os, glob, shutil
from kaggle_secrets import UserSecretsClient
try:
    os.environ["TABPFN_TOKEN"] = UserSecretsClient().get_secret("TABPFN_TOKEN")
except Exception:
    pass
src = glob.glob("/kaggle/input/**/s2_code_v15/code/run_s2_lot.sh", recursive=True)
print(src); assert len(src) == 1
shutil.rmtree("/kaggle/working/code", ignore_errors=True)
shutil.copytree(os.path.dirname(src[0]), "/kaggle/working/code", ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
for sub in ("s3_npz", "l_cache_s3"):                               # từ phiên 2: chạy tiếp
    os.makedirs(f"/kaggle/working/{sub}", exist_ok=True)
    for f in glob.glob(f"/kaggle/input/**/{sub}/*.npz", recursive=True):
        shutil.copy(f, f"/kaggle/working/{sub}/")
for f in glob.glob("/kaggle/input/**/s3_*.csv", recursive=True):
    shutil.copy(f, "/kaggle/working/")
print({s: len(glob.glob(f"/kaggle/working/{s}/*.npz")) for s in ("s3_npz", "l_cache_s3")})
```
- **Kết quả ô 1:** ở phiên 1, ô in ra `s3_npz` = 0. Khi lượt chạy đủ, `s3_npz` = 213, gồm 3 TFM × 60 ô cộng 33 ô của TabPFN v2.
- **Ô 2:** như ô 2 của V1C (tự dừng sau 11 giờ), đổi `LOT = "S3"`.
- **Kiểm nhanh sau khoảng 20 phút:**
  - dòng đầu của `s2_S3_master_log.txt` là "lượt S3 bắt đầu";
  - tệp `s3_dataset_hashes.csv` có 20 dòng, đều `ok`.
- **Phiên kế tiếp:** notebook mới, thêm output phiên trước, dán y hệt hai ô. Lặp lại cho tới khi log không còn việc mới.
- **Nếu OpenML lỗi tải (`[lỗi tải]` trong log):** không sửa gì, chỉ chạy thêm một phiên; ô nào chưa có sẽ được chạy lại.
- **Gửi lại** (từ output phiên cuối):
  - `s2_S3_*`;
  - `s3_dataset_hashes.csv`, mọi tệp `s3_*.csv` và `s3_*.log`;
  - thư mục `s3_npz/`.
  
  Không cần gửi `l_cache_s3/`.
- **Chấm:** `S3/s3_report.py`, viết và commit trước khi có số.

## V2 — lượt V1E dùng `s2_code_v13.zip` (10/11/2026) — ĐÃ HUỶ, không chạy

Kế hoạch ghi trước ở [V2/V1E_PLAN](../V2/V1E_PLAN.md), commit `75d0c61`. Lượt gồm ba phần:
- **E1:** hoàn tất phần PLL phụ thuộc mẫu, sau khi sửa lỗi NaN.
- **E2:** đếm dataset Study 3 trên toàn bộ OpenML.
- **E3:** chạy TabPFN-3 và TabDPT trên 96 ô của V1 mà trước đây chỉ có TabICL.

Ước lượng khoảng 1–2 phiên.

**Notebook mới:** GPU T4 × 2, Internet On, Secret `TABPFN_TOKEN`.
- **Add data:**
  - `icgibbs-g4` bản mới, có `s2_code_v13.zip`;
  - output của **V1B**;
  - output của **V1D**, lấy phiên mới nhất;
  - ở phiên 2, thêm output phiên 1 của V1E.
- **Ô 1:**
```python
!pip install -q tabpfn==9.2.0 tabicl==2.2.0 tabdpt==1.3.1 cleanlab==2.9.0 dill "git+https://github.com/ML-KULeuven/SAR-PU.git@6e4fc3d8c84ac3512669a4e36ffcb5086f5b42a7#subdirectory=sarpu"
import os, glob, shutil
from kaggle_secrets import UserSecretsClient
try:
    os.environ["TABPFN_TOKEN"] = UserSecretsClient().get_secret("TABPFN_TOKEN")
except Exception:
    pass
src = glob.glob("/kaggle/input/**/s2_code_v13/code/run_s2_lot.sh", recursive=True)
print(src); assert len(src) == 1
shutil.rmtree("/kaggle/working/code", ignore_errors=True)
shutil.copytree(os.path.dirname(src[0]), "/kaggle/working/code", ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
for sub in ("v1_npz", "v1d_npz", "v1e_npz", "l_cache"):
    os.makedirs(f"/kaggle/working/{sub}", exist_ok=True)
    for f in glob.glob(f"/kaggle/input/**/{sub}/*.npz", recursive=True):
        shutil.copy(f, f"/kaggle/working/{sub}/")
for pat in ("v1d_calib_tabicl_inst.csv", "v1e_*.csv"):
    for f in glob.glob(f"/kaggle/input/**/{pat}", recursive=True):
        shutil.copy(f, "/kaggle/working/")
print({s: len(glob.glob(f"/kaggle/working/{s}/*.npz")) for s in ("v1_npz", "v1d_npz", "v1e_npz", "l_cache")})
print("calib csv:", os.path.exists("/kaggle/working/v1d_calib_tabicl_inst.csv"))
```
- **Kết quả ô 1 ở phiên 1:**
  - `v1_npz` ≥ 165;
  - `v1d_npz` = 218;
  - `l_cache` = 114;
  - `v1e_npz` = 0;
  - `calib csv: True`.
- **Ô 2:** như ô 2 của V1C, đổi `LOT = "V1E"`.
- **Kiểm nhanh:** dòng đầu của `s2_V1E_master_log.txt` phải là "lượt V1E bắt đầu".
- **Gửi lại:**
  - `s2_V1E_*`, `s3_census_*`;
  - mọi tệp `v1e_*.csv`, `v1e_*.log`;
  - `v1d_calib_tabicl_inst.csv`, `v1d_q.txt`;
  - thư mục `v1e_npz/`;
  - 3 tệp `v1_tabicl_v2_36_pll_inst_0.4_s*.npz` trong `v1d_npz/`.

## V2 — lượt V1D dùng `s2_code_v12.zip` (08/11/2026, mở rộng tối 08/11)

Thầy duyệt ngày 08/11. Ghi trước ở [V2/V1D_PLAN](../V2/V1D_PLAN.md): §1–§2 ở commit `8138f9b`, phần mở rộng §3 ở commit `4203d9b`. **Không dùng `s2_code_v11.zip`.**

Lượt gồm 3 hàng việc chạy song song:
- **CPU:** đếm dataset cho Study 3, rồi tính trước learner L.
- **GPU 1:** hiệu chỉnh q của PLL phụ thuộc mẫu với TabICL, rồi TabDPT trên 69 ô chung, rồi các bối cảnh khó.
- **GPU 0:** TabPFN-3 trên 69 ô chung, rồi các bối cảnh khó với TabPFN-3 và TabPFN v2.

Ước lượng khoảng 18–22 giờ GPU, tức **2 phiên**.

**Notebook:** GPU T4 × 2, Internet On, Secret `TABPFN_TOKEN`.
- **Add data:**
  - `icgibbs-g4` bản mới, có `s2_code_v12.zip`;
  - output của **V1B**, vì quy tắc chọn q* cần npz của TabICL ở V1;
  - ở phiên 2, thêm output của phiên 1.
- **Ô 1:**
```python
!pip install -q tabpfn==9.2.0 tabicl==2.2.0 tabdpt==1.3.1 cleanlab==2.9.0 dill "git+https://github.com/ML-KULeuven/SAR-PU.git@6e4fc3d8c84ac3512669a4e36ffcb5086f5b42a7#subdirectory=sarpu"
import os, glob, shutil
from kaggle_secrets import UserSecretsClient
try:
    os.environ["TABPFN_TOKEN"] = UserSecretsClient().get_secret("TABPFN_TOKEN")
except Exception:
    pass
src = glob.glob("/kaggle/input/**/s2_code_v12/code/run_s2_lot.sh", recursive=True)
print(src); assert len(src) == 1
shutil.rmtree("/kaggle/working/code", ignore_errors=True)
shutil.copytree(os.path.dirname(src[0]), "/kaggle/working/code", ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
for sub in ("v1_npz", "v1d_npz", "l_cache"):                      # npz của V1; phiên 2: npz và L đã tính
    os.makedirs(f"/kaggle/working/{sub}", exist_ok=True)
    for f in glob.glob(f"/kaggle/input/**/{sub}/*.npz", recursive=True):
        shutil.copy(f, f"/kaggle/working/{sub}/")
for f in glob.glob("/kaggle/input/**/v1d_*.csv", recursive=True):  # phiên 2: chạy tiếp
    shutil.copy(f, "/kaggle/working/")
print({s: len(glob.glob(f"/kaggle/working/{s}/*.npz")) for s in ("v1_npz", "v1d_npz", "l_cache")})
```
  - Ở phiên 1, ô 1 phải in `v1_npz` ≥ 165 (thêm V1A thì 234 cũng được), và `v1d_npz`, `l_cache` bằng 0.
- **Ô 2:** như ô 2 của V1C, chỉ đổi `LOT = "V1D"`. Ô vẫn tự dừng sau 11 giờ.
- **Phiên 2:** notebook mới, cùng dữ liệu, thêm output phiên 1, dán y hệt ô 1 và ô 2.
- **Gửi lại** (từ output phiên cuối):
  - `s2_V1D_*`;
  - `s3_census_candidates.csv`, `s3_census_summary.txt`;
  - mọi tệp `v1d_*.csv`, `v1d_*.log` và `v1d_q.txt`;
  - thư mục `v1d_npz/`.
  
  Không cần gửi `l_cache/` và `v1_npz/`.

## V2 giai đoạn 1b — lượt V1C dùng `s2_code_v9.zip` (06/11/2026, mở rộng 07/11)

Thầy duyệt ngày 06/11. Kế hoạch ghi trước ở [V2/PHASE1B_PLAN](../V2/PHASE1B_PLAN.md): §1–§6 ở commit `15a5ea6`, phần bổ sung §7 (các phần D–G) ở commit `ebf2e4e`. **Không dùng `s2_code_v8.zip`** — bản đó chưa có D–G. Lượt này là **thăm dò**, chỉ trên dataset cũ (NC1, Study 2) và tập phát triển; không chạm dữ liệu của Study 3.

**Trước khi chạy (một lần):**
- Đăng nhập ux.priorlabs.ai bằng tài khoản sinh `TABPFN_TOKEN`, rồi chấp nhận giấy phép **TabPFN-3** và **TabPFN-3.5** ở tab License.
- Tải `s2_code_v9.zip` lên dataset `icgibbs-g4` dưới dạng *New Version*.

**Notebook:** GPU T4 × 2, Internet On, Secret `TABPFN_TOKEN`.
- **Add data:**
  - dataset `icgibbs-g4` (bản mới);
  - output của notebook **V1A**;
  - output của notebook **V1B**.
  
  Cần cả hai output vì phần Φ_chuỗi đọc thư mục `v1_npz/` của cả hai.
- **Ô 1:** như ô 1 của V1A và V1B, với ba thay đổi:
  - thêm `tabdpt==1.3.1` vào lệnh pip;
  - tìm `s2_code_v9`;
  - chép thêm các tệp `v1c_*` cũ để chạy tiếp được.
```python
!pip install -q tabpfn==9.2.0 tabicl==2.2.0 tabdpt==1.3.1 cleanlab==2.9.0 dill "git+https://github.com/ML-KULeuven/SAR-PU.git@6e4fc3d8c84ac3512669a4e36ffcb5086f5b42a7#subdirectory=sarpu"
import os, glob, shutil
from kaggle_secrets import UserSecretsClient
try:
    os.environ["TABPFN_TOKEN"] = UserSecretsClient().get_secret("TABPFN_TOKEN")
except Exception:
    pass
src = glob.glob("/kaggle/input/**/s2_code_v9/code/run_s2_lot.sh", recursive=True)
print(src); assert len(src) == 1
shutil.rmtree("/kaggle/working/code", ignore_errors=True)
shutil.copytree(os.path.dirname(src[0]), "/kaggle/working/code", ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
os.makedirs("/kaggle/working/v1_npz", exist_ok=True)
for f in glob.glob("/kaggle/input/**/v1_npz/*.npz", recursive=True):
    shutil.copy(f, "/kaggle/working/v1_npz/")
for f in glob.glob("/kaggle/input/**/v1c_*.csv", recursive=True):          # phiên 2: chạy tiếp
    shutil.copy(f, "/kaggle/working/")
for f in glob.glob("/kaggle/input/**/v1c_*_npz/*.npz", recursive=True):
    d = os.path.join("/kaggle/working", os.path.basename(os.path.dirname(f)))
    os.makedirs(d, exist_ok=True); shutil.copy(f, d)
print("npz V1:", len(glob.glob("/kaggle/working/v1_npz/*.npz")))   # cần 234
```
- **Ô 2 (V1C):** tự dừng sau 11 giờ để Kaggle kịp lưu output, trước giới hạn 12 giờ.
```python
LOT = "V1C"
import subprocess, os, signal
os.chdir("/kaggle/working/code")
log_path = f"/kaggle/working/s2_{LOT}_master_log.txt"
with open(log_path, "a") as log:
    p = subprocess.Popen(["bash", "run_s2_lot.sh", LOT, "/kaggle/working"], stdout=log,
                         stderr=subprocess.STDOUT, start_new_session=True)
    try:
        p.wait(timeout=11 * 3600)
        print("Lượt chạy xong hết.")
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGTERM); p.wait()
        print("Đã dừng sau 11 giờ để lưu output; cần chạy phiên 2.")
print(open(log_path).read()[-4000:])
```
- **Phiên 2 (08/11): dùng `s2_code_v10.zip`.** Bản này giống v9, chỉ khác `run_s2_lot.sh`: việc được chia lại cho cân giữa hai GPU (V2/V1C_REPORT §6).
  - Ô 1 đổi một chỗ: tìm `s2_code_v10` thay cho `s2_code_v9`.
  - *Add data:* như phiên 1, **thêm output của phiên 1**.
  - Ô 2 giữ nguyên.
- **Hai phiên.** Từ bản v9, lượt này gồm 7 phần A–G (PHASE1B_PLAN §7.3), khoảng 20–24 giờ GPU, tức **2 phiên Kaggle**.
  - Phiên 1 chạy tới giới hạn 12 giờ.
  - Phiên 2: tạo notebook mới với cùng dữ liệu, **thêm output của phiên 1** làm dữ liệu, rồi chạy lại y hệt. Ô 1 chép các tệp `v1c_*.csv` và các thư mục `v1c_*_npz/` cũ, nên mọi ô đã xong được bỏ qua.
  - Thứ tự ưu tiên: các phần quan trọng nhất chạy ở đầu mỗi GPU.
- **Ước lượng:** phần A–C khoảng 5–7 giờ; toàn bộ A–G khoảng 20–24 giờ GPU (xem "Hai phiên").
  - **GPU 0:**
    - đo tốc độ của 5 TFM (khoảng 30 phút, gồm tải trọng số);
    - ánh xạ phản ứng trên 8 dataset F1;
    - Φ_chuỗi trên các ô TabPFN v2 của V1.
  - **GPU 1:** Φ_chuỗi trên các ô TabICL v2 của V1.
- **Nếu hết 12 giờ:** chạy lại với output của chính lượt này làm dữ liệu. Các dòng đã xong được bỏ qua.
- **Nếu TabPFN-3 hoặc 3.5 báo lỗi giấy phép hay tải trọng số:** lượt vẫn chạy tiếp. Các dòng đó ghi `error`, và em sẽ báo lại.
- **Gửi lại:**
  - mọi tệp `s2_V1C_*`;
  - mọi tệp `v1c_*.csv`, `v1c_*.log` và `v1c_choice.txt`;
  - các thư mục `v1c_e_npz/`, `v1c_probe_npz/`, `v1c_f_npz/`, `v1c_g_npz/`.
  
  Không cần gửi lại `v1_npz/`.
- **Phân tích:** `V2/analysis/v1c_report.py`, viết và commit trước khi có số.

## V2 giai đoạn 1 — lượt V1A và V1B dùng `s2_code_v7.zip` (05/11/2026)

Thầy duyệt ngày 05/11. Kế hoạch ghi trước ở [V2/PHASE1_PLAN](../V2/PHASE1_PLAN.md), commit `7ec273b`. Cả hai lượt là **thăm dò**. Mỗi ô chạy 3 chuỗi (warm, base, cold) và lưu Q̄ cùng dự đoán của từng chuỗi vào `v1_npz/`. Learner khởi động có thêm dự đoán ngoài fold.
- **V1A** (TabPFN v2; GPU T4 × 2, Internet On, Secret `TABPFN_TOKEN`), ước 7–9 giờ:
  - GPU 0: LLP túi 8 trên 8 dataset còn lại của tập đánh giá NC1, rồi 6 dataset phát triển.
  - GPU 1: PLL q = 0,6 trên 10 + 3 dataset tương ứng.
- **V1B** (TabICL v2; cấu hình máy như trên), ước 7–9 giờ:
  - như V1A, thêm toàn bộ dataset của Study 2;
  - hai GPU chia đôi khối lượng.
- Hai lượt độc lập nhau. Nếu hết 12 giờ, chạy lại đúng lượt đó với output cũ làm dữ liệu; các ô đã xong được bỏ qua, kể cả tệp npz.
- **Ô 1:** như ô 1 của giai đoạn 0, nhưng tìm `s2_code_v7/code/run_s2_lot.sh` và chép kết quả cũ theo mẫu mới:
```python
!pip install -q tabpfn==9.2.0 tabicl==2.2.0 cleanlab==2.9.0 dill "git+https://github.com/ML-KULeuven/SAR-PU.git@6e4fc3d8c84ac3512669a4e36ffcb5086f5b42a7#subdirectory=sarpu"
import os, glob, shutil
from kaggle_secrets import UserSecretsClient
try:
    os.environ["TABPFN_TOKEN"] = UserSecretsClient().get_secret("TABPFN_TOKEN")
except Exception:
    pass
src = glob.glob("/kaggle/input/**/s2_code_v7/code/run_s2_lot.sh", recursive=True)
print(src); assert len(src) == 1
shutil.rmtree("/kaggle/working/code", ignore_errors=True)
shutil.copytree(os.path.dirname(src[0]), "/kaggle/working/code", ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
for pat in ("v1_*.csv", "v1_*.jsonl"):
    for f in glob.glob(f"/kaggle/input/**/{pat}", recursive=True):
        shutil.copy(f, "/kaggle/working/")
os.makedirs("/kaggle/working/v1_npz", exist_ok=True)
for f in glob.glob("/kaggle/input/**/v1_npz/*.npz", recursive=True):
    shutil.copy(f, "/kaggle/working/v1_npz/")
```
- **Ô 2:** `LOT = "V1A"` hoặc `"V1B"`, phần còn lại như các lượt trước.
- **Gửi lại:**
  - mọi tệp `s2_V1A_*` / `s2_V1B_*` và `v1_*`;
  - thư mục `v1_npz/`, nén thành zip.
- **Phân tích:** `V2/analysis/v1_aggregate.py`, viết và commit trước khi có số.

### V1A lần 2 — chạy bù 4 dataset LLP lỗi tải (05/11/2026)

> **Không cần chạy nữa:** thầy quyết bỏ lượt chạy bù (05/11 tối, D-V1-1 bổ sung). Hướng dẫn dưới đây chỉ giữ lại để tham khảo.

Lượt V1A đầu thiếu dataset 31, 37, 23381 và 38 vì OpenML trả lỗi 503 (deviation D-V1-1, [V2/V1_REPORT §3](../V2/V1_REPORT.md)). Chạy bù **đúng lượt V1A**, không đổi code:
- **Code:** vẫn `s2_code_v7.zip`, không cần tải lại.
- **Notebook:** GPU T4 × 2, Internet On, Secret `TABPFN_TOKEN`.
- **Thêm dữ liệu:** *Add data* → output của notebook V1A lần trước. **Chỉ thêm output V1A**, không thêm output V1B, vì ô 1 chép mọi tệp `v1_*` tìm thấy.
- **Ô 1:** giữ nguyên ô 1 ở trên. Ô này chép các tệp `v1_*.csv`, `v1_*.jsonl` và `v1_npz/*.npz` cũ vào `/kaggle/working`.
- **Ô 2:** `LOT = "V1A"`.
- **Những gì sẽ chạy:**
  - Bộ test mất khoảng 12 phút.
  - GPU 0 bỏ qua các ô đã có và chỉ chạy 12 ô thiếu, ước khoảng 2,5 giờ.
  - GPU 1 chỉ tải lại dữ liệu PLL rồi bỏ qua hết.
  - Tổng cộng khoảng 3 giờ.
- **Kiểm nhanh khi xong:** trong `v1_llp_eval_tabpfn.log` và `v1_llp_dev_tabpfn.log` không còn dòng `[lỗi tải]`. Nếu vẫn còn, chạy lại một lần nữa, lần này với output của chính lượt chạy bù làm dữ liệu.
- **Gửi lại:** như lượt V1A, gồm mọi tệp `s2_V1A_*`, `v1_*` và thư mục `v1_npz/` nén zip.

## V2 giai đoạn 0 — lượt V0A và V0B dùng `s2_code_v6.zip` (04/11/2026)

Thầy duyệt V2 ngày 04/11 ([V2_PLAN](../V2/V2_PLAN.md)). Dự đoán P1–P4 đã ghi trước ở [PHASE0_PREDICTIONS](../V2/PHASE0_PREDICTIONS.md), commit `1a8859f`. Cả hai lượt là **thăm dò**, chạy trên dataset cũ của NC1. Riêng R4 chạy trên tập đánh giá S2, nhưng không chạy lại theo nghĩa xác nhận.
- **V0A** (GPU T4 × 2, Internet On, Secret `TABPFN_TOKEN`), ước khoảng 5 giờ:
  - GPU 0, TabPFN v2: ánh xạ phản ứng PLL ở ρ cao và ánh xạ LLP (cho P2, P3); nhiều điểm xuất phát cho LLP túi 8 trên tic-tac-toe và cylinder-bands; ánh xạ với lỗi có cấu trúc.
  - GPU 1, TabICL v2: R4, tức chuỗi thứ hai trên mọi ô của Study 2, có lưu vết vào `r4_traces/`.
- **V0B** (GPU T4 × 2, như trên), ước khoảng 5 giờ:
  - Hai GPU chia đôi phần nhiều điểm xuất phát PLL q = 0,6 với TabPFN v2 trên mfeat-factors, optdigits, MiceProtein, semeion. Mỗi chuỗi TabPFN mất 7–26 phút.
  - Sau đó TabICL v2 chạy trên cùng các ô, cùng ánh xạ phản ứng của TabICL.
- Hai lượt độc lập nhau, chạy song song được nếu quota cho phép.
- **Cách chạy:** như các lượt S2 bên dưới, nhưng dùng **`s2_code_v6.zip`** và chép kết quả cũ theo mẫu mới:
  - ô 1 tìm `s2_code_v6/code/run_s2_lot.sh`;
  - ô 2 đặt `LOT = "V0A"` hoặc `"V0B"`.
  - Nếu hết 12 giờ mà chưa xong, chạy lại đúng lượt đó với output cũ làm dữ liệu. Các ô đã xong được bỏ qua.
  - Ô 1 cho V2 (chép cả `v0_*`, `v0x_*`, `s2r4_*` và thư mục `r4_traces` của lần trước):
```python
!pip install -q tabpfn==9.2.0 tabicl==2.2.0 cleanlab==2.9.0 dill "git+https://github.com/ML-KULeuven/SAR-PU.git@6e4fc3d8c84ac3512669a4e36ffcb5086f5b42a7#subdirectory=sarpu"
import os, glob, shutil
from kaggle_secrets import UserSecretsClient
try:
    os.environ["TABPFN_TOKEN"] = UserSecretsClient().get_secret("TABPFN_TOKEN")
except Exception:
    pass
src = glob.glob("/kaggle/input/**/s2_code_v6/code/run_s2_lot.sh", recursive=True)
print(src); assert len(src) == 1
shutil.rmtree("/kaggle/working/code", ignore_errors=True)
shutil.copytree(os.path.dirname(src[0]), "/kaggle/working/code", ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
for pat in ("v0_*.csv", "v0x_*.csv", "s2r4_*.csv", "v0_*.jsonl", "s2r4_*.jsonl"):
    for f in glob.glob(f"/kaggle/input/**/{pat}", recursive=True):
        shutil.copy(f, "/kaggle/working/")
os.makedirs("/kaggle/working/r4_traces", exist_ok=True)
for f in glob.glob("/kaggle/input/**/r4_traces/*.npz", recursive=True):
    shutil.copy(f, "/kaggle/working/r4_traces/")
```
- Nếu lỡ dùng bản cũ, lượt tự dừng và báo "BẢN CODE CŨ".
- **Gửi lại:**
  - mọi tệp `s2_V0A_*` / `s2_V0B_*`, `v0_*`, `v0x_*`, `s2r4_*`;
  - thư mục `r4_traces/`, nén thành zip.
- **Phân tích:** `V2/analysis/v0_report.py`, viết và commit trước khi có số.

## S2 — lượt S2E (thăm dò cho bài) dùng `s2_code_v5.zip` (03/11/2026)

Thầy duyệt D3 của [G9_PLAN](../G9/G9_PLAN.md) ngày 03/11. Lượt **S2E** chạy TabICL v2 trên **dataset cũ của NC1** (tập phát triển của S2), không chạm tập đánh giá S2. Kết quả là thăm dò.
- **GPU 0, R̂ hai chuỗi:** 3 dataset nhỏ mỗi nhóm như NC1; thêm 4 dataset ≥ 8 lớp ở PLL q = 0,6.
- **GPU 1, khối có sửa MH** (bản chính xác theo lý thuyết): LLP túi 8, PLL q = 0,3 và 0,6; thêm 4 dataset ≥ 8 lớp ở q = 0,6.
- **Phần cứng:** GPU T4 × 2, Internet On, Secret `TABPFN_TOKEN`. Ước khoảng 1–3 giờ.
- **Cách chạy:** như các lượt S2 bên dưới, nhưng dùng **`s2_code_v5.zip`**: ô 1 tìm `s2_code_v5/code/run_s2_lot.sh`; ô 2 đặt `LOT = "S2E"`. Không cần chép kết quả cũ vào.
- Nếu lỡ dùng bản cũ, lượt tự dừng và báo "BẢN CODE CŨ".
- Gửi lại các tệp `s2_*` và `s2e_*`.
- Phân tích: `S2/analysis/s2e_report.py`, viết trước khi có kết quả.

## S2 — lượt S2R (chạy lại ô lỗi) dùng `s2_code_v4.zip` (03/11/2026)

Lượt S2G có 5 ô lỗi: PRODEN mặc định trên gas-drift. Theo G7_PLAN §2, ô lỗi được chạy lại một lần. Lượt **S2R** chỉ chạy lại PRODEN mặc định trên gas-drift (6 ô).
- Phần cứng: Kaggle **CPU**, Internet On. Khoảng 10–15 phút, phần lớn là bộ test.
- Cách chạy: như các lượt S2 bên dưới, nhưng dùng **`s2_code_v4.zip`**: ô 1 tìm `s2_code_v4/code/run_s2_lot.sh`; ô 2 đặt `LOT = "S2R"`.
- Không cần chép kết quả cũ vào.
- Gửi lại các tệp `s2_*`.
- Lượt này không ảnh hưởng kết luận nào của S2 (S2/S2_RESULTS.md §5).
- **Đã chạy 03/11/2026:** lỗi lặp lại, 5 ô ghi "thiếu, có lý do".

## S2 — lượt S2G (gas-drift) dùng `s2_code_v3.zip` (02/11/2026)

S2T, S2C1, S2C2 đã chạy bằng `s2_code_v1.zip`, tức danh sách trước bước kiểm dữ liệu, nên **thiếu gas-drift**. Lượt **S2G** chỉ chạy gas-drift.
- Phần cứng: một phiên GPU T4 × 2, Internet On, Secret `TABPFN_TOKEN`. Phương pháp TFM chạy trên GPU; baseline PLL chạy trên CPU của cùng phiên, song song. Khoảng 1–3 giờ.
- Cách chạy: như các lượt S2 bên dưới, nhưng dùng **`s2_code_v3.zip`**: ô 1 tìm `s2_code_v3/code/run_s2_lot.sh`; ô 2 đặt `LOT = "S2G"`.
- Nếu lỡ dùng bản cũ, lượt tự dừng và báo "BẢN CODE CŨ".
- Gửi lại các tệp `s2_*`.

## S2 trên Kaggle — các lượt S2X, S2D, S2T, S2C1, S2C2 (02/11/2026; S2X và S2D đã chạy bằng `s2_code_v1.zip`)

Kế hoạch: [S2/S2_PLAN.md](../S2/S2_PLAN.md); danh sách: [S2/S2_DATASETS.md](../S2/S2_DATASETS.md). Một notebook cho mỗi lượt; đổi `LOT` ở ô 2.
- **S2X** (GPU T4 × 2, Internet On, Secrets `TABPFN_TOKEN`): thăm dò ánh xạ phản ứng trên dataset **cũ** của NC1. Chạy được ngay, khoảng 1–1,5 giờ.
- **S2D** (CPU): **chỉ sau khi thầy duyệt danh sách và em ghi commit đóng băng.** Tải dữ liệu S2, tính mã băm, kiểm tiêu chí. Vài phút.
- **S2T** (GPU T4 × 2), **S2C1**, **S2C2** (CPU): chạy sau S2D, khi em đã kiểm `s2_dataset_hashes.csv`.

1. Tải `s2_code_v2.zip` lên dataset `icgibbs-g4` dưới dạng *New Version*. Bản v2 chỉ thêm danh sách cuối và phần lấy mẫu con theo S2_PLAN §3.4. Bản zip có cả code G7, nên các notebook G7 cũ vẫn chạy nếu sửa đường dẫn.
2. Ô 1: như ô 1 của G7 ở dưới, nhưng tìm `s2_code_v1/code/run_s2_lot.sh` và chép các tệp `s2_*` / `s2x_*` của lần trước (nếu chạy tiếp):
```python
!pip install -q tabpfn==9.2.0 tabicl==2.2.0 cleanlab==2.9.0 dill "git+https://github.com/ML-KULeuven/SAR-PU.git@6e4fc3d8c84ac3512669a4e36ffcb5086f5b42a7#subdirectory=sarpu"
import os, glob, shutil
from kaggle_secrets import UserSecretsClient
try:
    os.environ["TABPFN_TOKEN"] = UserSecretsClient().get_secret("TABPFN_TOKEN")
except Exception:
    pass
src = glob.glob("/kaggle/input/**/s2_code_v2/code/run_s2_lot.sh", recursive=True)
print(src); assert len(src) == 1
shutil.rmtree("/kaggle/working/code", ignore_errors=True)
shutil.copytree(os.path.dirname(src[0]), "/kaggle/working/code", ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
for f in glob.glob("/kaggle/input/**/s2*_*.csv", recursive=True) + glob.glob("/kaggle/input/**/s2*_*.jsonl", recursive=True):
    shutil.copy(f, "/kaggle/working/")
```
3. Ô 2:
```python
LOT = "S2X"
%cd /kaggle/working/code
!bash run_s2_lot.sh {LOT} /kaggle/working 2>&1 | tee /kaggle/working/s2_{LOT}_master_log.txt
```
4. Gửi lại mọi tệp `s2_*`, `s2x_*` trong /kaggle/working.

## S2 — đếm dataset mới (30/10/2026; chỉ đọc metadata, vài phút)

Kế hoạch: [S2/S2_PLAN.md](../S2/S2_PLAN.md) §3. Script `code/s2_dataset_census.py` **chỉ dùng thư viện chuẩn của Python** và chỉ đọc metadata OpenML (không tải bảng dữ liệu).
- **Cách 1 (máy thầy):** `python3 s2_dataset_census.py --out .`
- **Cách 2 (Kaggle, CPU, Internet: On):** tải riêng tệp `s2_dataset_census.py` lên một dataset nhỏ (ví dụ `icgibbs-s2`), rồi một ô:
```python
!python /kaggle/input/icgibbs-s2/s2_dataset_census.py --out /kaggle/working
```
Gửi lại `s2_census_candidates.csv` và `s2_census_summary.txt`. OpenML hay trả lỗi 503: script tự thử lại, nếu vẫn lỗi thì chạy lại sau.

**Chạy lại L5 (21 ô khởi động lạnh còn thiếu):** dùng đúng notebook G7 bên dưới với `LOT = "L5"` và **thêm output của lần L5 trước làm dữ liệu** (ô 1 tự chép các `g7_*` cũ vào `/kaggle/working`, các ô đã xong được bỏ qua). Không thêm output cũ thì lượt chạy lại từ đầu (khoảng 7,5 giờ).

## G7 trên Kaggle — dùng `g7_code_v1.zip` (chỉ sau khi thầy ký "Qua" ở G6/GATE_REVIEW_G6.md)

Kế hoạch các lượt: [G7/G7_PLAN.md](../G7/G7_PLAN.md). Một notebook cho mỗi lượt; đổi `LOT` ở ô 2.
- **Lượt GPU** (L1–L6): *Accelerator: GPU T4 × 2, Internet: On, Add data `icgibbs-g4`, Secrets: tích `TABPFN_TOKEN`*.
- **Lượt CPU** (C1–C3): *Accelerator: None*.

Nên chạy bằng *Save Version → Save & Run All (Commit)*. Hết 12 giờ mà chưa xong thì chạy lại đúng lượt đó; các ô đã xong được bỏ qua. Để chạy tiếp, ô 1 chép lại các tệp `g7_*` của lần trước vào `/kaggle/working`: thêm output của phiên trước làm dữ liệu vào notebook.

1. Tải `g7_code_v1.zip` lên dataset `icgibbs-g4` dưới dạng *New Version*.
2. Ô 1:
```python
!pip install -q tabpfn==9.2.0 tabicl==2.2.0 cleanlab==2.9.0 dill "git+https://github.com/ML-KULeuven/SAR-PU.git@6e4fc3d8c84ac3512669a4e36ffcb5086f5b42a7#subdirectory=sarpu"
import os, glob, shutil
from kaggle_secrets import UserSecretsClient
try:
    os.environ["TABPFN_TOKEN"] = UserSecretsClient().get_secret("TABPFN_TOKEN")
except Exception:
    pass                                   # lượt CPU không cần TabPFN
src = glob.glob("/kaggle/input/**/g7_code_v1/code/run_g7_lot.sh", recursive=True)
print(src); assert len(src) == 1, "cập nhật dataset icgibbs-g4 lên phiên bản mới nhất"
shutil.rmtree("/kaggle/working/code", ignore_errors=True)
shutil.copytree(os.path.dirname(src[0]), "/kaggle/working/code", ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
for f in glob.glob("/kaggle/input/**/g7_*.csv", recursive=True) + glob.glob("/kaggle/input/**/g7_*.jsonl", recursive=True):
    shutil.copy(f, "/kaggle/working/")    # chạy tiếp lượt dở (nếu có)
```
3. Ô 2 (đổi `LOT`):
```python
LOT = "L1"
%cd /kaggle/working/code
!bash run_g7_lot.sh {LOT} /kaggle/working 2>&1 | tee /kaggle/working/g7_{LOT}_master_log.txt
```
4. Gửi lại toàn bộ `g7_*` trong /kaggle/working **và thư mục `/kaggle/working/code`**.

## G6 lượt 9 trên Kaggle — dùng `g6_code_v9.zip` (chỉ CPU, khoảng 1 giờ; 28/10/2026)

Mục đích: T3 — kiểm lại phần RC, CC lệch số công bố ở Yeast và Dermatology (lượt 8) bằng cách chọn siêu tham số riêng cho từng lần chia. **Chỉ cần CPU.**

1. Tải `g6_code_v9.zip` lên dataset `icgibbs-g4` dưới dạng *New Version*. Notebook mới: *Accelerator: None (CPU), Internet: On, Add data `icgibbs-g4`*.
2. Ô 1:
```python
import os, glob, shutil
src = glob.glob("/kaggle/input/**/g6_code_v9/code/run_g6_round9.sh", recursive=True)
print(src); assert len(src) == 1, "cập nhật dataset icgibbs-g4 lên phiên bản mới nhất"
shutil.rmtree("/kaggle/working/run_g6i", ignore_errors=True)
shutil.copytree(os.path.dirname(src[0]), "/kaggle/working/run_g6i", ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
```
3. Ô 2:
```python
%cd /kaggle/working/run_g6i
!bash run_g6_round9.sh /kaggle/working 2>&1 | tee /kaggle/working/g6r9_master_log.txt
```
4. Gửi lại các file `g6r9_*` trong /kaggle/working (kèm thư mục `run_g6i` nếu tiện, để em đối chiếu code đã chạy).

## G6 lượt 8 trên Kaggle — dùng `g6_code_v8.zip` (chỉ CPU, khoảng 1 giờ; 27/10/2026)

Mục đích: (1) chạy lại GLWS/ILL cho PLL sau khi thêm số hạng cân bằng entropy của bản chính thức (deviation 13), cả bằng script thí điểm lẫn bộ chạy G7, để chọn lại đối thủ chính của PLL; (2) T3: tái hiện số công bố của PRODEN, RC, CC trên các bộ UCI của chính bài gốc (Yeast, Texture, Dermatology, Synthetic Control, Har — không trùng tập phát triển hay tập đánh giá). **Chỉ cần CPU.**

1. Tải `g6_code_v8.zip` lên dataset `icgibbs-g4` dưới dạng *New Version*. Notebook mới: *Accelerator: None (CPU), Internet: On, Add data `icgibbs-g4`*. Nên chạy bằng *Save Version → Save & Run All (Commit)*.
2. Ô 1:
```python
import os, glob, shutil
src = glob.glob("/kaggle/input/**/g6_code_v8/code/run_g6_round8.sh", recursive=True)
print(src); assert len(src) == 1, "cập nhật dataset icgibbs-g4 lên phiên bản mới nhất"
shutil.rmtree("/kaggle/working/run_g6h", ignore_errors=True)
shutil.copytree(os.path.dirname(src[0]), "/kaggle/working/run_g6h", ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
```
3. Ô 2:
```python
%cd /kaggle/working/run_g6h
!bash run_g6_round8.sh /kaggle/working 2>&1 | tee /kaggle/working/g6r8_master_log.txt
```
4. Gửi lại các file `g6r8_*` trong /kaggle/working (csv, log, preds, `g6r8_pip_freeze.txt`).

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
