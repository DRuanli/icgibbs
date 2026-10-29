# Nhật ký thực nghiệm (G7)

> Một dòng cho mỗi lần chạy, kể cả lần lỗi hoặc bị huỷ. Không xoá dòng. Kết quả thô của từng lượt ở `G7/raw/<lượt>/` (không sửa tay); mỗi dòng kết quả tự ghi mã băm dữ liệu, lần chia, nhãn yếu, code và cấu hình.

| run_id | Ngày giờ (UTC) | Commit / mã băm code | Lượt, phần | Dataset | Phương pháp | Seed | Phần cứng | Thời gian | Trạng thái (ok / lỗi / huỷ) | Kết quả thô | Ghi chú |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 28/10 18:15 – 23:47 | `py-387059443932`; sha256 của mọi tệp code khớp kho (commit c5e55f7) | C1, baseline mảnh 0–3/12 | 10 nhị phân + 14 đa lớp | mọi baseline (mặc định, tune) | 0–2 | Kaggle CPU 4 nhân | 5 giờ 32 phút | ok | `G7/raw/C1/` | bộ test: 106 qua, 1 bỏ qua |
| C2 | 28/10 18:15 – 23:45 | như C1 | C2, baseline mảnh 4–7/12 | như C1 | như C1 | 0–2 | Kaggle CPU | 5 giờ 30 phút | ok | `G7/raw/C2/` | bộ test: 106 qua |
| C3 | 28/10 18:16 – 29/10 khoảng 06:16 | như C1 | C3, baseline mảnh 8–11/12; baseline PLL thật; đối thủ chính ở các cỡ dữ liệu | như C1; Lost, MSRCv2, BirdSong | như C1 | 0–2; PLL thật 0–9 | Kaggle CPU | 12 giờ (hết giờ) | **hết giờ, dở** | `G7/raw/C3_lan1_hetgio/` | 4 mảnh baseline xong (máy chậm hơn C1, C2: 10,5 giờ). Baseline PLL thật xong Lost, MSRCv2 (160 ô) và 1/80 ô BirdSong; phần đối thủ chính ở các cỡ dữ liệu chưa chạy. Chạy tiếp: lượt C3 lần 2 (chạy tiếp được) |
| C3 (lần 2) | 29/10 khoảng 07:00 – 09:52 | như C1; sha256 code khớp kho | chạy tiếp C3 với tệp kết quả của lần 1 | như C3 | như C3 | như C3 | Kaggle CPU | khoảng 3 giờ | ok | `G7/raw/C3/` (lần 1: `G7/raw/C3_lan1_hetgio/`) | bộ test 106 qua; các ô của lần 1 được bỏ qua đúng (không ô trùng) |

## Môi trường
- Python 3.13 (Kaggle); thư viện chính theo `G4/code/requirements-g7.txt` (tabpfn 9.2.0, tabicl 2.2.0, torch 2.11.0, numpy 2.1.3, pandas 2.3.3, scikit-learn 1.6.1, scipy 1.16.3, cleanlab 2.9.0, sarpu commit 6e4fc3d); `pip freeze` của từng lượt lưu ở `g7_<lượt>_pip_freeze.txt`.
- PRODEN: kho chính thức, commit 637b531 (clone lúc chạy).
- GPU: Tesla T4 × 2 (15 GB mỗi GPU).

## Kiểm sau mỗi lượt
| Lượt | Số ô dự kiến | Số ô ok | Lỗi | Mã băm dữ liệu và lần chia khớp `docs/dataset_hashes.csv` | Bộ test | Ghi chú |
|---|---|---|---|---|---|---|
| C1 + C2 + C3 (baseline chính) | 3.492 | 3.492 | 0 | 72/72 cặp (dataset, seed): bảng và lần chia khớp | 106 qua ở cả 3 lượt | 1 ô AUC đúng 0,5: `ls_tun`, dresses-sales, SSL 5%, seed 0 — label spreading không lan được nhãn tới mẫu test nào (`prior_fallback_rows=150`), dự đoán hằng; là kết quả hợp lệ của phương pháp, giữ nguyên |
| C3 (baseline PLL thật) | 240 | 240 | 0 | SHA-256 tệp gốc khớp DATASET_CARDS (bộ chạy kiểm) | 106 qua | đủ sau lần 2 (Lost, MSRCv2, BirdSong × 10 lần chia × 8 baseline) |
| C3 (đối thủ chính, cỡ dữ liệu) | 57 | 57 | 0 | — | — | đủ sau lần 2: cỡ 500, 1.000, 2.000 trên 6 dataset; cỡ 4.000 chỉ satimage (các dataset khác có train < 4.000, dùng kết quả phần chính — đúng quy tắc) |
