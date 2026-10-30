# Nhật ký thực nghiệm (G7)

> Một dòng cho mỗi lần chạy, kể cả lần lỗi hoặc bị huỷ. Không xoá dòng. Kết quả thô của từng lượt ở `G7/raw/<lượt>/` (không sửa tay); mỗi dòng kết quả tự ghi mã băm dữ liệu, lần chia, nhãn yếu, code và cấu hình.

| run_id | Ngày giờ (UTC) | Commit / mã băm code | Lượt, phần | Dataset | Phương pháp | Seed | Phần cứng | Thời gian | Trạng thái (ok / lỗi / huỷ) | Kết quả thô | Ghi chú |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 28/10 18:15 – 23:47 | `py-387059443932`; sha256 của mọi tệp code khớp kho (commit c5e55f7) | C1, baseline mảnh 0–3/12 | 10 nhị phân + 14 đa lớp | mọi baseline (mặc định, tune) | 0–2 | Kaggle CPU 4 nhân | 5 giờ 32 phút | ok | `G7/raw/C1/` | bộ test: 106 qua, 1 bỏ qua |
| C2 | 28/10 18:15 – 23:45 | như C1 | C2, baseline mảnh 4–7/12 | như C1 | như C1 | 0–2 | Kaggle CPU | 5 giờ 30 phút | ok | `G7/raw/C2/` | bộ test: 106 qua |
| C3 | 28/10 18:16 – 29/10 khoảng 06:16 | như C1 | C3, baseline mảnh 8–11/12; baseline PLL thật; đối thủ chính ở các cỡ dữ liệu | như C1; Lost, MSRCv2, BirdSong | như C1 | 0–2; PLL thật 0–9 | Kaggle CPU | 12 giờ (hết giờ) | **hết giờ, dở** | `G7/raw/C3_lan1_hetgio/` | 4 mảnh baseline xong (máy chậm hơn C1, C2: 10,5 giờ). Baseline PLL thật xong Lost, MSRCv2 (160 ô) và 1/80 ô BirdSong; phần đối thủ chính ở các cỡ dữ liệu chưa chạy. Chạy tiếp: lượt C3 lần 2 (chạy tiếp được) |
| C3 (lần 2) | 29/10 khoảng 07:00 – 09:52 | như C1; sha256 code khớp kho | chạy tiếp C3 với tệp kết quả của lần 1 | như C3 | như C3 | như C3 | Kaggle CPU | khoảng 3 giờ | ok | `G7/raw/C3/` (lần 1: `G7/raw/C3_lan1_hetgio/`) | bộ test 106 qua; các ô của lần 1 được bỏ qua đúng (không ô trùng) |
| L1 | 29/10 02:33 – 09:26 | như C1; sha256 code khớp kho | L1, phần chính (TFM) mảnh 0 và 1/6, 2 GPU song song | 10 nhị phân + 14 đa lớp | Gibbs ấm, SEM, hard-EM, TFM ngây thơ, oracle; Gibbs biết propensity (PU-SAR); CL + TFM (nhiễu) | 0–2 | Kaggle 2 × T4 | 6 giờ 53 phút | ok | `G7/raw/L1/` | bộ test 106 qua |
| L2 | 29/10 04:06 – 11:02 | như C1; sha256 code khớp kho | L2, phần chính mảnh 2 và 3/6 | như L1 | như L1 | 0–2 | Kaggle 2 × T4 | 6 giờ 56 phút | ok | `G7/raw/L2/` | bộ test 106 qua, 1 bỏ qua |
| L3 | 29/10 04:16 – 11:01 | như C1; sha256 code khớp kho | L3, phần chính mảnh 4 và 5/6 | như L1 | như L1 | 0–2 | Kaggle 2 × T4 | 6 giờ 44 phút | **dở**: 25 ô thiếu | `G7/raw/L3/` | bộ test 106 qua. Mảnh 4: OpenML trả HTTP 503 khi tải 5 dataset (splice 46, satimage 182, eucalyptus 188, analcatdata_authorship 458, analcatdata_dmft 469) ở PLL q = 0,6 seed 1 → 25 ô (5 dataset × 5 phương pháp) không chạy. Lỗi mạng, không phải lỗi code; chạy lại lượt L3 với output lần này (các ô đã xong tự bỏ qua) |
| L3 (lần 2) | 29/10 14:48 – 17:55 | như C1; sha256 code khớp kho (64/64) | chạy lại L3 **từ đầu** (không kèm output lần 1) | như L1 | như L1 | 0–2 | Kaggle 2 × T4 | 3 giờ 7 phút | ok (dở về số ô, đủ cho phần thiếu) | `G7/raw/L3_lan2/` | bộ test 106 qua. OpenML lại trả 503 cho nhiều dataset nên chỉ chạy được 204 ô, trong đó có đủ **25 ô còn thiếu** của lần 1. 179 ô trùng với lần 1 cho kết quả **giống hệt từng bit** (mọi metric, nhãn yếu, lần chia, vết chuỗi). Quy tắc gộp, đặt trước khi so số: giữ kết quả hợp lệ đầu tiên của mỗi ô (đúng nghĩa "chạy tiếp") |
| L4 | 29/10 18:00 – 30/10 02:00 | như C1; sha256 code khớp kho | L4: R̂; RQ5 PLL thật (TabICL); ablation K, số vòng quét, sửa MH, chuyển nạp, cỡ dữ liệu | 3 nhị phân + 3 đa lớp nhỏ nhất (R̂, K, vòng quét, MH); 10 nhị phân (chuyển nạp); 3 + 3 lớn nhất (cỡ dữ liệu); Lost, MSRCv2, BirdSong | theo G7_RUNNER_DESIGN §4 | 0–2; PLL thật 0–9 | Kaggle 2 × T4 | 8 giờ | ok | `G7/raw/L4/` | bộ test 106 qua. SHA-256 tệp PLL thật khớp C3 |
| L5 (lần 1) | 29/10 10:35 – 18:04 | như C1; sha256 code khớp kho | L5: ablation khởi động lạnh, prior η | mọi dataset của LLP túi 8, PLL; PU-SAR | `gibbs_cold`; `gibbs_warm_prior1`, `gibbs_warm_prior4` | 0–2 | Kaggle 2 × T4 | 7 giờ 28 phút | **dở**: thiếu 21 ô khởi động lạnh | `G7/raw/L5_lan1/` | bộ test 106 qua. OpenML trả 503 / "not found" khi tải 6 dataset đa lớp (458, 469, 1501, 40966, 40975, 40982) → 21 ô PLL của ablation khởi động lạnh không chạy. Prior η đủ. Chạy lại L5 với output lần này |
| L6 | 29/10 12:26 – 13:44 | như C1; sha256 code khớp kho | L6: ablation TFM (TabICL v2) cho LLP túi 8, PLL q 0,3 và 0,6 | 10 nhị phân + 14 đa lớp | Gibbs ấm, SEM, hard-EM, TFM ngây thơ, oracle | 0–2 | Kaggle 2 × T4 | 1 giờ 18 phút | ok | `G7/raw/L6/` | bộ test 106 qua |

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
| L1 (phần chính, mảnh 0–1/6) | 730 | 730 | 0 | 48/48 cặp khớp | 106 qua | không metric NaN, không AUC hằng 0,5; đỉnh bộ nhớ GPU 1,9 GB |
| L2 (phần chính, mảnh 2–3/6) | 730 | 730 | 0 | khớp | 106 qua | không metric NaN; đỉnh bộ nhớ GPU 1,9 GB. 1 ô AUC đúng 0,5: `hard_em_warm`, pc4, nhiễu r = 0,4, seed 2 — hard-EM trôi về dự đoán hằng; kết quả hợp lệ của phương pháp, giữ nguyên |
| L3 (phần chính, mảnh 4–5/6) | 730 | 705 | 25 ô không chạy (OpenML 503) | khớp ở mọi ô đã chạy | 106 qua | không metric NaN. **Tổng phần chính: 2.165/2.190**; chờ chạy lại L3 |
| L3 lần 2 (bù 25 ô) | 25 | 25 | 0 | khớp | 106 qua | **Tổng phần chính: 2.190/2.190.** 179 ô trùng lần 1: giống hệt |
| L4 | R̂ 117; RQ5 150; K 36; vòng quét 36; MH 18; chuyển nạp 30; cỡ dữ liệu 285 | đủ cả | 0 | khớp (cỡ dữ liệu: chỉ kiểm bảng, lần chia là mẫu con lồng nhau) | 106 qua | không metric NaN ở RQ5, ablation, cỡ dữ liệu; dòng R̂ không có AUC là đúng thiết kế (ghi thống kê chuỗi) |
| L5 lần 1 | khởi động lạnh 114; prior η 120 | 93; 120 | 21 ô không chạy (OpenML) | khớp ở mọi ô đã chạy | 106 qua | chờ chạy lại L5 |
| L6 | 570 | 570 | 0 | khớp | 106 qua | không metric NaN, không AUC hằng 0,5 |
