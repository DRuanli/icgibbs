# Lượt V1D — kết quả (phiên 1: 09/11; phiên 2: 10/11/2026)

- **Căn cứ:** [V1D_PLAN](V1D_PLAN.md): §1–§2 ở commit `8138f9b`, §3 ở commit `4203d9b`.
- **Code:** `874a669` (`s2_code_v12.zip`), khớp 84/84 tệp.
- **Chấm:** `V2/analysis/v1d_report.py`, viết trước, chạy y nguyên. Đầu ra ở `V2/results/v1d_report_s12.txt` và `v1d_summary.json`.
- **Tính chất:** thăm dò, trên tập phát triển và dataset cũ.
- **Phiên 1:** 03:22–12:04 UTC, xong mọi việc chạy được. Bộ test: 150 qua, 2 bỏ qua.
- **Phiên 2:** không còn việc gì để chạy, chỉ thử lại ô lỗi (§4).

## 0. Tóm tắt

| Phần | Kết quả |
|---|---|
| P16 (v2-đa-TFM) | **ĐÚNG.** X4 kém tiên tri 0,004, thua quá 0,02 ở 1/23 dataset, và không kém TFM nào tính trung bình |
| P17 (A1 với TFM mới) | **ĐÚNG.** TabPFN-3: G = +0,0115, R = 2. TabDPT: G = +0,0002, R = 2 |
| P18 (H8′) | **ĐÚNG.** AUC(c3 → T2) = 0,86 trên 138 ô (19 ô dương tính). Riêng TabPFN-3: 0,93; riêng TabDPT: 0,74 |
| q* (V1D-2) | **chưa chọn được:** 1 ô trong 27 lỗi vì một bug (§4) |
| V1D-4 túi 32 | xong với 4 TFM (§3) |
| V1D-4 PLL phụ thuộc mẫu | chưa chạy, vì chưa có q* |
| Đếm dataset Study 3 | xong; **kho vẫn thiếu** so với mục tiêu (§5) |
| Bộ nhớ đệm L | 98,6–100% ô trúng; tái lập: 14 ô của V1C phần E khớp **từng số** (chênh 0) |

## 1. P16: gộp 4 TFM trên 23 dataset × 3 seed

| | TabPFN v2 | TabICL v2 | TabPFN-3 | TabDPT |
|---|---|---|---|---|
| X4 − A1 của TFM đó, trung bình theo dataset | +0,0054 | +0,0003 | +0,0048 | +0,0076 |

- X4 kém TFM tốt nhất của từng dataset (bộ chọn tiên tri) trung bình 0,0041.
- X4 thua quá 0,02 ở đúng 1 dataset: 1464, với 0,630 so với 0,653.
- **Ý nghĩa:** gộp đều 4 TFM gần như không kém việc biết trước TFM nào tốt nhất, và không kém TFM nào trong 4.
- **Lưu ý:** so với TabICL, lợi gần bằng 0, vì TabICL thường đã là TFM tốt nhất trên các dataset này. Lợi của X4 nằm ở độ bền: không cần biết TFM nào tốt.

## 2. P17 và P18

- **P17:**
  - TabPFN-3 được lợi rõ từ A1: +0,0115, gấp khoảng 3 lần lợi của TabPFN v2 và TabICL ở V1. Lý do là các chuỗi `warm` của TabPFN-3 hay sụp hơn.
  - TabDPT gần như không đổi: +0,0002.
  - Cả hai TFM đều có 2 thụt lùi, đúng bằng ngưỡng.
- **P18:** H8′ được xác nhận trên dữ liệu mới, sau khi được đề xuất từ dữ liệu cũ (AUC 0,85 khi thăm dò).
  - Tín hiệu là độ lệch giữa Q̄ của chuỗi `warm` và chuỗi `base` trên train. Tín hiệu này không cần nhãn và không tốn thêm lượt gọi TFM.
  - Nó báo được ô mà v1 thất bại với AUC 0,86.
  - Với TabDPT thì yếu hơn (0,74). Phép xác nhận chính thức vẫn là Study 3.

## 3. V1D-4: LLP iid túi 32 với 4 TFM (trung bình 3 seed, AUC; chỉ báo cáo)

| Dataset | v1: TabPFN v2 / TabICL / TabPFN-3 / TabDPT | A1: TabPFN v2 / TabICL / TabPFN-3 / TabDPT | L |
|---|---|---|---|
| 31 | 0,636 / 0,620 / 0,603 / 0,589 | 0,618 / 0,678 / 0,628 / **0,692** | 0,649 |
| 37 | 0,705 / 0,615 / 0,647 / 0,614 | 0,617 / 0,667 / 0,667 / 0,601 | 0,643 |
| 1462 | 1,000 / 1,000 / 1,000 / 0,841 | 1,000 / 1,000 / 1,000 / 1,000 | 1,000 |
| 1464 | 0,598 / 0,581 / 0,620 / 0,564 | 0,677 / 0,613 / 0,680 / 0,544 | **0,681** |
| 1480 | 0,464 / 0,344 / 0,440 / 0,566 | 0,480 / 0,448 / 0,460 / **0,559** | 0,450 |
| 1510 | 0,984 / 0,992 / 0,985 / 0,989 | 0,982 / 0,987 / 0,960 / 0,990 | 0,966 |

- **Túi 32 thật sự khó với mọi TFM.** Ở 1480, mọi TFM trừ TabDPT cho AUC dưới 0,5: chuỗi hội tụ về nhãn đảo. 11 ô v1 có AUC < 0,5, rải ở cả 4 TFM.
- **A1 chỉ sửa được một phần.** Learner L ngang hoặc hơn ở 1464 và 1480.
- **Gợi ý cho Study 3:** giữ túi 32 làm bối cảnh "khó", và báo cáo riêng tỉ lệ ô nhãn đảo như một kiểu thất bại đã biết.

## 4. Bug: điểm xuất phát `base` khi learner cho xác suất 0 trên mọi ứng viên

- **Ô lỗi:** dataset 36 (segment), PLL phụ thuộc mẫu q = 0,4, seed 1, chuỗi `gibbs_init_base`. Lỗi: `ValueError: Probabilities contain NaN`.
- **Nguyên nhân:** learner L gán xác suất 0 cho mọi nhãn ứng viên của một số điểm. Khi đó `init_labels` chia cho 0.
- **Vì sao lỗi lặp lại:** lỗi này tất định, nên phiên 2 gặp lại đúng chỗ đó.
- **Hệ quả:** quy tắc chọn q* cần đủ 27/27 ô mà chỉ có 26, nên toàn bộ phần PLL phụ thuộc mẫu của V1D-4 bị bỏ qua, như thiết kế.
- **Sửa (10/11, deviation D-V1D-1):** hàng nào có khối lượng 0 thì quay về phân phối đều trên các ứng viên. Các hàng khác giữ nguyên, nên không kết quả cũ nào thay đổi: trước đây mọi ô gặp hàng như vậy đều báo lỗi. Có test kèm.

## 5. Đếm dataset cho Study 3

Ứng viên sạch theo từng bậc nới tiêu chí, chưa rà tay:

| Bậc | Nhị phân | Đa lớp | Đa lớp ≥ 8 lớp |
|---|---|---|---|
| 0 | 9 | 10 | 6 |
| 3 | 25 (7 có cờ cùng nguồn) | 18 (8 có cờ cùng nguồn) | 12 |

**Rà tay nhanh, em ước lượng:**
- **Nhị phân:** bỏ dữ liệu nhân tạo (monks 1–3, madelon), bỏ bản trùng (gina và gina_agnostic), và giữ một đại diện cho nhóm NASA (kc1, kc2, pc1, pc3, jm1). Còn khoảng **16**.
- **Đa lớp:** bỏ dữ liệu nhân tạo (waveform, synthetic_control, LED), và bỏ cả họ mfeat-* vì cùng nguồn với mfeat-factors đã dùng. Còn khoảng **9–10**.
- **Kết luận:** dưới mục tiêu ≥ 20 nhị phân và ≥ 12 đa lớp.
- **Đề xuất:** mở rộng phép đếm ra mọi dataset đang hoạt động trên OpenML thoả các tiêu chí, không chỉ trong các suite, rồi rà tay. Việc này nằm trong lượt V1E.
