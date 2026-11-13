# Báo cáo lượt bổ sung S3B / S3BC (13/11/2026)

- **Tính chất:** bổ sung, **ghi trước sau khi Study 3 đã chấm** (`S3/S3B_PLAN.md`, commit `d9e6062`). Không thay đổi kết luận nào của Study 3.
- **Chạy:** Kaggle, một phiên, 12/11 15:00 – 13/11 02:04 UTC. Code `b59ef14` (mã băm 87 file khớp ở cả hai lượt), pytest đầu lượt 170 qua.
- **Chấm:** `S3/s3b_report.py`, chạy không sửa ngày 13/11/2026 06:09 UTC. Đầu ra: `S3/results/s3b_report.txt`, `s3b_summary.json`, `s3b_cpu_datasets.csv`, `s3b_tfm_pairs.csv`, `s3b_long_cells.csv`.
- **Kiểm ô:** mọi dòng khớp mã băm lần chia và nhãn yếu của lượt S3.
- **Độ đầy đủ:**
  - B1 đủ 20 dataset.
  - B2 đủ 71 cặp (TFM, dataset).
  - **B3 thiếu 5/72 ô** (dataset 46933: 3 seed của TabICL v2 và 2 seed của TabPFN-3), vì lượt dừng ở mốc tự ngắt 11 giờ. Theo §5 của kế hoạch, B3 được chấm trên TFM đủ 24 ô, tức chỉ TabDPT, và P22 ghi "tạm thời". Mục 4 xem tạm cả 67 ô; việc xem tạm này được ghi trong `V2/EXPERIMENT_LOG.md`.

Quy ước: hiệu số dương là phương pháp thứ nhất tốt hơn; hoà khi |chênh| < 0,001; AUC cho LLP túi 8, accuracy cho PLL q 0,6.

## 1. B1 — baseline không dùng mạng nơ-ron (chính thức)

| So sánh (theo dataset) | n | Chênh trung bình [KTC 95%] | Trung vị | Thắng/hoà/thua | p (Holm) |
|---|---|---|---|---|---|
| X − GBDT tự huấn luyện | 20 | +0,135 [+0,051; +0,234] | +0,031 | 19/0/1 | 0,0002 |
| X − PL-kNN (chỉ PLL) | 8 | +0,188 [+0,091; +0,299] | +0,129 | 8/0/0 | 0,016 |
| L − baseline CPU tốt hơn | 20 | +0,022 [−0,024; +0,072] | −0,003 | 7/1/12 | 0,84 |

Mô tả (không kiểm định):

| So sánh | Chênh trung bình [KTC 95%] | Trung vị | Thắng/hoà/thua |
|---|---|---|---|
| v1 (một chuỗi, trung bình TFM) − GBDT | +0,031 [−0,003; +0,071] | +0,003 | 11/1/8 |
| A1 − GBDT | +0,089 [+0,025; +0,169] | +0,010 | 15/0/5 |
| X − baseline CPU tốt hơn | +0,085 [+0,034; +0,147] | +0,031 | 19/0/1 |

Trung bình theo bối cảnh:

| Bối cảnh | X | A1 | v1 | L | GBDT tự huấn luyện | PL-kNN |
|---|---|---|---|---|---|---|
| LLP (12) | 0,828 | 0,807 | 0,801 | 0,783 | 0,796 | — |
| PLL (8) | 0,808 | 0,725 | 0,589 | 0,717 | 0,519 | 0,620 |

Đọc:
- **Cấu hình khuyến nghị X hơn cả hai baseline không nơ-ron**, có ý nghĩa sau hiệu chỉnh Holm. Dataset duy nhất X thua GBDT là 4538 (PLL, 0,499 so với 0,534).
- **Một chuỗi đơn không hơn rõ một GBDT không tune.** Khoảng tin cậy chứa 0, trung vị +0,003. Ở LLP hai bên gần như bằng nhau. Ở PLL, trung bình của chuỗi đơn (0,589) **thấp hơn PL-kNN không tune** (0,620), do các dataset nhiều lớp (42: 0,350 so với 0,679; 40499: 0,734 so với 0,915).
- **L không mạnh hơn baseline cây không tune** (thua 12/20). Bài v2 không được gọi L là "baseline mạnh nhất" khi nói về Study 3.

## 2. B2 — TFM ngây thơ, hard-EM, oracle (chính thức)

| TFM | Bối cảnh | n | naive | hard-EM | v1 | A1 | oracle |
|---|---|---|---|---|---|---|---|
| TabDPT | LLP | 12 | 0,716 | 0,814 | 0,786 | 0,799 | 0,874 |
| TabDPT | PLL | 8 | 0,382 | 0,577 | 0,648 | 0,720 | 0,925 |
| TabICL v2 | LLP | 12 | 0,707 | 0,794 | 0,802 | 0,803 | 0,876 |
| TabICL v2 | PLL | 8 | 0,442 | 0,678 | 0,749 | 0,805 | 0,941 |
| TabPFN v2 | LLP | 10 | 0,726 | 0,781 | 0,784 | 0,798 | 0,858 |
| TabPFN v2 | PLL | 1 | 0,447 | 0,526 | 0,488 | 0,494 | 0,745 |
| TabPFN-3 | LLP | 12 | 0,729 | 0,796 | 0,817 | 0,817 | 0,880 |
| TabPFN-3 | PLL | 8 | 0,357 | 0,367 | 0,371 | 0,650 | 0,940 |

- **Kiểm định ghi trước** (log loss, theo dataset, trung bình TFM): chuỗi đơn tốt hơn hard-EM ở 17/20 dataset, chênh trung bình 0,906 nat (trung vị 0,214), p < 0,001 (hai phía). Đúng hướng của H3 ở Study 1–2.
- **Theo chỉ số chính**, chuỗi đơn chỉ hơn hard-EM +0,020 (11/3/6; KTC chạm 0), và ở LLP hai bên bằng nhau (+0,001).
- **Chuỗi đơn − naive:** +0,128 (18/1/1). Ở LLP +0,083, ở PLL +0,196.
- **Khoảng cách tới oracle:** chuỗi đơn 0,182 (LLP 0,074; PLL 0,345); A1 0,124 (LLP 0,068; PLL 0,209). X kém oracle tốt nhất trong các TFM 0,085 (LLP 0,052; PLL 0,135), ở 18/20 dataset.
- **TabPFN-3 trên PLL:** chuỗi đơn (0,371) và hard-EM (0,367) đều đứng gần mức ngây thơ (0,357); chỉ điểm xuất phát từ learner kéo lên được (A1 0,650).

## 3. B3 — chuỗi 100 vòng quét (chính thức, tạm thời: chỉ TabDPT đủ 24 ô)

| | TabDPT |
|---|---|
| Ô kẹt (T2 = 1 ở lượt S3) | 10 |
| Sửa được (đạt trong 0,02 của max(A1, L)) | 1 (10%) |
| Lợi trung bình 100 so với 30 vòng, 24 ô | +0,016 |
| Lợi trung bình ở ô kẹt | +0,046 |
| Ô còn đang tăng ở vòng 100 | 4 |

**P22: ĐÚNG (tạm thời, 1/3 TFM).**

## 4. B3 — xem tạm trên 67/72 ô (không phải kết quả chấm)

| TFM | Ô | Kẹt | Sửa được | Lợi trung bình | Lợi ở ô kẹt | Độ đúng nhãn vòng 30 → 100 | Còn đang tăng |
|---|---|---|---|---|---|---|---|
| TabDPT | 24 | 10 | 1 | +0,016 | +0,046 | 0,628 → 0,648 | 4 |
| TabICL v2 | 21 | 9 | 3 | +0,035 | +0,084 | 0,735 → 0,758 | 5 |
| TabPFN-3 | 22 | 16 | 1 | +0,121 | +0,167 | 0,351 → 0,445 | 10 |
| **Cả 67** | 67 | 35 | 5 (14%) | +0,056 | | | 19 |

- **Vế thứ nhất của P22 (dưới 25% ô kẹt được sửa) đứng vững:** 14%.
- **Vế thứ hai (lợi trung bình dưới 0,03) sẽ sai khi đủ ô.** Kể cả khi 5 ô còn thiếu đều có lợi bằng 0, lợi trung bình trên 72 ô vẫn là +0,053. Dataset còn thiếu (46933) là dataset dễ (mọi phương pháp ≈ 0,964), nên khả năng cao là lợi ở đó gần 0.
- **Hệ quả:** chuỗi TabPFN-3 không đứng yên trong một lưu vực sai. Nó leo chậm: độ đúng nhãn tăng từ 0,35 lên 0,45 trong 70 vòng thêm, 10/22 ô vẫn đang tăng. Ở TabDPT thì gần như đứng yên.

## 5. Bài v2 phải sửa gì

| # | Chỗ trong bài | Sửa |
|---|---|---|
| 1 | Câu "More sweeps then do not repair a start in the wrong basin" (`theory.tex`) và các câu tương tự | Viết lại: thêm vòng quét hiếm khi sửa được ô kẹt (14% ở 67 ô), nhưng chuỗi của một TFM leo chậm, nên "kẹt" ở đây là lẫn giữa lưu vực và trộn chậm. Chờ B3 đủ rồi mới chốt số |
| 2 | Mục thực nghiệm, phần hạn chế ("chỉ một baseline", "baseline toàn nơ-ron") | Thêm B1: X hơn GBDT tự huấn luyện 19/20 và PL-kNN 8/8, kèm điều kiện cả hai không tune |
| 3 | Mọi nơi gợi ý chuỗi đơn là cấu hình đủ dùng | Nói rõ: chuỗi đơn không hơn rõ GBDT không tune, và ở PLL nhiều lớp thua PL-kNN về trung bình. Cấu hình khuyến nghị là X |
| 4 | Mô tả L | L không mạnh hơn baseline cây không tune; bỏ mọi cách gọi "strongest" khi nói về Study 3 |
| 5 | Khoảng cách tới oracle | Thêm bảng B2 vào phụ lục; nêu X còn kém oracle tốt nhất 0,085 |
| 6 | So sánh với hard-EM | Báo cả hai: hơn rõ về log loss, gần bằng về chỉ số chính ở LLP |

Các con số này phải đi vào bài qua `make_v2.py` (khoá `s3b-*`), không gõ tay.

## 6. Việc còn lại

1. Thầy chạy phiên 2 của lượt S3B để xong 5 ô B3: notebook mới, thêm output phiên 1, dán y hệt hai ô ở `G4/README.md`. Ước khoảng 1 giờ, cộng 17 phút pytest đầu lượt.
2. Chấm B3 lần cuối bằng cùng script, ghi vào nhật ký.
3. Sửa bài v2 theo mục 5.
