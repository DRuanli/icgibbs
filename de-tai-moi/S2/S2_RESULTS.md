# Nghiên cứu 2: kết quả xác nhận (đang cập nhật; 02/11/2026)

- **Phân tích:** `S2/analysis/s2_confirmatory.py`, viết trước khi chạy trên số liệu. Script áp đúng S2_PLAN §1 và §4 (đóng băng ở commit aaee9fd) và danh sách cuối trong `s2_final.csv`.
- **Kết quả:** `S2/results/`.
- **Tình trạng dữ liệu:** LLP túi 8 đủ (18 dataset × 3 seed). PLL còn thiếu gas-drift (lượt S2G), nên **H4′ và H3′ chưa tính**. Script tự chặn khi danh sách đa lớp chưa đủ.

## H2-LLP′ (LLP túi 8, AUC; TFM: TabICL v2): **ĐẠT**

| So sánh | Chênh trung bình [CI 95%] | Thắng/hoà/thua | p (Holm) |
|---|---|---|---|
| Gibbs − DLLP tuyến tính (đối thủ chính) | **+0,052** [+0,025; +0,086] | 16/0/2 | **0,0001** |
| Gibbs − TFM ngây thơ | **+0,116** [+0,085; +0,150] | 18/0/0 | **0,00002** |
| Gibbs − baseline mạnh nhất trên tập S2 (DLLP tune) | +0,041 [+0,018; +0,072] | **16/18 = 89%** (cần ≥ 70%) | p không hiệu chỉnh 0,0004 |

- **Độ nhạy S1** (bỏ SpeedDating, eye_movements; còn 16 dataset): vẫn đạt. Gibbs − DLLP tuyến tính +0,051, 15/1, Holm 0,0001; trước baseline mạnh nhất (LLP-VAT tune) thắng 14/16 = 88%.
- **Hai dataset Gibbs thua DLLP tuyến tính, đều rất sát:** eye_movements (−0,005) và scene (−0,002).
- **Thắng đậm:** Marketing_Campaign (+0,27), SpeedDating (+0,13), Bank_Customer_Churn, philippine, jasmine, Fitness_Club (+0,09 tới +0,10).
- AUC trung bình: Gibbs 0,840; hard-EM 0,827; DLLP tune 0,799; DLLP tuyến tính 0,788; TFM ngây thơ 0,724; oracle 0,879.

Bảng theo dataset: `results/s2_table_llp8.csv`.

## H4-PLL′ và H3′
Chờ lượt S2G (gas-drift).
