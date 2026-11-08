# Lượt V1C — kết quả phiên 1 (08/11/2026)

- **Căn cứ:** [PHASE1B_PLAN](PHASE1B_PLAN.md), ghi trước: §1–§6 ở commit `15a5ea6`, §7 ở `ebf2e4e`. Code `1c7321e` (`s2_code_v9.zip`); 80/80 tệp khớp mã băm.
- **Chấm:** `V2/analysis/v1c_report.py`, viết trước khi có số, chạy y nguyên, không sửa. Đầu ra ở `V2/results/v1c_report_session1.txt` và `v1c_summary.json`.
- **Tính chất:** thăm dò, chỉ trên dataset cũ và tập phát triển.
- **Phiên 1:** 07/11, 10:25 – khoảng 21:25 UTC, tự dừng sau 11 giờ. Bộ test: 145 qua, 2 bỏ qua. Mọi dòng có trạng thái `ok`, không ô nào `mismatch`.

## 0. Tóm tắt

| Phần | Trạng thái | Kết quả |
|---|---|---|
| A, §4 | xong | Chọn **TabPFN-3**: TabPFN-3.5 chậm gấp 3,1 lần, vượt ngưỡng 1,5 |
| B, P11 | xong | **SAI.** TabPFN-3 vẫn có điểm hút thấp: đạt 0/4 dataset F1 |
| C, P8 | xong | **ĐÚNG.** Sai số trung vị khi dự đoán r_cuối từ Φ_chuỗi là **0,016** (ngưỡng ≤ 0,10) |
| C, P9 | xong | **ĐÚNG.** Gieo `chain` sai 0,016, gieo `uniform` sai 0,073 |
| D, P15 | **chưa xong**: 148/234 ô | Tạm thời: AUC 0,52; có khả năng SAI |
| E, P14 | xong | **SAI**, đạt 1/4. Lý do: A1 của chính TabPFN v2 đã thoát kiểu sụp F1 ở 3/4 dataset (§4) |
| E, P13 | chưa | Thiếu TabICL trên F1 và TabDPT |
| F, P12 | chưa | |
| G | chưa | |

## 1. Phần A: tốc độ, bộ nhớ, tính tất định

Thời gian trung vị mỗi lượt gọi trong chuỗi (giây, n_estimators = 1, T4):

| Backend | (500, 2) | (2.000, 2) | (500, 10) | (2.000, 10) | (2.000, 26) | Dự đoán cuối (2.000, 10) | Bộ nhớ đỉnh |
|---|---|---|---|---|---|---|---|
| TabPFN v2 | 0,170 | 0,477 | 0,154 | 0,542 | không hỗ trợ | 2,63 | 272 MB |
| TabICL v2 | 0,062 | 0,093 | 0,062 | 0,094 | 0,187 | 0,29 | 325 MB |
| **TabPFN-3** | 0,221 | **0,252** | 0,220 | **0,248** | 0,250 | 1,02 | 571 MB |
| TabPFN-3.5 | 0,648 | 0,768 | 0,644 | 0,772 | 0,852 | 2,23 | 1.207 MB |
| TabDPT | 0,053 | 0,222 | 0,053 | 0,222 | 0,447 | 1,78 | 436 MB |

- Cả 5 TFM đều tất định: chênh giữa hai lượt gọi giống hệt nhau bằng 0.
- Với n = 2.000, TabPFN-3 nhanh gấp khoảng 2 lần TabPFN v2.
- Tỉ số 3.5/3 là 3,08 > 1,5, nên theo §4 chọn **TabPFN-3**.

## 2. Phần B: ánh xạ phản ứng trên các ô F1 (P11 SAI)

Số dataset F1 (12, 1501, 40966, 28) đạt tiêu chí "g > 0 tại ρ = 0,2; 0,35; 0,5", với gieo `uniform`:

| TFM | TabPFN v2 | TabICL v2 | TabPFN-3 | TabPFN-3.5 | TabDPT |
|---|---|---|---|---|---|
| Số dataset đạt | 0/4 | 4/4 | **0/4** | 1/4 | 2/4 |

- **Kiểm nhất quán đạt:** TabPFN v2 trượt và TabICL đạt, đúng như S2X.
- **TabPFN-3 vẫn có điểm hút thấp gần ρ = 0,2:** g < 0 tại 0,2 ở 3/4 dataset.
- **Cũng TabPFN-3 đó lại có g dương lớn ở mức ρ cao hơn:** từ 0,35 ở 12 và 28 (khoảng +0,05 đến +0,15), và từ 0,5 ở 40966. Nghĩa là lưu vực trên rộng hơn so với TabPFN v2.
- Dataset 1501 (semeion) vẫn khó với mọi bản TabPFN.
- **Gieo `struct`, chỉ báo cáo:** cho cùng bức tranh.
- **LLP:** TabPFN-3 trên tic-tac-toe (50) có g > 0 trong khoảng 0,65–0,9, trong khi TabPFN v2 âm suốt từ 0,35 trở lên. Cylinder-bands (6332) có cùng hình dạng ở mọi TFM.

## 3. Phần C: Φ_chuỗi dự đoán r_cuối (P8 ĐÚNG, P9 ĐÚNG)

Sai số tuyệt đối |r̂_cuối − r_cuối| theo quy tắc §2, trên mọi ô V1 (234 ô):

| TFM | Setting | Ô | `chain` (trung vị) | `uniform` | `struct` |
|---|---|---|---|---|---|
| TabICL v2 | LLP | 96 | **0,026** | 0,081 | — |
| TabICL v2 | PLL | 69 | **0,011** | 0,042 | 0,053 |
| TabPFN v2 | LLP | 30 | **0,025** | 0,070 | — |
| TabPFN v2 | PLL | 39 | **0,009** | 0,083 | 0,079 |
| **Gộp** | | 234 | **0,016** | 0,073 | |

- **Ý nghĩa:** điểm bất động của ánh xạ một vòng quét, đo trên đúng cấu trúc lỗi của chuỗi, dự đoán độ đúng dừng của chuỗi Gibbs thật với sai số khoảng 1–3 điểm phần trăm. Đây là kiểm chứng thực nghiệm trực tiếp cho Mệnh đề 4′ và 6, và tiêu chí "Xong khi" của W1.1 đã đạt.
- Lỗi gieo đều sai gấp khoảng 4–5 lần. Điều này khẳng định F2: cấu trúc lỗi quyết định điểm bất động.
- **Giới hạn:** Q̄ lấy từ chính chuỗi đã chạy. Vì vậy đây là phép kiểm mô hình động học, không phải phép chẩn đoán đo trước được.

## 4. Phần E: chuỗi với TabPFN-3 (P14 SAI, lý do quan trọng)

A1 và từng chuỗi, seed 0, trên 4 dataset F1 PLL (accuracy):

| Dataset | TabPFN v2: warm / base / cold / **A1** / L | TabPFN-3: warm / base / cold / **A1** |
|---|---|---|
| 12 | 0,098 / 0,975 / 0,098 / **0,973** / 0,948 | 0,209 / 0,985 / 0,209 / **0,983** |
| 1501 | 0,272 / 0,766 / 0,205 / **0,774** / 0,897 | 0,098 / 0,944 / 0,130 / **0,948** |
| 28 | 0,961 / 0,982 / 0,958 / **0,977** / 0,972 | 0,098 / 0,989 / 0,098 / **0,989** |
| 40966 | 0,432 / 0,969 / 0,531 / **0,966** / 0,957 | 0,333 / 0,966 / 0,333 / **0,966** |

- **P14 sai vì giả định sai.** P14 coi A1 của TabPFN v2 vẫn sụp trên F1. Thực tế A1 đã sửa được F1 ở 12, 28 và 40966, vì chuỗi `base` (khởi động từ learner L) đi vào lưu vực trên. TabPFN-3 chỉ hơn rõ ở 1501: 0,948 so với 0,774.
- **Thăm dò: A1 tự đặt trọng số.** Các chuỗi `warm` và `cold` sụp, với cả hai TFM, lại cho dự đoán gần đều: xác suất lớn nhất trung bình chỉ 0,11–0,43, so với 0,76–0,99 của chuỗi `base`. Vì vậy trung bình đều gần như giữ nguyên độ chính xác của chuỗi tốt mà không cần chọn chuỗi.
- **Hệ quả cho bài:** kiểu sụp F1 là vấn đề của điểm xuất phát, không riêng của TFM. A1 (v2) giải quyết được ở phần lớn trường hợp, kể cả với TFM có điểm hút thấp.
- **LLP, chỉ báo cáo:**
  - TabPFN-3 trên tic-tac-toe đạt AUC 0,987, so với 0,746 của TabPFN v2.
  - Trên 6332: 0,736 so với 0,615.
  - Trên tập phát triển, TabPFN-3 ngang TabICL:
    - hơn ở 23 và 1480, nhỉnh hơn chút ở 36 và 1510;
    - bằng ở 54 và 1462;
    - kém ở 1464, 31 và 37.

## 5. Phần D: chuỗi dò (tạm thời, 148/234 ô)

- **AUC:** AUC(s1 → T1) = 0,52, với 10 ô dương tính; AUC(s1 → T2) = 0,49.
  - Riêng TabICL: 0,70. Riêng TabPFN v2: 0,40.
  - Nếu giữ nguyên khi đủ ô, **P15 sai**: độ lệch giữa các chuỗi dò sau 3 vòng quét không báo trước được ô nhiều lưu vực.
- **Chốt:** chỉ chốt sau phiên 2.

## 6. Việc còn lại: phiên 2 (`s2_code_v10.zip`)

- **Phần còn thiếu:**
  - D: 86 ô;
  - E: TabICL trên F1, TabDPT;
  - F: toàn bộ;
  - G: toàn bộ.
- **Chia lại việc giữa hai GPU (08/11, không đổi protocol: cùng ô, cùng seed, cùng code chạy):**
  - Ở phiên 1, GPU 1 mất khoảng 7 giờ cho phần D trên các dataset LLP của Study 2 (S2). Gần hết thời gian đó là tune learner L trên CPU.
  - Ở bản v10, G và phần F của TabICL ngoài S2-LLP chuyển sang GPU 0.
  - Phần F của TabICL trên S2-LLP được chia 3 mảnh.
