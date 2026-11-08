# Lượt V1C — kết quả (phiên 1: 07/11; phiên 2: 08/11/2026)

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
| D, P15 | xong (234/234) | **SAI.** AUC 0,45 |
| E, P14 | xong | **SAI**, đạt 1/4. Lý do: A1 của chính TabPFN v2 đã thoát kiểu sụp F1 ở 3/4 dataset (§4) |
| E, P13 | xong | **ĐÚNG.** Gộp 4 TFM kém TFM tốt nhất (tiên tri) trung bình 0,008; thua quá 0,02 ở 2/13 dataset |
| F, P12 | 145/234 ô; TabPFN v2 đủ | **SAI**, đã chắc dù thiếu ô: riêng TabPFN v2 đã có G < 0 và 5 thụt lùi |
| G | xong | chỉ báo cáo (§8) |

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

## 5. Phần D: chuỗi dò (P15 SAI)

- **Đủ 234 ô** (18 ô T1, 24 ô T2): AUC(s1 → T1) = **0,45**; AUC(s1 → T2) = 0,51; AUC(s2 → T1) = 0,38. Riêng TabICL: 0,49. Riêng TabPFN v2: 0,37.
- **Kết luận:** sau 3 vòng quét, độ lệch giữa các chuỗi dò không mang thông tin về việc các chuỗi đầy đủ có về khác lưu vực hay không. Ở 3 vòng quét, mọi chuỗi còn đang di chuyển.
- **H8 cần tín hiệu khác.** Nếu không tìm được trước cổng V2, H8 trở thành giới hạn của bài.

## 6. Phần E: gộp chéo 4 TFM (P13 ĐÚNG) và TabDPT

A1 theo TFM, seed 0. X là trung bình đều dự đoán A1 của 4 TFM. Dấu — nghĩa là V1A thiếu ô (D-V1-1).

| Dataset | Setting | TabPFN v2 | TabICL v2 | TabPFN-3 | TabDPT | **X** |
|---|---|---|---|---|---|---|
| 50 | LLP | 0,746 | 0,999 | 0,987 | 0,999 | 0,996 |
| 1462 | LLP | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 |
| 1464 | LLP | 0,710 | 0,683 | 0,605 | 0,689 | 0,670 |
| 1480 | LLP | 0,790 | 0,756 | 0,765 | 0,760 | 0,789 |
| 1510 | LLP | 0,992 | 0,993 | 0,994 | 0,995 | 0,994 |
| 6332 | LLP | 0,615 | 0,647 | 0,736 | 0,760 | 0,751 |
| 12 | PLL | 0,973 | 0,988 | 0,983 | 0,983 | 0,987 |
| 23 | PLL | 0,586 | 0,575 | 0,612 | 0,570 | 0,591 |
| 28 | PLL | 0,977 | 0,990 | 0,989 | 0,988 | 0,989 |
| 36 | PLL | 0,981 | 0,974 | 0,976 | 0,981 | 0,979 |
| 54 | PLL | 0,811 | 0,835 | 0,835 | 0,862 | 0,854 |
| 1501 | PLL | 0,774 | 0,948 | 0,948 | 0,912 | 0,946 |
| 40966 | PLL | 0,966 | 0,982 | 0,966 | 0,982 | 0,972 |
| 31 | LLP | — | 0,790 | 0,777 | 0,774 | — |
| 37 | LLP | — | 0,837 | 0,798 | 0,841 | — |

- **P13.**
  - Trên 13 dataset, X kém TFM tốt nhất của từng dataset trung bình 0,0076. Ngưỡng là −0,01, nên đạt.
  - X thua quá 0,02 ở 2 dataset: 1464 (−0,040) và 23 (−0,021). Ngưỡng là ≤ 2, nên đạt.
  - **Ý nghĩa:** không cần biết trước TFM nào tốt, gộp đều cho gần bằng bộ chọn tiên tri. TFM nào không có dataset nào là tốt nhất ổn định.
- **TabDPT:** mạnh ngang hoặc hơn các TFM khác ở nhiều dataset. Tốt nhất ở 6332, 54, 37 và 1510; kém ở 1501. Không dataset nào ở đây thuộc tập huấn luyện của TabDPT.

## 7. Phần F: A1 cùng chi phí (P12 SAI)

A1e gồm 3 chuỗi × 10 vòng quét, burn-in 3. Số ô có đủ dữ liệu: TabPFN v2 69/69, TabICL 76/165. Phần F của TabICL bị cắt ở giới hạn 11 giờ của phiên 2.

| TFM | Setting | Dataset | G(A1e − v1) | G(A1 − v1), từ V1 | Thụt lùi R(A1e) |
|---|---|---|---|---|---|
| TabICL v2 | LLP | 22 | +0,0025 | +0,0051 | 1 |
| TabICL v2 | PLL | 10 | −0,0010 | +0,0021 | 0 |
| TabPFN v2 | LLP | 10 | −0,0044 | +0,0025 | 1 |
| TabPFN v2 | PLL | 13 | −0,0091 | +0,0043 | 4 |

- **P12 sai, và kết luận này không đổi dù có thêm ô TabICL.** Riêng TabPFN v2 đủ ô đã có G < 0 ở cả hai setting, với 5 thụt lùi, vượt ngưỡng 3.
- **Ý nghĩa:** chuỗi 10 vòng quét quá ngắn. Thăm dò `v1_explore2` thấy r gần ổn định từ vòng quét 8–10, nhưng dự đoán cuối cần đủ mẫu sau burn-in. **v2 giữ 3 × 30 vòng quét**, chi phí gấp 3 lần v1 vẫn là cái giá của A1.

## 8. Phần G: cơ chế nhãn yếu mới (TabICL v2, chỉ báo cáo)

Trung bình 3 seed. Cột "iid" là A1 với cơ chế cũ: LLP iid túi 8, hoặc PLL iid q 0,6.

| Dataset | Túi gom theo feature, 8: v1 / **A1** / L | Túi iid 32: v1 / **A1** / L | iid túi 8: A1 |
|---|---|---|---|
| 31 | 0,723 / **0,766** / 0,720 | 0,620 / **0,678** / 0,649 | 0,764 |
| 37 | 0,829 / **0,835** / 0,807 | 0,615 / **0,667** / 0,643 | 0,818 |
| 1462 | 1,000 / 1,000 / 1,000 | 1,000 / 1,000 / 1,000 | 1,000 |
| 1464 | 0,709 / **0,730** / 0,726 | 0,581 / 0,613 / **0,681** | 0,646 |
| 1480 | 0,762 / **0,766** / 0,690 | 0,344 / 0,448 / **0,450** | 0,679 |
| 1510 | 0,991 / **0,992** / 0,985 | 0,992 / 0,987 / 0,966 | 0,992 |

| Dataset | PLL phụ thuộc mẫu q 0,6: v1 / **A1** / L | PLL iid q 0,6: A1 |
|---|---|---|
| 23 | 0,517 / **0,523** / 0,508 | 0,562 |
| 36 | 0,509 / **0,517** / 0,396 | 0,972 |
| 54 | 0,734 / 0,731 / 0,690 | 0,845 |

- **Túi gom theo feature dễ hơn túi iid, trái với giả định ban đầu.** Túi đồng nhất về feature thì tỉ lệ nhãn gần 0 hoặc 1 hơn, nên mang nhiều thông tin hơn. Ở dạng hiện tại, cơ chế này không tạo được độ khó cho Study 3.
- **Túi 32 rất khó.** Ở 1480, v1 có AUC 0,344, thấp hơn mức ngẫu nhiên: chuỗi hội tụ về nhãn đảo. A1 kéo lên rõ, nhưng không hơn L ở 1464 và 1480.
- **PLL phụ thuộc mẫu khó hơn nhiều,** vì distractor luôn là các lớp dễ nhầm. Ví dụ 36: 0,517 so với 0,972 khi iid. A1 vẫn hơn L ở 2/3 dataset.
- **Đề xuất cho Study 3 (cần thầy quyết):**
  - giữ túi 32 và PLL phụ thuộc mẫu, nhưng PLL phụ thuộc mẫu ở mức q thấp hơn;
  - với túi gom theo feature: bỏ, hoặc đổi thành túi theo một feature phân loại, như cách gom túi thường gặp trong thực tế.

## 9. Việc còn lại (cũ)



- **Phần còn thiếu:**
  - D: 86 ô;
  - E: TabICL trên F1, TabDPT;
  - F: toàn bộ;
  - G: toàn bộ.
- **Chia lại việc giữa hai GPU (08/11, không đổi protocol: cùng ô, cùng seed, cùng code chạy):**
  - Ở phiên 1, GPU 1 mất khoảng 7 giờ cho phần D trên các dataset LLP của Study 2 (S2). Gần hết thời gian đó là tune learner L trên CPU.
  - Ở bản v10, G và phần F của TabICL ngoài S2-LLP chuyển sang GPU 0.
  - Phần F của TabICL trên S2-LLP được chia 3 mảnh.
