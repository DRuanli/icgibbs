# Giai đoạn 1 của V2: chọn bộ gộp chuỗi — kết quả và cổng V1-A (05/11/2026)

- **Căn cứ:** [PHASE1_PLAN](PHASE1_PLAN.md), ghi trước ở commit `7ec273b`. Code ở `a8e355c`, bản gửi thầy là `s2_code_v7.zip`. Script chấm `V2/analysis/v1_aggregate.py` được viết và commit trước khi có số, và chạy y nguyên, không sửa.
- **Tính chất:** phát triển phương pháp, tức **thăm dò** (PHASE1_PLAN, phần đầu). Phép xác nhận là Study 3.
- **Thiếu dữ liệu:** 4 trong 14 dataset LLP của TabPFN v2 không có, vì OpenML trả lỗi 503 lúc tải (§3, deviation D-V1-1).
- **Cổng V1-A đã chốt (05/11/2026, tối):** thầy quyết định **không chạy bù** (D-V1-1, bổ sung). Cổng được chốt trên số liệu hiện có, với 10/14 dataset của nhóm TabPFN LLP (§6).

## 0. Tóm tắt

1. **Bộ gộp được chọn: A1, trung bình đều 3 chuỗi** (warm, base, cold). Theo quy tắc §4:
   - tổng số thụt lùi R: A1 = 2, A2 = 3, A3 = 3;
   - A1 cũng có G lớn nhất: +0,0032, so với +0,0021 (A2) và +0,0026 (A3).
2. **Dự đoán ghi trước:**
   - **P5 SAI.** Với TabPFN, A3 (stacking có learner L) không ít thụt lùi hơn: cả ba bộ đều có 1 thụt lùi, cùng ở blood-transfusion (1464).
   - **P6 ĐÚNG.** Với TabICL, chênh G giữa ba bộ nhỏ: tối đa 0,0031 với LLP và 0,0003 với PLL, đều dưới 0,005.
   - **P7 ĐÚNG.** A1 có G dương ở cả bốn nhóm, từ +0,0019 đến +0,0043.
3. **Cổng V1-A:** không nhóm nào có ≥ 3 thụt lùi với A1, nên không phải ghi giới hạn nhóm theo §7.
   - Cả 2 thụt lùi của A1 đều ở cùng một dataset, 1464, với hai TFM. Ở đó learner khởi động L (DLLP-MLP) tốt hơn mọi chuỗi.
   - **A1 không thụt lùi so với v1 ở bất kỳ dataset nào.**
4. **Độ vững của lựa chọn:** A1 chỉ hơn A2/A3 đúng 1 thụt lùi. A1 chỉ mất vị trí nếu, trên 4 dataset còn thiếu, nó có nhiều hơn ít nhất 2 thụt lùi so với bộ tốt hơn trong A2 và A3 (§3.2). Trên chính 4 dataset này, TabICL không có thụt lùi nào với bộ nào. Đây chỉ là chỉ báo, không phải bằng chứng cho TabPFN.
5. **Thăm dò (viết sau khi xem số, §4):**
   - Ở 83–97% ô, ba chuỗi kết thúc cùng một lưu vực.
   - Lợi ích của A1 so với v1 nhỏ (+0,002 đến +0,004) ở các ô đó, khớp với việc giảm phương sai Monte Carlo.
   - Lợi ích lớn hơn ở số ít ô mà các chuỗi kết thúc ở các lưu vực khác nhau.
   - So từng dataset: A1 thắng v1 ở 39 dataset, thua ở 11 dataset, và hoà ở các dataset còn lại.

## 1. Lượt chạy và kiểm tra

| Lượt | Thời gian (UTC, 05/11) | Phần cứng | Bộ test | Dòng ok / tổng | Ghi chú |
|---|---|---|---|---|---|
| V1A (TabPFN v2) | 03:56 – 11:51, tức 7 giờ 55 phút | Kaggle 2 × T4 | 130 qua, 2 bỏ qua | 345 / 345 | dataset 31, 37, 23381, 38 lỗi tải (HTTP 503), nên không có dòng nào |
| V1B (TabICL v2) | 02:43 – 13:12, tức 10 giờ 29 phút | Kaggle 2 × T4 | 130 qua, 2 bỏ qua | 825 / 825 | đủ 32 dataset LLP và 23 dataset PLL |

Các phép kiểm:
- **Mã băm code:** 73/73 tệp của V1A và V1B khớp code trong repo (commit `a8e355c`). Hai lượt có cùng mã băm.
- **Lần chia và nhãn yếu:**
  - 138/138 bộ (dataset, setting, seed) có bản ghi Study 1 hoặc Study 2 ở cùng mức đều khớp cả `sha256_split` và `weak_hash`.
  - 27 bộ thuộc tập phát triển không có bản ghi cũ ở cùng mức. Với các bộ này, hai TFM cho cùng mã băm.
- **Tính tất định:**
  - `gibbs_init_warm` (chuỗi v1) bằng `gibbs_warm` của Study 1 hoặc Study 2 ở **186/186 ô** (138 TabICL, 48 TabPFN), chênh 0.
  - Learner L cho cùng kết quả với hai TFM ở 69/69 ô chung.
- **Script chấm:**
  - A0 trùng chuỗi có WPLL lớn nhất (`gibbs_ms`) ở 234/234 ô;
  - không có hàng nào của dự đoán ngoài fold là không hữu hạn.

## 2. Chấm theo PHASE1_PLAN §4–§5

Số ô: 234. Số dataset theo nhóm: TabICL LLP 32, TabICL PLL 23, TabPFN LLP **10/14**, TabPFN PLL 13. Chỉ số là AUC với LLP và accuracy với PLL, lấy trung bình 3 seed. Toàn bộ số ở `V2/results/v1_cells.csv` và `v1_summary.json`.

### 2.1 Trung bình theo nhóm

| Nhóm | v1 | L | A0 | **A1** | A2 | A3 |
|---|---|---|---|---|---|---|
| TabICL, LLP | 0,8497 | 0,8191 | 0,8487 | **0,8538** | 0,8506 | 0,8507 |
| TabICL, PLL | 0,8116 | 0,7786 | 0,8127 | **0,8135** | 0,8136 | 0,8139 |
| TabPFN, LLP | 0,8974 | 0,8809 | 0,8968 | **0,8999** | 0,8980 | 0,8981 |
| TabPFN, PLL | 0,7683 | 0,7319 | 0,7705 | **0,7725** | 0,7733 | 0,7746 |

### 2.2 Thụt lùi và G

Thụt lùi được tính so với max(v1, L) − 0,02. G là trung bình của m_A − m_v1.

| Nhóm | n | R_A0 | R_A1 | R_A2 | R_A3 | G_A0 | G_A1 | G_A2 | G_A3 |
|---|---|---|---|---|---|---|---|---|---|
| TabICL, LLP | 32 | 2 | 1 | 2 | 2 | −0,0010 | +0,0041 | +0,0009 | +0,0010 |
| TabICL, PLL | 23 | 0 | 0 | 0 | 0 | +0,0011 | +0,0019 | +0,0019 | +0,0023 |
| TabPFN, LLP | 10 | 1 | 1 | 1 | 1 | −0,0006 | +0,0025 | +0,0006 | +0,0007 |
| TabPFN, PLL | 13 | 0 | 0 | 0 | 0 | +0,0022 | +0,0043 | +0,0050 | +0,0064 |
| **Tổng / trung bình nhóm** | 78 | 3 | **2** | 3 | 3 | +0,0004 | **+0,0032** | +0,0021 | +0,0026 |

- **Quy tắc chọn:** R nhỏ nhất thuộc về A1, nên chọn **A1**. Không cần dùng đến bước 2 và bước 3.
- **Các thụt lùi:**
  - **1464 (blood-transfusion, LLP):** có với cả hai TFM và cả ba bộ A1–A3. L đạt 0,6790, còn A1 đạt 0,6529 (TabPFN) và 0,6463 (TabICL).
  - **470 (TabICL, LLP):** chỉ có với A2 và A3. Ở đây A2 đạt 0,5566 và A3 đạt 0,5539, trong khi v1 đạt 0,5803.
- **Tách theo nguồn:** so riêng với v1, A1 có **0** thụt lùi ở cả bốn nhóm. A2 và A3 mỗi bộ có 1 thụt lùi (470). Mọi thụt lùi của A1 là so với L.

### 2.3 Dự đoán ghi trước

| | Nội dung (§5) | Số đo | Kết luận |
|---|---|---|---|
| P5 | TabPFN: R(A3) < R(A1) và R(A3) < R(A2) | R = 1, 1, 1 | **SAI** |
| P6 | TabICL: chênh G mọi cặp < 0,005 trong từng setting | chênh lớn nhất: 0,0031 (LLP), 0,0003 (PLL) | **ĐÚNG** |
| P7 | bộ được chọn có G ≥ −0,002 ở mọi nhóm | G của A1: +0,0041, +0,0019, +0,0025, +0,0043 | **ĐÚNG** |

**Vì sao P5 sai:**
- Stacking chỉ cho L trọng số trung bình 0,02–0,07. Tỉ lệ ô có w_L > 0,5 là 0–3%.
- Ở 1464, A3 cho đúng kết quả của A2: 0,6408 với TabPFN. Nghĩa là stacking theo hợp lý của nhãn yếu không nhận ra rằng L tốt hơn ở dataset này.
- P5 dựa trên giả định rằng thụt lùi của TabPFN là do chuỗi bị kéo xa khỏi L, và trọng số của L sẽ sửa được. Số liệu không ủng hộ giả định này.

### 2.4 Báo cáo kèm

| Nhóm | A2: w_warm / w_base / w_cold | A3: w_warm / w_base / w_cold / w_L | Tỉ lệ ô có w_L > 0,5 |
|---|---|---|---|
| TabICL, LLP | 0,307 / 0,403 / 0,290 | 0,290 / 0,378 / 0,261 / 0,072 | 1,0% |
| TabICL, PLL | 0,285 / 0,446 / 0,270 | 0,278 / 0,440 / 0,259 / 0,022 | 0% |
| TabPFN, LLP | 0,333 / 0,396 / 0,271 | 0,334 / 0,393 / 0,238 / 0,034 | 3,3% |
| TabPFN, PLL | 0,265 / 0,486 / 0,248 | 0,250 / 0,468 / 0,235 / 0,047 | 2,6% |

- Stacking nghiêng về chuỗi `base` nhưng không nhiều.
- A0 (argmax WPLL, tức cách chọn của v1 nhiều điểm xuất phát) có G âm với LLP và R = 3. Điều này khớp với kết luận ở giai đoạn 0 (V0_REPORT §6), rằng WPLL không phải tiêu chí chọn tốt.
- Bảng theo dataset có ở `v1_cells.csv`, và in đầy đủ trong đầu ra của script.

## 3. Deviation D-V1-1 (05/11/2026): thiếu 4 dataset LLP của TabPFN

### 3.1 Sự việc và cách xử lý

- **Sự việc:** trong lượt V1A, GPU 0 không tải được dataset 31, 37 (tập phát triển) và 23381, 38 (tập đánh giá), do OpenML trả `HTTPError 503: Service Temporarily Unavailable`. Script ghi `[lỗi tải]` rồi bỏ qua; không có dòng lỗi trong CSV.
  - Thiếu 12 ô (4 dataset × 3 seed), tức 10/14 dataset của nhóm TabPFN LLP.
  - TabICL (V1B) có đủ 4 dataset này.
- **Xử lý (không đổi protocol):** chạy bù đúng lượt V1A, với output cũ làm dữ liệu.
  - `run_g7.py` bỏ qua ô đã có dòng `ok`, nên chỉ 12 ô thiếu được chạy.
  - Script chấm và quy tắc chọn giữ nguyên. Kết quả chốt là kết quả chạy lại `v1_aggregate.py` trên đủ 14 dataset.
- **Nếu OpenML vẫn lỗi sau hai lần chạy bù:** chốt trên 10/14, và ghi giới hạn này vào báo cáo cổng.
- **Bổ sung (05/11/2026, 20:57 giờ Việt Nam): thầy quyết định không chạy bù.** Cổng V1-A chốt trên 10/14 dataset của nhóm TabPFN LLP. Lý do:
  - giai đoạn này là thăm dò;
  - TabICL đã phủ đủ 4 dataset;
  - lựa chọn chỉ đổi trong trường hợp ở §3.2;
  - tiết kiệm khoảng 3 giờ GPU cho các lượt sau.
- **Hệ quả:** 4 dataset này (31, 37, 23381, 38) không có kết quả nhiều chuỗi với TabPFN v2. Khi nhắc tới bước chọn bộ gộp, bài phải ghi rõ con số 10/14. Các dataset này vẫn thuộc tập phát triển của Study 3, như PHASE1_PLAN đã định.

### 3.2 Độ vững của lựa chọn

Lựa chọn A1 chỉ đổi nếu trên 4 dataset bù, số thụt lùi thêm của A1 (x₁) và của A2, A3 (x₂, x₃) thoả **x₁ ≥ 2 + min(x₂, x₃)**. Có hai lý do:
- Nếu x₁ = 1 + min(x₂, x₃), R hoà. Khi đó:
  - Chênh G hiện tại là 0,0011 (A1 so với A2) và 0,0006 (A1 so với A3). Nếu chênh G vẫn dưới 0,002 thì bước 3 chọn A1, vì A1 đơn giản nhất.
  - Nếu chênh G vượt 0,002 thì bước 2 chọn bộ có G lớn nhất. A1 hiện dẫn về G.
- Nếu x₁ ≤ min(x₂, x₃), A1 vẫn có R nhỏ nhất.

Với TabICL trên đúng 4 dataset này, cả ba bộ đều không thụt lùi. Ở 31, A2 và A3 sát ngưỡng: 0,7559, so với ngưỡng 0,7521.

## 4. Phân tích THĂM DÒ (viết sau khi xem số)

Script `V2/analysis/v1_explore.py`; đầu ra ở `V2/results/v1_explore.txt` và `v1_explore_cells.csv`. Không phải dự đoán ghi trước. Ngưỡng 0,03 cho "cùng lưu vực" chọn sau khi xem số.

**E1. Các điểm xuất phát có về cùng một lưu vực không.** Với mỗi chuỗi, r_cuối là trung bình 5 vòng quét cuối của `monitor`, tức độ đúng của nhãn chuỗi trên train, tính bằng nhãn thật. Ô được tính là "cùng lưu vực" nếu max − min của r_cuối giữa ba chuỗi < 0,03.

| Nhóm | Ô | Cùng lưu vực | Trung vị spread | Phân vị 90 của spread |
|---|---|---|---|---|
| TabICL, LLP | 96 | 91,7% | 0,008 | 0,028 |
| TabICL, PLL | 69 | 94,2% | 0,005 | 0,021 |
| TabPFN, LLP | 30 | 83,3% | 0,008 | 0,037 |
| TabPFN, PLL | 39 | 97,4% | 0,006 | 0,019 |

- **Liên hệ với lý thuyết (W1_R1A_REPORT, Mệnh đề 4′ và 6):** phần lớn ô trông như chế độ một điểm hút. Ba chuỗi xuất phát rất khác nhau, nhưng sau 30 vòng quét đều nằm trong một dải rộng cỡ 0,01.
  - Độ rộng này cùng bậc với dao động dừng trong mô phỏng của Mệnh đề 6: sd mỗi vòng quét 0,008–0,035, với n từ 500 đến 3.000.
  - Chưa so định lượng được, vì chưa có Φ′ đo trên TFM thật.
- **Giới hạn của suy luận:** với 18 ô còn lại (7,7%), 30 vòng quét không đủ để phân biệt hai khả năng:
  - nhiều lưu vực thật sự;
  - một lưu vực nhưng trộn chậm.

**E2. Lợi ích của A1 tách theo lưu vực** (trung bình theo ô):

| Nhóm | Cùng lưu vực? | Ô | A1 − v1 | A1 − trung bình 3 chuỗi đơn | Trung vị spread chỉ số |
|---|---|---|---|---|---|
| TabICL, LLP | không | 8 | +0,0119 | +0,0122 | 0,054 |
| TabICL, LLP | có | 88 | +0,0034 | +0,0041 | 0,005 |
| TabICL, PLL | không | 4 | +0,0041 | +0,0025 | 0,023 |
| TabICL, PLL | có | 65 | +0,0018 | +0,0012 | 0,006 |
| TabPFN, LLP | không | 5 | +0,0051 | +0,0243 | 0,082 |
| TabPFN, LLP | có | 25 | +0,0020 | +0,0028 | 0,003 |
| TabPFN, PLL | không | 1 | +0,0045 | +0,0211 | 0,032 |
| TabPFN, PLL | có | 38 | +0,0043 | +0,0027 | 0,008 |

- **Ô cùng lưu vực:** A1 hơn cả v1 lẫn trung bình của ba chuỗi đơn khoảng 0,001–0,004. Đây là phần lợi do lấy trung bình dự đoán của ba ước lượng Monte Carlo của cùng một phân phối dừng.
- **Ô khác lưu vực:** lợi ích lớn hơn, nhưng chỉ có 18 ô.
  - Một bộ chọn **tiên tri** (chuỗi có r_cuối cao nhất, dùng nhãn thật nên không triển khai được) hơn A1 khoảng 0,019 với TabICL LLP và 0,006 với TabICL PLL.
  - Với TabPFN, chuỗi `warm` luôn có r_cuối cao nhất (6/6 ô), và A1 còn hơn chuỗi đó.
  - A0 (WPLL lớn nhất, triển khai được) chọn trúng chuỗi tiên tri ở 60–100% số ô này. Cỡ mẫu quá nhỏ để kết luận.

**E3. Blood-transfusion (1464).** Chỉ số của chuỗi đơn dao động mạnh theo seed: AUC 0,48–0,76, và L dao động 0,57–0,75. Mức r_cuối của chuỗi không xếp hạng đúng chuỗi theo AUC test. Ví dụ với TabICL seed 0: chuỗi `warm` có r = 0,675 và AUC = 0,585, chuỗi `base` có r = 0,729 và AUC = 0,757.
- Dataset nhỏ (748 dòng, túi 8), nên thụt lùi ở đây nhiều khả năng là nhiễu giữa các seed cộng với lợi thế của L, hơn là một lỗi có hệ thống của chuỗi.
- Đây là nhận định thăm dò, chưa kiểm.

**E4. A1 so với v1 theo dataset.** Hoà nghĩa là |chênh| < 0,001. Kiểm định Wilcoxon hai phía, không hiệu chỉnh đa kiểm định.

| Nhóm | Dataset | Thắng / hoà / thua | Trung bình | Trung vị | Nhỏ nhất | Lớn nhất | p (Wilcoxon) |
|---|---|---|---|---|---|---|---|
| TabICL, LLP | 32 | 17 / 12 / 3 | +0,0041 | +0,0019 | −0,0086 | +0,0428 | 0,0007 |
| TabICL, PLL | 23 | 12 / 7 / 4 | +0,0019 | +0,0015 | −0,0166 | +0,0279 | 0,036 |
| TabPFN, LLP | 10 | 4 / 5 / 1 | +0,0025 | +0,0002 | −0,0057 | +0,0118 | 0,16 |
| TabPFN, PLL | 13 | 6 / 4 / 3 | +0,0043 | +0,0006 | −0,0120 | +0,0362 | 0,28 |

## 5. Ý nghĩa và bước tiếp

- **Cho v2:**
  - A1 (trung bình đều 3 chuỗi) là mặc định của v2 (cổng V1-A, §6). Đây là bộ đơn giản nhất, không cần learner hay trọng số học, và không thụt lùi so với v1 ở dataset nào.
  - Đổi lại, A1 tốn gấp 3 lần TFM so với v1.
  - Lợi ích trung bình nhỏ, +0,2 đến +0,4 điểm. Đây là cải thiện ổn định chứ không phải bước nhảy.
- **Cho bài:**
  - Kết luận "nhiều chuỗi và trung bình đều" đi vào phương pháp v2 dưới dạng đã chọn trên tập phát triển.
  - Kết quả E1 (phần lớn ô một lưu vực) là bằng chứng thực nghiệm cho kịch bản của Mệnh đề 4′. Phải ghi rõ đây là thăm dò.
- **Việc tiếp theo:**
  1. ~~Chạy bù V1A~~: thầy quyết bỏ (D-V1-1, bổ sung); cổng đã chốt ở §6.
  2. Viết bản tiếng Anh của Mệnh đề 4′, 5 và 6 vào bài.
  3. **Vẫn chờ thầy duyệt:** đo Φ_chuỗi trên GPU, gieo theo Q̄ đã lưu trong npz của V1.

## 6. Cổng V1-A (chốt 05/11/2026)

Theo PHASE1_PLAN §7:
- **Bộ gộp mặc định của v2 cho Study 3: A1**, trung bình đều dự đoán của 3 chuỗi `warm`, `base`, `cold`, mỗi chuỗi với cấu hình v1 (K = 10, T = 30, burn-in 10, S = 10, ngữ cảnh tối đa 2.000).
- **Dự đoán ghi trước:** P5 sai, P6 đúng, P7 đúng.
- **Giới hạn theo nhóm:** không có nhóm nào ≥ 3 thụt lùi.
  - Theo V2_PLAN H7, ngưỡng thụt lùi đã biết trước cho Study 3 lấy từ số ở đây: 2/78 dataset, tức 2,6%, thụt lùi quá 0,02 so với max(v1, L).
  - Cả 2 thụt lùi đều so với L, ở cùng một dataset.
- **Không đổi gì khác của v2** trong giai đoạn này: K, T, các điểm xuất phát, TFM.
- **Giới hạn của lần chốt:** nhóm TabPFN LLP chỉ có 10/14 dataset (D-V1-1).

## 7. Tệp

- **Kết quả thô:** `V2/raw/V1A/`, `V2/raw/V1B/` (CSV, log, jsonl dự đoán, `v1_npz/` với 69 và 165 tệp, nhật ký pip, GPU, mã băm code).
- **Chấm ghi trước:** `V2/analysis/v1_aggregate.py`, cho ra `V2/results/v1_cells.csv` và `V2/results/v1_summary.json`.
- **Thăm dò:** `V2/analysis/v1_explore.py`, cho ra `V2/results/v1_explore.txt` và `V2/results/v1_explore_cells.csv`.
