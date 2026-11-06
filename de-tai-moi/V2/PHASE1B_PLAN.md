# Giai đoạn 1b của V2: lượt V1C — thử TFM mới và đo Φ_chuỗi (ghi trước, 06/11/2026)

- **Căn cứ:**
  - [W4_TFM_REPORT](W4_TFM_REPORT.md) §4 (lượt thử GPU) và §2 (nhiễm dữ liệu);
  - V2_PLAN W1.1 (ánh xạ trên lỗi có cấu trúc; "Xong khi": sai số tuyệt đối trung vị ≤ 0,1);
  - đề xuất Φ_chuỗi trong W1_R1A_REPORT §6.
- **Duyệt:** thầy duyệt ngày 06/11/2026 ("Đồng ý và phát triển đi"). Lần duyệt gồm:
  - lượt thử GPU;
  - quy tắc loại dataset (§5);
  - đo Φ_chuỗi.
- **Tính chất:** thăm dò. Không có số test mới của phương pháp: lượt này chỉ đo thời gian, tính tất định, ánh xạ phản ứng, và so ánh xạ với chuỗi V1 đã chạy.
- **Thời điểm ghi:** tài liệu này được commit **trước** khi viết code của V1C. Script chấm `V2/analysis/v1c_report.py` được viết và commit cùng code, trước khi có số. Mọi thay đổi về sau phải ghi thành deviation có ngày.

## 1. Ba phần của lượt V1C

**A. Tốc độ và tính tất định (W4), script `w4_pilot.py`.**
- **Backend:** `tabpfn_v2`, `tabicl_v2` (làm mốc), `tabpfn_v3`, `tabpfn_v3_5`, `tabdpt`.
- **Dữ liệu:** tổng hợp (`make_classification`), d = 20, truy vấn 200 điểm.
- **Lưới:** (n_ngữ_cảnh, C) ∈ {(500, 2), (2.000, 2), (500, 10), (2.000, 10), (2.000, 26)}. TabPFN v2 bỏ C = 26, vì nó chỉ nhận tối đa 10 lớp.
- **Mỗi ô:**
  - 1 lượt khởi động, rồi 5 lượt gọi đo giờ với cấu hình trong chuỗi (n_estimators = 1);
  - 2 lượt với cấu hình dự đoán cuối, chỉ ở ô (2.000, 10);
  - chênh tuyệt đối lớn nhất giữa hai lượt gọi giống hệt nhau;
  - bộ nhớ GPU đỉnh.

**B. Ánh xạ phản ứng của TFM mới trên các ô F1 (W4, W1), script có sẵn `v0_response_map.py`, không sửa.**
- **Backend:** như phần A. TabPFN v2 và TabICL chạy lại trên cùng lưới để so trực tiếp.
- **Dữ liệu:**
  - PLL q = 0,6 trên 12, 1501, 40966, 28, 188, 40982 (`--plant uniform` và `--plant struct`);
  - LLP túi 8 trên 50, 6332 (`--plant uniform`).
  - Không dataset nào trong số này thuộc tập huấn luyện của TabDPT.
- **Seed:** 0. **Lưới ρ:** 0,2; 0,35; 0,5; 0,65; 0,8; 0,9; 0,97.

**C. Φ_chuỗi trên các ô V1 (W1.1), script mới `v1c_chain_map.py`.**
- **Ô:** mọi ô của V1A (TabPFN v2, 69 ô) và V1B (TabICL v2, 165 ô), mỗi ô chỉ với chuỗi `warm`.
  - Dựng lại đúng ô như `run_g7.py`, gồm cả lấy mẫu con của S2.
  - Kiểm `sha256_split`, `weak_hash` và `ytr` khớp tệp npz của V1. Không khớp thì ghi `mismatch` và bỏ ô.
- **Ba cách gieo lỗi.** Với mỗi ρ, gieo một vector nhãn z có tỉ lệ đúng ρ, rồi đo Φ bằng một vòng quét kiểu Jacobi: K = 10 khối, ngữ cảnh tối đa 2.000, cùng TFM với chuỗi, n_estimators = 1.
  - `uniform`: như S2X và V0; LLP thì hoán đổi cặp trong túi.
  - `struct`: chỉ PLL, như V0, với P0 = khởi động ấm (TFM ngây thơ chéo 5 fold).
  - `chain` (mới): dùng Q̄ của chính chuỗi `warm` đã lưu ở V1.
    - PLL: dùng `planted_struct` của V0 với P0 = Q̄_warm. Điểm sai được chọn với xác suất ∝ 1 − Q̄_i(y_i), nhãn sai ∝ Q̄_i trên các ứng viên sai.
    - LLP: trong mỗi túi, số cặp đổi s ~ Binomial(m, 1 − ρ) như `uniform`. Điểm dương bị đổi được chọn với xác suất ∝ 1 − Q̄_i(1), điểm âm ∝ 1 − Q̄_i(0).
- **Lưới ρ:** như phần B.
- **r_cuối thật** của chuỗi `warm`: trung bình 5 vòng quét cuối của `monitor_warm` (như `v1_explore.py`).
- **r_đầu:** `rstart_warm` trong npz.

## 2. Quy tắc dự đoán r_cuối từ một ánh xạ (dùng cho P8, P9)

Mỗi ô và mỗi cách gieo cho các điểm (r_k, g_k), với r_k là tỉ lệ đúng thực của z và g_k = Φ_k − r_k. Sắp các điểm theo r_k và nội suy tuyến tính g(r).
1. Kẹp r_đầu vào [r_min, r_max].
2. **Nếu g(r_đầu) ≥ 0:** đi lên tới điểm cắt đầu tiên mà g đổi từ ≥ 0 sang < 0, nội suy tuyến tính. Nếu không có điểm cắt nào, dự đoán là min(1, Φ tại r_max).
3. **Nếu g(r_đầu) < 0:** đi xuống tới điểm cắt đầu tiên mà g đổi từ < 0 sang ≥ 0, nội suy tuyến tính. Nếu không có điểm cắt nào, dự đoán là Φ tại r_min.
4. **Sai số:** |r̂_cuối − r_cuối thật|.

## 3. Dự đoán ghi trước

- **P8 (tiêu chí "Xong khi" của W1).** Sai số tuyệt đối trung vị của cách gieo `chain`, gộp mọi ô của hai TFM, **≤ 0,10**. Em dự đoán: đúng.
- **P9 (F2: lỗi gieo đều chỉ đúng về định tính).** Trên các ô có cả hai cách gieo, sai số trung vị của `chain` **nhỏ hơn** của `uniform`. Em dự đoán: đúng.
- **P11 (TFM mới sửa được kiểu sụp F1).**
  - **Tiêu chí "không có điểm hút thấp" của một (TFM, dataset):** với cách gieo `uniform` ở phần B, g > 0 tại cả ba mức ρ ∈ {0,2; 0,35; 0,5}.
  - **Dự đoán:** TabPFN-3.x được chọn theo §4 đạt tiêu chí này trên **≥ 3/4** dataset F1 (12, 1501, 40966, 28). Em dự đoán: đúng, độ tin vừa phải.
- **Kiểm nhất quán, không phải dự đoán:** với cùng tiêu chí, TabPFN v2 phải trượt trên ≥ 3/4 dataset F1, và TabICL v2 phải đạt trên ≥ 3/4. S2X đã cho kết quả này: TabPFN v2 trượt 4/4, TabICL đạt 4/4. Nếu V1C cho khác, đó là dấu hiệu môi trường hoặc code đã đổi, cần kiểm trước khi đọc tiếp.
- **Báo cáo kèm, không dự đoán:**
  - kết quả P11 của TabDPT và của bản TabPFN-3.x không được chọn;
  - ánh xạ LLP trên 50 và 6332;
  - sai số theo nhóm (TFM × setting) và theo cách gieo `struct`;
  - tỉ lệ ô có điểm cắt.

## 4. Chọn TabPFN-3 hay 3.5 (ghi trước, từ W4_TFM_REPORT §4, làm rõ thêm)

- **Hợp lệ:** một bản hợp lệ nếu cả hai điều sau đúng:
  - mọi ô của phần A chạy không lỗi với C ≤ 10;
  - chênh giữa hai lượt gọi giống hệt nhau ≤ 1e−4.
- **Tỉ số tốc độ:** trung vị theo C ∈ {2, 10} của [thời gian trung vị mỗi lượt gọi của v3.5 / của v3], ở n = 2.000.
- **Chọn 3.5** nếu cả hai bản hợp lệ và tỉ số ≤ 1,5.
- **Chọn v3** nếu cả hai hợp lệ và tỉ số > 1,5, hoặc nếu chỉ v3 hợp lệ.
- **Chọn 3.5** nếu chỉ 3.5 hợp lệ.
- **Không bản nào hợp lệ:** báo thầy, và W4 chỉ còn TabDPT.

## 5. Quy tắc loại dataset cho Study 3 (thầy duyệt 06/11)

- Loại khỏi tập ứng viên của Study 3 mọi dataset nằm trong tập huấn luyện công bố của một TFM dùng trong Study 3. Hiện danh sách là của TabDPT: `V2/data/tabdpt_pretrain_ids.csv`, 123 ID.
- Loại cả các bản trùng của chúng, tìm bằng mã băm bảng như S2, khi đếm dataset ở giai đoạn 2.
- Ở tập phát triển, kết quả TabDPT trên 13 dataset Study 2 bị nhiễm chỉ được báo cáo tách riêng.

## 6. Chi phí và cách chạy

- **Một lượt, Kaggle 2 × T4, khoảng 5–7 giờ:**
  - **GPU 0:** phần A (khoảng 30 phút, gồm cả tải trọng số), rồi phần B, rồi phần C cho TabPFN v2 (khoảng 1,5 giờ).
  - **GPU 1:** phần C cho TabICL v2 (khoảng 2–3 giờ).
- **Đầu vào:** output của V1A và V1B (thư mục `v1_npz/`) thêm làm dữ liệu.
- **Chạy tiếp được:** mỗi script bỏ qua các dòng đã `ok`.
- **Code:** `s2_code_v8.zip`, dựng bằng `git archive` từ commit code.

## 7. Bổ sung ngày 07/11/2026, trước khi chạy V1C

- **Lý do:** thầy yêu cầu lượt V1C chạy được nhiều việc trong một lần, giải quyết các vấn đề còn mở và làm phương pháp mạnh hơn.
- **Tình trạng dữ liệu:** V1C chưa chạy, nên chưa có số nào của lượt này. Các phần A, B, C và mọi dự đoán ở §1–§5 **giữ nguyên**.
- **Các phần mới D–G:** cùng một lượt V1C, cùng `s2_code_v9.zip`.
- **Cơ sở thăm dò:** em viết `V2/analysis/v1_explore2.py` ngày 07/11, trên số V1 **đã xem**, để thiết kế phần F và phần gộp chéo TFM. Vì vậy:
  - kết quả của phép thăm dò đó **không** được dùng làm bằng chứng;
  - mọi dự đoán dưới đây chỉ chấm trên **dữ liệu mới** của V1C.

### 7.1 Vấn đề còn mở và phần giải quyết

| Vấn đề | Phần |
|---|---|
| W1.3/H8: dữ liệu V1 chỉ có tín hiệu sau khi chạy chuỗi, nên chưa phát triển được phép chẩn đoán đo trước | **D**: chuỗi dò 3 vòng quét, không cần nhãn |
| A1 tốn gấp 3 lần TFM so với v1 | **F**: A1 cùng chi phí (3 chuỗi × 10 vòng quét) |
| F1: TabPFN v2 sụp; mới có ánh xạ (P11), chưa có chuỗi | **E**: chuỗi A1 với TabPFN-3.x được chọn và TabDPT |
| Phương pháp mạnh hơn và bền với TFM (H7) | **E**: gộp chéo TFM (trung bình A1 của mọi TFM trên cùng ô) |
| Các cơ chế nhãn yếu mới của Study 3 chưa chạy lần nào | **G**: TabICL v2 với túi gom theo feature, túi 32, PLL phụ thuộc mẫu |

### 7.2 Các phần mới

**D. Chuỗi dò (H8).**
- **Ô:** mọi ô của V1, gồm 69 ô TabPFN v2 và 165 ô TabICL v2.
- **Cách chạy:** `run_g7.py --part v0 --sweeps 3`, 3 điểm xuất phát (burn-in 1), lưu npz.
- **Tín hiệu, không dùng nhãn thật:**
  - **s1** = trung bình theo cặp, trên các cặp điểm xuất phát, của khoảng cách biến phân toàn phần (TV) trung bình theo điểm giữa các Q̄ sau 3 vòng quét;
  - **s2** = max − min của WPLL giữa ba điểm xuất phát.
- **Đích (phát triển):** T1 = ô V1 có các chuỗi về khác lưu vực (spread r_cuối ≥ 0,03, như V1_REPORT §4). Ngưỡng 0,03 được chọn sau khi xem V1, nên phần này là **phát triển**, không phải xác nhận.
- **P15.** AUC của s1 cho T1, gộp hai TFM, **≥ 0,75**. Em dự đoán: đúng, độ tin thấp.
- **Báo cáo kèm:** AUC của s2. Thêm đích T2 = ô mà v1 thua max(A1, L) quá 0,02.
- **Phép xác nhận của H8 vẫn là Study 3**, với ngưỡng AUC 0,8 như V2_PLAN.

**E. Chuỗi với TFM mới, và gộp chéo TFM.**
- **TFM:** bản TabPFN-3.x được chọn theo §4 (chọn tự động trong lượt, bằng đúng quy tắc §4), cùng TabDPT.
- **Cấu hình:** 3 điểm xuất phát, cấu hình V1, seed 0.
- **Dataset:**
  - PLL q = 0,6: F1 gồm 12, 1501, 40966, 28; tập phát triển gồm 54, 36, 23;
  - LLP túi 8: F1 gồm 50, 6332; tập phát triển gồm 31, 37, 1464, 1462, 1480, 1510.
- **Mốc so sánh:** chạy thêm TabPFN v2 và TabICL v2 trên 6 dataset F1, seed 0, vì V1 chưa có các ô này. Các ô phát triển lấy từ V1.
- **P14 (chuỗi sửa được F1).** Trên 4 dataset F1 PLL, A1 của TabPFN-3.x có accuracy ≥ A1 của TabPFN v2 + 0,05 ở **≥ 3/4** dataset. Em dự đoán: đúng nếu P11 đúng.
- **P13 (gộp chéo TFM).**
  - **Cách gộp:** trên mỗi ô có đủ 4 TFM (TabPFN v2, TabICL v2, TabPFN-3.x, TabDPT), X = trung bình đều dự đoán A1 của 4 TFM.
  - **Điều kiện:** trung bình theo dataset của (X − A1 của TFM tốt nhất trên ô đó, là bộ chọn tiên tri) **≥ −0,01**, và X thua TFM tốt nhất quá 0,02 ở **≤ 2/15** dataset.
  - **Dự đoán:** đúng.
  - **Ô thiếu:** ô nào thiếu một TFM thì bỏ khỏi P13 và báo cáo kèm.
- **Nhiễm dữ liệu:** không dataset nào của phần E thuộc tập huấn luyện của TabDPT (đã kiểm ngày 06/11).

**F. A1 cùng chi phí.**
- **Ô:** mọi ô của V1, với cả hai TFM.
- **Cấu hình:** `--sweeps 10` (burn-in 3, dùng 7 mẫu), 3 điểm xuất phát. Gọi kết quả là A1e.
- **Chi phí:** số lượt gọi TFM trong chuỗi của 3 chuỗi × 10 vòng quét bằng 1 chuỗi × 30 vòng quét của v1.
- **P12.**
  - **Lợi so với v1:** trung bình theo dataset của (A1e − v1) **> 0** ở cả bốn nhóm (TFM × setting), với v1 lấy từ V1.
  - **Thụt lùi:** tổng số thụt lùi R(A1e), theo định nghĩa của PHASE1_PLAN, **≤ 3**.
  - **Dự đoán:** đúng.

**G. Cơ chế nhãn yếu mới, chỉ báo cáo, không dự đoán.**
- **Cấu hình:** TabICL v2, 3 điểm xuất phát, seed 0, 1, 2, trên tập phát triển.
  - LLP `--weak_variant feat` túi 8 trên 6 dataset LLP phát triển;
  - LLP iid túi 32 trên cùng 6 dataset;
  - PLL `--weak_variant inst` q = 0,6 trên 54, 36, 23.
- **Báo cáo:** v1, A1, L, và chênh với LLP iid túi 8 hoặc PLL iid q 0,6 của V1 trên cùng ô.

### 7.3 Thứ tự chạy và chi phí

- **Thứ tự:** mỗi GPU chạy theo thứ tự ưu tiên dưới đây.
  - **GPU 0:** A → B → chọn TabPFN-3.x → E (TabPFN-3.x, rồi TabPFN v2 trên F1) → C (TabPFN v2) → D (TabPFN v2) → F (TabPFN v2).
  - **GPU 1:** C (TabICL) → D (TabICL) → E (TabICL trên F1, rồi TabDPT) → G → F (TabICL).
- **Chi phí:** ước khoảng 20–24 giờ GPU, tức **2 phiên Kaggle**. Phiên 2 là chạy lại đúng lượt với output phiên 1 làm dữ liệu; mọi ô đã xong được bỏ qua.
- **Chấm:** `V2/analysis/v1c_report.py`, mở rộng và commit cùng code, trước khi có số.
