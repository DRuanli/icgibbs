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
