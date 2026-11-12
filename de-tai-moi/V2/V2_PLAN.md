# V2: kế hoạch phát triển đề tài A thành một bài mạnh hơn (lập 04/11/2026)

- **Trạng thái:** **thầy duyệt toàn bộ ngày 04/11/2026**, gồm V-Q1–V-Q5 theo đề xuất (§10). Giai đoạn 0 bắt đầu; dự đoán ghi trước ở [PHASE0_PREDICTIONS](PHASE0_PREDICTIONS.md). **05/11/2026: giai đoạn 0 xong** ([V0_REPORT](V0_REPORT.md)): P1 đúng, P2 sai, P3 đúng, P4 sai (79,2%); **cổng V0 qua một phần**. Giữ nhiều điểm xuất phát; thay cách chọn argmax WPLL bằng bộ gộp chuỗi, chọn trên tập phát triển. **Thầy duyệt 05/11/2026** (V0_REPORT §6); giai đoạn 1 ghi trước ở [PHASE1_PLAN](PHASE1_PLAN.md). **05/11/2026, trong lúc chờ V1:** bản nháp W1 (Mệnh đề 4) và R1a trên CPU, thăm dò, ở [W1_R1A_REPORT](W1_R1A_REPORT.md). **05/11/2026, giai đoạn 1 xong, cổng V1-A chốt** ([V1_REPORT](V1_REPORT.md)): bộ gộp mặc định của v2 là A1 (trung bình đều 3 chuỗi); P5 sai, P6 đúng, P7 đúng. Thầy quyết không chạy bù 4 dataset LLP của TabPFN bị lỗi tải (D-V1-1). **05/11 tối, W3 và W4 trên CPU** ([W4_TFM_REPORT](W4_TFM_REPORT.md)): đề xuất TabPFN-3 (hoặc 3.5) và TabDPT; TabDPT đã học trên 13 dataset Study 2 của ta, nên đề xuất loại các dataset huấn luyện của TFM khỏi Study 3; code cho túi gom theo feature và PLL phụ thuộc mẫu đã có, chưa chạy GPU. **08/11: V1C xong** ([V1C_REPORT](V1C_REPORT.md)); thầy duyệt các quyết định ở [V1D_PLAN](V1D_PLAN.md): v2 = A1 3 × 30; thêm v2-đa-TFM; bỏ túi k-means; H8 viết lại thành H8′ (tín hiệu sau chuỗi, không cần nhãn); Study 3 lấy thêm CC18 và loại dataset huấn luyện của TabDPT. Lượt V1D (đếm dataset, hiệu chỉnh q) chờ thầy chạy. **10/11: V1D xong** ([V1D_REPORT](V1D_REPORT.md): P16, P17, P18 đúng). **Thầy quyết dừng phát triển ở V1D và chạy Study 3 bản gọn** ([S3/STUDY3_PLAN](../S3/STUDY3_PLAN.md)); V1E huỷ; PLL phụ thuộc mẫu, túi 32, chẩn đoán đo trước, mở rộng Mệnh đề 5 và đếm dataset toàn OpenML thành future work. **12/11: Study 3 bản gọn xong** ([S3_REPORT](../S3/S3_REPORT.md)): H1, H2 đúng, C5 đạt; H3 sai, C4 không đạt. Thực nghiệm của V2 kết thúc.
- **Kế thừa:** [S2_PLAN §5.1–5.2](../S2/S2_PLAN.md) (lý thuyết kiểu sụp; làm phương pháp bền hơn), [REVIEW_RESPONSE_PLAN](../G9/REVIEW_RESPONSE_PLAN.md) R1–R8, [G9_PLAN](../G9/G9_PLAN.md) D2. Kế hoạch này gom các việc đó quanh một mục tiêu chung và thêm những gì rút ra từ phân tích lại số liệu ngày 04/11.
- **Số liệu:** mọi con số ở §2 sinh bằng [`V2/analysis/v2_diagnosis.py`](analysis/v2_diagnosis.py), kết quả ở [`V2/results/v2_diagnosis.txt`](results/v2_diagnosis.txt). Script chỉ đọc kết quả đã có, không chạy gì mới. Đây là **phân tích thăm dò trên dữ liệu đã dùng**, không phải kiểm định.

## 0. Tóm tắt

Bài hiện tại có một phương pháp đúng và một nghiên cứu xác nhận đạt. Nhưng nó còn ba điểm yếu lớn:
1. Chạy tốt hay sụp là tùy TFM.
2. Lý thuyết không phủ bộ lấy mẫu thực sự đang chạy.
3. Ở PLL chỉ thắng nhỏ, và thắng trên các dataset dễ.

Phân tích lại số liệu cho thấy điểm yếu 1 và 3 có chung một gốc: **điểm cuối của chuỗi do ánh xạ phản ứng của TFM và điểm xuất phát quyết định.** Bản v2 lấy chính gốc chung này làm trục của bài:
1. **Lý thuyết động học của chuỗi:** ánh xạ phản ứng, lưu vực hút, điều kiện thoát. Lý thuyết này dự đoán được khi nào phương pháp chạy.
2. **Phương pháp v2 tự chẩn đoán và tự sửa:** chạy từ nhiều điểm xuất phát, chọn chuỗi bằng độ khớp với nhãn yếu, không cần nhãn thật.
3. **Chặn độ lệch của cập nhật khối không phụ thuộc n** (R2).
4. **Study 3 đăng ký trước, trên dataset mới:** nhiều TFM, cơ chế nhãn yếu khó và gần thực tế hơn, dự đoán của lý thuyết được ghi trước.

Câu hỏi bài v2 trả lời: *Khi nào một mô hình nền tảng đông cứng học được từ nhãn yếu, và làm sao để nó luôn học được?*

## 1. Bài hiện tại đứng ở đâu

**Điểm mạnh**
- Ý tưởng mới: PPD của TFM là xác suất có điều kiện đầy đủ của một bộ lấy mẫu Gibbs (Bổ đề 1).
- Có lý thuyết cho các biến thể chính xác và cho trường hợp TFM xấp xỉ, kể cả không tương thích (Định lý 1, 2).
- Hai nghiên cứu đăng ký trước. Study 2 đạt cả LLP và PLL.
- Chi phí thấp: trung vị 41 giây mỗi lần chạy trên một T4.

**Điểm yếu**, theo §7 của bài và reviewer ngoài 01:
- Phụ thuộc TFM.
- Bộ lấy mẫu mặc định không có đảm bảo khác rỗng.
- Độ trộn ở mức từng nhãn chưa đủ.
- Phạm vi khẳng định hẹp: chỉ LLP túi 8 và PLL.
- Lợi thế PLL nhỏ.
- Thiếu dữ liệu thật, nhất là LLP thật.

## 2. Chẩn đoán từ số liệu (thăm dò, 04/11)

**F1. Mọi lỗi lớn đã thấy đều cùng một kiểu: chuỗi bắt đầu ở nơi gain gần bằng 0 và không thoát lên được vùng nhãn tốt.**
- **PLL q = 0,6, TabPFN v2, bốn dataset từ 8 lớp trở lên** (mục A, B):
  - Sau vòng quét đầu, tỉ lệ nhãn huấn luyện đúng chỉ là r = 0,16–0,22.
  - Ánh xạ phản ứng có điểm hút thấp ở r ≈ 0,17–0,21. Điểm bất động không ổn định nằm ở 0,35 (optdigits), 0,43 (mfeat-factors) và 0,56 (MiceProtein).
  - semeion không có lưu vực trên trong khoảng đã đo: gain vẫn là −0,103 ở r = 0,81.
  - Sau 30 vòng quét, r trung bình đạt 0,17 (mfeat-factors), 0,36 (semeion), 0,57 (MiceProtein) và 0,72 (optdigits). optdigits có điểm không ổn định thấp nhất và là dataset duy nhất có seed thoát (2/3).
- **Cùng bốn dataset đó, CC đã tune đạt accuracy test 0,91–0,96**, tức nằm xa phía trên các điểm không ổn định.
- **LLP túi 8 ở Study 1** (mục D, G):
  - tic-tac-toe với TabPFN v2: Gibbs đạt 0,706 AUC, trong khi oracle là 1,000, DLLP tuyến tính 0,812 và baseline tốt nhất 0,982. Cùng dataset, Gibbs với TabICL v2 đạt 0,993.
  - cylinder-bands: 0,658 với TabPFN, 0,715 với TabICL.
- **Hàm ý:** kiểu lỗi này sửa được.
  - Nếu TFM có lưu vực trên, sửa bằng điểm xuất phát tốt hơn.
  - Nếu không có lưu vực trên (semeion với TabPFN), phải đổi TFM.

**F2. Ánh xạ phản ứng đo bằng lỗi gieo đều chỉ đúng về định tính** (mục A, B; S2X_REPORT đã ghi trường hợp eucalyptus).
- Ở eucalyptus và steel-plates-fault (q = 0,6), ánh xạ gieo đều cho điểm hút ở 0,35 và 0,29 với TabPFN. Gain còn âm ở r ≈ 0,8 với cả hai TFM.
- Thế nhưng chuỗi thật với TabPFN đạt r = 0,69 và 0,77.
- **Hàm ý:** muốn dự đoán đúng về định lượng, phải đo ánh xạ trên lỗi có cấu trúc giống lỗi của chính chuỗi. Đây là việc lý thuyết chính (W1).

**F3. TFM tốt không thỏa điều kiện co Dobrushin** (mục A).
- Độ dốc lớn nhất của Φ, đo trên lưới 6 điểm:
  - TabICL: trung vị 1,10, tối đa 1,77.
  - TabPFN: trung vị 1,19, tối đa 1,68.
- TabICL không sụp ở ô nào, dù độ dốc lớn hơn 1.
- **Em sửa lại đề xuất trong chat ngày 04/11.** Hướng dùng điều kiện α < 1 cùng kết quả về Gibbs bất đồng bộ của De Sa et al. (ICML 2016) không áp được.
  - TFM tốt khuếch đại tín hiệu nhãn, nên độ dốc lớn hơn 1 là bình thường.
  - Điều kiện đúng là dấu của gain trên đoạn từ điểm xuất phát tới vùng nhãn tốt (S2X_REPORT §3).
  - Chặn độ lệch của cập nhật khối đi theo hướng total correlation của R2, không cần α < 1.

**F4. Dư địa cải thiện nằm ở đâu** (mục C, F).
- **PLL ở Study 2 đã gần trần.**
  - Khoảng cách tới oracle trung bình chỉ 0,008 (q = 0,3) và 0,019 (q = 0,6).
  - Chỉ 1/20 ô có khoảng cách từ 0,05 trở lên.
  - Trên các dataset này, phương pháp tốt đến đâu cũng không thể thắng lớn ở PLL.
- **LLP ở Study 2:** khoảng cách trung bình 0,039, nhưng dồn vào 4/18 dataset:
  - philippine 0,160;
  - eye_movements 0,137;
  - Is-this-a-good-customer 0,104;
  - profb 0,095.
  
  Gibbs thua baseline tốt nhất của từng dataset ở 3 trong 4 dataset này, với chênh lệch từ −0,005 đến −0,023.
- **PLL thật còn cách xa oracle:** BirdSong 0,076; Lost 0,140; MSRCv2 0,196.
- **Hàm ý:** muốn thắng lớn phải đổi sân chơi. Đó là PLL thật, PLL phụ thuộc mẫu, LLP khó (túi lớn, túi gom theo feature). Thêm dataset dễ không giúp.

**F5. Túi lớn chưa có bằng chứng** (mục D).
- Ở Study 1 (TabPFN), Gibbs thua baseline tốt nhất của từng dataset ở 5/10 dataset với túi 8, 5/10 với túi 16 và 6/10 với túi 32.
- Khoảng cách tới oracle tăng dần: 0,068, rồi 0,100, rồi 0,147.
- TabICL chưa được chạy với túi 16 và 32.
- Lưu ý: "baseline tốt nhất của từng dataset" là đường bao trên, chọn sau khi đã thấy kết quả. Nó khắt khe hơn tiêu chí đã đăng ký.

**F6. Sân chơi tự nhiên của phương pháp là dữ liệu nhỏ** (mục E). Đây chỉ là dấu hiệu, vì mỗi bối cảnh có 3 dataset.
- Ở n = 500, lợi thế so với đối thủ chính là +0,082 (LLP) và +0,032 (PLL).
- Ở cỡ đầy đủ, lợi thế còn +0,065 và +0,012.

**F7. Lấy mẫu con ngữ cảnh không có ảnh hưởng rõ** (mục C).
- LLP: khoảng cách tới oracle là 0,042 ở nhóm có lấy mẫu con (10 dataset), so với 0,034 ở nhóm không (8 dataset).
- PLL q = 0,6 lại ngược chiều: 0,014 so với 0,022.
- **Hàm ý:** hạ ưu tiên R7, không đầu tư mở rộng lên 100.000 dòng.

## 3. Định vị mới của bài

- **Câu hỏi:** khi nào một TFM đông cứng học được từ nhãn yếu, và làm sao để nó luôn học được?
- **Luận điểm:** In-Context Gibbs đúng khi TFM chính xác. Khi TFM xấp xỉ, kết quả do động học của chuỗi quyết định, và động học đó đo được trước khi chạy. Phương pháp v2 dùng chính phép đo ấy để tự chẩn đoán và tự sửa.
- **Tiêu đề làm việc:** *When can a frozen foundation model learn from weak labels? In-Context Gibbs and the dynamics of its chains.*

| Đóng góp | Đã có | Thêm ở v2 |
|---|---|---|
| C1. Phương pháp | In-Context Gibbs v1 | Nhiều điểm xuất phát; chọn chuỗi bằng độ khớp nhãn yếu, không cần nhãn thật; tùy chọn tham số quan sát chưa biết (W3) |
| C2. Lý thuyết đúng và bền | Định lý 1, 2; Mệnh đề 1–3 | Giữ nguyên |
| C3. Lý thuyết động học | Hình 2 (ánh xạ đo bằng lỗi gieo đều) | Mệnh đề về lưu vực và điều kiện thoát; ánh xạ đo trên lỗi có cấu trúc; phép chẩn đoán không cần nhãn (W1) |
| C4. Lý thuyết khối | Mệnh đề 3(b), đang rỗng | Chặn không phụ thuộc n theo total correlation (W2) |
| C5. Bằng chứng | Study 1 (không đạt), Study 2 (đạt) | Study 3 xác nhận cho v2, với dự đoán ghi trước cho từng TFM (W5) |
| C6. Phạm vi | LLP túi 8, PLL độc lập | Túi lớn, túi gom theo feature, PLL phụ thuộc mẫu, dữ liệu nhỏ, PLL thật |

**Vì sao bài này nổi bật:**
- Hiếm có bài phương pháp nào **dự đoán được chỗ mình thất bại, sửa được chỗ đó, rồi kiểm lại bằng nghiên cứu đăng ký trước.**
- Ba nghiên cứu đăng ký trước, kể cả nghiên cứu không đạt, là mức chặt chẽ khó tìm thấy ở NeurIPS.
- Phương pháp không huấn luyện TFM và dùng được với nhiều TFM. Mỗi khi TFM tốt lên, phương pháp tốt lên theo, và lý thuyết cho biết TFM mới có hợp hay không.

## 4. Các mảng việc

### W1. Lý thuyết động học và phép chẩn đoán (trụ chính)

**Mục tiêu:** chứng minh mệnh đề sau. Nếu gain dương trên [r₀, r*) thì chuỗi đi tới vùng nhãn tốt. Nếu có một điểm hút nằm dưới r₀ thì chuỗi kẹt ở đó. Bản định lượng phải đúng cả với lỗi có cấu trúc.

**Việc cần làm:**
1. **Ánh xạ trên lỗi có cấu trúc (Φ_struct).**
   - Gieo nhãn sai theo xác suất dự đoán của TFM ngây thơ bên trong tập ứng viên, thay vì gieo đều.
   - Kiểm xem Φ_struct có dự đoán đúng r cuối của chuỗi thật hay không, trên 6 dataset của F1 và F2.
2. **Chứng minh.**
   - Khi biết ngữ cảnh, mỗi vòng quét cập nhật n nhãn gần như độc lập, nên r_{t+1} = Φ(r_t) + O_p(n^{-1/2}).
   - Dùng lập luận xấp xỉ ngẫu nhiên, kiểu phương trình vi phân của Wormald, dưới giả định lỗi trao đổi được trong từng lớp.
   - Ghi rõ giả định trong bài.
3. **Chẩn đoán không cần nhãn.**
   - Đo độ khớp nhãn yếu trên phần giữ lại:
     - PLL: log của tổng xác suất dự đoán trên tập ứng viên;
     - LLP: log P(số đếm | xác suất dự đoán), tính bằng quy hoạch động Poisson–binomial.
   - Kết hợp với mức bất đồng giữa các chuỗi xuất phát từ những điểm khác nhau.

**Sản phẩm:** Mệnh đề 4 kèm chứng minh ở phụ lục; hình so sánh dự đoán với thực tế; phép chẩn đoán.

**Xong khi:**
- Tự rà toán bằng skill dm-paper-math-review không còn lỗi Critical hay Major.
- Trên tập phát triển, sai số tuyệt đối trung vị giữa r cuối dự đoán và r cuối thật ≤ 0,1. Đây là ngưỡng em đề xuất, thầy duyệt.

**Chi phí:** chủ yếu là CPU và thời gian suy nghĩ; Φ_struct cần khoảng 1 phiên GPU.

**Rủi ro:** giả định lỗi trao đổi được có thể quá mạnh. Khi đó giữ bản định tính (có hay không lưu vực trên), kèm bằng chứng số.

### W2. Lý thuyết cập nhật khối (R2) và R1a

- **Hướng chứng minh:** thay lập luận lai trên TV bằng total correlation.
  - Với prior tham số d chiều, TC ≈ d/(4(K−1)²), không phụ thuộc n.
  - Làm PLL trước. Với LLP, đi qua I(θ; c_B).
  - Chi tiết ở REVIEW_RESPONSE_PLAN §3.R2.
- **R1a trên CPU:** mở rộng code kiểm chứng của phụ lục E với nhân khối không sửa và bag-block không sửa. Đo TV theo n và K, để kiểm thang 1/K².
- **Hạn:** 20/12, trùng hạn D2. Nếu không xong, giữ đoạn "Scope" trung thực như hiện tại.

### W3. Phương pháp v2

1. **Nhiều điểm xuất phát.** Mỗi điểm xuất phát chạy một chuỗi:
   - khởi động ấm như v1;
   - TFM ngây thơ;
   - một learner nhãn yếu đã tune (CC hoặc GLWS cho PLL, DLLP-MLP cho LLP).
   
   Chi phí gấp 3 lần v1. Với TabICL là khoảng 2 phút mỗi lần chạy.
2. **Chọn hoặc gộp chuỗi** bằng độ khớp nhãn yếu trên 20% dữ liệu giữ lại. Đây đúng là giao thức validation mà các baseline đã được dùng để tune.
3. **Tham số quan sát chưa biết** (tùy chọn, có thể cắt):
   - Lấy mẫu θ cùng với y trong mỗi vòng quét: Dirichlet cho các hàng của ma trận nhiễu, Beta cho tham số c của PU.
   - Việc này bỏ được giả định "biết ℓ". Giả định (A0) vốn đã cho phép có tham số phiền như vậy.
4. **Tôi luyện:** chỉ giữ làm phương án dự phòng. F1 cho thấy khi TFM không có lưu vực trên thì không cách lấy mẫu nào cứu được.

**Lưu ý về định vị:** nếu dùng một learner đã huấn luyện làm điểm xuất phát, TFM vẫn đông cứng nhưng cả hệ thống không còn "hoàn toàn không huấn luyện". Phương án thay thế là bản chỉ dùng TFM: khởi động ngây thơ, hard-EM, và khởi động ngẫu nhiên thỏa ràng buộc. Thầy quyết ở §10.

**Nơi phát triển:** chỉ trên các dataset đã dùng (tập phát triển, Study 1, Study 2). Tất cả các dataset này trở thành tập phát triển của Study 3.

### W4. Nhiều TFM

- Đã có TabICL v2 và TabPFN v2. Thêm 2–3 TFM trong số: TabPFN-2.5/v3, TabDPT, TabFM, Mitra.
- Với từng TFM cần kiểm: giấy phép, chạy được trên T4, số lớp tối đa.
- Đo Φ và Φ_struct trên tập phát triển cho từng TFM, rồi ghi trước dự đoán TFM nào hợp với dataset nào.

### W5. Study 3

Xem §5.

### W6. Đóng các điểm P0 của bài hiện tại

Giữ từ REVIEW_RESPONSE_PLAN:
- **R4:** một phiên GPU, làm ở giai đoạn 0.
- **R1b:** đo độ lệch của khối ở trạng thái gần dừng. Dùng lại hạ tầng nhiều chuỗi của W3.
- **R3:** chẩn đoán độ trộn đầy đủ. Chạy nhiều điểm xuất phát tự nhiên cho ra nhiều chuỗi để dùng.
- **R8:** đưa vào phụ lục, tính từ vết đã lưu.

Các mục còn lại được xếp lại:
- R5 gộp vào W4 và Study 3.
- R6(b) gộp vào Study 3.
- R7 hạ ưu tiên (F7).

## 5. Study 3 (bản nháp thiết kế; đóng băng ở cổng V2)

- **Mục đích:** kiểm xác nhận cho phương pháp v2 và cho lý thuyết, trên dataset chưa từng dùng.
- **Dataset:**
  - Lấy từ các bộ benchmark lớn chưa dùng (ví dụ TALENT, TabArena; cần kiểm).
  - Loại mọi dataset của Study 1, Study 2, tập phát triển, và các bản trùng.
  - Dùng cùng tiêu chí n, d, số lớp như S2.
  - Mục tiêu: ít nhất 20 dataset nhị phân và 12 dataset đa lớp, trong đó ít nhất 4 dataset có từ 8 lớp trở lên.
- **Dữ liệu thật:**
  - 3 bộ PLL thật hiện có, cộng bộ mới nếu tìm được. Có thể nới tiêu chí số lớp nếu TFM hỗ trợ.
  - PU thật (disdat) chỉ đưa vào nếu W3.3 xong.
- **Bối cảnh:**
  - LLP: túi ngẫu nhiên cỡ 8 và 32; túi gom theo feature (k-means trên feature, cỡ 8).
  - PLL: distractor độc lập với q = 0,6; PLL phụ thuộc mẫu, với distractor lấy từ các lớp dễ nhầm theo một mô hình phụ, giữ số ứng viên trung bình bằng nhau.
  - Dữ liệu nhỏ: mỗi dataset thêm mức n_train = 500, lấy mẫu con lồng nhau.
  - Nếu W3.3 xong: nhiễu phụ thuộc lớp với ma trận nhiễu chưa biết.
- **TFM:** 3–4 TFM (W4). Mỗi TFM chạy cả v1 (một điểm xuất phát) và v2.
- **Baseline:** như S2, cùng không gian tune.
- **Giả thuyết (nháp):**
  - **H6 (v2 so với baseline):** trong từng bối cảnh chính (LLP túi 32, LLP gom theo feature, PLL phụ thuộc mẫu), v2 vượt đối thủ chính và thắng baseline có điểm trung bình cao nhất ở ít nhất 60–70% dataset.
  - **H7 (bền với TFM):** với mọi TFM, tỉ lệ ô mà v2 kém learner khởi động tốt nhất hơn 0,02 không quá 5%; và v2 không kém v1 một cách có ý nghĩa ở bất kỳ TFM nào.
  - **H8 (dự đoán):**
    - Phép chẩn đoán đo trước khi chạy chuỗi, không cần nhãn thật, phân biệt ô v1 sụp với ô không sụp với AUC ≥ 0,8.
    - Dự đoán ghi trước về việc TFM nào hợp với dataset nào đúng ở ít nhất 80% ô.
  - **H9 (Gibbs so với hard-EM):** giữ như H3.
- **Kiểm định:** như S2. Wilcoxon kèm Holm theo dataset; Bayesian signed-rank cho H9; có phân tích độ nhạy.
- **Chi phí** (ước tính thô, sẽ đo lại bằng một lượt thử):
  - TabICL: khoảng 2 phút mỗi lần chạy v2, tức khoảng 15–20 giờ GPU cho một TFM.
  - TabPFN chậm hơn khoảng 4–5 lần.
  - Tổng khoảng 60–120 giờ GPU, tức khoảng 3–5 tuần quota Kaggle, cộng thêm CPU cho baseline.

## 6. Hai tuần tới: giai đoạn 0 (kiểm nhanh, thăm dò)

**Mục tiêu:** trước khi đầu tư lớn, kiểm xem kiểu lỗi ở F1 có thật sự sửa được không.

**Việc cần làm:**
1. Ghi trước dự đoán P1–P4 và commit trước khi chạy.
2. Viết code:
   - cờ `--init` với các lựa chọn: TFM ngây thơ, khởi động ấm, baseline đã tune, ngẫu nhiên thỏa ràng buộc;
   - chạy nhiều điểm xuất phát;
   - độ khớp nhãn yếu trên phần giữ lại;
   - mở rộng lưới ánh xạ phản ứng tới r ∈ {0,85; 0,90; 0,95};
   - ánh xạ phản ứng cho LLP, gieo lỗi bằng cách đổi nhãn trong túi để giữ số đếm;
   - Φ_struct.
3. Chạy R4 trong cùng đợt, vì công cụ đã có sẵn.
4. Thầy chạy 1–2 phiên Kaggle (2 × T4).

**Dự đoán ghi trước.** Đây là thăm dò, vì dữ liệu Study 1 đã được xem.
- **P1.** TabPFN v2, PLL q = 0,6, khởi động từ nhãn của CC đã tune. Trên MiceProtein, mfeat-factors và optdigits, chuỗi không sụp:
  - tỉ lệ nhãn huấn luyện đúng sau 30 vòng quét ≥ 0,80 ở cả 9/9 lần chạy;
  - accuracy test trung bình ≥ accuracy của CC đã tune trừ 0,02, trên từng dataset.
- **P2.** semeion: đo Φ ở r ∈ {0,85; 0,90; 0,95} trước.
  - Nếu gain âm ở cả ba mức, dự đoán chuỗi vẫn sụp (r cuối < 0,5) dù khởi động từ CC.
  - Nếu gain dương ở một mức không vượt accuracy khởi động, dự đoán chuỗi không sụp.
  - Dự đoán được ghi sau khi đo Φ, trước khi chạy chuỗi.
- **P3.** TabPFN v2, LLP túi 8, tic-tac-toe và cylinder-bands, khởi động từ DLLP-MLP đã tune. Đo ánh xạ phản ứng LLP trước. Nếu gain dương quanh điểm khởi động, dự đoán AUC của Gibbs ≥ AUC của điểm khởi động trừ 0,01.
- **P4.** Độ khớp nhãn yếu trên phần giữ lại chọn đúng điểm xuất phát có chỉ số test cao nhất ở ít nhất 80% số ô. Chỉ tính các ô mà kết quả giữa các điểm xuất phát chênh nhau hơn 0,01.

**Cổng V0** (khoảng 17/11):
- P1 và P4 đạt: tiếp tục W3 và lấy nó làm mặc định của v2.
- P1 không đạt: xem lại cơ chế trước khi làm Study 3.

**Chi phí:** khoảng 8–12 giờ GPU (chuỗi TabPFN trung vị khoảng 3–4 phút mỗi ô), cộng R4 khoảng 2,5–4,5 giờ.

## 7. Lịch và các cổng

| Giai đoạn | Thời gian | Việc chính | Cổng |
|---|---|---|---|
| 0 | 05–17/11/2026 | P1–P4, R4, code nhiều điểm xuất phát | V0 |
| 1 | 18/11–20/12 | W1 (chứng minh, Φ_struct); W2 và R1a; W3 (tham số chưa biết, túi gom theo feature, PLL phụ thuộc mẫu); W4 (bọc TFM mới, kiểm giấy phép); R1b với TabICL | V1 (20/12, trùng hạn D2): có mệnh đề kèm chứng minh nháp; phương pháp v2 được chốt trên tập phát triển |
| 2 | 21/12/2026–15/01/2027 | Đếm và chọn dataset cho Study 3; viết giao thức; ghi trước dự đoán; đóng băng | V2 |
| 3 | 16/01–15/03/2027 | Chạy Study 3 (GPU và CPU); dữ liệu thật | V3: kết quả tính bằng script viết trước |
| 4 | 16/03–30/04/2027 | Viết lại bài theo mạch mới; G10 (đọc độc lập, tái lập một bảng từ clone sạch); cập nhật tìm kiếm tính mới | V4 |
| Nộp | Khoảng 14/05 (abstract), 21/05 (bài) | NeurIPS 2027 | |

## 8. Thứ tự ưu tiên và danh sách cắt

- **Bắt buộc**, để bài v2 đứng được:
  - W1, ít nhất là bản định tính kèm phép chẩn đoán;
  - W3.1–W3.2;
  - W4 với ít nhất 3 TFM;
  - phần lõi của Study 3: LLP túi 32, LLP gom theo feature, PLL phụ thuộc mẫu;
  - R4 và R1b.
- **Nên có:** W2 (chặn khối), mở rộng PLL thật, mức dữ liệu nhỏ.
- **Cắt trước nếu trễ**, theo thứ tự:
  1. W3.3 (tham số quan sát chưa biết), chuyển sang bản JMLR;
  2. PU thật;
  3. TFM thứ 4 và thứ 5;
  4. R8.
- **Không làm:** mở rộng lên 100.000 dòng (F6, F7); tôi luyện (F1).

## 9. Rủi ro

- **P1 không đạt:** cơ chế ở F1 sai một phần. Quay lại W1 trước khi làm tiếp.
- **Không đủ dataset mới đạt tiêu chí:** dùng thang nới tiêu chí như S2.
- **TFM mới không chạy được trên T4, hoặc giấy phép không cho phép:** giữ 3 TFM.
- **v2 không hơn v1 đáng kể trên TFM tốt** (TabICL): khi đó lợi ích chính của v2 là bền với TFM yếu, và bài viết đúng như vậy qua H7.
- **Có người công bố trước:** cập nhật tìm kiếm G2 mỗi tháng; theo dõi riêng các bài về TFM với nhãn ngữ cảnh nhiễu.
- **Phạm vi phình to:** dùng danh sách cắt ở §8 và xem lại ở mỗi cổng.
- **9 trang không chứa nổi ba nghiên cứu:** bài chính chỉ tóm tắt Study 1–2, chi tiết chuyển sang phụ lục.

## 10. Quyết định (thầy duyệt 04/11/2026: chấp nhận kế hoạch, mọi câu theo đề xuất)

| # | Câu hỏi | Quyết định |
|---|---|---|
| V-Q1 | Có duyệt hướng v2 không: phương pháp mới cùng Study 3 đăng ký trước | Duyệt |
| V-Q2 | Có cho khởi động từ một learner nhãn yếu đã huấn luyện (hệ lai) không, hay chỉ khởi động bằng TFM | Cho phép, và báo cáo cả bản chỉ dùng TFM |
| V-Q3 | Thêm TFM nào (sau khi em kiểm giấy phép và khả năng chạy trên T4) | 2 TFM: một từ họ TabPFN mới, một TFM học trên dữ liệu thật |
| V-Q4 | Có duyệt P1–P4 và ngưỡng của cổng V0 không | Duyệt |
| V-Q5 | Có thay Q1–Q5 cũ của REVIEW_RESPONSE_PLAN bằng W6 không | Thay |

## 11. Đối chiếu với các kế hoạch cũ

| Kế hoạch cũ | Nằm ở đâu trong V2 |
|---|---|
| S2_PLAN §5.1 (a)–(d) | W1 |
| S2_PLAN §5.2 | W3 |
| R1 | W2 (R1a), W6 (R1b) |
| R2 | W2 |
| R3 | W6 |
| R4 | Giai đoạn 0 |
| R5 | W4, Study 3 |
| R6(b) | Study 3 |
| R6(a), R6(c) | Phần dữ liệu thật của Study 3 |
| R7 | Hạ ưu tiên |
| R8 | W6, phụ lục |
| D2 | W1, giữ hạn 20/12 |
