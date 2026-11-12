# Bài v3: chương trình nghiên cứu (12/11/2026)

- **ĐÃ CHUYỂN (12/11/2026):** thầy quyết định tách v3 thành bài riêng ở kho `DRuanli/neur-v3`. Bản đang dùng là `v3/V3_PLAN.md` ở kho đó (bản 2, luận điểm đã đổi sang "tỉ giá quy đổi"). Tài liệu này là bản 1, giữ làm hồ sơ, không cập nhật nữa. Mục 7 (tách hay gộp) đã được quyết: tách.
- **Trạng thái lúc viết:** bản đề xuất của em, chờ thầy duyệt. Chưa có thí nghiệm nào của v3 được ghi trước hay chạy.
- **Mục tiêu thầy đặt:** bài v3 ở mức đột phá, 8–10 điểm NeurIPS 2027 (hạn nộp ước 05/2027).
- **Điều em không hứa được:** mức 8–10 không lập kế hoạch ra được. Tài liệu này chọn hướng có trần cao nhất mà em thấy, và đặt các cổng dừng sớm để không tốn sức vào hướng chết. Ước lượng chủ quan của em: khoảng 40–50% đạt mức 6 trở lên, khoảng 10–15% đạt mức 8 trở lên.

## 1. Hai phép tính định hướng trên dữ liệu đã có (thăm dò, `V3/v3_headroom.py`)

| Phép tính | Kết quả | Hệ quả |
|---|---|---|
| **H-A.** Trần của việc *chọn* chuỗi ở Study 3: bộ chọn hoàn hảo (biết nhãn test) so với bản gộp X | chọn đúng chuỗi tốt nhất chỉ hơn X **+0,011**; chọn theo điểm khớp nhãn yếu thì −0,004 đến +0,001 | **Bỏ** hướng chẩn đoán và chọn chuỗi: dù làm hoàn hảo cũng gần như không còn gì để lấy |
| **H-B.** Study 2 (TabICL v2, một chuỗi): phần khoảng cách "TFM ngây thơ → oracle có nhãn thật" đã lấy lại | LLP túi 8: **75%** (baseline tune tốt nhất: 48%); PLL q 0,3: **73%** (31%); PLL q 0,6: **91%** (86%) | Phương pháp đã ở gần trần. Lợi thêm trong các bối cảnh này tối đa 0,01–0,04. Muốn kết quả lớn phải đổi **câu hỏi**, không phải vắt thêm điểm |

"Oracle" ở đây là chính TFM đó khi được cho nhãn thật.

## 2. Luận điểm của v3

> Với một TFM đóng băng, nhãn yếu gần như "miễn phí" trong một vùng rộng; ngoài vùng đó có một vùng "khó" (chỉ thành công khi xuất phát có thông tin) và một vùng "bất khả" (không đủ thông tin). Bài vẽ ba vùng này, giải thích chúng, và cho quy tắc biết mình đang ở vùng nào.

Bài v2 hỏi "phương pháp có hơn baseline không". Bài v3 hỏi "**nhãn yếu đắt bao nhiêu, và giới hạn nằm ở đâu, vì thuật toán hay vì thông tin**".

Vì sao em chọn luận điểm này:
1. **Nó xuất phát từ số liệu đã có**, không từ một phép so sánh đẹp: lấy lại 73–91% khoảng cách tới oracle là con số mạnh hơn "+0,04 so với baseline".
2. **Dấu hiệu "vùng khó" đã thấy trong dữ liệu:** chuỗi xuất phát từ learner thành công ở nơi chuỗi xuất phát không thông tin thất bại. Trong vật lý thống kê của suy luận, đó là dấu hiệu chuẩn của pha khó. Ba điểm xuất phát của v2 trở thành dụng cụ đo.
3. **Các thất bại cũ thành dữ liệu:** túi 32, PLL nhập nhằng cao nhiều lớp, PLL phụ thuộc mẫu là các điểm trên sơ đồ pha, không còn là "hạn chế".
4. **Định lý 3 có việc để làm:** một điểm bất động là vùng dễ, hai điểm bất động là vùng khó.

## 3. Năm tuyên bố cần có để xứng mức 8

| Mã | Tuyên bố | Đang có | Còn thiếu |
|---|---|---|---|
| C1 | Trong vùng dễ, bộ lấy mẫu lấy lại ≥ 70% khoảng cách naive → oracle, không huấn luyện TFM, trên nhiều loại nhãn yếu và nhiều dataset | 3 bối cảnh, 1 TFM, 28 dataset | Study 3 (chờ oracle và naive của S3B); thêm loại nhãn yếu |
| C2 | Có ba vùng dễ / khó / bất khả theo độ yếu của nhãn (q, cỡ túi, số lớp, n), đo bằng ba điểm xuất phát cùng hai mốc naive và oracle | vài điểm rời rạc | quét có hệ thống |
| C3 | Lý thuyết đặt được hai đường biên: biên thuật toán (ánh xạ phản ứng) và biên thông tin (cận Bayes hoặc Fisher) | Định lý 3; Mệnh đề về thông tin của nhãn ngây thơ | cận thông tin; dự báo biên ghi trước |
| C4 | Một quy tắc không cần nhãn: từ (q, số lớp, cỡ túi, n) biết đang ở vùng nào và nên chạy cấu hình nào | chưa có | suy ra từ C2 và C3, kiểm trên dataset mới |
| C5 | Cùng một bộ lấy mẫu chạy cho nhãn bù (complementary labels), học đa thể hiện (MIL) và số đếm có nhiễu | chưa có | cài đặt (rẻ: tập ứng viên và quy hoạch động theo túi đã có) và thí điểm |

## 4. Các cổng (mỗi cổng có điều kiện dừng)

| Cổng | Việc | Chi phí | Dừng hoặc đổi hướng nếu |
|---|---|---|---|
| **G0** | H-A và H-B ở mục 1 | xong | — |
| **G1** | Dùng kết quả S3B: X lấy lại bao nhiêu phần khoảng cách naive → oracle trên 20 dataset của Study 3 | 0 (chờ S3B) | dưới 50% ở cả LLP và PLL: luận điểm "gần oracle" không đứng, phải thu hẹp |
| **G2** | Thí điểm sơ đồ pha trên dataset phát triển: PLL q ∈ {0,3; 0,5; 0,7; 0,8; 0,9}, LLP túi ∈ {8; 16; 32; 64}; ba điểm xuất phát, naive, oracle; TabICL v2 và TabPFN-3; 6 dataset | 1–2 phiên Kaggle | không thấy vùng giữa nơi kết quả phụ thuộc điểm xuất phát vượt nhiễu Monte Carlo, hoặc chuyển tiếp không đơn điệu theo độ yếu |
| **G3** | Thí điểm tính tổng quát: nhãn bù và MIL trên 6 dataset phát triển, so với một baseline chuẩn của mỗi loại | 1 phiên | không loại mới nào lấy lại ≥ 50% khoảng cách |
| **G4** | Lý thuyết: cận thông tin cho biên bất khả; dự báo biên khó từ ánh xạ phản ứng; viết dự báo **trước** khi chạy xác nhận | CPU | dự báo sai lệch có hệ thống trên dữ liệu phát triển |
| **Study 4** | Nghiên cứu xác nhận, ghi trước, trên dataset mới | 4–6 phiên | — |

Mọi thí nghiệm của G2 trở đi đều ghi trước theo quy trình đang dùng (kế hoạch và script chấm commit trước khi chạy).

## 5. Việc đã cân nhắc và loại

| Hướng | Lý do loại |
|---|---|
| Chẩn đoán không cần nhãn, chọn chuỗi | H-A: trần chỉ +0,011 |
| Khung "giải mã / EXIT chart" làm đóng góp chính | kiểm tài liệu ngày 12/11: phép so sánh LLP với giải mã đã có; phần còn mới không đủ làm lõi |
| Trộn TFM trong bộ lấy mẫu | phép thử trên ánh xạ đã lưu: cần 35–85% trọng số của TFM tốt |
| Huấn luyện trước một TFM cho nhãn yếu | vượt tài nguyên (GPU miễn phí) |
| Lấy mẫu tuần tự (SMC) thay Gibbs | vẫn gặp đúng vấn đề khởi đầu không thông tin; trần lợi ích không rõ |

## 6. Rủi ro chính

1. **Tính mới của khung "ba vùng":** khái niệm dễ / khó / bất khả đã có trong vật lý thống kê của suy luận cho mô hình tổng hợp. Phần mới sẽ là đo nó trên dữ liệu thật với mô hình nền tảng và nhãn yếu. Em chưa tra tài liệu riêng cho điểm này; phải làm trước G2.
2. **Sơ đồ pha có thể nhoè** ở cỡ dữ liệu vài nghìn điểm: kết quả từng seed dao động mạnh. Cần nhiều seed hơn (5–10) ở vùng chuyển tiếp.
3. **"Oracle" là TFM có nhãn thật**, không phải giới hạn tuyệt đối. Reviewer có thể đòi thêm mốc: mô hình có giám sát tốt nhất (ví dụ cây tăng cường đã tune).
4. **Trùng với v2:** nếu v2 được đăng trước hạn NeurIPS thì v3 phải đứng được bằng phần mới. Xem mục 7.

## 7. Quyết định cần thầy chốt

**Số phận của v2.** Hai phương án:
- **(a) Tách:** v2 gửi TMLR sau khi có S3B; v3 nộp NeurIPS như một bài khác, trích v2. Được hai bài, nhưng v3 không được dựa vào Study 1–3 làm đóng góp.
- **(b) Gộp:** không gửi v2 đi đâu; v3 nuốt toàn bộ v2 và chỉ nộp một bài NeurIPS. Bài mạnh hơn, nhưng mất phương án dự phòng nếu G2 thất bại muộn.

Em nghiêng về **(a) nhưng hoãn quyết định tới sau G2** (khoảng 2 tuần): nếu G2 đạt thì cân lại giữa (a) và (b); nếu G2 trượt thì v2 đi TMLR ngay.

## 8. Mốc thời gian (nếu các cổng đều qua)

| Thời gian | Việc |
|---|---|
| 11/2026 | S3B xong; G1; tra tài liệu cho khung ba vùng; ghi trước và chạy G2 |
| 12/2026 | G3; G4 (lý thuyết); quyết định (a) hay (b) |
| 01–02/2027 | Ghi trước và chạy Study 4 |
| 03/2027 | Phân tích; bản nháp; các vòng phản biện |
| 04/2027 | Sửa; style chính thức; giấy phép; checklist |
| 05/2027 | Nộp |
