# G7 — kế hoạch chạy tập đánh giá (bản 29/10/2026)

> **Trạng thái 30/10/2026:** mọi lượt đã chạy.
> - Phần chính đủ 2.190/2.190 ô; baseline 3.492/3.492; R̂, RQ5, mọi ablation đủ.
> - Còn thiếu 21/114 ô của ablation khởi động lạnh (OpenML 503 ở L5) → chạy lại L5 với output lần 1.
> - LLP-Bench bỏ theo quy tắc §3 (deviation 14).
> - Kết quả phân tích: [G8_REPORT](../G8/G8_REPORT.md).

**Điều kiện bắt đầu:** thầy ký "Qua" ở [GATE_REVIEW_G6](../G6/GATE_REVIEW_G6.md). Từ lượt G7 đầu tiên trở đi, mọi số trên tập đánh giá là **kết quả xác nhận**: không đổi giao thức nữa, trừ deviation có lý do ghi ở giao thức §11, và kết quả phần đổi được gắn nhãn *exploratory*.

**Tiêu chí qua cổng G7** (02 §G7): mọi ô (dataset × bối cảnh × mức × seed × phương pháp) trong giao thức đều có kết quả, hoặc có lý do ghi rõ (hết giờ, hết bộ nhớ, lỗi) và lý do đó được báo cáo trong bài.

## 1. Các lượt
Một script cho mọi lượt: `G4/code/run_g7_lot.sh <LOT>`. Mọi lượt đều **chạy bộ test trước** từ bản code vừa tải lên, trong môi trường Kaggle mới. Đây cũng là bước "chạy từ clone sạch" còn thiếu của G6 (Phát hiện 3). Test hỏng thì lượt dừng, không chạy ô nào.

Các lượt **chạy tiếp được**: hết 12 giờ thì chạy lại đúng lượt đó, các ô đã xong được bỏ qua.

| Lượt | Phần cứng | Nội dung | Ước tính |
|---|---|---|---|
| L1, L2, L3 | GPU T4 × 2 | Phần chính: Gibbs ấm (SEM lấy từ cùng chuỗi), hard-EM, TFM ngây thơ, oracle, Gibbs biết propensity (PU-SAR), CL + TFM (nhãn nhiễu). 330 ô nhị phân và 84 ô PLL, chia 6 mảnh, mỗi lượt 2 mảnh song song | khoảng 7 giờ mỗi lượt |
| L4 | GPU T4 × 2 | GPU 0: R̂ → ablation K → số vòng quét → sửa MH. GPU 1: RQ5 PLL thật (TabICL, 10 lần chia) → chuyển nạp → cỡ dữ liệu | khoảng 8–10 giờ |
| L5 | GPU T4 × 2 | Ablation khởi tạo lạnh, prior η (PU-SAR), mỗi cái 2 mảnh | khoảng 8 giờ |
| L6 | GPU T4 × 2 | Ablation TFM: TabICL v2 cho LLP túi 8 và PLL | khoảng 2–3 giờ |
| C1, C2, C3 | CPU (4 nhân) | Mọi baseline, bản mặc định và bản tune, chia 12 mảnh, mỗi lượt 4 mảnh. C3 thêm baseline của PLL thật và đối thủ chính của ablation cỡ dữ liệu | khoảng 9 giờ mỗi lượt |

- Thứ tự đề xuất: **L1 trước**; lượt này còn là lần thử cuối của cách chạy. Sau đó L2, L3, rồi L4–L6.
- C1–C3 không tốn quota GPU, chạy song song với các lượt GPU được nếu Kaggle cho nhiều phiên cùng lúc.
- Sau mỗi lượt, thầy gửi về toàn bộ `g7_*` trong `/kaggle/working`, **kèm thư mục code đã chạy**.
- Sau mỗi lượt, em kiểm: số ô so với số dự kiến; dòng lỗi; mã băm dữ liệu và lần chia so với `docs/dataset_hashes.csv`; mã băm code. Kết quả kiểm ghi vào [EXPERIMENT_LOG](EXPERIMENT_LOG.md). Em **không xem metric tổng hợp** của khẳng định chính cho tới khi đủ dữ liệu cho G8. Chỉ kiểm tính đầy đủ và tính hợp lệ (ví dụ AUC = 0,5 hằng, NaN), để tránh điều chỉnh theo kết quả giữa chừng.

## 2. Chạy lại và lỗi
- Ô lỗi: ghi lý do; chạy lại một lần trong lượt kế tiếp bằng chính lệnh của lượt đó. Lỗi lặp lại thì ghi là "thiếu, có lý do".
- Không sửa code giữa các lượt, trừ khi một lỗi làm hỏng ô. Khi đó: sửa, ghi deviation, chạy lại **mọi** ô của phương pháp bị ảnh hưởng (không chỉ ô lỗi), và gắn nhãn trong bài.

## 3. LLP-Bench (RQ5, kiểm tra áp lực)
Chưa có quyền truy cập dữ liệu Criteo ([G7_RUNNER_DESIGN §4b](../G6/G7_RUNNER_DESIGN.md)). **Quy tắc ghi trước:**
- Nếu tới khi xong L6 vẫn chưa có dữ liệu thì bỏ LLP-Bench, ghi deviation; RQ5 và H5 chỉ còn PLL thật (H5 vốn đánh giá trên PLL thật).
- Nếu có dữ liệu thì cài theo đề xuất §4b và chạy như một lượt riêng.

## 4. Sau G7
G8 phân tích theo giao thức Mục 6:
- H2-LLP túi 8, H4-PLL: Wilcoxon, Holm.
- H3: Bayesian signed-rank với ROPE.
- H5: thứ hạng.
- Phân tích thăm dò D1–D3; ablation; phân tích lỗi.

Sau đó điền ma trận khẳng định–bằng chứng và đối chiếu với tiêu chí thành công (RESEARCH_PLAN §5): tối thiểu cần H2-LLP *hoặc* H4-PLL, cùng H3.
