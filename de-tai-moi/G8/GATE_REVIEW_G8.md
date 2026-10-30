# Biên bản kiểm duyệt cổng G8

Cổng: **G8 — Phân tích và đối chiếu khẳng định** · Dự án: đề tài A, *In-Context Gibbs* · Ngày lập: 30/10/2026

- Người tạo sản phẩm: Claude (script phân tích, báo cáo), thầy (chạy G7 trên Kaggle)
- Người kiểm duyệt (không tham gia tạo sản phẩm): **thầy** ☐ Người ☐ Agent AI

Mục 3a là **tự rà soát của người tạo**, không thay kiểm duyệt độc lập. Người kiểm duyệt điền cột cuối của Mục 2, Mục 3b và Mục 4.

## 1. Sản phẩm được kiểm
| File | Ghi chú |
|---|---|
| [G8_REPORT](G8_REPORT.md) | báo cáo chính; thay bản sơ bộ [G8_PRELIM_REPORT](G8_PRELIM_REPORT.md) |
| `G8/analysis/` | `g8_confirmatory.py`, `g8_descriptive.py`, `g8_pll_collapse.py`, `g8_figures.py`: tạo lại mọi bảng và hình từ `G7/raw/` |
| `G8/results/`, `G8/figures/` | kết quả (json, csv) và 3 hình |
| [CLAIM_EVIDENCE_MATRIX](../G3/CLAIM_EVIDENCE_MATRIX.md) | đã điền cột trạng thái |
| [G7/EXPERIMENT_LOG](../G7/EXPERIMENT_LOG.md) | mọi lượt G7, kể cả lượt lỗi |
| Giao thức §11 | deviation 14–16 (LLP-Bench; cách đọc quy tắc 60% của H4; phân tích thăm dò với TabICL) |

## 2. Tiêu chí qua cổng (02_QUY_TRINH_CONG_KIEM_DUYET §G8)
| Tiêu chí | Người tạo tự đánh giá | Người kiểm duyệt |
|---|---|---|
| Bảng và hình tính bằng script từ kết quả thô | **Đạt**: 4 script, chỉ đọc `G7/raw/`; chạy lại cho cùng số | ☐ |
| Phân tích thống kê đúng kế hoạch G5 | **Đạt**: H2, H3, H4, H5, D1–D3 đúng giao thức Mục 6. Câu tiêu chí 60% của H4 có nhiều cách đọc: báo cáo mọi cách đọc, kết luận theo cách đọc yếu nhất (deviation 15) | ☐ |
| Ablation và phân tích độ nhạy | **Đạt một phần**: mọi ablation ở Mục 7 và S1 đã chạy; khởi động lạnh thiếu 21/114 ô (OpenML 503) — cần chạy lại L5 | ☐ |
| Phân tích lỗi | **Đạt**: PLL q = 0,6 với dataset ≥ 8 lớp (G8_REPORT Mục 5); LLP tic-tac-toe | ☐ |
| Mọi khẳng định được đánh dấu ủng hộ / một phần / không; khẳng định không được ủng hộ bị bỏ hoặc thu hẹp | **Đạt**: ma trận đã điền; #6 và #8 không được ủng hộ theo kiểm định xác nhận | ☐ |

## 3a. Phát hiện — tự rà soát
| # | Mức | Mô tả | Đề xuất | Trạng thái |
|---|---|---|---|---|
| 1 | **Lớn** | H2-LLP và H4-PLL đều không đạt với TFM chính (TabPFN v2). Thành công tối thiểu không đạt, nên điều kiện chuyển sang phương án B đã thoả (RESEARCH_PLAN §5) | Thầy chọn ở Mục 4: phương án A (phương án B đã đăng ký) hay B′ (nghiên cứu xác nhận thứ hai) | Chờ quyết |
| 2 | **Lớn** | Với TabICL v2 (ablation đã đăng ký), mọi kiểm định của H2 và H4 đều đạt. Đây là kết quả chọn sau khi xem số, nên chỉ được dùng như kết quả thăm dò | Không đổi TFM chính của nghiên cứu 1. Nếu muốn khẳng định thì cần nghiên cứu 2 đăng ký trước, trên dữ liệu mới (G8_REPORT Mục 8) | Đã ghi (deviation 16) |
| 3 | Lớn | Tập phát triển của PLL không có dataset ≥ 8 lớp, nên thí điểm không thấy kiểu lỗi sụp. Đây là điểm mù của thiết kế | Ghi ở phần giới hạn. Nếu làm nghiên cứu 2: tập phát triển phải phủ số lớp của tập đánh giá | Đề xuất |
| 4 | Nhỏ | Tiêu chí 60% của H4 không nói rõ (a) tính theo từng mức hay gộp, (b) baseline mạnh nhất là một baseline chung hay tốt nhất mỗi dataset | Lấy cách đọc yếu nhất, theo nguyên tắc S1 (deviation 15). Nghiên cứu sau phải ghi rõ | Đã ghi |
| 5 | Nhỏ | R̂: chuỗi chưa trộn hết ở mức nhãn (trung vị 5–29% mẫu có R̂ > 1,1), dù dự đoán của hai chuỗi thường gần nhau | Bài nói rõ: dự đoán ổn định, không khẳng định posterior đã hội tụ | Đề xuất |
| 6 | Nhỏ | Ablation khởi động lạnh thiếu 21 ô | Chạy lại L5 với output cũ; nếu lỗi lặp lại thì ghi "thiếu, có lý do" (G7_PLAN §2) | Chờ chạy |
| 7 | Nhỏ | H5 đạt theo đơn vị đã đăng ký ("nguồn"), không đạt nếu tính từng dataset (Lost: thứ 3) | Báo cáo cả hai cách | Đã ghi |

## 3b. Phát hiện — người kiểm duyệt
| # | Mức | Mô tả | Đề xuất | Trạng thái |
|---|---|---|---|---|
| | | | | |

## 4. Quyết định
**Hướng đi sau G8** (chọn một):

☐ **A — Phương án B đã đăng ký.** Bài phân tích/benchmark, nhắm NeurIPS D&B 2027 hoặc TMLR. Sang G9.

☐ **B′ — Nghiên cứu xác nhận thứ hai.** TFM chính là TabICL v2, trên tập dataset mới chưa từng dùng; giao thức G5′ đóng băng trước khi chạm dữ liệu. Không đạt thì quay về A.

☐ Khác: ____________

**Cổng G8:** ☐ **Qua** ☐ **Sửa** (quay lại G__) ☐ **Dừng**

Chữ ký người kiểm duyệt: ____________ · Ngày: ____________
