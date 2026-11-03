# Biên bản kiểm duyệt: nghiên cứu xác nhận S2

Cổng: **S2: kết quả nghiên cứu xác nhận thứ hai** (theo cùng tiêu chí với cổng G8) · Dự án: đề tài A, *In-Context Gibbs* · Ngày lập: 03/11/2026

- Người tạo sản phẩm: Claude (kế hoạch, code, phân tích, báo cáo); thầy chạy các lượt trên Kaggle.
- Người kiểm duyệt (không tham gia tạo sản phẩm): **thầy** ☒ Người ☐ Agent AI

**Kết quả: thầy duyệt toàn bộ biên bản ngày 03/11/2026, cổng S2 qua, sang G9.** Lời duyệt trong trao đổi: "duyệt toàn bộ GATE_REVIEW_S2". Em ghi các ô dưới đây theo lời duyệt đó.

Mục 3a là **tự rà soát của người tạo**, không thay kiểm duyệt độc lập. Người kiểm duyệt điền cột cuối của Mục 2, Mục 3b và Mục 4.

## 1. Sản phẩm được kiểm
| File | Ghi chú |
|---|---|
| [S2_RESULTS](S2_RESULTS.md) | báo cáo kết quả xác nhận |
| [S2_PLAN](S2_PLAN.md), [S2_DATASETS](S2_DATASETS.md) | đóng băng ở commit `aaee9fd` (01/11/2026 15:19 UTC), trước khi tải dữ liệu đánh giá |
| `S2/analysis/` | `s2_confirmatory.py` (phép thử đã đăng ký), `s2_verify_independent.py` (kiểm lại độc lập), `s2_figures.py`, `s2x_response_map_report.py` |
| `S2/results/`, `S2/figures/` | json, bảng theo dataset, hình |
| [EXPERIMENT_LOG](EXPERIMENT_LOG.md) | mọi lượt S2 và kiểm sau lượt |
| [S2X_REPORT](S2X_REPORT.md) | cơ chế (thăm dò, chỉ trên dataset cũ của NC1) |
| [CLAIM_EVIDENCE_MATRIX](../G3/CLAIM_EVIDENCE_MATRIX.md), [RESEARCH_PLAN §9](../G3/RESEARCH_PLAN.md) | đã cập nhật theo S2 |

## 2. Tiêu chí
| Tiêu chí | Người tạo tự đánh giá | Người kiểm duyệt |
|---|---|---|
| Kế hoạch đóng băng trước khi chạm dữ liệu đánh giá | **Đạt**: commit `aaee9fd` có trước lượt S2D, lượt đầu tiên tải dữ liệu. Mọi thay đổi sau đó đều theo quy tắc đã ghi trước: §3.4 bỏ 2 dataset, bổ sung gas-drift | ☒ Đạt |
| Bảng và hình tính bằng script từ kết quả thô | **Đạt**: chỉ đọc `S2/raw/`. Một script độc lập cho cùng số | ☒ Đạt |
| Phân tích đúng kế hoạch | **Đạt**: H2′, H4′, H3′, S1 và phân tích phụ đúng §1, §4. Có một sửa lỗi làm tròn sau khi thấy số, không đổi kết luận (S2_RESULTS §6) | ☒ Đạt |
| Đủ ô | **Đạt** (sau S2R, 03/11): mọi ô có kết quả hoặc có lý do. 5 ô của PRODEN mặc định (gas-drift) lỗi lặp lại y hệt khi chạy lại, nên ghi "thiếu, có lý do"; không thuộc phép thử chính | ☒ Đạt |
| Khẳng định được đánh dấu và thu hẹp đúng | **Đạt**: #6, #7, #8, #13 ủng hộ, với điều kiện TFM bền với nhãn ngữ cảnh nhiễu | ☒ Đạt |

## 3a. Phát hiện — tự rà soát
| # | Mức | Mô tả | Thầy duyệt 03/11 | Trạng thái |
|---|---|---|---|---|
| 1 | **Lớn** | **S2 đạt thành công mạnh.** H2-LLP′: +0,052 AUC, 16/2, Holm 0,0001. H4-PLL′: +0,012 accuracy ở cả hai mức, Holm 0,016; quy tắc 60% đạt 100% và 90%. H3′: 78,9% ô. S1 không đổi kết luận | Bài giữ là bài phương pháp, nhắm NeurIPS 2027. Sang G9 (viết bài) | Thầy duyệt 03/11 |
| 2 | **Lớn** | Rủi ro phản biện "đổi TFM tới khi đạt" | Bài trình bày theo thứ tự thời gian: NC1 (TabPFN, không đạt) → phát hiện thăm dò (TabICL) → S2 đăng ký trước trên dữ liệu mới (đạt). Dẫn commit đóng băng có dấu thời gian; khi công bố, mở kho với đủ lịch sử. Không bỏ kết quả nào của NC1 | Thầy duyệt 03/11 |
| 3 | Vừa | Khẳng định chỉ đúng với TFM bền với nhãn ngữ cảnh nhiễu. Hai TFM được thử trên **hai tập dataset khác nhau** (TabPFN: NC1; TabICL: NC1 thăm dò + S2) | Không viết "TabICL tốt hơn TabPFN" như một thí nghiệm đối chứng. So sánh hai TFM cùng dữ liệu chỉ có ở NC1 (thăm dò) và S2X (cơ chế) | Đã ghi (ma trận #13); thầy duyệt 03/11 |
| 4 | Vừa | Lợi thế ở PLL khiêm tốn: +1,2 điểm accuracy so với CC tune. Phân tích phụ: 7/16 cặp baseline × mức có ý nghĩa sau Holm, dù chênh dương ở cả 16 | Abstract viết đúng độ lớn: "vượt nhất quán", không viết "vượt xa". Nhấn vào độ bền ở nhãn yếu mạnh (q = 0,6, ≥ 8 lớp), nơi TFM ngây thơ và hard-EM sụp | Thầy duyệt 03/11 |
| 5 | Vừa | Lượt S2T, S2C1, S2C2 chạy bằng bản code đóng băng v1, với danh sách trước bước kiểm dữ liệu. Gas-drift chạy riêng bằng v3 | Code phương pháp giống hệt giữa các bản (đã so). Dataset trượt tiêu chí bị bỏ khi phân tích theo §3.4. Ghi trong EXPERIMENT_LOG | Đã ghi; thầy duyệt 03/11 |
| 6 | Nhỏ | Sửa lỗi làm tròn số dấu phẩy động trong `s2_confirmatory.py` sau khi đã thấy số (cjs, q 0,3: hoà bị đếm là thua) | Báo cáo số trước và sau; mọi kết luận không đổi; NC1 không bị ảnh hưởng | Đã ghi (S2_RESULTS §6); thầy duyệt 03/11 |
| 7 | Nhỏ | 5 ô PRODEN mặc định trên gas-drift lỗi (xác suất không hữu hạn) | Lượt S2R chạy lại một lần (CPU, khoảng 10–15 phút). Lặp lại thì ghi "thiếu, có lý do" | **Xong 03/11:** lỗi lặp lại y hệt ở cả 5 ô (lỗi tất định; ô chạy được cho đúng 0,9430 như lần đầu) → thiếu, có lý do; báo cáo trong bài |
| 8 | Nhỏ | S2 chỉ kiểm LLP túi 8 và PLL. LLP túi 16 và 32, PU, SSL, nhãn nhiễu không chạy lại với TabICL | Phạm vi khẳng định chỉ gồm LLP túi 8 và PLL. Các bối cảnh khác báo cáo theo NC1 (giới hạn, D3) | Thầy duyệt 03/11 |
| 9 | Nhỏ | Chưa đo R̂ với TabICL. Các con số về trộn chuỗi hiện chỉ có cho TabPFN (NC1) | Tuỳ chọn: một lượt R̂ thăm dò trên tập phát triển S2 (dataset cũ của NC1) với TabICL | Thầy duyệt 03/11 |

## 3b. Phát hiện — người kiểm duyệt
| # | Mức | Mô tả | Thầy duyệt 03/11 | Trạng thái |
|---|---|---|---|---|
| — | — | Thầy duyệt toàn bộ, không thêm phát hiện (03/11/2026) | — | — |

## 4. Quyết định
**Hướng đi sau S2** (chọn một):

☒ **Sang G9: viết bài phương pháp.** Gồm phương pháp và lý thuyết (Định lý 1–2, Mệnh đề 3–6); NC1 đầy đủ, kể cả phân tích lỗi; S2 là phép xác nhận chính; S2X và lý thuyết ngưỡng sụp (S2_PLAN §5.1) là phần cơ chế.

☐ **Sửa:** ____________ (quay lại bước ____)

☐ **Dừng**

**Cổng S2:** ☒ **Qua** ☐ **Sửa** ☐ **Dừng**

Người kiểm duyệt: thầy (duyệt trong trao đổi) · Ngày: 03/11/2026
