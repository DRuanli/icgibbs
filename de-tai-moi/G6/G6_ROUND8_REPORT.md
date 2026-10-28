# G6 lượt 8 — GLWS/ILL cho PLL đã sửa, chọn lại đối thủ chính của PLL; tái hiện PRODEN, RC, CC trên UCI (28/10/2026)

Dữ liệu ở `G4/pilot/`: `g6r8_pll.csv` (script thí điểm), `g6r8_runner_pll.csv` (bộ chạy G7), `g6r8_pll_uci.csv` (tái hiện), log gộp `g6r8_logs.txt`. Chưa phương pháp nào chạy trên tập đánh giá.

## 1. Đầy đủ
- Kaggle CPU, 16:38–18:50 UTC 27/10 (2 giờ 12 phút; phần tái hiện chậm nhất vì lưới 36 cấu hình × 2000 epoch ở Har).
- GLWS/ILL cho PLL: 36/36 dòng `ok` ở mỗi nguồn. Số mẫu train/test khớp lượt 2 ở 18/18 ô.
- Bản tune chọn số hạng cân bằng entropy ở 10/18 ô, bỏ ở 8/18 ô.
- Bản nén không kèm thư mục code đã chạy nên lần này em không đối chiếu được code với kho bằng `cmp`. Script tự ghi tham số trong từng dòng kết quả (vd. `average_entropy_loss`), và các tham số này khớp code đã gửi.

## 2. Đối thủ chính của PLL: chọn lại theo quy tắc → **CC tune**
| Ứng viên (trung bình accuracy, tập phát triển) | Script thí điểm (nguồn của quy tắc) | Bộ chạy G7 (đối chiếu) |
|---|---|---|
| **cc_tun** | **0,7422** | **0,7472** |
| glwspll_tun (đã sửa) | 0,7412 (bản cũ 0,7451) | 0,7436 (bản cũ 0,7493) |
| rc_tun | 0,7411 | 0,7452 |
| proden_tun | 0,7378 | 0,7451 |
| glwspll_def (đã sửa) | 0,7376 | 0,7426 |

- Hai nguồn dữ liệu cho cùng thứ hạng đầu. Khoảng cách giữa các bản tune rất nhỏ (0,001–0,004).
- Thêm số hạng cân bằng entropy của bản chính thức làm GLWS/ILL hơi kém trên dữ liệu bảng (−0,004 tới −0,006), dù tune được chọn bật/tắt.
- **Phạm vi:** Gibbs ấm vẫn đạt quy tắc 60% ở 6/6 ô. Trên bộ chạy G7, Gibbs hơn CC tune ở mọi ô, từ +0,016 (vehicle q = 0,3) tới +0,069 (cmc q = 0,6); trung bình 0,785 so với 0,747.
- Đã ghi ở docs/primary_comparators.md, giao thức §11 và RESEARCH_PLAN §9.

## 3. Tái hiện số công bố (T3)
| Baseline | Đạt | Lệch |
|---|---|---|
| PRODEN (Bảng 1, binomial, 4 bộ × 2 mức q) | **8/8**; chênh lớn nhất −1,96 ở Yeast q = 0,1, vẫn trong 1 độ lệch chuẩn công bố (4,66) | — |
| RC (Bảng 3, ứng viên đều) | Texture (+0,32), Har (−0,32) | Yeast (54,26 so với 59,89), Dermatology (94,59 so với 99,41) |
| CC (Bảng 3) | Texture (+1,53, trong độ lệch chuẩn 2,91), Har (−0,32) | Yeast (54,26 so với 59,97), Dermatology (94,86 so với 99,73) |

**Chẩn đoán phần lệch.** Script chọn lr và weight decay **một lần**, trên validation 10% của lần chia đầu: khoảng 130 mẫu ở Yeast và 30 mẫu ở Dermatology. Ở Yeast, cách chọn này ra weight decay 0,1 — rất mạnh. Khi đó RC và CC cho kết quả **trùng nhau ở cả 10 lần chia**, tức mô hình gần như chỉ còn phản ánh số hạng điều chuẩn. Ở Dermatology, cách chọn ra lr 1e-4, có thể chưa hội tụ sau 2000 bước full batch. Bài không nói chọn siêu tham số một lần hay mỗi lần chia.

**Lượt 9** (CPU, khoảng 1 giờ) chạy phân tích độ nhạy, ghi trước: chọn riêng cho từng lần chia, chỉ trên Yeast và Dermatology. Báo cáo cả hai cách; số của lượt 8 giữ nguyên. Nếu cách chọn mới đạt, xếp loại (a): nguyên nhân rõ, do chọn siêu tham số. Nếu không, xếp loại (b).

PRODEN đạt cả 8 bộ dùng cùng vòng huấn luyện (`PLLNet`) với RC và CC. Vì vậy em cho rằng phần lệch nằm ở cách chọn siêu tham số, không ở vòng huấn luyện chung. CC nay là đối thủ chính của H4 nên cần làm rõ phần này trước G7.
