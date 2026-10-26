# Biên bản kiểm duyệt cổng G1 (chọn bài toán)

Dự án: đề tài mới 2026–2027 · Ngày lập: 24/10/2026

- **Người tạo sản phẩm:** Claude (trợ lý AI), theo yêu cầu của thầy
- **Người kiểm duyệt độc lập:** người hướng dẫn hoặc đồng tác giả của thầy (điền tên) ☐ · Ngày: ____

> Mục 3a là kết quả **tự rà soát của người tạo**. Theo quy tắc 10 của sổ tay, nó *không thay thế* kiểm duyệt độc lập. Người kiểm duyệt điền Mục 2, 3b và 4.

## 1. Sản phẩm được kiểm
| File | Nội dung |
|---|---|
| [RESOURCE_INVENTORY.md](RESOURCE_INVENTORY.md) | G0, còn 2 ô cần bổ sung |
| [G1_NOVELTY_SCAN_LOG.md](G1_NOVELTY_SCAN_LOG.md) | Quét tính mới sơ bộ, 5 ứng viên |
| [TOPIC_CARD_A.md](TOPIC_CARD_A.md) | Ứng viên được đề xuất |
| [TOPIC_CARDS_B_to_E.md](TOPIC_CARDS_B_to_E.md) | 4 ứng viên còn lại và bảng tổng hợp |

## 2. Tiêu chí qua cổng G1 (theo [02](../research-handbook/02_QUY_TRINH_CONG_KIEM_DUYET.md))
| Tiêu chí | Đạt? | Ghi chú của người kiểm duyệt |
|---|---|---|
| Sinh 3–5 ứng viên, có Heilmeier | | |
| Ứng viên được chọn đạt C1–C7 | | Lưu ý C1 (nhãn yếu bán tổng hợp), C6 (compute sát) |
| Ứng viên được chọn có điểm cao nhất | | |
| Có venue đích + 2 dự phòng + chu kỳ nộp | | Hạn NeurIPS 2027 mới là ước tính |
| Có tiêu chí dừng đo được | | |

## 3a. Tự rà soát của người tạo
| # | Mức | Phát hiện | Đề xuất |
|---|---|---|---|
| 1 | Lớn | Nhãn yếu sinh từ nhãn thật là *bán tổng hợp*, dễ bị coi là va với quy tắc "bằng chứng chính từ dữ liệu thật" | Thầy quyết định: chấp nhận như chuẩn của mảng, với điều kiện bắt buộc có ≥2 nguồn nhãn yếu thật (LLP-Bench, partial label thật) |
| 2 | Lớn | Ước tính compute (200–250 giờ GPU, ×2) dựa trên giả định khoảng 1 giây mỗi forward, chưa đo | Đo thời gian forward thật trên Kaggle T4 trong tuần đầu G2, trước khi đóng băng phạm vi |
| 3 | Lớn | Quét tính mới mới ở mức sơ bộ (công cụ tìm web); chưa quét Scopus, IEEE, ACM, DBLP, chưa snowballing | Làm ở G2, ưu tiên các bài trích dẫn GLWS, ILL, In-Context PU |
| 4 | Nhỏ | Hạn NeurIPS 2027 lấy từ trang tổng hợp, chưa có call for papers chính thức | Xác minh khi call for papers ra (thường khoảng tháng 2–3) |
| 5 | Nhỏ | Trọng số chấm điểm là mặc định của sổ tay, chưa được thầy điều chỉnh | Thầy xem lại; nếu đổi trọng số thì chấm lại cả 5 ứng viên |

## 3c. Quyết định của thầy về các điểm tự rà soát (24/10/2026)
- **Điểm 1 (dữ liệu bán tổng hợp):** đồng ý chấp nhận nhãn yếu sinh từ nhãn thật theo cơ chế công bố, với điều kiện bắt buộc có ≥2 nguồn nhãn yếu thật để kiểm chứng ngoại tại.
- **Điểm 2 (compute):** đồng ý đo thật trên Kaggle ở G2 (script: [G2/05_COMPUTE_BENCHMARK.md](G2/05_COMPUTE_BENCHMARK.md)).
- **Điểm 3 (quét tính mới đầy đủ):** đồng ý thực hiện ở G2 (kết quả: [G2/04_NOVELTY_VERDICT.md](G2/04_NOVELTY_VERDICT.md)).
- Thầy đồng ý cho bắt đầu G2 song song với việc chờ người hướng dẫn kiểm duyệt G1.

## 3b. Phát hiện của người kiểm duyệt
| # | Mức | Mô tả | Đề xuất | Trạng thái |
|---|---|---|---|---|
| | | | | |

## 4. Quyết định
☑ **Qua** → sang G2 với ứng viên A  ☐ **Sửa** (ghi rõ)  ☐ **Dừng**

> Ghi nhận ngày 24/10/2026: thầy xác nhận người kiểm duyệt độc lập đã kiểm duyệt xong G1. Bản ký và các phát hiện của người kiểm duyệt (Mục 3b) do thầy lưu; nếu có phát hiện cần xử lý, bổ sung vào Mục 3b.

Chữ ký người kiểm duyệt: ____________
