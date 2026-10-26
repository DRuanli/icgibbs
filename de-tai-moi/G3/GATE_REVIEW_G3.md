# Biên bản kiểm duyệt cổng G3 (câu hỏi nghiên cứu, giả thuyết, đóng góp)

Đề tài A · Ngày lập: 24/10/2026
- **Người tạo sản phẩm:** Claude (bản nháp), thầy (chủ sở hữu kế hoạch)
- **Người kiểm duyệt độc lập:** người hướng dẫn hoặc đồng tác giả (điền tên) ☐ · Ngày: ____

## 1. Sản phẩm
| File | Nội dung |
|---|---|
| [RESEARCH_PLAN.md](RESEARCH_PLAN.md) | Đóng góp, dữ liệu (tách tập phát triển và tập đánh giá), 5 RQ kèm giả thuyết, dự đoán, điều kiện bác bỏ; tiêu chí dừng; mốc thời gian; CRediT; phương án B |
| [THEORY_DRAFT.md](THEORY_DRAFT.md) | Bổ đề 1, Định lý 1–2, Mệnh đề 3–4, kế hoạch kiểm chứng số |
| [CLAIM_EVIDENCE_MATRIX.md](CLAIM_EVIDENCE_MATRIX.md) | 11 khẳng định dự kiến, mỗi khẳng định gắn một RQ hoặc định lý |

## 2. Tiêu chí qua cổng G3 (theo [02](../../research-handbook/02_QUY_TRINH_CONG_KIEM_DUYET.md))
| Tiêu chí | Đạt? | Ghi chú của người kiểm duyệt |
|---|---|---|
| Mỗi khẳng định dự kiến trong abstract có RQ hoặc định lý, thí nghiệm dự kiến, và tiêu chí bác bỏ | | |
| RQ cụ thể, có metric và kết quả dự đoán bằng con số | | |
| Có tiêu chí dừng đo được | | |
| Thoả thuận vai trò và thứ tự tác giả (CRediT) | | **Thầy cần điền tên ở Mục 7 của RESEARCH_PLAN** |
| Đóng góp được phân loại (Shaw 2003) | | Loại câu hỏi: phát triển phương pháp; kết quả: kỹ thuật + mô hình giải tích; kiểm chứng: phân tích (chứng minh) + đánh giá thực nghiệm có hệ thống |

## 3a. Tự rà soát của người tạo (không thay thế kiểm duyệt độc lập)
| # | Mức | Phát hiện | Đề xuất |
|---|---|---|---|
| 1 | Lớn | Các con số dự đoán ở RQ1–RQ5 (ví dụ AUC hơn ≥0,02; thắng ≥7/10) do Claude đề xuất từ hiểu biết về văn liệu, chưa có thí điểm. Nếu đặt quá cao, khẳng định chính dễ bị bác bỏ; quá thấp thì không còn ý nghĩa | Thầy và người kiểm duyệt duyệt từng con số **trước khi đóng băng**; sau khi đóng băng thì không sửa |
| 2 | Lớn | Bổ đề 1 chỉ đúng cho cập nhật một mẫu; cập nhật theo khối (cần cho compute) là xấp xỉ và Mệnh đề 4 chưa chắc chứng minh được | Đã ghi rõ trong THEORY_DRAFT §1 và §5; RQ1 đo sai lệch theo K; ghi trong Limitations nếu không chứng minh được |
| 3 | Lớn | Chặn ở Định lý 2 có hệ số n và có thể rất lỏng | Ghi trung thực; bổ sung thực nghiệm; tìm chặn theo hệ số co ở G4 |
| 4 | Lớn | Lịch G7–G8 không còn dư địa (khoảng 10 tuần cho 8–9 tuần GPU) | Đã ghi quy tắc thu hẹp (bớt RQ4 trước, không cắt RQ2) |
| 5 | Nhỏ | Tiêu chí thí điểm đã đổi sang tập phát triển (≥6 dataset, thắng ≥4/6) thay cho Mốc 2 của TOPIC_CARD_A | Thầy xác nhận việc đổi này |
| 6 | Nhỏ | PU-SAR với propensity chưa biết cần prior trên $e(x)$; kết quả phụ thuộc prior | Phân tích độ nhạy bắt buộc (đã ghi trong RESEARCH_PLAN §3) |

## 3b. Phát hiện của người kiểm duyệt
| # | Mức | Mô tả | Đề xuất | Trạng thái |
|---|---|---|---|---|
| | | | | |

## 4. Quyết định
☑ **Qua** → ghi ngày đóng băng và commit vào RESEARCH_PLAN, sang G4  ☐ **Sửa**  ☐ **Dừng**

> Ghi nhận ngày 24/10/2026: thầy xác nhận người kiểm duyệt độc lập đã kiểm duyệt xong G3, gồm các con số dự đoán, việc đổi tiêu chí thí điểm sang tập phát triển, và kế hoạch lý thuyết. RESEARCH_PLAN được đóng băng từ ngày này; mọi thay đổi sau đó ghi ở Mục 9 của RESEARCH_PLAN. **Còn mở:** tên và thứ tự tác giả (Mục 7) chưa được điền.

Chữ ký người kiểm duyệt: ____________
