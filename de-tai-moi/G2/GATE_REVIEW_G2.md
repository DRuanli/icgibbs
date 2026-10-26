# Biên bản kiểm duyệt cổng G2 (tổng quan và kiểm tra tính mới)

Đề tài A · Ngày lập: 24/10/2026
- **Người tạo sản phẩm:** Claude, theo yêu cầu của thầy
- **Người kiểm duyệt độc lập:** người hướng dẫn hoặc đồng tác giả (điền tên) ☐ · Ngày: ____

## 1. Sản phẩm
| File | Nội dung |
|---|---|
| [00_SEARCH_PROTOCOL.md](00_SEARCH_PROTOCOL.md) | Protocol viết trước khi tìm |
| [01_QUERY_LOG.md](01_QUERY_LOG.md) | 18 truy vấn + 2 vòng snowballing |
| [related_work_matrix.csv](related_work_matrix.csv) | 36 công trình |
| [02_CLOSEST_WORKS.md](02_CLOSEST_WORKS.md) | Bảng định vị, đọc sâu, câu "khác ở đâu" |
| [03_QUERIES_FOR_THAY.md](03_QUERIES_FOR_THAY.md) | Truy vấn Scopus, WoS, IEEE, ACM, Google Scholar để thầy chạy |
| [04_NOVELTY_VERDICT.md](04_NOVELTY_VERDICT.md) | Kết luận có điều kiện, câu khẳng định tính mới, baseline cập nhật |
| [05_COMPUTE_BENCHMARK.md](05_COMPUTE_BENCHMARK.md) + `compute/` | Script đo compute và tính ngân sách |

## 2. Tiêu chí qua cổng G2 (theo [02](../../research-handbook/02_QUY_TRINH_CONG_KIEM_DUYET.md))
| Tiêu chí | Trạng thái | Ghi chú |
|---|---|---|
| Tìm kiếm có nhật ký trên ≥4 nguồn | **Đạt một phần** | 6 nhóm nguồn qua tìm kiếm có giới hạn domain; **Scopus, WoS, Google Scholar chưa chạy** |
| Snowballing đến bão hoà | **Đạt với nguồn truy cập được** | Phần xuôi cần Google Scholar "Cited by" |
| ≥3 công trình gần nhất đọc lượt 3, có câu "khác ở đâu" | **Đạt** | In-Context PU, GLWS, TabPFN (lượt 3); Sepin, ILL, Focused PU (lượt 2) |
| Không có công trình làm đúng đóng góp dự kiến | **Đạt (có điều kiện)** | Chờ kết quả §1 và §2 |
| Baseline đạt C2 | **Đạt** | ≥2 baseline peer-review cho mỗi loại nhãn yếu, cộng GLWS và ILL |
| Compute (C6) đo thật | **Chưa** | Chờ CSV từ Kaggle |

## 3a. Tự rà soát của người tạo (không thay thế kiểm duyệt độc lập)
| # | Mức | Phát hiện | Đề xuất |
|---|---|---|---|
| 1 | Lớn | Tìm kiếm qua công cụ web có giới hạn domain không tương đương tìm nâng cao trong Scopus hay IEEE Xplore; có thể sót bài hội thảo nhỏ hay bài tạp chí | Thầy chạy [03](03_QUERIES_FOR_THAY.md) trước khi coi G2 là qua |
| 2 | Lớn | Hai tài liệu lý thuyết làm nền cho định lý (ii) (Liu et al. 2014; Mitrophanov 2005) mới được nhận diện, chưa đọc | Đọc ở đầu G4; nếu không áp dụng được thì định lý (ii) phải đổi hướng |
| 3 | Lớn | Rủi ro bị đi trước tăng: In-Context PU ghi hướng "các dạng nhãn yếu khác" | Ưu tiên PU và LLP, đưa preprint sớm |
| 4 | Nhỏ | arXiv 2610.23765 chưa đọc được (giới hạn tốc độ) | Thầy mở thủ công, đã ghi trong [03](03_QUERIES_FOR_THAY.md) |
| 5 | Nhỏ | Một số baseline (PU ExtraTrees, VIME, Optimal LLP 2509.15145) chưa kiểm tra venue và code | Kiểm tra ở G5 |

## 3b. Phát hiện của người kiểm duyệt
| # | Mức | Mô tả | Đề xuất | Trạng thái |
|---|---|---|---|---|
| | | | | |

## 3c. Cập nhật sau khi thầy hoàn tất phần còn mở (24/10/2026)
- **Scopus, WoS, IEEE, ACM, Google Scholar "Cited by" (file 03):** thầy đã chạy; **không có kết quả nào khác** so với nhật ký của Claude. Tiêu chí "không có công trình làm đúng đóng góp" chuyển từ *có điều kiện* sang **đạt**.
- **Compute (C6):** đã đo trên Kaggle T4 (64/64 cấu hình). Kết luận **đạt với giao thức rút gọn** (xem [05_COMPUTE_BENCHMARK.md](05_COMPUTE_BENCHMARK.md)).

## 4. Quyết định
☑ **Qua** → sang G3 (câu hỏi nghiên cứu, giả thuyết, kế hoạch đóng băng)  ☐ **Sửa**  ☐ **Dừng hoặc thu hẹp**

> Ghi nhận ngày 24/10/2026: thầy xác nhận người kiểm duyệt độc lập đã kiểm duyệt xong G2. Bản ký và các phát hiện (Mục 3b) do thầy lưu.

Chữ ký người kiểm duyệt: ____________
