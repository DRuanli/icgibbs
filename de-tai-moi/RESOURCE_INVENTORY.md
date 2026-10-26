# Kiểm kê tài nguyên (G0): đề tài mới

Người điền: (thầy)  Ngày: 24/10/2026  Nguồn: câu trả lời của thầy ngày 24/10/2026

> Các ô *cần bổ sung* phải có giá trị cụ thể thì G0 mới qua cổng.

## 1. Thời gian
| Mục | Giá trị |
|---|---|
| Số giờ mỗi tuần cho dự án | **Trên 30 giờ/tuần** |
| Hạn chót cứng | **Không có**. Venue đích là hội nghị nên mốc thực tế là một chu kỳ nộp cụ thể, sẽ chốt ở G1 |
| Giai đoạn bận | *cần bổ sung* |

## 2. Con người
| Vai trò | Có? | Ghi chú |
|---|---|---|
| Người hướng dẫn / giảng viên | **Có** | Nên là người kiểm duyệt độc lập ở G1, G5, G10 |
| Đồng tác giả cùng chuyên môn | **Có** | Nên là người kiểm duyệt độc lập ở G4 (chứng minh) và G6 (code từ clone sạch) |
| Chuyên gia lĩnh vực ứng dụng | Không | |

## 3. Compute
| Mục | Giá trị |
|---|---|
| Máy cá nhân | **MacBook Pro M1** (RAM: *cần bổ sung*) |
| GPU | **Colab và Kaggle miễn phí** (giới hạn giờ mỗi tuần và độ dài phiên; kiểm tra quota hiện hành) |
| GPU riêng, của trường, hoặc cloud trả phí | Không |

## 4. Dữ liệu
| Loại | Có? | Ghi chú |
|---|---|---|
| Dữ liệu công khai | **Có** | Nguồn chính cho tiêu chí C1 (≥10 bộ độc lập) |
| Dữ liệu tự thu thập | **Có** | Nếu là dữ liệu học sinh dưới 18 tuổi: cần đồng ý của người đại diện theo Luật 91/2025/QH15, và ẩn danh hoá trước khi phân tích |
| Dữ liệu riêng của tổ chức | Không | |

## 5. Mảng muốn đi sâu (C4)
**Deep learning** và **data-centric AI**.

## 6. Tài chính và nơi công bố
| Mục | Giá trị |
|---|---|
| Phí xuất bản (APC) | **Cơ quan chi trả đầy đủ** |
| Phí đăng ký hội nghị, đi lại, visa | *cần bổ sung*: cơ quan có chi trả không? |
| Venue đích | **Hội nghị hạng A/A*** (NeurIPS, ICML, ICLR, KDD, AAAI…) |

## 7. Ràng buộc và loại trừ
- Không làm: suy luận nhân quả, fairness, kiểm định hay thẩm định như là chủ đề nghiên cứu.
- Tiêu chí bắt buộc: ≥10 dataset độc lập; ≥2 baseline đã peer-review; venue đích rõ ràng.

## 8. Hệ quả cho bước chọn đề tài (G1)
1. **Compute là ràng buộc chặt nhất.** Chỉ có GPU miễn phí theo phiên và một máy M1, nên loại các đề tài cần pretrain mô hình lớn từ đầu. Ví dụ, TabPFN gốc cần 8 GPU trong khoảng 2 tuần. Ưu tiên đề tài mà phần tốn compute nằm ở mô hình nhỏ đến vừa, fine-tune nhẹ, hoặc dùng mô hình nền tảng có sẵn ở chế độ suy luận.
2. **Hội nghị A/A* đặt chuẩn cao về tính mới và độ mạnh của baseline.** Đề tài cần có một ý tưởng phương pháp rõ, lý tưởng là kèm phần lý thuyết. Một cải tiến nhỏ trên benchmark sẽ không đủ.
3. **Dữ liệu công khai là nguồn chính.** Dữ liệu tự thu thập chỉ nên làm case study bổ sung, vì nó khó đạt tính tái lập mà hội nghị ML đòi hỏi, và vướng thủ tục đồng ý nếu là dữ liệu trẻ em.
4. **Hai người đồng hành** đủ để bố trí kiểm duyệt độc lập ở mọi cổng quan trọng.
