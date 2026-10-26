# G2 · Kết luận tính mới (có điều kiện)

Ngày: 24/10/2026 · Đề tài A: *In-Context Gibbs: học từ nhãn yếu bằng tabular foundation model, không huấn luyện lại*

## 1. Kết luận
**Mới. Đi tiếp sang G3.**

Cập nhật 24/10/2026: thầy đã chạy toàn bộ [03_QUERIES_FOR_THAY.md](03_QUERIES_FOR_THAY.md) (Scopus, WoS, IEEE, ACM, Google Scholar "Cited by") và không thấy kết quả nào khác. Điều kiện ban đầu đã được thoả. Nhật ký tìm kiếm phải được chạy lại trong vòng 30 ngày trước ngày nộp (G11).

Căn cứ:
- Trên arXiv, OpenReview, PMLR/NeurIPS, IEEE/ACM (tìm theo domain), DBLP/Semantic Scholar (tìm theo domain), cộng 2 vòng snowballing: **không có công trình nào dùng TFM có sẵn để học từ nhãn yếu bằng suy luận trên nhãn ẩn**, dù cho một hay nhiều loại nhãn yếu.
- Công trình gần nhất (In-Context PU) pretrain một mô hình riêng cho một loại nhãn yếu dưới SCAR.
- Hai khung thống nhất (GLWS, ILL) huấn luyện bằng gradient trên ảnh và văn bản.
- Hai bài độc lập ghi đúng khoảng trống này trong future work: In-Context PU ("mở rộng sang các dạng nhãn yếu khác") và Focused PU ("tích hợp tabular foundation model").

## 2. Bản nháp câu khẳng định tính mới (để dùng trong bài)
> To our knowledge, based on searches of arXiv, OpenReview, PMLR, NeurIPS Proceedings, IEEE Xplore, ACM DL, DBLP and Semantic Scholar (queries in Appendix X, last updated <ngày>), no prior work learns from weak labels with an off-the-shelf tabular foundation model, without retraining, by Bayesian inference over the latent labels. The closest work, In-Context PU learning (Liu et al., 2026), pretrains a dedicated model for positive-unlabeled data under the SCAR assumption. Unified weak-supervision frameworks (GLWS, Chen et al., 2024; ILL, Chen et al., 2024) require gradient training, treat instance posteriors as conditionally independent, and are evaluated on images and text.

## 3. Thu hẹp và làm rõ đóng góp sau G2
| Đóng góp (từ TOPIC_CARD_A) | Sau G2 |
|---|---|
| 1. Gibbs dùng TFM làm xác suất có điều kiện | **Giữ.** Phải tách rõ khỏi EM kiểu GLWS bằng ablation "Gibbs so với posterior từng mẫu độc lập" |
| 2. Lý thuyết (tính đúng, độ bền, định danh) | **Giữ.** Neo vào lý thuyết imputation lặp với các xác suất có điều kiện không tương thích (Liu et al., *Biometrika* 2014) và nhiễu loạn chuỗi Markov (Mitrophanov 2005). Hai tài liệu này **chưa được đọc** ở G2, cần đọc và kiểm tra ở G4 |
| 3. Tính toán theo khối | **Giữ.** Có thể mượn forward-backward trên NFA của GLWS cho ràng buộc tổ hợp |
| 4. 5 loại nhãn yếu | **Giữ, nhưng ưu tiên thứ tự:** PU (có SAR) và LLP trước, vì đây là chỗ có đối thủ và có dữ liệu thật; sau đó PLL, nhiễu, SSL |

## 4. Rủi ro bị đi trước: tăng
Nhóm In-Context PU ghi rõ hướng "các dạng nhãn yếu khác". Tuy nhiên, cách của họ là pretrain riêng cho từng loại; cách không cần huấn luyện của ta là khác biệt phương pháp, không chỉ khác phạm vi. **Hành động:** đưa preprint lên arXiv sớm, ngay khi có kết quả thí điểm PU và LLP cùng định lý (i), không chờ đến hạn NeurIPS.

## 5. Baseline sau G2 (đầu vào cho G5)
| Loại nhãn yếu | Baseline đã peer-review | Baseline khác |
|---|---|---|
| PU | uPU (ICML 2015), nnPU (NeurIPS 2017), SAR-EM (ECML-PKDD 2019), VPU (NeurIPS 2020), PU ExtraTrees (NeurIPS 2022, cần kiểm tra) | In-Context PU (preprint), iFPU (preprint), TabPFN ngây thơ |
| LLP | DLLP (ICDM 2017), Easy-LLP (NeurIPS 2023), BP-LLP (ICLR 2024), Count-based (NeurIPS 2023) | Các phương pháp trong LLP-Bench, TabPFN ngây thơ |
| PLL | PRODEN (ICML 2020), IPAL (IJCAI 2015) | PL-kNN, TabPFN ngây thơ |
| Nhãn nhiễu | Confident Learning (JAIR 2021), Forward correction (CVPR 2017) | TabPFN thô (theo Noise Immunity) |
| SSL | VIME (NeurIPS 2020, cần kiểm tra), self-training | TabPFN chỉ với phần có nhãn |
| Thống nhất | **GLWS (ICML 2024), ILL (NeurIPS 2024)** | TabPFN + hard-EM |

Mỗi loại nhãn yếu đều có ≥2 baseline đã peer-review, nên **C2 vẫn đạt**.

## 6. Việc còn mở trước khi G2 chính thức qua (đã xong ngày 24/10/2026)
1. ☑ Thầy chạy [03_QUERIES_FOR_THAY.md](03_QUERIES_FOR_THAY.md): không có kết quả khác.
2. ☑ Đo compute trên Kaggle: xong ([05_COMPUTE_BENCHMARK.md](05_COMPUTE_BENCHMARK.md)).
3. ☑ Người kiểm duyệt độc lập duyệt [GATE_REVIEW_G2.md](GATE_REVIEW_G2.md): xong.
