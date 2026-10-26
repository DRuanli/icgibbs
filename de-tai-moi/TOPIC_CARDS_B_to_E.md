# Thẻ đề tài rút gọn cho các ứng viên B–E (G1)

Ngày: 24/10/2026. Bằng chứng quét tính mới: [G1_NOVELTY_SCAN_LOG.md](G1_NOVELTY_SCAN_LOG.md).

---

## B. Dự đoán đa nhãn và đa target Bayes-optimal bằng phân phối đồng thời của TFM (không huấn luyện lại)
- **Ý tưởng:** dùng quy tắc chuỗi (classifier chain) với TFM có sẵn để có phân phối dự đoán đồng thời của các nhãn, từ đó ra quyết định tối ưu cho từng hàm loss (Hamming, subset accuracy, F1).
- **Tiêu chí bắt buộc:**
  - C1 đạt: Mulan multi-target regression có 13 nguồn độc lập; multi-label có khoảng 7–10 bộ.
  - C2 đạt: Classifier Chains (*Machine Learning* 2011), ML-kNN (*Pattern Recognition* 2007), RAkEL (*TKDE* 2011), ERC (*Machine Learning* 2016).
  - C3 đạt nhưng yếu cho A/A*.
  - C4: deep learning, không có data-centric AI.
  - C5, C6, C7 đạt.
- **Vấn đề:** TabPFN + BR/LP/PCC/ECC đã được dùng (arXiv 2608.30337, 2026). Suy luận Bayes-optimal cho PCC đã có từ Dembczyński et al. (ICML 2010). Phần mới còn lại là phép ghép hai kết quả đã biết, nên khó thuyết phục hội nghị A/A*. Hợp với tạp chí (*Machine Learning*, *Pattern Recognition*) hơn.

| Tính mới | Quan trọng | Lý thuyết | Kỹ năng | Ít bị đi trước | Dự phòng | Khớp venue | **Tổng** |
|---|---|---|---|---|---|---|---|
| 2 | 3 | 3 | 4 | 3 | 3 | 3 | **2,90** |

## C. PFN đa đầu ra gốc (sửa kiến trúc + pretrain)
- **Ý tưởng:** đặt nhiều cột target vào lưới 2 chiều, pretrain trên prior có nhiều target.
- **Tiêu chí bắt buộc:** C1–C5 đạt. **C6 không đạt.** TabPFN gốc cần khoảng 2 tuần × 8 GPU; tiếp tục pretrain từ trọng số v2 trên GPU miễn phí theo phiên là rủi ro cao, và mô hình nhỏ khó cạnh tranh ở hội nghị A/A*.
- **Kết luận:** **loại ở bước lọc cứng.** Giữ làm hướng tương lai nếu có GPU.

## D. Sinh dữ liệu tổng hợp hoặc tăng cường dữ liệu bằng TFM cho bảng nhỏ hoặc mất cân bằng
- **Tiêu chí bắt buộc:**
  - C1 đạt.
  - C2 đạt: SMOTE (*JAIR* 2002), CTGAN (NeurIPS 2019), TabDDPM (ICML 2023).
  - C3–C7 đạt.
- **Vấn đề:** đã có TabPFGen, TabMDA, chức năng sinh dữ liệu của TabPFN, và nhiều mô hình khuếch tán (diffusion) cho bảng. Cạnh tranh cao, khó có lý thuyết mới.

| Tính mới | Quan trọng | Lý thuyết | Kỹ năng | Ít bị đi trước | Dự phòng | Khớp venue | **Tổng** |
|---|---|---|---|---|---|---|---|
| 2 | 3 | 2 | 3 | 2 | 3 | 3 | **2,50** |

## E. Chọn, tinh chỉnh ngữ cảnh và định giá dữ liệu cho TFM
- **Tiêu chí bắt buộc:** C1–C7 đạt. C2: Data Shapley (ICML 2019), KNN-Shapley (VLDB 2019), Data-OOB (ICML 2023).
- **Vấn đề:** TICDA (10/2026), VIP-COP, DemoShapley và các phương pháp ngữ cảnh truy hồi ra liên tục. Rủi ro bị đi trước rất cao.

| Tính mới | Quan trọng | Lý thuyết | Kỹ năng | Ít bị đi trước | Dự phòng | Khớp venue | **Tổng** |
|---|---|---|---|---|---|---|---|
| 2 | 4 | 3 | 3 | 1 | 3 | 4 | **2,80** |

---

## Bảng tổng hợp G1

| Ứng viên | Qua C1–C7? | Điểm | Quyết định |
|---|---|---|---|
| **A. Học từ nhãn yếu bằng TFM (In-Context Gibbs)** | Có (C6 sát) | **4,25** | **Chọn** |
| B. Đa nhãn Bayes-optimal bằng TFM | Có | 2,90 | Dự phòng; hợp tạp chí hơn hội nghị |
| E. Ngữ cảnh và định giá dữ liệu | Có | 2,80 | Loại: quá đông |
| D. Sinh dữ liệu bằng TFM | Có | 2,50 | Loại: quá đông |
| C. PFN đa đầu ra gốc | Không (C6) | — | Loại: thiếu compute |
