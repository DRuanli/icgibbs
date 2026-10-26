# Thí điểm RQ4 trên tập phát triển — báo cáo cuối (lượt v4 + v5, 26/10/2026)

Script: `code/kaggle_g4_rq4_pilot.py` (cấu hình mặc định sau G4: khởi động ấm, khối trên mọi mẫu, K = 10, 30 vòng, bỏ 10; TabPFN v2, tabpfn 9.2.0, Kaggle T4). Dữ liệu: `g4_rq4_pilot.csv` (612 dòng ok + 42 dòng lỗi của v4 giữ lại làm vết), `g4_rq4_pilot_preds.jsonl`, `g4_rq4_pilot_log.txt`. Exploratory, chỉ trên tập phát triển. Nguồn gốc: GATE_REVIEW_G4 R5.

## 1. Tình trạng dữ liệu

| Bối cảnh | Lượt | Tình trạng |
|---|---|---|
| Nhãn nhiễu (6 dataset × 2 mức × 3 seed) | v4 | **Đủ**, mọi phương pháp |
| SSL (6 × 2 × 3) | v4 (phương pháp dựa trên TFM), v5 (baseline bị lỗi ở v4) | **Đủ** |
| PLL (vehicle, segment, cmc × 2 mức × 3 seed) | v4 (vehicle, một phần segment), v5 (phần còn lại) | **Đủ** |

Kiểm tra: không có khoá (backend, dataset, bối cảnh, mức, seed, phương pháp) nào bị trùng; phần đầu của file v5 giống hệt file v4 (chạy tiếp đúng cách); số dòng dự đoán (612) bằng số dòng ok.

**Lỗi của v4 và cách sửa ở v5.** LabelSpreading (kernel kNN, scikit-learn) trả 0/0 = NaN cho điểm test mà cả 7 láng giềng nằm ở thành phần đồ thị không chứa mẫu có nhãn; lỗi làm hỏng cả ô nên mất luôn self_training. Ở v5 hàng NaN nhận tiên nghiệm lớp của phần có nhãn và số hàng được ghi vào cột `extra`. Số hàng phải gán như vậy lớn ở mức 0,05: blood-transfusion 39–66/225 hàng test, banknote 40–47/412. Vậy label spreading kNN mặc định là baseline yếu ở tỉ lệ nhãn thấp; ở G6 bản tune (20 cấu hình) phải gồm số láng giềng và kernel RBF.

## 2. Kết quả (trung bình theo seed)

**Nhãn nhiễu phụ thuộc lớp** (ROC AUC; dương→âm với r, âm→dương với r/2, T đã biết):

| mức | dataset | naive | gibbs_warm | sem_warm | hard_em_warm | forward | cl_tfm | oracle |
|--:|:--|--:|--:|--:|--:|--:|--:|--:|
| 0,2 | banknote | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 |
| 0,2 | blood-transfusion | 0,747 | 0,751 | 0,754 | 0,730 | 0,709 | 0,743 | 0,768 |
| 0,2 | credit-g | 0,798 | 0,792 | 0,789 | 0,797 | 0,711 | 0,795 | 0,798 |
| 0,2 | diabetes | 0,837 | 0,838 | 0,825 | 0,815 | 0,715 | 0,837 | 0,843 |
| 0,2 | ilpd | 0,752 | 0,759 | 0,747 | 0,749 | 0,735 | 0,760 | 0,742 |
| 0,2 | wdbc | 0,983 | 0,985 | 0,984 | 0,988 | 0,960 | 0,986 | 0,992 |
| 0,4 | banknote | 1,000 | 1,000 | 1,000 | 1,000 | 0,987 | 1,000 | 1,000 |
| 0,4 | blood-transfusion | 0,702 | 0,720 | 0,662 | 0,654 | 0,631 | 0,711 | 0,768 |
| 0,4 | credit-g | 0,740 | 0,668 | 0,616 | 0,703 | 0,668 | 0,767 | 0,798 |
| 0,4 | diabetes | 0,826 | 0,811 | 0,798 | 0,775 | 0,659 | 0,828 | 0,843 |
| 0,4 | ilpd | 0,749 | 0,690 | 0,621 | 0,567 | 0,666 | 0,752 | 0,742 |
| 0,4 | wdbc | 0,968 | 0,969 | 0,977 | 0,947 | 0,858 | 0,980 | 0,992 |

**SSL** (ROC AUC):

| mức | dataset | naive | gibbs_warm | sem_warm | hard_em_warm | label_spreading | self_training | oracle |
|--:|:--|--:|--:|--:|--:|--:|--:|--:|
| 0,05 | banknote | 1,000 | 0,999 | 0,999 | 1,000 | 0,991 | 0,996 | 1,000 |
| 0,05 | blood-transfusion | 0,704 | 0,697 | 0,619 | 0,658 | 0,620 | 0,680 | 0,768 |
| 0,05 | credit-g | 0,632 | 0,620 | 0,583 | 0,638 | 0,654 | 0,633 | 0,798 |
| 0,05 | diabetes | 0,800 | 0,799 | 0,751 | 0,768 | 0,675 | 0,798 | 0,843 |
| 0,05 | ilpd | 0,663 | 0,656 | 0,639 | 0,656 | 0,629 | 0,654 | 0,742 |
| 0,05 | wdbc | 0,984 | 0,966 | 0,942 | 0,975 | 0,979 | 0,979 | 0,992 |
| 0,2 | banknote | 1,000 | 1,000 | 1,000 | 1,000 | 0,999 | 1,000 | 1,000 |
| 0,2 | blood-transfusion | 0,761 | 0,756 | 0,721 | 0,716 | 0,665 | 0,767 | 0,768 |
| 0,2 | credit-g | 0,697 | 0,701 | 0,710 | 0,680 | 0,694 | 0,692 | 0,798 |
| 0,2 | diabetes | 0,825 | 0,803 | 0,786 | 0,811 | 0,739 | 0,798 | 0,843 |
| 0,2 | ilpd | 0,729 | 0,734 | 0,718 | 0,706 | 0,623 | 0,698 | 0,742 |
| 0,2 | wdbc | 0,983 | 0,976 | 0,975 | 0,977 | 0,978 | 0,984 | 0,992 |

**PLL** (accuracy; mỗi nhãn sai vào tập ứng viên với xác suất q):

| q | dataset | naive | gibbs_warm | sem_warm | hard_em_warm | PRODEN | oracle |
|--:|:--|--:|--:|--:|--:|--:|--:|
| 0,3 | cmc | 0,530 | 0,546 | 0,538 | 0,525 | 0,482 | 0,548 |
| 0,3 | segment | 0,945 | 0,979 | 0,978 | 0,980 | 0,968 | 0,989 |
| 0,3 | vehicle | 0,749 | 0,827 | 0,831 | 0,824 | 0,811 | 0,853 |
| 0,6 | cmc | 0,458 | 0,529 | 0,514 | 0,498 | 0,474 | 0,548 |
| 0,6 | segment | 0,530 | 0,970 | 0,965 | 0,915 | 0,953 | 0,989 |
| 0,6 | vehicle | 0,457 | 0,799 | 0,801 | 0,748 | 0,752 | 0,853 |

PLL, log loss:

| q | dataset | naive | gibbs_warm | sem_warm | hard_em_warm | PRODEN | oracle |
|--:|:--|--:|--:|--:|--:|--:|--:|
| 0,3 | cmc | 0,952 | 0,926 | 0,934 | 1,031 | 1,946 | 0,917 |
| 0,3 | segment | 0,880 | 0,071 | 0,075 | 0,069 | 0,133 | 0,038 |
| 0,3 | vehicle | 0,800 | 0,313 | 0,349 | 0,352 | 0,567 | 0,280 |
| 0,6 | cmc | 1,048 | 0,985 | 1,039 | 1,463 | 2,512 | 0,917 |
| 0,6 | segment | 1,719 | 0,113 | 0,127 | 0,339 | 0,208 | 0,038 |
| 0,6 | vehicle | 1,311 | 0,416 | 0,437 | 0,729 | 0,931 | 0,280 |

## 3. Áp tiêu chí đã viết trước

Tiêu chí (hàm `summarize` của script, viết trước khi chạy; cùng ngưỡng 60% với H4): một loại nhãn yếu ở lại khẳng định chính nếu `gibbs_warm` đạt hoặc vượt **cả** baseline peer-review tốt nhất **lẫn** TFM ngây thơ ở ≥ 60% ô (dataset × mức).

| Loại | Ô đạt | Kết luận |
|---|---|---|
| **PLL** | 3/3 (q = 0,3) + 3/3 (q = 0,6) = **6/6** | **Chính** |
| Nhãn nhiễu | 3/6 + 1/6 = **4/12** (33%) | **Phụ** |
| SSL | 0/6 + 3/6 = **3/12** (25%) | **Phụ** |

**Đính chính.** Báo cáo tạm thời (sau v4) ghi SSL "tối đa 2/12, thêm baseline chỉ có thể làm số ô đạt giảm". Lập luận đó sai vì hai lẽ: (1) ô thiếu cả hai baseline bị so với NaN nên đã bị tính là *không đạt*, và (2) thêm seed còn thiếu làm thay đổi trung bình của baseline (credit-g 0,2: self_training từ 0,707 với 2 seed xuống 0,692 với 3 seed). Số đúng là 3/12. Kết luận "SSL là bối cảnh phụ" không đổi.

**RQ3 trên các loại nhãn của RQ4:** `gibbs_warm` có log loss tốt hơn `hard_em_warm` ở **27/30** ô (SSL 11/12, nhãn nhiễu 11/12, PLL 5/6). Gộp với lượt v2 (30/36): 57/66 ô = 86%, trên ngưỡng 70% của H3.

## 4. Đọc kết quả

1. **PLL là chỗ phương pháp giúp rõ nhất.** Gibbs vượt PRODEN 0,011–0,064 accuracy ở cả 6 ô và vượt naive 0,016–0,44. Ở q = 0,6 Gibbs vượt hard-EM 0,03–0,05 accuracy, log loss tốt hơn rõ (segment 0,113 so với 0,339). Ở cmc (bài toán khó, oracle chỉ 0,548), Gibbs gần bằng oracle. Lưu ý: PRODEN ở đây chạy mặc định, chưa tune; ở G6–G7 nó nhận 20 cấu hình nên cách biệt có thể hẹp lại.
2. **Nhãn nhiễu phụ thuộc lớp và SSL: TFM ngây thơ đã gần trần về AUC.** Ở nhiễu r = 0,2, naive cách oracle chưa tới 0,02 AUC ở 5/6 dataset. Điều này khớp một lập luận giống Elkan–Noto cho PU-SCAR: với ma trận nhiễu phụ thuộc lớp, $p(w{=}1\mid x)=\rho_{01}+(1-\rho_{10}-\rho_{01})\,p(y{=}1\mid x)$ là hàm affine tăng khi $\rho_{10}+\rho_{01}<1$, nên xếp hạng của TFM ngây thơ trùng xếp hạng Bayes. Ở SSL, TFM chỉ trên phần có nhãn đạt hoặc vượt cả hai baseline peer-review ở 9/12 ô.
3. **Ở mức yếu nặng, Gibbs làm hỏng log loss** (nhiễu r = 0,4: 0,74–1,04 so với naive 0,55–0,56 ở diabetes, credit-g, ilpd; SSL 0,05 cũng vậy) — cùng hiện tượng khuếch đại sai số đã thấy ở LLP túi 32 (lượt v2). SEM và hard-EM còn tệ hơn, nên vấn đề nằm ở vòng lặp tự gán nhãn chứ không riêng Gibbs.
4. **Bức tranh chung sau RQ2 và RQ4:** In-Context Gibbs giúp khi nhãn yếu ở dạng **tập, tổng hoặc phụ thuộc mẫu** (PLL, LLP túi nhỏ, PU-SAR), nơi bộ ngây thơ không nhất quán; không giúp về AUC khi xếp hạng ngây thơ đã đúng (PU-SCAR, nhiễu phụ thuộc lớp) hoặc khi phần có nhãn đã đủ (SSL). Đề xuất viết thành một mệnh đề ở G6 và dùng làm phần giải thích giới hạn trong bài.

## 5. Quyết định

- G5 Mục 0: **PLL chính** (đã chốt); SSL và nhãn nhiễu phụ, vẫn chạy và báo cáo đầy đủ trên tập đánh giá. Ghi ở RESEARCH_PLAN §9, CLAIM_EVIDENCE_MATRIX #8.
- RQ5 dùng PLL thật làm nguồn thứ hai (G5 §1.4); backend nhiều lớp chọn ở [PILOT_MANYCLASS_REPORT.md](PILOT_MANYCLASS_REPORT.md).
