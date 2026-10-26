# Thí điểm PLL nhiều lớp trên tập phát triển — báo cáo (lượt v5, 26/10/2026)

Script: `code/kaggle_g4_manyclass_pilot.py` (Kaggle T4; tabpfn 9.2.0, tabpfn-extensions 0.6.3, tabicl 2.2.0 với checkpoint `tabicl-classifier-v2-20260212.ckpt`). Dữ liệu: `g4_manyclass_pilot.csv` (88 dòng, đều ok), `g4_manyclass_pilot_preds.jsonl`, `g4_manyclass_pilot_log.txt`. Exploratory, chỉ trên tập phát triển. Mục đích: việc F6 của G5 — chọn backend cho RQ5-PLL (bộ PLL thật có 13–23 lớp) và đo chi phí.

Thiết kế: vowel (11 lớp, n = 990) và letter (26 lớp, mẫu con phân tầng 2.000 hàng); chia 70/30; PLL với q = r/(C−1), r ∈ {1, 2} → trung bình 2 hoặc 3 ứng viên (khớp dữ liệu thật: 2,18–3,16); 2 seed. Cấu hình phương pháp giống RQ4; dự đoán cuối dùng `n_estimators = 2` cho mọi phương pháp dựa trên TFM.

## 1. Đủ và đúng

Đủ 2 dataset × 2 mức × 2 seed × 2 backend × 5 phương pháp + PRODEN = 88 dòng, không lỗi. Số ứng viên trung bình thực tế 1,97–2,01 (r = 1) và 3,00–3,05 (r = 2), đúng thiết kế. Mọi chuỗi Gibbs có tỉ lệ nhãn đổi ở vòng cuối ≤ 4,1%.

## 2. Kết quả (accuracy, trung bình 2 seed)

| dataset | r | backend | naive | gibbs_warm | sem_warm | hard_em_warm | PRODEN (MLP) | oracle |
|:--|--:|:--|--:|--:|--:|--:|--:|--:|
| vowel | 1 | TabICL v2 | 0,843 | **0,987** | 0,980 | 0,975 | 0,938 | 0,987 |
| vowel | 1 | TabPFN v2 + ECOC | 0,823 | **0,970** | 0,958 | 0,961 | 0,938 | 0,976 |
| vowel | 2 | TabICL v2 | 0,613 | **0,971** | 0,958 | 0,943 | 0,906 | 0,987 |
| vowel | 2 | TabPFN v2 + ECOC | 0,657 | 0,946 | **0,958** | 0,929 | 0,906 | 0,976 |
| letter | 1 | TabICL v2 | 0,818 | 0,886 | 0,882 | **0,891** | 0,811 | 0,897 |
| letter | 1 | TabPFN v2 + ECOC | 0,775 | 0,878 | 0,872 | **0,882** | 0,811 | 0,883 |
| letter | 2 | TabICL v2 | 0,731 | **0,877** | 0,876 | 0,873 | 0,794 | 0,897 |
| letter | 2 | TabPFN v2 + ECOC | 0,671 | **0,868** | 0,858 | 0,857 | 0,794 | 0,883 |

Log loss: `gibbs_warm` tốt hơn `hard_em_warm` ở **8/8** ô (dataset × r × backend), ví dụ vowel r = 2, TabICL: 0,131 so với 0,338.

Thời gian trung bình mỗi ô (giây) và chi phí mỗi lần gọi TFM:

| dataset | backend | gibbs_warm | hard_em_warm | giây / lần gọi |
|:--|:--|--:|--:|--:|
| vowel | TabICL v2 | 35 | 11 | 0,11 |
| vowel | TabPFN v2 + ECOC | 499 | 157 | 1,58 |
| letter | TabICL v2 | 51 | 16 | 0,16 |
| letter | TabPFN v2 + ECOC | 1.074 | 343 | 3,41 |

## 3. Áp quy tắc đã viết trong script trước khi chạy

1. *Loại backend có thời gian trung bình một ô `gibbs_warm` trên letter > 30 phút:* TabPFN v2 + ECOC 17,9 phút, TabICL v2 0,9 phút → không loại backend nào.
2. *Chọn backend có accuracy oracle trung bình cao hơn; chênh < 0,5 điểm % thì chọn backend rẻ hơn:* TabICL v2 0,9416, TabPFN v2 + ECOC 0,9295 → chênh 1,2 điểm % → **chọn TabICL v2** (đồng thời rẻ hơn khoảng 20 lần).
3. *Chia dữ liệu cho RQ5-PLL:* 10 × 90/10 nếu chi phí ước tính cho 3 bộ ≤ 6 giờ T4. Ước tính với TabICL: mỗi lần chạy khoảng 430 lần gọi (Gibbs 315, hard-EM 106, còn lại khởi động ấm và dự đoán). Giả định bi quan 0,5 giây/lần gọi (gấp 3 số đo trên letter, để tính tới d = 108 của Lost và n = 4.498 của BirdSong) → ≤ 4 phút mỗi lần chạy → 3 bộ × 10 lần chia ≈ 2 giờ, kể cả baseline. **→ 10 × 90/10**, đúng giao thức của văn liệu (Feng et al. 2020).

## 4. Đọc kết quả

- In-Context Gibbs **chạy được với 11–26 lớp** qua cả hai backend, và lợi ích của PLL ở RQ4 (≤ 7 lớp) **vẫn giữ** khi nhiều lớp và ít ứng viên: với TabICL, Gibbs vượt PRODEN 0,05–0,08 và vượt naive 0,07–0,36; ở vowel Gibbs bằng hoặc gần bằng oracle.
- So với hard-EM: ở r = 1 chênh accuracy nhỏ (letter: hard-EM nhỉnh hơn 0,004–0,005; vowel: Gibbs hơn khoảng 0,01), ở r = 2 Gibbs hơn ở 4/4 ô; log loss Gibbs tốt hơn ở mọi ô — cùng chiều với H3.
- TabICL v2 vừa tốt hơn vừa nhanh hơn rõ rệt so với TabPFN v2 bọc ECOC ở bài nhiều lớp. Với ≤ 10 lớp, so sánh hai TFM vẫn là ablation đã định ở G5 Mục 7; kết quả này **không** đổi TFM chính của các bối cảnh khác (không có lý do đã viết trước để đổi).
- PRODEN ở đây là bản mặc định, chưa tune.

## 5. Quyết định (ghi vào G5 §1.4, §4, §9)

- RQ5-PLL thật: backend **TabICL v2** (`support_many_classes=True`, checkpoint v2-20260212, `n_estimators = 1` trong chuỗi, 2 khi dự đoán cuối như thí điểm này), dùng chung cho mọi phương pháp dựa trên TFM ở RQ5.
- Chia **10 lần ngẫu nhiên 90/10** cho Lost, MSRCv2, BirdSong.
- Chi phí RQ5-PLL khoảng 2 giờ T4 (thay cho 6 giờ dự trù).
