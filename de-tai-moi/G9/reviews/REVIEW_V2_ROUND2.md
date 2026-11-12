# REVIEW_V2_ROUND2 — Phản biện vòng 2 (bản nháp đang viết, ghi chú tạm)

Trạng thái: ĐANG LÀM. Tệp này sẽ được thay bằng báo cáo đầy đủ. Các ghi chú dưới đây là kết quả đã kiểm xong tính đến lúc ghi.

## Ghi chú tạm (đã kiểm)

- PDF 36 trang; chính văn kết thúc ở trang 9 (Conclusion, dòng 343); References bắt đầu trang 10. Dấu nháp còn lại: chỉ `[Pending: licences ...]` ở checklist mục 12. Không còn TODO, `[C#]`, `??`.
- `\claim{}` bị ẩn nhưng để lại khoảng trắng trước dấu câu trong PDF ("weak labels , with averages over starts and over TFMs . (2)", "exact TFM , stability with an approximate one , and", "datasets ."). COSMETIC nhưng lộ ngay ở trang 2.
- Bảng 2 vẫn in "gap +0.000" (thân bài đã sửa thành "0.000"): `make_v2.py` ghi `s3_tests.tex` (dòng 191) trước khi ghi đè `s3-c5-gap` (dòng 570).
- `experiments.tex:90-92`: "for $n_F$ between 756 to 3,934 it needs margins above about 0.17 to 0.30": 0,17 và 0,30 là biên ở n_F = 3 000 và 1 000 (khoá `th-margin-3k`, `th-margin-1k`), không phải ở 3 934 và 756 (tính lại: 0,15 và 0,35). Lỗi ngữ pháp "between ... to".
- `experiments.tex:79-80`: "three of the four TabPFN v2 datasets climbed through the region of negative gain": semeion chỉ leo VÀO vùng gain âm (0,17 → 0,34–0,37; gain âm tới r = 0,8), MiceProtein 2/3 seed dừng ở 0,44 (trong vùng âm 0,21–0,55), optdigits 2/3 seed vượt qua.
- Bảng 6 (`app_s1_chains.tex`): cột "start" là `monitor[0]` = độ đúng nhãn SAU vòng quét thứ nhất, không phải giá trị xuất phát (Study 1 không lưu r_start).
- Contributions (2) vẫn viết "invariant floors and ceilings".
- `appendix.tex:229-233` (đoạn Regressions, audit m3) chưa sửa.
- Tính lại từ 213 npz của Study 3: mọi số xác nhận khớp (H1″ +0,0585, trung vị +0,0025, 13/6/1; H2″ +0,0457, 18/2/0; H3″ AUC 0,7661, 66/213 ô dương; C4″ 10/71, thua chuỗi đơn 2, thua L 9, cả hai 1; C5″ gap +0,00002; X−L +0,0637, 20/0/0; X_warm−L: LLP 12/0/0, PLL 6/1/1, nhỏ nhất −0,200).
- Study 1, bốn dataset: s1-fin-pfn-* khớp (mfeat 0,161–0,172; semeion 0,342–0,370; MiceProtein 0,437/0,442/0,636; optdigits 0,265/0,887/0,934; TabICL nhỏ nhất 0,925).
- `v2-uni-above-all` = 224/234 = 95,7% (in 96%); `v2-p1` 0,953–0,989; `th-gain-absmax` = 0,1506 (TabICL v2, optdigits, r = 0,6); TabPFN v2 lớn nhất |gain| = 0,111.
