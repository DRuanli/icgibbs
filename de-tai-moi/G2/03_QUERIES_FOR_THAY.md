# G2 · Truy vấn thầy cần chạy (cần tài khoản cơ quan hoặc trình duyệt)

Môi trường làm việc của Claude không truy cập được Scopus, Web of Science, Google Scholar, và API của DBLP, Semantic Scholar, arXiv. Các truy vấn dưới đây hoàn tất phần còn thiếu của G2. Thời gian ước tính khoảng 60–90 phút.

**Cách ghi lại:** điền bảng ở Mục 5. Bài nào thoả tiêu chí chọn 1 (TFM hoặc PFN × nhãn yếu) hoặc tiêu chí 4 (Gibbs hoặc EM với xác suất có điều kiện do mạng học sẵn) trong [00_SEARCH_PROTOCOL.md](00_SEARCH_PROTOCOL.md) thì ghi tên, link, và một câu "làm gì".

## 1. Scopus (Advanced search)

```
TITLE-ABS-KEY ( ( "tabular foundation model*" OR tabpfn OR tabicl OR "prior-data fitted network*" OR ( "in-context learning" AND tabular ) )
AND ( "weak supervision" OR "weakly supervised" OR "label proportion*" OR "positive-unlabeled" OR "PU learning"
      OR "partial label*" OR "candidate label*" OR "noisy label*" OR "label noise" OR "semi-supervised"
      OR "complementary label*" OR "imprecise label*" OR "aggregate label*" ) )
AND PUBYEAR > 2021
```

```
TITLE-ABS-KEY ( ( tabpfn OR "prior-data fitted network*" OR "tabular foundation model*" )
AND ( gibbs OR "expectation maximization" OR "latent label*" OR "multiple imputation" OR "chained equation*" ) )
AND PUBYEAR > 2021
```

## 2. Web of Science (Advanced search)

```
TS=(("tabular foundation model*" OR TabPFN OR TabICL OR "prior-data fitted network*")
AND ("weak supervision" OR "weakly supervised" OR "label proportion*" OR "positive-unlabeled"
     OR "partial label*" OR "noisy label*" OR "label noise" OR "semi-supervised" OR "imprecise label*"))
AND PY=(2022-2026)
```

## 3. IEEE Xplore (Command Search) và ACM Digital Library

IEEE Xplore:
```
("All Metadata":TabPFN OR "All Metadata":"tabular foundation model" OR "All Metadata":"prior-data fitted")
AND ("All Metadata":"weak supervision" OR "All Metadata":"label noise" OR "All Metadata":"positive-unlabeled"
     OR "All Metadata":"label proportions" OR "All Metadata":"partial label" OR "All Metadata":"semi-supervised")
```
ACM DL (Advanced search, trường Abstract):
```
[Abstract: tabpfn] OR [Abstract: "tabular foundation model"]
AND ([Abstract: "weak supervision"] OR [Abstract: "noisy labels"] OR [Abstract: "positive-unlabeled"] OR [Abstract: "label proportions"] OR [Abstract: "partial label"])
```

## 4. Google Scholar: snowballing xuôi ("Cited by")

Mở trang "Cited by" của từng bài dưới đây, lọc *Since 2025*, đọc tiêu đề (và abstract nếu cần):

| Bài hạt giống | Cách tìm |
|---|---|
| A General Framework for Learning from Weak Supervision (Chen et al., ICML 2024) | Tìm tiêu đề → "Cited by" |
| Imprecise Label Learning (Chen et al., NeurIPS 2024) | Như trên |
| In-Context Positive-Unlabeled Learning (Liu et al., 2026) | Như trên (có thể chưa có trích dẫn) |
| A Unified Approach to Count-Based Weakly-Supervised Learning (Shukla et al., NeurIPS 2023) | Như trên |
| Accurate predictions on small data with a tabular foundation model (Hollmann et al., Nature 2025) | "Cited by" → **Search within citing articles**: `weak OR noisy OR "positive-unlabeled" OR proportions OR "partial label" OR "semi-supervised"` |

Thêm một bài chưa đọc được do arXiv giới hạn tốc độ: **arXiv 2610.23765** (*Universal Semantic Enhancement for Weakly Supervised Anomaly Detection on Tabular Data*). Mở và ghi lại: bài có dùng TFM không, có không cần huấn luyện không.

## 5. Bảng ghi kết quả

| Ngày | Nguồn | Truy vấn (mã ở trên) | Số kết quả | Số bài thoả tiêu chí 1 hoặc 4 | Tên bài và link | Ghi chú |
|---|---|---|---|---|---|---|
| | Scopus | §1 truy vấn 1 | | | | |
| | Scopus | §1 truy vấn 2 | | | | |
| | WoS | §2 | | | | |
| | IEEE Xplore | §3 | | | | |
| | ACM DL | §3 | | | | |
| | Google Scholar | §4 GLWS | | | | |
| | Google Scholar | §4 ILL | | | | |
| | Google Scholar | §4 In-Context PU | | | | |
| | Google Scholar | §4 Count-based | | | | |
| | Google Scholar | §4 TabPFN Nature, lọc trong bài trích dẫn | | | | |
| | arXiv | 2610.23765 | — | | | |

**Quy tắc quyết định:** nếu có bất kỳ bài nào dùng TFM có sẵn để học từ ≥2 loại nhãn yếu bằng suy luận trên nhãn ẩn, áp dụng tiêu chí dừng của Mốc 1 (thu hẹp hoặc dừng) và báo lại để cập nhật [04_NOVELTY_VERDICT.md](04_NOVELTY_VERDICT.md).
