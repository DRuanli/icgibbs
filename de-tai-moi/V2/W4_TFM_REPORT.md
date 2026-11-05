# W4: chọn hai TFM mới — giấy phép, khả năng chạy trên T4, nhiễm dữ liệu (05/11/2026)

- **Căn cứ:** V2_PLAN §4 (W4) và quyết định V-Q3, thầy duyệt 04/11. V-Q3 chọn thêm 2 TFM: một TFM họ TabPFN mới, và một TFM học trên dữ liệu thật.
- **Tính chất:** tra cứu và viết code bọc. **Chưa chạy trên GPU.** Mọi con số về tốc độ trên T4 đều chờ lượt thử (§4).
- **Lưu ý:** em không phải luật sư. Phần giấy phép tóm tắt từ trang của nhà phát hành; thầy nên đọc bản gốc khi bấm chấp nhận.

## 0. Tóm tắt và đề xuất

1. **TFM họ TabPFN mới: TabPFN-3**, hoặc TabPFN-3.5 nếu lượt thử cho thấy đủ nhanh (quy tắc chọn ở §4).
   - Cả hai đều chạy được bằng gói `tabpfn` 9.2.0 mà ta đang dùng: `ModelVersion.V3` và `ModelVersion.V3_5`.
   - Cả hai hỗ trợ tới 160 lớp mà không cần ECOC.
   - Cả hai học trên dữ liệu tổng hợp. Báo cáo kỹ thuật của 3.5 không nhắc tới bước tinh chỉnh trên dữ liệu thật.
   - **Giấy phép phi thương mại.** Nghiên cứu và đánh giá được phép; dùng cho mục đích thương mại hay sản xuất thì cấm.
2. **TFM học trên dữ liệu thật: TabDPT** (Layer6), gói `tabdpt` 1.3.1, code Apache-2.0.
   - Chạy được trên T4 khi tắt FlashAttention; bộ bọc đã tự làm việc này.
   - Không giới hạn số lớp: TabDPT mã hoá nhãn theo cơ số C_max và tốn ⌈log_{C_max} C⌉ lượt forward.
3. **Phát hiện quan trọng: nhiễm dữ liệu.**
   - TabDPT được huấn luyện trên **123 dataset OpenML**, với nhãn thật. Trong đó có **13/65 dataset ta đã dùng**, tất cả thuộc Study 2: 24, 312, 375, 1466, 1476, 23380, 40536, 41143, 41145, 41146, 41156, 45026, 45039. Study 1 không trùng dataset nào.
   - Với bài toán nhãn yếu, đây là rò rỉ trực tiếp, vì TFM đã thấy nhãn thật của chính dataset đó.
   - **Đề xuất cho Study 3:** loại cả 123 dataset này, cùng các bản trùng của chúng, khỏi tập ứng viên.
   - **Với tập phát triển:** kết quả TabDPT trên 13 dataset trên chỉ được báo cáo tách riêng, ghi rõ là bị nhiễm.
4. **Code:** `run_g7.py` có thêm ba backend `tabpfn_v3`, `tabpfn_v3_5` và `tabdpt`, kèm test với mô hình giả (`tests/test_w4_backends.py`).
   - Với TabDPT: mỗi lượt Gibbs dùng 1 lượt forward; dự đoán cuối dùng ensemble 8, theo cấu hình gốc của bài TabDPT.
5. **Việc của thầy trước lượt thử:**
   - Đăng nhập ux.priorlabs.ai bằng đúng tài khoản sinh `TABPFN_TOKEN`.
   - Ở tab License, chấp nhận giấy phép **TabPFN-3** và **TabPFN-3.5**.
   - Token hiện tại có thể chỉ mở được trọng số v2.

## 1. So sánh các ứng viên

| | TabPFN v2 (đang dùng) | TabPFN-2.5 / 2.6 | **TabPFN-3** | **TabPFN-3.5** | **TabDPT 1.3.1** |
|---|---|---|---|---|---|
| Gọi trong code | `tabpfn_v2` | `tabpfn_v2_5` | `tabpfn_v3` | `tabpfn_v3_5` | `tabdpt` |
| Giấy phép trọng số | Prior Labs License (Apache 2.0 kèm yêu cầu ghi công) | phi thương mại | phi thương mại (tabpfn-3-license-v1.0) | phi thương mại | trang Hugging Face ghi apache-2.0, kèm nhãn "modified-mit"; repo huấn luyện có giấy phép riêng, dễ dãi |
| Giấy phép code | Apache 2.0 (từ `tabpfn` 9.0.0) | như trái | như trái | như trái | Apache-2.0 |
| Số lớp tối đa | 10 | 10 | 160 | 160 | không giới hạn (mã hoá theo cơ số) |
| Dữ liệu huấn luyện | tổng hợp | 2.5 mặc định: tinh chỉnh trên dữ liệu thật | tổng hợp | tổng hợp (theo báo cáo kỹ thuật) | **123 dataset OpenML thật** |
| Tham số | — | — | — | khoảng 220 triệu; có thể chậm gấp đôi v3 khi train lớn | 78 triệu |
| T4 | đã chạy | — | chưa rõ, cần thử | chưa rõ, cần thử | được, nếu tắt FlashAttention (FA cần compute capability ≥ 8) |

**Vì sao không chọn TabPFN-2.5 hay 2.6:**
- chỉ hỗ trợ 10 lớp, nên không phủ được dataset nhiều lớp của Study 3;
- bản 2.5 mặc định đã tinh chỉnh trên dữ liệu thật, nên vướng cùng vấn đề nhiễm như TabDPT nhưng không công bố danh sách dataset.

## 2. Nhiễm dữ liệu

- **Nguồn danh sách:** tệp `data_splits/noleak_training_datasets.csv` của repo `layer6ai-labs/tabdpt-training`, commit `af0340c`. Tệp có 123 ID, đúng bằng số dataset bài TabDPT nêu. Code huấn luyện của repo đọc đúng tệp này.
  - Bản sao trong repo của ta: `V2/data/tabdpt_pretrain_ids.csv`.
  - Danh sách in trong PDF của bài bị vỡ bảng, nên em dùng tệp CSV.
- **Trùng với dữ liệu đã dùng:**

| Nghiên cứu | Số dataset | Trùng với tập huấn luyện của TabDPT |
|---|---|---|
| Study 1 và tập phát triển (G7) | 26 | 0 |
| Study 2 | 42 | 13: 24, 312, 375, 1466, 1476, 23380, 40536, 41143, 41145, 41146, 41156, 45026, 45039 |

- **Chưa kiểm:** các bản trùng gần, tức cùng dữ liệu nhưng khác ID trên OpenML. Ở Study 3, phép loại trùng bằng mã băm bảng của S2 phải chạy cả trên 123 dataset này.
- **TabPFN-3 và 3.5:** không công bố dùng dữ liệu thật khi huấn luyện. Báo cáo của 3.5 có nhắc rằng một số dataset được dùng để phát triển TabPFN-2 và TabICLv2. Đây là nhiễm qua việc chọn mô hình, nhẹ hơn nhiều, và cũng áp dụng cho hai TFM ta đang dùng.
- **Hệ quả cho bài:** H7 của Study 3 (bền với TFM) chỉ có nghĩa nếu không TFM nào đã thấy nhãn thật của dataset. Vì vậy:
  - Study 3 loại dataset trong tập huấn luyện của mọi TFM có công bố danh sách.
  - Phần Limitations ghi rằng TFM không công bố danh sách thì không kiểm được.

## 3. Code đã viết

- **`run_g7.make_estimator`:**
  - `tabpfn_v3` gọi `TabPFNClassifier.create_default_for_version(ModelVersion.V3, ignore_pretraining_limits=True, n_estimators=…)`; `tabpfn_v3_5` tương tự với `ModelVersion.V3_5`.
  - `tabdpt` dùng lớp `TabDPTAdapter`.
- **`TabDPTAdapter`:**
  - Giữ một `TabDPTClassifier` trong bộ nhớ cho mỗi tiến trình, vì trọng số được nạp ở `__init__`.
  - Mã hoá nhãn ngữ cảnh về 0..k−1 và trả `classes_`, vì TabDPT không tự làm hai việc này.
  - Chỉ bật FlashAttention khi compute capability ≥ 8. Tắt `torch.compile`.
  - Dùng toàn bộ ngữ cảnh (`context_size=None`); giới hạn ngữ cảnh do `max_context` của bộ lấy mẫu đặt.
  - Với n_ensembles = 1: một lượt forward, không hoán vị, tất định. Với n_ensembles > 1: `ensemble_predict_proba(seed=0)`.
- **Test** (`tests/test_w4_backends.py`, 3 test, dùng mô hình giả):
  - mã hoá nhãn khi ngữ cảnh vắng lớp;
  - mô hình chỉ nạp một lần;
  - đi đúng nhánh đơn hoặc ensemble;
  - cột được đặt đúng lớp bên trong `SklearnTFMConditional`.
- **Cài đặt trên Kaggle cho lượt sau:** thêm `tabdpt==1.3.1` vào lệnh pip. Gói này kéo theo `faiss-cpu`, `omegaconf` và `torch>=2.6`.
  - Phải kiểm bản torch có sẵn trên Kaggle để pip không cài lại torch.
  - Trọng số TabDPT tải từ Hugging Face (`Layer6/TabDPT`, tệp `tabdpt1_3.safetensors` với gói 1.3.1), nên cần Internet On.

## 4. Lượt thử GPU đề xuất (chưa viết code chạy, chờ thầy đồng ý)

- **Mục đích:**
  - đo thời gian và bộ nhớ mỗi lượt gọi TFM trên T4;
  - kiểm tính tất định;
  - đo ánh xạ phản ứng trên vài dataset của tập phát triển cho ba TFM mới.
- **Cấu hình đo thời gian:** ngữ cảnh n ∈ {500, 2.000}, số lớp C ∈ {2, 10, 26}, dữ liệu tổng hợp (`--source synthetic --synthetic_shape`, đã có từ G6).
- **Quy tắc chọn giữa TabPFN-3 và 3.5, ghi trước:** chọn **3.5** nếu thời gian trung vị mỗi lượt gọi ở n = 2.000 không quá 1,5 lần của v3; ngược lại chọn **v3**.
- **Lý do của quy tắc:** chuỗi Gibbs gọi TFM vài trăm lần mỗi ô, nên tốc độ quyết định chi phí của Study 3 (V2_PLAN §5).
- **Ước tính:** dưới 1 phiên GPU. Có thể ghép vào lượt GPU kế tiếp, cùng với Φ_struct nếu thầy duyệt.

## Nguồn

- [Prior Labs: danh sách mô hình và giấy phép](https://docs.priorlabs.ai/models)
- [TabPFN README (chọn phiên bản, token, giấy phép)](https://cdn.jsdelivr.net/gh/PriorLabs/TabPFN@main/README.md)
- [TabPFN-2.5 trên Hugging Face](https://huggingface.co/Prior-Labs/tabpfn_2_5)
- [TabPFN-3.5: Technical Report](https://arxiv.org/pdf/2609.17895)
- [TabPFN-3: Technical Report](https://arxiv.org/pdf/2605.13986)
- [TabDPT: Scaling Tabular Foundation Models on Real Data](https://arxiv.org/pdf/2410.18164v3)
- [tabdpt trên PyPI](https://pypi.org/project/tabdpt/)
- [TabDPT-inference (GitHub)](https://github.com/layer6ai-labs/TabDPT-inference)
- [Thẻ mô hình TabDPT trên Hugging Face](https://huggingface.co/Layer6/TabDPT/commit/34e2b7dfbee1e3846781472aa612bb68bfefe27c)
- [AutoGluon: cách bọc TabDPT (tắt flash trên compute capability 7.5)](https://auto.gluon.ai/stable/_modules/autogluon/tabular/models/tabdpt/tabdpt_model.html)
- Repo `layer6ai-labs/tabdpt-training`, commit `af0340c`, tệp `data_splits/noleak_training_datasets.csv`.
- **Trước khi trích dẫn trong bài:** ba báo cáo (TabPFN-3, TabPFN-3.5, TabDPT) phải được thêm vào G9/REFERENCE_CHECK.
