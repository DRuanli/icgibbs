# 08 · Tái lập và mã nguồn (G6–G7, G11–G12)

[← Về README](README.md)

## 1. Thuật ngữ (ACM Artifact Review and Badging v1.1)

| Thuật ngữ | Ai làm | Dùng gì | Câu hỏi |
|---|---|---|---|
| Repeatability | Cùng nhóm | Cùng thiết lập | Nhóm tác giả chạy lại có ra cùng kết quả không? |
| Reproducibility | Nhóm khác | Artifact của tác giả | Người khác dùng code và dữ liệu của tác giả có ra cùng kết quả không? |
| Replicability | Nhóm khác | Tự cài đặt lại | Người khác tự làm lại từ mô tả trong bài có ra kết luận tương tự không? |

Mục tiêu tối thiểu của mọi dự án: **reproducibility**. Một người lạ, từ một clone sạch và các lệnh trong README, tái tạo được mọi bảng và hình chính.

## 2. Cấu trúc repo chuẩn

```
project/
├── README.md               mô tả, cài đặt, bảng kết quả kèm lệnh tái tạo từng dòng
├── LICENSE
├── CITATION.cff            cách trích dẫn
├── environment.yml / requirements.lock   phiên bản cố định (pin)
├── Dockerfile              (khuyến nghị) môi trường đóng gói
├── configs/                một file YAML cho mỗi thí nghiệm; không hard-code siêu tham số
├── src/<package>/          mã nguồn dạng thư viện, thiết kế hướng đối tượng
│   ├── data/               Dataset, Splitter, Preprocessor
│   ├── models/             BaseModel và các cài đặt cụ thể
│   ├── baselines/          adapter cho code chính thức của baseline
│   ├── evaluation/         Metric, StatisticalAnalysis
│   └── experiment.py       Experiment: đọc config → chạy → ghi kết quả thô
├── scripts/                điểm vào: download_data.py, run_experiment.py, make_tables.py, make_figures.py
├── tests/                  unit test, known-answer test, kiểm chứng số
├── results/raw/            kết quả thô (không sửa tay); hoặc chỉ lưu đường dẫn tới kho lưu trữ
├── analysis/               notebook và script phân tích, đọc từ results/raw/
└── docs/                   các template của sổ tay đã điền
```

**Thiết kế hướng đối tượng:** mỗi khái niệm của giao thức (dataset, split, mô hình, metric, thí nghiệm) là một lớp có giao diện thống nhất. Thêm một baseline hay dataset mới chỉ cần thêm một lớp, không sửa vòng thí nghiệm.

## 3. Kiểm thử (G6)

Theo thứ tự:
1. **Shape và dtype:** assert ở ranh giới hàm.
2. **Đáp án biết trước:** ví dụ tính tay được (ví dụ tập itemset của một CSDL 5 giao dịch).
3. **Kiểm chứng số:** gradient check bằng sai phân hữu hạn (sai số tương đối dưới 1e-6); so với thư viện chuẩn (scikit-learn, scipy, statsmodels, SPMF); khôi phục tham số trên dữ liệu tổng hợp có đáp án.
4. **Bất biến:** tính chất mà bài tuyên bố (bất biến hoán vị, đơn điệu, tính đầy đủ của cắt tỉa) có test riêng.
5. **Hồi quy (regression test):** lưu kết quả của một cấu hình nhỏ cố định; mọi thay đổi code phải giữ nguyên kết quả đó, hoặc có giải thích.

## 4. Ngẫu nhiên và môi trường

- Đặt seed cho mọi nguồn ngẫu nhiên (Python, NumPy, PyTorch, hoặc framework tương ứng), và ghi seed vào log.
- GPU có thể không tất định. Bật chế độ tất định khi cần, hoặc báo cáo độ dao động giữa các lần chạy cùng seed.
- Pin phiên bản mọi thư viện. Ghi phiên bản Python, CUDA, driver, và phần cứng (CPU, GPU, RAM) cho mỗi lần chạy.
- Dữ liệu: lưu checksum (SHA-256) của file dữ liệu đã tải, và ghi phiên bản hoặc ID nguồn.

## 5. Checklist độ đầy đủ của code ML (ML Code Completeness Checklist)

Đi kèm hướng dẫn phát hành code cho NeurIPS 2020:
1. Đặc tả phụ thuộc (dependencies).
2. Code huấn luyện.
3. Code đánh giá.
4. Mô hình đã huấn luyện (nếu có).
5. README có bảng kết quả và **lệnh chính xác** để tạo ra từng kết quả.

## 6. Lưu trữ và huy hiệu artifact

- Khi nộp double-blind: đưa code lên kho ẩn danh, ví dụ anonymous GitHub hoặc file đính kèm.
- Khi được chấp nhận: phát hành bản cố định (release tag) và **lưu trữ có DOI** trên Zenodo hoặc kho của cơ quan. Liên kết DOI trong bài.
- Huy hiệu ACM (khi venue có đánh giá artifact):
  - *Artifacts Available*: artifact được lưu trữ vĩnh viễn, truy xuất được.
  - *Artifacts Evaluated – Functional*: được tài liệu hoá, nhất quán, đầy đủ, chạy được.
  - *Artifacts Evaluated – Reusable*: chất lượng vượt mức tối thiểu, dễ tái sử dụng.
  - *Results Reproduced*: người khác tái tạo được kết quả chính bằng artifact của tác giả.
  - *Results Replicated*: người khác tái tạo được kết quả chính mà không dùng artifact của tác giả.
- Dữ liệu: công bố theo giấy phép cho phép. Nếu không được phép thì công bố script tải và tiền xử lý.

## 7. Kiểm tra từ clone sạch (G10)

Một người hoặc agent không tham gia viết code làm các bước sau:
1. `git clone` vào máy hoặc container mới.
2. Cài môi trường theo README.
3. Chạy `tests/`.
4. Chạy lệnh tái tạo **ít nhất một bảng chính**, hoặc một tập con đã ghi trước nếu toàn bộ quá tốn compute.
5. So sánh với số trong bản thảo. Sai lệch vượt ngưỡng đã ghi là lỗi Nghiêm trọng.
