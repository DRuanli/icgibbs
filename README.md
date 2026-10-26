# In-Context Gibbs — học từ nhãn yếu bằng tabular foundation model, không huấn luyện lại

Kho riêng tư của đề tài A (đích NeurIPS 2027). Làm theo quy trình cổng G0–G12 trong `research-handbook/`.

| Thư mục | Nội dung |
|---|---|
| `research-handbook/` | Sổ tay nghiên cứu: quy trình cổng, thiết kế thực nghiệm, mẫu biểu |
| `de-tai-moi/` | Hồ sơ đề tài: chọn đề tài (G1), tính mới (G2), kế hoạch (G3), phương pháp và lý thuyết (G4), giao thức đã đóng băng (G5), cài đặt và kiểm chứng (G6) |
| `de-tai-moi/G4/code/` | Thư viện `icgibbs`, kiểm thử, script thí điểm chạy trên Kaggle |
| `de-tai-moi/G4/pilot/` | Kết quả thí điểm trên tập phát triển (exploratory) |
| `de-tai-moi/docs/` | Tài liệu đi kèm cho tái lập (khai báo dùng AI, tái hiện baseline, …) |

Trạng thái (26/10/2026): G1–G5 đã qua. Giao thức thực nghiệm đóng băng ngày 26/10/2026 (`de-tai-moi/G5/EXPERIMENT_PROTOCOL.md`, mã băm trong `de-tai-moi/G5/FREEZE_RECORD.md`). Đang ở G6 (`de-tai-moi/G6/G6_PLAN.md`).

Chạy kiểm thử (CPU):

```bash
cd de-tai-moi/G4/code
pip install numpy scipy scikit-learn pandas torch pytest
python -m pytest -q tests
```

Mã truy cập TabPFN chỉ đặt qua biến môi trường `TABPFN_TOKEN` (Kaggle Secrets); không bao giờ ghi vào kho.
