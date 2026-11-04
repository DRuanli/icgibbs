# In-Context Gibbs — học từ nhãn yếu bằng tabular foundation model, không huấn luyện lại

Kho riêng tư của đề tài A (đích NeurIPS 2027). Làm theo quy trình cổng G0–G12 trong `research-handbook/`.

| Thư mục | Nội dung |
|---|---|
| `research-handbook/` | Sổ tay nghiên cứu: quy trình cổng, thiết kế thực nghiệm, mẫu biểu |
| `de-tai-moi/` | Hồ sơ đề tài: chọn đề tài (G1), tính mới (G2), kế hoạch (G3), phương pháp và lý thuyết (G4), giao thức đóng băng (G5), cài đặt (G6), Study 1 (G7–G8), Study 2 (`S2/`), bài báo (G9, `paper/`) |
| `de-tai-moi/G4/code/` | Thư viện `icgibbs`, kiểm thử, script thí điểm chạy trên Kaggle |
| `de-tai-moi/G4/pilot/` | Kết quả thí điểm trên tập phát triển (exploratory) |
| `de-tai-moi/docs/` | Tài liệu đi kèm cho tái lập (khai báo dùng AI, tái hiện baseline, …) |

Trạng thái (04/11/2026):
- G1–G8 và cổng của Study 2 đã qua.
- Đang ở G9 (viết bài, `de-tai-moi/paper/`).
- Đã có nhận xét của reviewer ngoài 01, cùng kế hoạch sửa ở `de-tai-moi/G9/REVIEW_RESPONSE_PLAN.md`.
- **Bắt đầu từ `de-tai-moi/HANDOFF.md`**: trạng thái, con số chính, quy trình, ràng buộc, việc mở.

Chạy kiểm thử (CPU):

```bash
cd de-tai-moi/G4/code
pip install numpy scipy scikit-learn pandas torch pytest
python -m pytest -q tests
```

Mã truy cập TabPFN chỉ đặt qua biến môi trường `TABPFN_TOKEN` (Kaggle Secrets); không bao giờ ghi vào kho.
