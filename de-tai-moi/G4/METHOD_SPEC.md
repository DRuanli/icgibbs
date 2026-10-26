# G4 · Đặc tả phương pháp In-Context Gibbs (bản dùng cho thí điểm)

> Trạng thái: **bản G4**. Mọi siêu tham số ở đây là mặc định cho thí điểm trên tập phát triển; giá trị dùng cho thực nghiệm chính được chốt ở G5 (EXPERIMENT_PROTOCOL), chỉ dựa trên tập phát triển. Lý thuyết đi kèm: [PROOFS.md](PROOFS.md). Kiểm chứng số: [RQ1_REPORT.md](RQ1_REPORT.md). Mã: `code/icgibbs/`.

## 1. Bài toán
- Vào: feature $X\in\mathbb R^{n\times d}$, quan sát yếu $W$ cùng mô hình quan sát $p(W\mid X,y)$ (Bảng 3), một TFM có sẵn $q(y_*\mid x_*,D)$, điểm test $X_*$.
- Ra: $\hat p(y_*\mid x_*)$ cho từng điểm test, cùng các mẫu $y^{(s)}\sim\pi(y)=p(y\mid X,W)$ (dùng cho chẩn đoán).
- Không cập nhật trọng số nào của TFM.

## 2. Thuật toán

```
InContextGibbs(q, weak, X):
  y ← khởi tạo hợp lệ (y_i ~ ℓ_i chuẩn hoá; LLP: hoán vị ngẫu nhiên khớp số đếm từng túi)
  for sweep = 1..T:
     if cấp mẫu:
        'single': for i in hoán_vị(F):  y_i ~ q(· | x_i, D_{-i}) · ℓ_i(·)                 # n forward/vòng
        'block' : chia hoán_vị(MỌI mẫu) thành K khối; for khối B:
                     ngữ cảnh = D_{-B};  với i ∈ B ∩ F: y_i ~ q(· | x_i, D_{-B}) · ℓ_i(·)       # K forward/vòng
                     [mh_correct: chấp nhận/loại cả khối bằng MH, tỉ số π qua Brook]          # +1 forward/nhãn đổi
     if LLP:
        'swap'      : lặp n/2 lần: chọn túi, chọn cặp (i,j) khác nhãn, đổi, chấp nhận min(1,R)   # 2 forward/đề xuất
        'bag-block' : chia các túi thành K nhóm; for nhóm G: ngữ cảnh = D_{-G};
                      mỗi túi: y_B ~ Π q(y_i | x_i, D_{-G}) · 1[counts = w_B]  (DP chính xác)   # K forward/vòng
                      [mh_correct như trên]
     cập nhật tham số phiền nhiễu (liên hợp hoặc MH; không cần TFM)
     nếu sweep > burn-in: lưu y
  return các mẫu

Dự đoán quy nạp (mặc định):  p̂(y*|x*) = (1/S) Σ_s q_final(y* | x*, (X, y^(s)))     # S forward (S = 10)
Dự đoán chuyển nạp:          đưa X_* vào chuỗi như mẫu tự do (ℓ ≡ 1); p̂ = trung bình các xác suất có
                             điều kiện tại X_* trong lúc chạy chuỗi (không tốn forward thêm).
                             Chính xác (Định lý 1c.3) chỉ với sơ đồ 'single'; với sơ đồ khối, xác suất ghi
                             tại X_* dùng ngữ cảnh đã bỏ cả khối nên là xấp xỉ.
```

## 3. Mô hình quan sát (`icgibbs/weak.py`)

| Loại | $\ell_i(k)$ | Tham số phiền nhiễu | Cập nhật | Mẫu cố định |
|---|---|---|---|---|
| PU-SCAR | $s_i=1$: $(0,c)$; $s_i=0$: $(1,1-c)$ | $c$ (đã biết, hoặc $\mathrm{Beta}(a,b)$) | $c\mid y\sim\mathrm{Beta}(a+L,b+P-L)$ | mẫu $s=1$ |
| PU-SAR | như trên với $e(x_i)=\sigma(\eta^\top[1,z_i])$ | $\eta\sim N(0,\sigma^2I)$ | random-walk MH, 5 bước/vòng | mẫu $s=1$ |
| PLL | $\mathbb 1[k\in S_i]$ | — | — | $|S_i|=1$ |
| Nhãn nhiễu | $T_{k,w_i}$ | $T$ (đã biết, hoặc Dirichlet theo hàng: chéo 10, ngoài chéo 1) | $T_{k\cdot}\mid y\sim\mathrm{Dir}(\beta_k+N_{k\cdot})$ | — |
| SSL | không nhãn: $1$; có nhãn: $\mathbb 1[k=w_i]$ | — | — | mẫu có nhãn |
| LLP | ràng buộc $\mathrm{counts}(y_{B_g})=w_g$ | — | — | — |

## 4. Chi phí

| Sơ đồ | Forward mỗi vòng quét | Chính xác khi $q=p$? | Dùng khi |
|---|---|---|---|
| single | $|F|$ | Có (Định lý 1a) | kiểm chứng; $n$ nhỏ |
| block, K khối | $K$ | Không (Mệnh đề 4b) | **mặc định thực nghiệm** |
| block + MH | $K$ + số nhãn đổi | Có (Mệnh đề 4a) | phân tích độ nhạy |
| swap | $\approx n$ | Có (Định lý 1b) | kiểm chứng LLP |
| bag-block, K nhóm | $K$ | Không — chưa có chặn khác rỗng, chưa đo sai lệch (PROOFS G4-5, kiểm duyệt độc lập 26/10/2026) | **mặc định LLP** |
| bag-block + MH | $K$ + số nhãn đổi | Có | phân tích độ nhạy |

Với cấu hình thí điểm (T = 30, K = 10, S = 10): 300 forward với `n_estimators=1` + 10 forward với ensemble mặc định. Theo số đo G2 trên T4 (TabPFN v2, $n\le2000$): khoảng 0,14–0,26 s mỗi forward khối ⇒ 45–80 s mỗi lần chạy.

## 5. Quyết định thiết kế (mỗi quyết định có căn cứ)

| # | Quyết định | Căn cứ |
|---|---|---|
| D1 | **Chia khối trên mọi mẫu** (`block_over='all'`), không chỉ trên mẫu tự do. Mẫu cố định rơi vào khối bị loại khỏi ngữ cảnh nhưng không bị lấy mẫu lại | Chia chỉ trên mẫu tự do làm ngữ cảnh **lệch thành phần**: với PU, ngữ cảnh giữ mọi dương có nhãn nhưng bỏ bớt mẫu không nhãn ⇒ PPD bị đẩy về lớp dương. Chuỗi gộp một ô (E5b/E5c): ở $n=1000$, K = 10, lệch dự đoán 0,031 khi chia trên mẫu tự do, còn 0,00017 khi chia trên mọi mẫu (giảm theo $O(1/n)$). Cái giá: ở SSL không có lệch thành phần, thêm sai số $O(1/n)$ (0,0004 ở $n=1000$). Bài nhỏ có feature mang thông tin (E5a2): sai số dự đoán giảm khoảng 4 lần |
| D2 | Dự đoán **quy nạp** là mặc định; chuyển nạp là tuỳ chọn | Quy nạp hội tụ về $\bar p$, không phải Bayes chính xác khi $x$ mang thông tin (Định lý 1c, E2). Chuyển nạp chính xác nhưng tăng kích thước chuỗi thêm $|X_*|$. Chọn ở G5 bằng tập phát triển (việc mở G4-2) |
| D3 | `n_estimators=1` trong chuỗi; ensemble mặc định của gói khi dự đoán cuối | Ngân sách G2; mỗi bước Gibbs chỉ cần một mẫu, phương sai giảm nhờ trung bình qua mẫu |
| D4 | `max_context=2000`: ngữ cảnh lớn hơn được lấy mẫu con đều | Ngân sách G2; mẫu con đều không lệch thành phần (cùng lý do D1) |
| D5 | PU: $c$ **đã biết** trong so sánh chính (nnPU/uPU được cho biết tỉ lệ lớp, và với một tập huấn luyện thì $c$ và $\pi_p$ suy ra nhau) | Công bằng giữa các phương pháp. Bản học $c$ là phân tích phụ (PU-SCAR không định danh được; posterior của $c$ dựa vào prior của TFM) |
| D6 | **Khởi động ấm (mặc định từ 25/10/2026):** PU — $P_0(y{=}1\mid x)=\min(1,\hat p(s{=}1\mid x)/e(x))$ với $\hat p$ là TFM ngây thơ tính chéo 5 fold ($e$ = c đã biết; PU-SAR học η thì dùng propensity ban đầu của chuỗi); LLP — DLLP tuyến tính, mỗi túi lấy mẫu có ràng buộc số đếm từ $P_0$. Hard-EM xuất phát từ cùng điểm | Khởi tạo không đổi phân phối mục tiêu. Thí điểm lượt 2: xoá các ca sụp, giúp ở LLP túi 16 và PU-SCAR c = 0,1, không hại ở LLP túi 8 (PILOT_V2_REPORT §0) |
| D7 | 30 vòng quét, bỏ 10 vòng đầu, Rao–Blackwell với 10 mẫu cách đều | Ngân sách G2 (giao thức rút gọn); chẩn đoán $\hat R$ với ≥ 2 chuỗi làm ở G6 |
| D8 | Ngữ cảnh chỉ có một lớp ⇒ trả về xác suất làm trơn Laplace $(n_k+1)/(n+C)$ | TFM không fit được một lớp; tình huống hiếm (đầu chuỗi, túi rất thuần) |
| D9 | Sàn xác suất $10^{-6}$ cho đầu ra TFM | Giả định (A2); tránh $\log 0$ trong tỉ số MH |

## 6. Baseline và cách hiện thực RQ3

| Tên | Mô tả | Ghi chú |
|---|---|---|
| naive (TFM) | PU: không nhãn = âm, xác suất $\min(1,\hat p(s{=}1\mid x)/c)$ (Elkan & Noto 2008). LLP: một lần gán ngẫu nhiên khớp số đếm | cùng TFM, cùng ensemble cuối |
| hard-EM (TFM) | cùng vòng lặp khối, gán **argmax** có ràng buộc (LLP: top-k theo log-odds trong túi); **cùng điểm khởi động ấm với Gibbs**; dự đoán bằng trạng thái cuối | **Hiện thực "EM với posterior từng mẫu độc lập" của RQ3** (classification EM, Celeux & Govaert 1992). Đã quyết 25/10/2026 (GATE_REVIEW_G4 §3c, Q2) |
| SEM | cùng chuỗi Gibbs, dự đoán chỉ bằng mẫu cuối (stochastic EM, Celeux & Diebolt 1985) | Phân tích phụ của RQ3: SEM vs hard-EM tách tác dụng của lấy mẫu; Gibbs vs SEM tách tác dụng của lấy trung bình Bayes |
| nnPU / uPU | Kiryo et al. NeurIPS 2017 / du Plessis et al. ICML 2015; MLP 2×64, loss sigmoid, Adam 1e-3, 200 epoch, prior lớp đã biết; U = toàn bộ tập huấn luyện | **Cài lại cho thí điểm.** G5 quyết định dùng cài đặt chính thức và ngân sách tune bằng nhau |
| DLLP | Ardehaly & Culotta ICDM 2017; cùng MLP; CE giữa tỉ lệ túi và trung bình softmax | như trên |
| oracle | TFM với nhãn sạch | trần tham chiếu, không đưa vào so sánh |

## 6b. Metric (đã quyết 25/10/2026, GATE_REVIEW_G4 §3c, Q1)
- PU-SCAR: chính = log loss sau **hiệu chỉnh khớp tiên nghiệm lớp** (dịch logit một hằng số để trung bình xác suất dự đoán bằng tỉ lệ lớp đã biết; như nhau cho mọi phương pháp; không dùng nhãn test; không đổi AUC). Phụ: AUC (không kém naive, ngưỡng 0,01), log loss thô, Brier.
- PU-SAR, LLP: chính = ROC AUC (như đã đóng băng); phụ: log loss, Brier.
- RQ3: log loss (PU-SCAR: bản khớp tiên nghiệm).

## 6c. Phạm vi áp dụng (sau thí điểm, 25/10/2026)
- **Chế độ được khẳng định:** LLP với túi ≤ 16; PU-SAR. *(Cập nhật 26/10/2026: sau G6 lượt 2 chỉ còn LLP túi 8 là khẳng định chính, PU-SAR thành bối cảnh phụ — giao thức §11.)* Trong chế độ này chuỗi hội tụ trong 30 vòng quét (ba cách khởi tạo cho cùng kết quả ở túi 8).
- **Kiểm tra áp lực, báo cáo như giới hạn:** LLP túi 32 — đa mode, trộn chậm, phụ thuộc khởi tạo; phù hợp với Định lý 2 ($\tau$ lớn khuếch đại sai số của TFM).
- **Kết quả phụ:** PU-SCAR — TFM ngây thơ + Elkan–Noto với c đã biết gần tối ưu; Gibbs không cải thiện đo được (chờ thầy xác nhận ở G5).

## 7. Rủi ro đã biết (đưa vào kế hoạch thí điểm và G5)

1. **PU-SCAR và AUC.** Dưới SCAR, $p(s{=}1\mid x)=c\,p(y{=}1\mid x)$ nên điểm xếp hạng của TFM ngây thơ đã có cùng thứ tự với posterior thật (Elkan & Noto 2008, Lemma 1). Khi $n\to\infty$, AUC của naive và của bộ dự đoán đúng bằng nhau; lợi ích AUC của Gibbs so với naive ở PU-SCAR chỉ có thể đến từ hiệu ứng mẫu hữu hạn. Dự đoán "hơn TFM ngây thơ ≥ 0,03 AUC ở mức yếu nhất" của RQ2 vì vậy **khó đạt ở PU-SCAR**; log loss/accuracy (cần hiệu chỉnh) và PU-SAR (thứ tự bị bóp méo bởi $e(x)$) là nơi kỳ vọng lợi ích rõ hơn. RESEARCH_PLAN đã đóng băng; nếu đổi metric chính cho PU-SCAR thì phải ghi ở Mục 9 và quyết **trước** khi chạm tập đánh giá. Thí điểm ghi cả AUC, log loss, accuracy, Brier để thầy quyết ở G5.
2. **Lệch do khối.** D1 sửa nguồn lệch thành phần; phần còn lại (phụ thuộc trong khối) được đo ở RQ1_REPORT §5. Bản sửa MH là phương án dự phòng chính xác nhưng tốn hơn.
3. **Trộn chậm** khi nhãn yếu rất thông tin (túi LLP lớn, PU với $c$ nhỏ) hoặc khi TFM quá tự tin: chuỗi có thể kẹt ở mode xấu. Thí điểm ghi tỉ lệ nhãn đổi mỗi vòng (`flips_last`); G6 thêm nhiều chuỗi và $\hat R$.
4. **Prior ngoài phân phối.** Mệnh đề 3 chỉ có nghĩa dưới prior của TFM; với dữ liệu thật, bằng chứng duy nhất là thực nghiệm.

## 8. Bản đồ mã và kiểm thử

| Tệp | Nội dung |
|---|---|
| `icgibbs/conditional.py` | `ConditionalModel`; `AnalyticCellPPD`, `DiscreteMixturePPD` (PPD chính xác); `PerturbedConditional` (không tương thích); `SklearnTFMConditional` (TabPFN/TabICL) |
| `icgibbs/weak.py` | PU-SCAR, PU-SAR, PLL, nhãn nhiễu, SSL, LLP; `make_bags` |
| `icgibbs/llp.py` | lấy mẫu chính xác có ràng buộc số đếm (DP), top-k |
| `icgibbs/sampler.py` | `InContextGibbs` (4 sơ đồ, sửa MH, argmax, chuyển nạp), `fit_transductive` |
| `icgibbs/exact.py` | liệt kê, ma trận chuyển chính xác, phân phối dừng, hệ số Dobrushin, hằng số Mitrophanov |
| `icgibbs/baselines.py` | naive, hard-EM, nnPU/uPU, DLLP |
| `tests/test_exact.py` | 23 kiểm thử: Bổ đề 1; Định lý 1 cho mọi loại nhãn yếu và cả hai prior; cân bằng chi tiết; DP; Monte Carlo của cài đặt thật; quy nạp vs chuyển nạp; tính không tương thích |
| `experiments/rq1_exact.py`, `experiments/cell_pfn.py` | kiểm chứng RQ1 (E1–E5) |
| `kaggle_g4_pilot.py` | thí điểm trên 6 dataset phát triển (GPU Kaggle) |
