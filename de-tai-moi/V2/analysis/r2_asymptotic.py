"""r2_asymptotic.py — V2 R2 (05/11/2026, thăm dò): giới hạn n → ∞ của độ lệch số đếm Ã_c/ã − A_c/a trong Mệnh đề 5(c)
(W1_R1A_REPORT §5.7), mô hình Dirichlet–categorical C lớp, khối 'all', lịch khối ngẫu nhiên.

Khai triển Taylor bậc hai theo l ~ siêu bội (số điểm có nhãn trong khối); mô-men bậc ba của l chỉ góp O(n⁻²) tương
đối. Kết quả in ra: α(1 − λ)(1 − C·p_c)/(λ(K − 1)) (đơn vị: số đếm), tức sai lệch dự đoán α(1−λ)(1−Cp_c)/(λ(K−1)n).
Dạng đóng chính xác (không tiệm cận): E_π̃ k_c = U(L_c + α′)/(n_lab + Cα′), α′ = αK(n−1)/((K−1)n) — xem
W1_REVIEW_INDEPENDENT_02, Phụ lục A, và kiểm số ở G4/results/r1a/e7f_alpha_prime_check.csv.
Chạy: python V2/analysis/r2_asymptotic.py   (cần sympy)
"""
import sympy as sp

e = sp.symbols("e", positive=True)                       # e = 1/n
K, lam, pc, al, C = sp.symbols("K lambda p_c alpha C", positive=True)
n = 1 / e
M = n / K
n_lab = lam * n
U = n - n_lab
Lc = pc * n_lab
El = M * lam
Vl = M * lam * (1 - lam) * (n - M) / (n - 1)
u, l = sp.symbols("u l")


def E2(expr, var, mean, var2):
    return expr.subs(var, mean) + sp.Rational(1, 2) * sp.diff(expr, var, 2).subs(var, mean) * var2


A = E2(u * (Lc + al) / (n - u + C * al), u, M - El, Vl)
a = E2(u / U - u * (1 - u / U) / (n - u + C * al), u, M - El, Vl)
At = E2((M - l) * (Lc - l * Lc / n_lab + al) / (n - M + C * al), l, El, Vl)
at = E2(u / U - u * (1 - u / U) / (n - M + C * al), u, M - El, Vl)
print(sp.factor(sp.simplify(sp.limit(sp.simplify(At / at - A / a), e, 0))))
