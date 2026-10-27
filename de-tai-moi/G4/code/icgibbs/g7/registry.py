"""Danh sách dataset của tập đánh giá (G5/DATASET_CARDS.md, giao thức Mục 1.2) và tập phát triển (G4)."""

# ID OpenML → (tên, phiên bản OpenML ghi trong DATASET_CARDS)
EVAL_BINARY = {
    3: ("kr-vs-kp", 1), 29: ("credit-approval", 1), 44: ("spambase", 1), 50: ("tic-tac-toe", 1),
    1049: ("pc4", 1), 1489: ("phoneme", 1), 1494: ("qsar-biodeg", 1), 6332: ("cylinder-bands", 2),
    23381: ("dresses-sales", 2), 38: ("sick", 1),
}
EVAL_MULTICLASS = {
    11: ("balance-scale", 1), 12: ("mfeat-factors", 1), 28: ("optdigits", 1), 46: ("splice", 1),
    182: ("satimage", 1), 188: ("eucalyptus", 1), 458: ("analcatdata_authorship", 1), 469: ("analcatdata_dmft", 1),
    1475: ("first-order-theorem-proving", 1), 1497: ("wall-robot-navigation", 1), 1501: ("semeion", 1),
    40966: ("MiceProtein", 4), 40975: ("car", 3), 40982: ("steel-plates-fault", 3),
}
# Phân tích độ nhạy S1 (giao thức Mục 1.5): loại các bộ có cấu trúc nhóm/chuỗi hoặc bài toán xác định
S1_EXCLUDE = {40966, 1497, 11, 40975}

# Chẩn đoán R̂ (giao thức Mục 5: "2 chuỗi trên mẫu con 3 dataset"): quy tắc ghi trước — 3 dataset nhỏ nhất theo n
# (DATASET_CARDS) của mỗi nhóm
RHAT_EVAL = {"binary": [23381, 6332, 29], "multiclass": [11, 188, 469]}
RHAT_DEV = {"binary": [1510, 1480, 1464], "multiclass": [54, 23, 36]}

# Ablation (giao thức Mục 7; quy tắc ghi trước 27/10/2026, giao thức §11 deviation 11): "3 dataset nhỏ" và "3 dataset,
# 1 mức" dùng đúng danh sách của R̂; "3 dataset lớn nhất" = 3 dataset lớn nhất theo n sau loại trùng mỗi nhóm
# (docs/dataset_hashes.csv): phoneme 5395, spambase 4210, sick 3711; satimage 6430, first-order-theorem-proving 5636,
# optdigits 5620. Tập phát triển: banknote 1348, credit-g 1000, diabetes 768; segment 2086, cmc 1425, vehicle 846.
ABL_SMALL_EVAL, ABL_SMALL_DEV = RHAT_EVAL, RHAT_DEV
ABL_LARGE_EVAL = {"binary": [1489, 44, 38], "multiclass": [182, 1475, 28]}
ABL_LARGE_DEV = {"binary": [1462, 31, 37], "multiclass": [36, 23, 54]}

# RQ5 — PLL thật (G5/DATASET_CARDS Mục D): tên → (mã số dùng cho seed, SHA-256 của tệp gốc trên trang PALM)
REAL_PLL = {
    "Lost": (1, "ce4c1e0d5ad117e1b48f9f83805cba46f2a955ca5bdf75e578026ae4f6c7a960"),
    "MSRCv2": (2, "8734cbd7b2f96623bfa8a0ea037daa4111fbce3954f077bf6c53d23dfb5e7a3c"),
    "BirdSong": (3, "8c0c9b1828e523b6bd499242a94a360a55326c326fe31b788138b3537bc1c33f"),
}

DEV_BINARY = {31: "credit-g", 37: "diabetes", 1464: "blood-transfusion-service-center",
              1462: "banknote-authentication", 1480: "ilpd", 1510: "wdbc"}
DEV_MULTICLASS = {54: "vehicle", 36: "segment", 23: "cmc"}

assert not (set(EVAL_BINARY) | set(EVAL_MULTICLASS)) & (set(DEV_BINARY) | set(DEV_MULTICLASS)), "tập phát triển và tập đánh giá phải rời nhau"
