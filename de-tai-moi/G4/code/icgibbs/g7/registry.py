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

# ---------------------------------------------------------------------------------------------------------------------
# Nghiên cứu 2 (S2): tập đánh giá mới (de-tai-moi/S2/S2_DATASETS.md; chọn tất định bằng S2/s2_select.py từ kết quả đếm
# metadata và bảng rà tay; bậc 0 = đúng tiêu chí NC1). ID OpenML → (tên, phiên bản OpenML lúc đếm, 01/11/2026).
# Danh sách dưới đây là danh sách CUỐI (S2/s2_final.csv), sau bước kiểm dữ liệu S2D.
S2_BINARY = {
    24: ("mushroom", 1), 312: ("scene", 1), 470: ("profb", 1), 934: ("socmob", 2),
    40536: ("SpeedDating", 1), 41143: ("jasmine", 1), 41145: ("philippine", 1), 41146: ("sylvine", 1),
    41156: ("ada", 1), 44130: ("eye_movements", 7), 45026: ("heloc", 2), 45039: ("compas-two-years", 5),
    46911: ("Bank_Customer_Churn", 1), 46927: ("Fitness_Club", 1), 46930: ("hazelnut-spread-contaminant-detection", 1),
    46938: ("Is-this-a-good-customer", 2), 46940: ("Marketing_Campaign", 1), 46969: ("NATICUSdroid", 3),
}
S2_MULTICLASS = {
    181: ("yeast", 1), 375: ("JapaneseVowels", 1), 1466: ("cardiotocography", 1), 23380: ("cjs", 3),
    40498: ("wine-quality-white", 1), 46906: ("anneal", 18),
    46960: ("students_dropout_and_academic_success", 1), 46963: ("website_phishing", 1), 46980: ("MIC", 3),
    1476: ("gas-drift", 1),
}
# Sau lượt S2D (02/11/2026), áp S2_PLAN §3.4 bằng S2/s2_select.py --hashes (ra S2/s2_final.csv): bỏ irish (451, n = 470
# sau loại trùng) và maternal_health_risk (46941, n = 452) vì trượt n ≥ 500; nhóm đa lớp còn 9 nên bổ sung dự phòng 1,
# gas-drift (1476), lấy mẫu con 10.000 hàng phân tầng, seed 0, sau loại trùng và trước khi chia.
S2_SUBSAMPLE = {1476: 10000}
# Dự phòng đa lớp, đúng thứ tự dùng (S2_PLAN §3.4): chỉ dùng khi sau loại trùng nhóm đa lớp còn < 10 dataset
S2_RESERVE_MULTICLASS = [(1476, "gas-drift", 1), (42, "soybean", 1), (1515, "micro-mass", 2), (41163, "dilbert", 1),
                         (41164, "fabert", 1), (46933, "hiva_agnostic", 2)]
# Độ nhạy kiểu S1 (S2_PLAN §1): cấu trúc nhóm/chuỗi (gas-drift chỉ khi được dùng)
S2_S1_EXCLUDE = {40536, 44130, 375, 23380, 1476}
_S2_ALL = set(S2_BINARY) | set(S2_MULTICLASS) | {d for d, _, _ in S2_RESERVE_MULTICLASS}
_NC1_ALL = set(EVAL_BINARY) | set(EVAL_MULTICLASS) | set(DEV_BINARY) | set(DEV_MULTICLASS)
assert not _S2_ALL & _NC1_ALL, "tập đánh giá S2 phải rời mọi dataset của NC1"

# ---------------------------------------------------------------------------------------------------------------------
# Study 3, bản gọn (de-tai-moi/S3/STUDY3_PLAN.md, ghi trước 10/11/2026; rà tay ở S3/s3_review.csv).
# ID OpenML → (tên, phiên bản OpenML lúc đếm ở V1D, 09/11/2026). Tiêu chí tới bậc 3 của S2_PLAN §3.2.
S3_BINARY = {
    1038: ("gina_agnostic", 1), 1046: ("mozilla4", 1), 1467: ("climate-model-simulation-crashes", 1),
    1487: ("ozone-level-8hr", 1), 4134: ("Bioresponse", 1), 46916: ("coil2000_insurance_policies", 1),
    46924: ("E-CommereShippingData", 1), 46935: ("HR_Analytics_Job_Change_of_Data_Scientists", 1),
    46937: ("in_vehicle_coupon_recommendation", 1), 46947: ("online_shoppers_intention", 1),
    46950: ("polish_companies_bankruptcy", 1), 46956: ("seismic-bumps", 4),
}
S3_MULTICLASS = {
    42: ("soybean", 1), 307: ("vowel", 2), 1468: ("cnae-9", 1), 1478: ("har", 1), 1515: ("micro-mass", 2),
    4538: ("GesturePhaseSegmentationProcessed", 1), 40499: ("texture", 1), 46933: ("hiva_agnostic", 2),
}
S3_SUBSAMPLE = {d: 10000 for d in (1046, 46924, 46935, 46937, 46947, 1478)}     # n > 10.000 (STUDY3_PLAN §2)
S3_S1_FLAG = {307, 1467, 1478, 1487, 4538}                                      # cấu trúc nhóm/chuỗi
# TabPFN v2 chỉ chạy trong giới hạn công bố (d ≤ 500, C ≤ 10): STUDY3_PLAN §3
S3_TABPFN_V2 = sorted(set(S3_BINARY) - {1038, 4134}) + [4538]
_S3_ALL = set(S3_BINARY) | set(S3_MULTICLASS)
assert not _S3_ALL & (_NC1_ALL | set(S2_BINARY) | set(S2_MULTICLASS)), "Study 3 phải rời mọi dataset đã chạy"
