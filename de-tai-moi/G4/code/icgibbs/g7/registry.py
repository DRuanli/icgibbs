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

DEV_BINARY = {31: "credit-g", 37: "diabetes", 1464: "blood-transfusion-service-center",
              1462: "banknote-authentication", 1480: "ilpd", 1510: "wdbc"}
DEV_MULTICLASS = {54: "vehicle", 36: "segment", 23: "cmc"}

assert not (set(EVAL_BINARY) | set(EVAL_MULTICLASS)) & (set(DEV_BINARY) | set(DEV_MULTICLASS)), "tập phát triển và tập đánh giá phải rời nhau"
