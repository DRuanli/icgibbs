#!/usr/bin/env bash
# G6 lượt 9 (chỉ CPU, Internet: On; khoảng 1 giờ): T3 — RC, CC lệch số công bố ở Yeast và Dermatology ở lượt 8 khi chọn
# lr, weight decay MỘT lần trên lần chia đầu (validation 10% rất nhỏ: khoảng 130 và 30 mẫu). Phân tích độ nhạy ghi trước:
# chọn riêng cho từng lần chia (--select per_trial). Báo cáo cả hai cách; không thay số của lượt 8.
# Chạy từ thư mục chứa code:  bash run_g6_round9.sh /kaggle/working
set -u
OUT=${1:-/kaggle/working}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ICGIBBS_DEVICE=cpu
echo "bắt đầu: $(date)"
for M in rc cc; do
  for D in yeast dermatology; do
    python g6_reproduce_pll_uci.py --only $M --datasets $D --select per_trial --out "$OUT/g6r9_${M}_${D}.csv" \
      > "$OUT/g6r9_${M}_${D}.log" 2>&1 &
  done
done
wait
tail -n 4 "$OUT"/g6r9_*.log
echo "kết thúc: $(date)"
