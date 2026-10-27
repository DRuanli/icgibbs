#!/usr/bin/env bash
# G6 lượt 5 (GPU T4): tổng duyệt bộ chạy G7 (run_g7.py) trên TẬP PHÁT TRIỂN cho hai bối cảnh chính (LLP túi 8; PLL q = 0,3
# và 0,6), kèm (a) PROOFS G4-5: Gibbs bag-block có sửa MH so với bản xấp xỉ ở LLP túi 8; (b) chẩn đoán R̂ (2 chuỗi).
# Phần baseline chạy CPU song song trong lúc GPU chạy phần TFM. Không chạm tập đánh giá.
# Chạy từ thư mục chứa code:  bash run_g6_round5.sh /kaggle/working
set -u
OUT=${1:-/kaggle/working}
if [ ! -d "$OUT/PRODEN" ]; then
  git clone -q https://github.com/Lvcrezia77/PRODEN "$OUT/PRODEN" && git -C "$OUT/PRODEN" checkout -q 637b531a627a149ca494f966f4f41a20ad235853
fi
export PRODEN_REPO="$OUT/PRODEN"
COMMON=(--split dev --settings llp,pll --levels "llp=8;pll=0.3,0.6")
echo "bắt đầu: $(date)"
# baseline trên CPU (3 tiến trình nền, mỗi tiến trình 1 luồng)
for i in 0 1 2; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ICGIBBS_DEVICE=cpu python run_g7.py --part baselines "${COMMON[@]}" --shard $i/3 \
    --out "$OUT/g6r5_base_$i.csv" > "$OUT/g6r5_base_$i.log" 2>&1 &
done
# phần TFM trên GPU (tuần tự). Ghi chú 27/10/2026: khi chạy lượt 5, cờ --mh-check trần áp cho MỌI bối cảnh (LLP và PLL);
# từ sau lượt 5 cờ trần chỉ áp cho LLP — muốn chạy lại đúng như lượt 5 thì dùng --mh-check llp,pll
python run_g7.py --part tfm "${COMMON[@]}" --mh-check --backend tabpfn_v2 --out "$OUT/g6r5_tfm.csv" > "$OUT/g6r5_tfm.log" 2>&1
echo "xong phần TFM: $(date)"
python run_g7.py --part rhat "${COMMON[@]}" --backend tabpfn_v2 --out "$OUT/g6r5_rhat.csv" > "$OUT/g6r5_rhat.log" 2>&1
echo "xong phần R̂: $(date)"
wait
echo "xong phần baseline: $(date)"
echo "kết thúc: $(date)"
