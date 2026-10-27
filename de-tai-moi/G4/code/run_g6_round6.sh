#!/usr/bin/env bash
# G6 lượt 6 (GPU T4, khoảng 2 giờ; CPU song song): ĐO THỜI GIAN VÀ BỘ NHỚ ở cỡ của tập đánh giá bằng dữ liệu TỔNG HỢP
# (run_g7.py --source synthetic --synthetic_shape), để thay ngoại suy trong G6_ROUND45_REPORT Mục 5 bằng số đo và bắt lỗi
# hết bộ nhớ trước G7. Không tải, không chạm tập đánh giá; số metric trên dữ liệu tổng hợp không dùng cho kết luận nào.
# Cỡ mô phỏng 4 dataset lớn nhất của tập đánh giá (sau loại trùng):
#   31 → 5395 × 5      (phoneme: n lớn nhất của nhóm nhị phân)
#   37 → 4210 × 57     (spambase: d lớn nhất trong các bộ nhị phân lớn)
#   54 → 6430 × 36, 6 lớp   (satimage: n lớn nhất của nhóm đa lớp)
#   36 → 1593 × 256, 10 lớp (semeion: d lớn nhất, 10 lớp)
# Chạy từ thư mục chứa code:  bash run_g6_round6.sh /kaggle/working
set -u
OUT=${1:-/kaggle/working}
if [ ! -d "$OUT/PRODEN" ]; then
  git clone -q https://github.com/Lvcrezia77/PRODEN "$OUT/PRODEN" && git -C "$OUT/PRODEN" checkout -q 637b531a627a149ca494f966f4f41a20ad235853
fi
export PRODEN_REPO="$OUT/PRODEN"
SHAPE="31=5395x5;37=4210x57;54=6430x36x6;36=1593x256x10"
SYN=(--split dev --source synthetic --synthetic_shape "$SHAPE" --seeds 0)
ALLB=(--settings llp,pu_sar,pu_scar,ssl,noisy --levels "llp=8;pu_sar=0.1;pu_scar=0.1;ssl=0.05;noisy=0.2")
echo "bắt đầu: $(date)"
pip freeze > "$OUT/g6r6_pip_freeze.txt" 2>&1   # ghi phiên bản thư viện (để chốt phiên bản cho G7)
# baseline trên CPU (3 tiến trình nền, mỗi tiến trình 1 luồng): bộ nhị phân 37 (mọi bối cảnh) và bộ đa lớp 54 (PLL)
for i in 0 1 2; do
  ( OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ICGIBBS_DEVICE=cpu python run_g7.py --part baselines "${SYN[@]}" --datasets 37 \
      "${ALLB[@]}" --shard $i/3 --out "$OUT/g6r6_base_b_$i.csv" > "$OUT/g6r6_base_b_$i.log" 2>&1
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ICGIBBS_DEVICE=cpu python run_g7.py --part baselines "${SYN[@]}" --datasets 54 \
      --settings pll --levels "pll=0.3" --shard $i/3 --out "$OUT/g6r6_base_m_$i.csv" > "$OUT/g6r6_base_m_$i.log" 2>&1 ) &
done
# phần TFM trên GPU (tuần tự): bộ 37 mọi bối cảnh nhị phân; bộ 31 chỉ LLP; hai bộ đa lớp PLL q = 0,3
python run_g7.py --part tfm "${SYN[@]}" --datasets 37 "${ALLB[@]}" --backend tabpfn_v2 \
  --out "$OUT/g6r6_tfm.csv" > "$OUT/g6r6_tfm_37.log" 2>&1
python run_g7.py --part tfm "${SYN[@]}" --datasets 31 --settings llp --levels "llp=8" --backend tabpfn_v2 \
  --out "$OUT/g6r6_tfm.csv" > "$OUT/g6r6_tfm_31.log" 2>&1
python run_g7.py --part tfm "${SYN[@]}" --datasets 54,36 --settings pll --levels "pll=0.3" --backend tabpfn_v2 \
  --out "$OUT/g6r6_tfm.csv" > "$OUT/g6r6_tfm_pll.log" 2>&1
echo "xong phần TFM: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv > "$OUT/g6r6_gpu.txt" 2>&1
wait
echo "xong phần baseline: $(date)"
echo "kết thúc: $(date)"
