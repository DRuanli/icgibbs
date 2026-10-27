#!/usr/bin/env bash
# G6 lượt 7 (GPU T4 × 2, khoảng 2,5 giờ; CPU song song): TỔNG DUYỆT phần ablation của bộ chạy G7 (giao thức Mục 7,
# deviation 11), ablation cỡ dữ liệu và backend TabICL v2 trên TẬP PHÁT TRIỂN — cùng quy tắc chọn dataset như tập đánh
# giá, áp vào danh sách của tập phát triển. Dùng cả hai GPU (mỗi luồng một GPU qua CUDA_VISIBLE_DEVICES) để thử cách chạy
# song song cho G7. Không chạm tập đánh giá.
# Chạy từ thư mục chứa code:  bash run_g6_round7.sh /kaggle/working
set -u
OUT=${1:-/kaggle/working}
if [ ! -d "$OUT/PRODEN" ]; then
  git clone -q https://github.com/Lvcrezia77/PRODEN "$OUT/PRODEN" && git -C "$OUT/PRODEN" checkout -q 637b531a627a149ca494f966f4f41a20ad235853
fi
export PRODEN_REPO="$OUT/PRODEN"
echo "bắt đầu: $(date)"
pip freeze > "$OUT/g6r7_pip_freeze.txt" 2>&1
nvidia-smi --query-gpu=index,name,memory.total --format=csv > "$OUT/g6r7_gpu.txt" 2>&1
DEV=(--split dev --backend tabpfn_v2)

# luồng GPU 0: khởi tạo lạnh, số khối K, chuyển nạp; ablation cỡ dữ liệu (phần TFM)
( export CUDA_VISIBLE_DEVICES=0
  for A in init blocks trans; do
    python run_g7.py --part ablation --ablation $A "${DEV[@]}" --out "$OUT/g6r7_abl_$A.csv" > "$OUT/g6r7_abl_$A.log" 2>&1
    echo "GPU0 xong ablation $A: $(date)"
  done
  for M in 500 1000; do
    python run_g7.py --part tfm "${DEV[@]}" --n_train_max $M --out "$OUT/g6r7_size_tfm.csv" >> "$OUT/g6r7_size_tfm.log" 2>&1
  done
  echo "GPU0 xong cỡ dữ liệu: $(date)" ) > "$OUT/g6r7_gpu0_log.txt" 2>&1 &

# luồng GPU 1: số vòng quét, prior của η (PU-SAR); TabICL v2 cho hai bối cảnh chính; TabICL nhiều lớp (tổng hợp, 16 lớp)
( export CUDA_VISIBLE_DEVICES=1
  for A in sweeps prior; do
    python run_g7.py --part ablation --ablation $A "${DEV[@]}" --out "$OUT/g6r7_abl_$A.csv" > "$OUT/g6r7_abl_$A.log" 2>&1
    echo "GPU1 xong ablation $A: $(date)"
  done
  python run_g7.py --part tfm --split dev --backend tabicl_v2 --settings llp,pll --levels "llp=8;pll=0.3,0.6" \
    --out "$OUT/g6r7_tabicl.csv" > "$OUT/g6r7_tabicl.log" 2>&1
  echo "GPU1 xong TabICL: $(date)"
  python run_g7.py --part tfm --split dev --source synthetic --synthetic_shape "36=2000x40x16" --datasets 36 \
    --settings pll --levels "pll=0.3" --seeds 0 --backend tabicl_v2 --out "$OUT/g6r7_tabicl_manyclass.csv" \
    > "$OUT/g6r7_tabicl_manyclass.log" 2>&1
  echo "GPU1 xong TabICL nhiều lớp: $(date)" ) > "$OUT/g6r7_gpu1_log.txt" 2>&1 &

# CPU: đối thủ chính ở các cỡ dữ liệu của ablation cỡ dữ liệu
( for M in 500 1000; do
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ICGIBBS_DEVICE=cpu python run_g7.py --part baselines --split dev --n_train_max $M \
      --only dllplin_def,glwspll_tun --out "$OUT/g6r7_size_base.csv" >> "$OUT/g6r7_size_base.log" 2>&1
  done
  echo "CPU xong: $(date)" ) > "$OUT/g6r7_cpu_log.txt" 2>&1 &
wait
cat "$OUT/g6r7_gpu0_log.txt" "$OUT/g6r7_gpu1_log.txt" "$OUT/g6r7_cpu_log.txt"
echo "kết thúc: $(date)"
