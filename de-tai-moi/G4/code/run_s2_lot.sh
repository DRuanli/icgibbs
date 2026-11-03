#!/usr/bin/env bash
# Nghiên cứu 2 (S2) — chạy tập đánh giá MỚI theo de-tai-moi/S2/S2_PLAN.md (chỉ sau khi S2 đã đóng băng).
# Dùng:  bash run_s2_lot.sh <LOT> /kaggle/working        LOT ∈ {S2X S2D S2T S2C1 S2C2 S2G S2R}
#   S2X  (GPU T4 × 2): THĂM DÒ, chỉ trên dataset cũ của NC1 (S2_PLAN §5.1): ánh xạ phản ứng Φ(r) của TabPFN v2 (GPU 0) và
#                      TabICL v2 (GPU 1). Không chạm tập đánh giá S2, nên chạy được trước khi đóng băng
#   S2D  (CPU)       : bộ test + tải dữ liệu, mã băm, kiểm tiêu chí sau loại trùng (S2_PLAN §3.4) → s2_dataset_hashes.csv
#   S2T  (GPU T4 × 2): mọi phương pháp TFM với TabICL v2 (cấu hình F6, như lượt L6 của NC1), LLP túi 8 và PLL q 0,3/0,6
#   S2C1, S2C2 (CPU) : mọi baseline LLP và PLL (mặc định và tune), 8 mảnh, mỗi lượt 4 mảnh song song
#   S2G  (GPU T4 × 2): chỉ gas-drift (1476, dự phòng bổ sung theo S2_PLAN §3.4, lấy mẫu con 10.000 hàng): phương pháp TFM
#                      trên GPU 0 và mọi baseline PLL trên CPU của cùng phiên (3 mảnh song song). Cần bản code có 1476
#                      trong S2_MULTICLASS (s2_code_v3 trở đi)
#   S2R  (CPU)       : chạy lại một lần các ô lỗi của S2G (G7_PLAN §2): PRODEN mặc định trên gas-drift, đủ 6 ô. Cần
#                      s2_code_v4 trở đi
# Mỗi lượt chạy tiếp được: chạy lại đúng lượt với output cũ trong thư mục kết quả, các ô đã xong được bỏ qua.
set -u
LOT=${1:?thiếu tên lượt}
OUT=${2:-/kaggle/working}
EV=(--split s2 --eval-ok --settings llp,pll --levels "llp=8;pll=0.3,0.6")
if [ ! -d "$OUT/PRODEN" ]; then
  git clone -q https://github.com/Lvcrezia77/PRODEN "$OUT/PRODEN" && git -C "$OUT/PRODEN" checkout -q 637b531a627a149ca494f966f4f41a20ad235853
fi
export PRODEN_REPO="$OUT/PRODEN"
echo "lượt $LOT bắt đầu: $(date)"
pip freeze > "$OUT/s2_${LOT}_pip_freeze.txt" 2>&1
nvidia-smi --query-gpu=index,name,memory.total --format=csv > "$OUT/s2_${LOT}_gpu.txt" 2>&1
sha256sum $(find . -name "*.py" -o -name "*.sh" | sort) > "$OUT/s2_${LOT}_code_sha256.txt"
if ! python -m pytest -q tests > "$OUT/s2_${LOT}_pytest.log" 2>&1; then
  echo "TEST HỎNG — dừng lượt $LOT. Xem s2_${LOT}_pytest.log"; tail -5 "$OUT/s2_${LOT}_pytest.log"; exit 1
fi
tail -1 "$OUT/s2_${LOT}_pytest.log"

gpu () {   # gpu <số GPU> <tên> <đối số run_g7...>
  local g=$1 tag=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g python run_g7.py "${EV[@]}" "$@" --out "$OUT/s2_${tag}.csv" >> "$OUT/s2_${tag}.log" 2>&1
  echo "GPU$g xong $tag: $(date)"
}
cpu () {   # cpu <tên> <đối số run_g7...>
  local tag=$1; shift
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ICGIBBS_DEVICE=cpu python run_g7.py "${EV[@]}" --part baselines "$@" \
    --out "$OUT/s2_${tag}.csv" >> "$OUT/s2_${tag}.log" 2>&1
  echo "CPU xong $tag: $(date)"
}

case $LOT in
  S2X)
    ( CUDA_VISIBLE_DEVICES=0 python s2x_response_map.py --backends tabpfn_v2 --out "$OUT/s2x_tabpfn.csv" \
        >> "$OUT/s2x_tabpfn.log" 2>&1; echo "GPU0 xong s2x_tabpfn: $(date)" ) &
    ( CUDA_VISIBLE_DEVICES=1 python s2x_response_map.py --backends tabicl_v2 --out "$OUT/s2x_tabicl.csv" \
        >> "$OUT/s2x_tabicl.log" 2>&1; echo "GPU1 xong s2x_tabicl: $(date)" ) &
    wait ;;
  S2D)
    python g6_dataset_hashes.py --split s2 --out "$OUT/s2_dataset_hashes.csv" > "$OUT/s2_dataset_hashes.log" 2>&1
    echo "xong mã băm: $(date)"; tail -3 "$OUT/s2_dataset_hashes.log" ;;
  S2T)
    gpu 0 tfm_0 --part tfm --backend tabicl_v2 --shard 0/2 & gpu 1 tfm_1 --part tfm --backend tabicl_v2 --shard 1/2 & wait ;;
  S2G)
    python -c "from icgibbs.g7.registry import S2_MULTICLASS, S2_SUBSAMPLE; assert 1476 in S2_MULTICLASS and S2_SUBSAMPLE.get(1476) == 10000" \
      || { echo "BẢN CODE CŨ: registry chưa có gas-drift — dùng s2_code_v3.zip"; exit 3; }
    gpu 0 g_tfm --part tfm --backend tabicl_v2 --datasets 1476 &
    for j in 0 1 2; do cpu "g_base_$j" --datasets 1476 --shard $j/3 & done; wait ;;
  S2R)
    python -c "from icgibbs.g7.registry import S2_MULTICLASS, S2_SUBSAMPLE; assert 1476 in S2_MULTICLASS and S2_SUBSAMPLE.get(1476) == 10000" \
      || { echo "BẢN CODE CŨ: registry chưa có gas-drift — dùng s2_code_v4.zip"; exit 3; }
    cpu r_base --datasets 1476 --only proden_def ;;
  S2C1|S2C2)
    b=$(( (${LOT#S2C} - 1) * 4 ))
    for j in 0 1 2 3; do cpu "base_$((b+j))" --shard $((b+j))/8 & done; wait ;;
  *) echo "lượt không hợp lệ: $LOT"; exit 2 ;;
esac
echo "lượt $LOT kết thúc: $(date)"
