#!/usr/bin/env bash
# G7 — chạy TẬP ĐÁNH GIÁ theo đúng giao thức đã đóng băng (G5 + deviation đến 13), chia thành các lượt Kaggle.
# Kế hoạch và thứ tự: de-tai-moi/G7/G7_PLAN.md. Mỗi lượt chạy tiếp được: hết giờ thì chạy lại ĐÚNG lượt đó với cùng
# thư mục kết quả (các ô đã xong được bỏ qua). Không xoá lần chạy hỏng; mọi lần chạy ghi vào G7/EXPERIMENT_LOG.md.
# Dùng:  bash run_g7_lot.sh <LOT> /kaggle/working        LOT ∈ {L1 L2 L3 L4 L5 L6 C1 C2 C3}
set -u
LOT=${1:?thiếu tên lượt}
OUT=${2:-/kaggle/working}
EV=(--split eval --eval-ok)
if [ ! -d "$OUT/PRODEN" ]; then
  git clone -q https://github.com/Lvcrezia77/PRODEN "$OUT/PRODEN" && git -C "$OUT/PRODEN" checkout -q 637b531a627a149ca494f966f4f41a20ad235853
fi
export PRODEN_REPO="$OUT/PRODEN"
echo "lượt $LOT bắt đầu: $(date)"
pip freeze > "$OUT/g7_${LOT}_pip_freeze.txt" 2>&1
nvidia-smi --query-gpu=index,name,memory.total --format=csv > "$OUT/g7_${LOT}_gpu.txt" 2>&1
sha256sum $(find . -name "*.py" -o -name "*.sh" | sort) > "$OUT/g7_${LOT}_code_sha256.txt"

# Kiểm từ bản code mới tải (việc T8 của G6): mọi lượt đều chạy bộ test trước; test hỏng thì dừng, không chạy ô nào
if ! python -m pytest -q tests > "$OUT/g7_${LOT}_pytest.log" 2>&1; then
  echo "TEST HỎNG — dừng lượt $LOT, không chạy ô nào. Xem g7_${LOT}_pytest.log"; tail -5 "$OUT/g7_${LOT}_pytest.log"; exit 1
fi
tail -1 "$OUT/g7_${LOT}_pytest.log"

gpu () {   # gpu <số GPU> <tên> <đối số run_g7...>
  local g=$1 tag=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g python run_g7.py "${EV[@]}" "$@" --out "$OUT/g7_${tag}.csv" >> "$OUT/g7_${tag}.log" 2>&1
  echo "GPU$g xong $tag: $(date)"
}
cpu () {   # cpu <tên> <đối số run_g7...>
  local tag=$1; shift
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ICGIBBS_DEVICE=cpu python run_g7.py "${EV[@]}" --part baselines "$@" \
    --out "$OUT/g7_${tag}.csv" >> "$OUT/g7_${tag}.log" 2>&1
  echo "CPU xong $tag: $(date)"
}
rq5_data () {   # tạo lại bản .npz của PLL thật (F7); bộ chạy kiểm SHA-256 tệp gốc với DATASET_CARDS
  [ -f "$OUT/pll_real/Lost.npz" ] || python g5_verify_pll_real.py --out "$OUT/g7_pll_real_check.csv" --save_dir "$OUT/pll_real" \
    > "$OUT/g7_pll_real_check.log" 2>&1
}

case $LOT in
  L1|L2|L3)   # phần chính (TFM): 6 mảnh, mỗi lượt 2 mảnh song song trên 2 GPU
    i=$(( (${LOT#L} - 1) * 2 ))
    gpu 0 "tfm_$i" --part tfm --shard $i/6 & gpu 1 "tfm_$((i+1))" --part tfm --shard $((i+1))/6 & wait ;;
  L4)         # R̂, RQ5 (PLL thật, TabICL), ablation K, số vòng quét, sửa MH, chuyển nạp, cỡ dữ liệu
    rq5_data
    ( gpu 0 rhat --part rhat; gpu 0 abl_blocks --part ablation --ablation blocks
      gpu 0 abl_sweeps --part ablation --ablation sweeps; gpu 0 abl_mh --part ablation --ablation mh ) &
    ( gpu 1 rq5 --part tfm --backend tabicl_v2 --settings pll_real --rq5_dir "$OUT/pll_real"
      gpu 1 abl_trans --part ablation --ablation trans
      for M in 500 1000 2000 4000; do gpu 1 size_tfm --part tfm --n_train_max $M; done ) &
    wait ;;
  L5)         # ablation khởi tạo lạnh và prior η, mỗi cái 2 mảnh
    ( gpu 0 abl_init_0 --part ablation --ablation init --shard 0/2; gpu 0 abl_prior_0 --part ablation --ablation prior --shard 0/2 ) &
    ( gpu 1 abl_init_1 --part ablation --ablation init --shard 1/2; gpu 1 abl_prior_1 --part ablation --ablation prior --shard 1/2 ) &
    wait ;;
  L6)         # ablation TFM: TabICL v2 cho hai bối cảnh chính
    gpu 0 tabicl_0 --part tfm --backend tabicl_v2 --settings llp,pll --levels "llp=8;pll=0.3,0.6" --shard 0/2 &
    gpu 1 tabicl_1 --part tfm --backend tabicl_v2 --settings llp,pll --levels "llp=8;pll=0.3,0.6" --shard 1/2 &
    wait ;;
  C1|C2|C3)   # baseline (CPU, 4 tiến trình): 12 mảnh, mỗi lượt 4 mảnh; C3 thêm đối thủ chính cho ablation cỡ dữ liệu
    b=$(( (${LOT#C} - 1) * 4 ))
    for j in 0 1 2 3; do cpu "base_$((b+j))" --shard $((b+j))/12 & done; wait
    if [ "$LOT" = C3 ]; then
      rq5_data
      cpu rq5_base --settings pll_real --rq5_dir "$OUT/pll_real"
      for M in 500 1000 2000 4000; do cpu size_base --n_train_max $M --only dllplin_def,cc_tun; done
    fi ;;
  *) echo "lượt không hợp lệ: $LOT"; exit 2 ;;
esac
echo "lượt $LOT kết thúc: $(date)"
