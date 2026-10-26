#!/usr/bin/env bash
# G6 lượt 2 (việc T3–T4): mọi baseline peer-review, bản mặc định và bản tune 20 cấu hình, trên TẬP PHÁT TRIỂN.
# Chạy trên notebook Kaggle CPU (4 nhân, Internet: On), từ thư mục chứa code: bash run_g6_round2.sh /kaggle/working
# Mỗi phần chia 4 mảnh chạy song song (mỗi tiến trình 1 luồng); chạy lại thì tự bỏ qua phần đã xong.
set -u
OUT=${1:-/kaggle/working}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# Kho chính thức của PRODEN (để dùng đúng partial_loss của tác giả), ghim commit
if [ ! -d "$OUT/PRODEN" ]; then
  git clone -q https://github.com/Lvcrezia77/PRODEN "$OUT/PRODEN" && git -C "$OUT/PRODEN" checkout -q 637b531a627a149ca494f966f4f41a20ad235853
fi
export PRODEN_REPO="$OUT/PRODEN"

LLP_M=dllp_def,dllp_tun,dllplin_def,dllplin_tun,llpvat_def,llpvat_tun,easyllp_def,easyllp_tun,glws_def,glws_tun
PLL_M=proden_def,proden_tun,glwspll_def,glwspll_tun,rc_def,rc_tun,cc_def,cc_tun
PU_M=nnpu_tun,upu_tun,sar_e_known

run4 () {   # $1 = tên phần, phần còn lại = lệnh (không có --shard/--out)
  local tag=$1; shift
  for i in 0 1 2 3; do
    "$@" --shard $i/4 --out "$OUT/g6r2_${tag}_$i.csv" > "$OUT/g6r2_${tag}_$i.log" 2>&1 &
  done
  wait
  echo "xong phần $tag: $(date)"
}

echo "bắt đầu: $(date)"
# tải trước dữ liệu OpenML một lần (tránh 4 tiến trình cùng ghi bộ nhớ đệm)
python - <<'PY'
import numpy as np
from kaggle_g4_pilot import DEV_DATASETS, load_dataset
from kaggle_g4_rq4_pilot import DEV_MULTI, load_multiclass
for did in DEV_DATASETS:
    load_dataset(did, False, np.random.default_rng(did))
for did in DEV_MULTI:
    load_multiclass(did, False)
print("đã tải trước dữ liệu phát triển")
PY
run4 pll python kaggle_g4_rq4_pilot.py --backend tabpfn_v2 --settings pll --only $PLL_M
run4 pu  python kaggle_g4_pilot.py --backend tabpfn_v2 --settings pu_scar,pu_sar --levels "pu_scar=0.1,0.3;pu_sar=0.1,0.3" --only $PU_M
run4 llp python kaggle_g4_pilot.py --backend tabpfn_v2 --settings llp --levels "llp=8,16" --only $LLP_M

# Kiểm giả thuyết về chênh lệch khi tái hiện SAR-EM trên Breast Cancer (G6_T1_SAREM_REPORT §2)
for mode in orig_plus1 all; do
  python g6_reproduce_sarem.py --repo "$OUT/SAR-PU" --datasets breastcancer --cls_mode $mode \
    --out "$OUT/g6r2_sarem_repro_$mode.csv" 2>&1 | grep -v "saved to" > "$OUT/g6r2_sarem_repro_$mode.log"
done
echo "kết thúc: $(date)"
