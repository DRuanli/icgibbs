#!/usr/bin/env bash
# G6 lượt 3 (26/10/2026): (1) PRODEN tune với không gian có thêm Adam (G6_ROUND2_REPORT §5);
# (2) bản tune của baseline cho hai bối cảnh phụ SSL và nhãn nhiễu (label spreading, self-training, forward correction).
# Chỉ CPU, Internet: On. Chạy từ thư mục chứa code:  bash run_g6_round3.sh /kaggle/working
set -u
OUT=${1:-/kaggle/working}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ ! -d "$OUT/PRODEN" ]; then
  git clone -q https://github.com/Lvcrezia77/PRODEN "$OUT/PRODEN" && git -C "$OUT/PRODEN" checkout -q 637b531a627a149ca494f966f4f41a20ad235853
fi
export PRODEN_REPO="$OUT/PRODEN"

run4 () {   # $1 = tên phần, phần còn lại = lệnh (không có --shard/--out)
  local tag=$1; shift
  for i in 0 1 2 3; do
    "$@" --shard $i/4 --out "$OUT/g6r3_${tag}_$i.csv" --preds "$OUT/g6r3_${tag}_${i}_preds.jsonl" > "$OUT/g6r3_${tag}_$i.log" 2>&1 &
  done
  wait
  echo "xong phần $tag: $(date)"
}

echo "bắt đầu: $(date)"
python - <<'PY'
from kaggle_g4_rq4_pilot import DEV_MULTI, load_multiclass
from kaggle_g4_pilot import DEV_DATASETS, load_dataset
import numpy as np
for did in DEV_MULTI: load_multiclass(did, False)
for did in DEV_DATASETS: load_dataset(did, False, np.random.default_rng(did))
print("đã tải trước dữ liệu phát triển")
PY
run4 pll   python kaggle_g4_rq4_pilot.py --backend tabpfn_v2 --settings pll --only proden_tun
run4 sslnz python kaggle_g4_rq4_pilot.py --backend tabpfn_v2 --settings ssl,noisy --only ls_tun,st_tun,forward_def,forward_tun
echo "kết thúc: $(date)"
