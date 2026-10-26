#!/usr/bin/env bash
# G6 lượt 4 (26/10/2026): baseline còn thiếu của các bối cảnh PHỤ — GLWS cho PU và VPU (PU-SCAR, PU-SAR);
# ILL cho nhãn nhiễu; ILL và VIME cho SSL. Mỗi baseline có bản mặc định và bản tune 20 cấu hình.
# Kèm việc T5: mã băm của mọi dataset (tập đánh giá và tập phát triển), chỉ tải và băm, không chạy phương pháp.
# Chỉ CPU, Internet: On. Chạy từ thư mục chứa code:  bash run_g6_round4.sh /kaggle/working
set -u
OUT=${1:-/kaggle/working}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

run4 () {   # $1 = tên phần, phần còn lại = lệnh (không có --shard/--out)
  local tag=$1; shift
  for i in 0 1 2 3; do
    "$@" --shard $i/4 --out "$OUT/g6r4_${tag}_$i.csv" > "$OUT/g6r4_${tag}_$i.log" 2>&1 &
  done
  wait
  echo "xong phần $tag: $(date)"
}

echo "bắt đầu: $(date)"
python - <<'PY'
import numpy as np
from kaggle_g4_pilot import DEV_DATASETS, load_dataset
for did in DEV_DATASETS: load_dataset(did, False, np.random.default_rng(did))
print("đã tải trước dữ liệu phát triển")
PY
# T5: mã băm và thống kê của mọi dataset sau đúng cách chuẩn hoá của bộ chạy G7 (không chạy phương pháp nào)
python g6_dataset_hashes.py --out "$OUT/g6r4_dataset_hashes.csv" > "$OUT/g6r4_dataset_hashes.log" 2>&1
echo "xong mã băm dữ liệu: $(date)"
run4 sslnz python kaggle_g4_rq4_pilot.py --backend tabpfn_v2 --settings ssl,noisy --only illnz_def,illnz_tun,illssl_def,illssl_tun,vime_def,vime_tun
run4 pu    python kaggle_g4_pilot.py --backend tabpfn_v2 --settings pu_scar,pu_sar --levels "pu_scar=0.1,0.3;pu_sar=0.1,0.3" --only glwspu_def,glwspu_tun,vpu_def,vpu_tun
echo "kết thúc: $(date)"
