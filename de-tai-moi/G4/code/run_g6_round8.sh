#!/usr/bin/env bash
# G6 lượt 8 (chỉ CPU, Internet: On; khoảng 1 giờ): (1) GLWS/ILL cho PLL nay có số hạng cân bằng entropy trung bình như
# mọi config PLL chính thức (mặc định bật, tune bật/tắt) → chạy lại glwspll_def/_tun trên tập phát triển, cả bằng
# script thí điểm (nguồn dữ liệu của quy tắc chọn đối thủ chính) lẫn bằng bộ chạy G7 (để đối chiếu); (2) T3: tái hiện
# số công bố của PRODEN, RC, CC trên các bộ UCI của chính bài gốc. Không chạm tập đánh giá.
# Chạy từ thư mục chứa code:  bash run_g6_round8.sh /kaggle/working
set -u
OUT=${1:-/kaggle/working}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ICGIBBS_DEVICE=cpu
if [ ! -d "$OUT/PRODEN" ]; then
  git clone -q https://github.com/Lvcrezia77/PRODEN "$OUT/PRODEN" && git -C "$OUT/PRODEN" checkout -q 637b531a627a149ca494f966f4f41a20ad235853
fi
export PRODEN_REPO="$OUT/PRODEN"
echo "bắt đầu: $(date)"
pip freeze > "$OUT/g6r8_pip_freeze.txt" 2>&1
python - <<'PY'
from kaggle_g4_rq4_pilot import DEV_MULTI, load_multiclass
for did in DEV_MULTI: load_multiclass(did, False)
print("đã tải trước dữ liệu phát triển")
PY
# (1a) script thí điểm, 2 mảnh
for i in 0 1; do
  python kaggle_g4_rq4_pilot.py --backend tabpfn_v2 --settings pll --only glwspll_def,glwspll_tun --shard $i/2 \
    --out "$OUT/g6r8_pll_$i.csv" --preds "$OUT/g6r8_pll_${i}_preds.jsonl" > "$OUT/g6r8_pll_$i.log" 2>&1 &
done
# (1b) bộ chạy G7
python run_g7.py --part baselines --split dev --settings pll --levels "pll=0.3,0.6" --only glwspll_def,glwspll_tun \
  --out "$OUT/g6r8_runner_pll.csv" > "$OUT/g6r8_runner_pll.log" 2>&1 &
# (2) tái hiện PRODEN, RC, CC trên UCI
python g6_reproduce_pll_uci.py --out "$OUT/g6r8_pll_uci.csv" > "$OUT/g6r8_pll_uci.log" 2>&1 &
wait
tail -20 "$OUT/g6r8_pll_uci.log"
echo "kết thúc: $(date)"
