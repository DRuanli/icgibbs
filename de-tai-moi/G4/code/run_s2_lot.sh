#!/usr/bin/env bash
# Nghiên cứu 2 (S2) — chạy tập đánh giá MỚI theo de-tai-moi/S2/S2_PLAN.md (chỉ sau khi S2 đã đóng băng).
# Dùng:  bash run_s2_lot.sh <LOT> /kaggle/working        LOT ∈ {S2X S2D S2T S2C1 S2C2 S2G S2R S2E V0A V0B V1A V1B V1C V1D V1E S3 S3B S3BC}
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
#   S2E  (GPU T4 × 2): THĂM DÒ (G9_PLAN O3, O4; thầy duyệt D3 ngày 03/11/2026), TabICL v2, chỉ trên dataset cũ của NC1
#                      (tập phát triển của S2). GPU 0: R̂ hai chuỗi (3 dataset nhỏ mỗi nhóm như NC1; thêm 4 dataset ≥ 8
#                      lớp ở PLL q 0,6). GPU 1: khối có sửa MH (LLP túi 8, PLL q 0,3 và 0,6; thêm 4 dataset ≥ 8 lớp ở
#                      q 0,6). Cần s2_code_v5 trở đi
#   V0A  (GPU T4 × 2): V2 giai đoạn 0, THĂM DÒ (V2/PHASE0_PREDICTIONS.md; thầy duyệt V2 ngày 04/11/2026). GPU 0
#                      (TabPFN v2, dataset cũ của NC1): ánh xạ phản ứng PLL ρ cao và LLP (cho P2, P3), nhiều điểm xuất phát
#                      LLP túi 8 (P3, P4), ánh xạ lỗi có cấu trúc (W1). GPU 1 (TabICL v2): R4 — chuỗi thứ hai trên mọi ô của
#                      Study 2, lưu vết vào r4_traces/. Cần s2_code_v6 trở đi
#   V0B  (GPU T4 × 2): V2 giai đoạn 0. Hai GPU chia đôi nhiều điểm xuất phát PLL q 0,6 với TabPFN v2 (P1, P2, P4); sau đó
#                      TabICL v2 trên cùng các ô (P4, kiểm v2 không hại TFM tốt) và ánh xạ phản ứng của TabICL
#   V1A  (GPU T4 × 2): V2 giai đoạn 1, THĂM DÒ (V2/PHASE1_PLAN.md), TabPFN v2. GPU 0: LLP túi 8 trên 8 dataset còn lại của
#                      tập đánh giá NC1 rồi 6 dataset phát triển; GPU 1: PLL q 0,6 trên 10 + 3 dataset tương ứng. Mỗi ô 3 chuỗi
#                      (warm, base, cold), lưu npz vào v1_npz/, learner khởi động có dự đoán ngoài fold. Cần s2_code_v7
#   V1B  (GPU T4 × 2): như V1A với TabICL v2, thêm toàn bộ dataset của Study 2 (--split s2 --explore). Hai GPU chia đôi
#   V1C  (GPU T4 × 2): V2 giai đoạn 1b, THĂM DÒ (V2/PHASE1B_PLAN.md). GPU 0: thời gian/tất định của 5 TFM (w4_pilot.py),
#                      ánh xạ phản ứng trên các ô F1 cho 5 TFM (v0_response_map.py), rồi Φ_chuỗi trên ô V1 của TabPFN v2.
#                      GPU 1: Φ_chuỗi trên ô V1 của TabICL v2 (v1c_chain_map.py). Cần output V1A, V1B (v1_npz/) làm dữ liệu,
#                      giấy phép TabPFN-3/3.5 đã chấp nhận, và gói tabdpt==1.3.1. Bổ sung 07/11 (PHASE1B_PLAN §7): GPU 0
#                      thêm chuỗi TabPFN-3.x (chọn tự động) và TabPFN v2 trên F1, chuỗi dò 3 vòng quét (D), A1 cùng chi
#                      phí (F); GPU 1 thêm D, TabICL trên F1, TabDPT, cơ chế nhãn yếu mới (G), F. Khoảng 2 phiên. Cần s2_code_v9 (phiên 1) hoặc v10 (phiên 2, chia lại việc giữa hai GPU)
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
  S2E)
    python -c "import run_g7; assert 'mh06' in run_g7.ABLATIONS" 2>/dev/null \
      || { echo "BẢN CODE CŨ: chưa có ablation mh06 — dùng s2_code_v5.zip"; exit 3; }
    MANY=12,28,1501,40966     # mfeat-factors, optdigits, semeion, MiceProtein: ≥ 8 lớp, nơi TabPFN sụp ở NC1
    xg () {   # xg <số GPU> <tên> <đối số run_g7...>: tập đánh giá cũ của NC1, TabICL v2
      local g=$1 tag=$2; shift 2
      CUDA_VISIBLE_DEVICES=$g python run_g7.py --split eval --eval-ok --backend tabicl_v2 "$@" \
        --out "$OUT/s2e_${tag}.csv" >> "$OUT/s2e_${tag}.log" 2>&1
      echo "GPU$g xong $tag: $(date)"
    }
    ( xg 0 rhat --part rhat --settings llp,pll --levels "llp=8;pll=0.3,0.6"
      xg 0 rhat_many --part rhat --settings pll --levels "pll=0.6" --datasets $MANY ) &
    ( xg 1 mh --part ablation --ablation mh
      xg 1 mh06 --part ablation --ablation mh06
      xg 1 mh06_many --part ablation --ablation mh06 --datasets $MANY ) &
    wait ;;
  V0A|V0B)
    { [ -f v0_response_map.py ] && python run_g7.py --help 2>/dev/null | grep -q -- "--starts"; } \
      || { echo "BẢN CODE CŨ: chưa có --part v0 — dùng s2_code_v6.zip"; exit 3; }
    E1=(--split eval --eval-ok)          # dataset cũ của NC1 (tập phát triển của S2 và V2)
    rm_ () {   # rm_ <số GPU> <tên> <đối số v0_response_map...>
      local g=$1 tag=$2; shift 2
      CUDA_VISIBLE_DEVICES=$g python v0_response_map.py "$@" --out "$OUT/v0x_${tag}.csv" >> "$OUT/v0x_${tag}.log" 2>&1
      echo "GPU$g xong v0x_$tag: $(date)"
    }
    ms_ () {   # ms_ <số GPU> <tên> <đối số run_g7...>
      local g=$1 tag=$2; shift 2
      CUDA_VISIBLE_DEVICES=$g python run_g7.py "${E1[@]}" --part v0 "$@" --out "$OUT/v0_${tag}.csv" >> "$OUT/v0_${tag}.log" 2>&1
      echo "GPU$g xong v0_$tag: $(date)"
    }
    PLL4=12,28,40966,1501; LLP2=50,6332
    if [ "$LOT" = V0A ]; then
      ( rm_ 0 pll_hi_tabpfn --setting pll --plant uniform --backends tabpfn_v2 --datasets $PLL4 --rho 0.85,0.9,0.95
        rm_ 0 llp_tabpfn --setting llp --backends tabpfn_v2 --datasets $LLP2 --rho u,0.1,0.2,0.3,0.45,0.6,0.8,0.9
        ms_ 0 llp_tabpfn --backend tabpfn_v2 --settings llp --levels "llp=8" --datasets $LLP2 --starts warm,base,cold
        rm_ 0 struct_tabpfn --setting pll --plant struct --backends tabpfn_v2 --datasets $PLL4,188,40982 \
            --rho 0.2,0.45,0.6,0.8,0.9 ) &
      ( CUDA_VISIBLE_DEVICES=1 python run_g7.py --split s2 --eval-ok --explore --part rhat --backend tabicl_v2 \
            --settings llp,pll --levels "llp=8;pll=0.3,0.6" --trace_dir "$OUT/r4_traces" --out "$OUT/s2r4_tabicl.csv" \
            >> "$OUT/s2r4_tabicl.log" 2>&1; echo "GPU1 xong s2r4_tabicl: $(date)" ) &
      wait
    else
      ( ms_ 0 pll_tabpfn_0 --backend tabpfn_v2 --settings pll --levels "pll=0.6" --datasets $PLL4 --starts warm,base --shard 0/2
        ms_ 0 pll_tabicl --backend tabicl_v2 --settings pll --levels "pll=0.6" --datasets $PLL4 --starts warm,base,cold
        rm_ 0 struct_tabicl --setting pll --plant struct --backends tabicl_v2 --datasets $PLL4,188,40982 \
            --rho 0.2,0.45,0.6,0.8,0.9 ) &
      ( ms_ 1 pll_tabpfn_1 --backend tabpfn_v2 --settings pll --levels "pll=0.6" --datasets $PLL4 --starts warm,base --shard 1/2
        ms_ 1 llp_tabicl --backend tabicl_v2 --settings llp --levels "llp=8" --datasets $LLP2 --starts warm,base,cold
        rm_ 1 llp_tabicl --setting llp --backends tabicl_v2 --datasets $LLP2 --rho u,0.1,0.2,0.3,0.45,0.6,0.8,0.9
        rm_ 1 pll_hi_tabicl --setting pll --plant uniform --backends tabicl_v2 --datasets $PLL4 --rho 0.85,0.9,0.95 ) &
      wait
    fi ;;
  V1A|V1B)
    python run_g7.py --help 2>/dev/null | grep -q -- "--oof" \
      || { echo "BẢN CODE CŨ: chưa có --oof — dùng s2_code_v7.zip"; exit 3; }
    NPZ="$OUT/v1_npz"; mkdir -p "$NPZ"
    v1_ () {   # v1_ <số GPU> <tên> <backend> <đối số run_g7...>
      local g=$1 tag=$2 be=$3; shift 3
      CUDA_VISIBLE_DEVICES=$g python run_g7.py --part v0 --backend $be --starts warm,base,cold --save_dir "$NPZ" --oof \
        "$@" --out "$OUT/v1_${tag}.csv" >> "$OUT/v1_${tag}.log" 2>&1
      echo "GPU$g xong v1_$tag: $(date)"
    }
    LLP_E=3,29,44,1049,1489,1494,23381,38; LLP_D=31,37,1464,1462,1480,1510
    PLL_E=11,46,182,188,458,469,1475,1497,40975,40982; PLL_D=54,36,23
    if [ "$LOT" = V1A ]; then
      ( v1_ 0 llp_eval_tabpfn tabpfn_v2 --split eval --eval-ok --settings llp --levels "llp=8" --datasets $LLP_E
        v1_ 0 llp_dev_tabpfn tabpfn_v2 --split dev --settings llp --levels "llp=8" --datasets $LLP_D ) &
      ( v1_ 1 pll_eval_tabpfn tabpfn_v2 --split eval --eval-ok --settings pll --levels "pll=0.6" --datasets $PLL_E
        v1_ 1 pll_dev_tabpfn tabpfn_v2 --split dev --settings pll --levels "pll=0.6" --datasets $PLL_D ) &
      wait
    else
      S2X_=(--split s2 --eval-ok --explore)
      ( v1_ 0 llp_s2_tabicl_0 tabicl_v2 "${S2X_[@]}" --settings llp --levels "llp=8" --shard 0/2
        v1_ 0 llp_eval_tabicl tabicl_v2 --split eval --eval-ok --settings llp --levels "llp=8" --datasets $LLP_E
        v1_ 0 llp_dev_tabicl tabicl_v2 --split dev --settings llp --levels "llp=8" --datasets $LLP_D ) &
      ( v1_ 1 llp_s2_tabicl_1 tabicl_v2 "${S2X_[@]}" --settings llp --levels "llp=8" --shard 1/2
        v1_ 1 pll_s2_tabicl tabicl_v2 "${S2X_[@]}" --settings pll --levels "pll=0.6"
        v1_ 1 pll_eval_tabicl tabicl_v2 --split eval --eval-ok --settings pll --levels "pll=0.6" --datasets $PLL_E
        v1_ 1 pll_dev_tabicl tabicl_v2 --split dev --settings pll --levels "pll=0.6" --datasets $PLL_D ) &
      wait
    fi ;;
  S3)
    # STUDY 3 bản gọn — XÁC NHẬN (de-tai-moi/S3/STUDY3_PLAN.md, ghi trước 10/11/2026). 20 dataset mới, LLP túi 8 và PLL
    # q 0,6, seed 0–2, 4 TFM × 3 chuỗi (warm, base, cold), cấu hình V1. CPU: kiểm mã băm dataset rồi tính trước learner L.
    # Chạy tiếp được: chạy lại đúng lượt với output cũ (s3_*.csv, s3_npz/, l_cache_s3/). Cần s2_code_v14
    python -c "from icgibbs.g7.registry import S3_BINARY, S3_MULTICLASS; assert len(S3_BINARY) == 12 and len(S3_MULTICLASS) == 8" \
      || { echo "BẢN CODE CŨ: dùng s2_code_v15.zip"; exit 3; }
    LC="$OUT/l_cache_s3"; ND="$OUT/s3_npz"; mkdir -p "$LC" "$ND"
    PFN2=$(python -c "from icgibbs.g7.registry import S3_TABPFN_V2, S3_BINARY; print(','.join(str(d) for d in S3_TABPFN_V2 if d in S3_BINARY))")
    s3_ () {   # s3_ <GPU ('' = CPU)> <tên> <setting> <đối số run_g7...>
      local g=$1 tag=$2 st=$3 lv; shift 3
      [ $st = llp ] && lv="llp=8" || lv="pll=0.6"
      CUDA_VISIBLE_DEVICES=$g python run_g7.py --split s3 --eval-ok --part v0 --starts warm,base,cold --seeds 0,1,2 \
        --settings $st --levels "$lv" --l_cache "$LC" "$@" --out "$OUT/s3_${tag}_${st}.csv" >> "$OUT/s3_${tag}_${st}.log" 2>&1
      if [ -n "$g" ]; then echo "GPU$g xong s3_${tag}_${st}: $(date)"; else echo "CPU xong s3_${tag}_${st}: $(date)"; fi
    }
    ( CUDA_VISIBLE_DEVICES= python g6_dataset_hashes.py --split s3 --out "$OUT/s3_dataset_hashes.csv" > "$OUT/s3_dataset_hashes.log" 2>&1
      echo "CPU xong mã băm dataset: $(date)"
      s3_ "" lprefetch llp --backend tabicl_v2 --l_only
      s3_ "" lprefetch pll --backend tabicl_v2 --l_only ) &
    # 11/11 (từ phiên 2, s2_code_v15): chia lại việc cho cân giữa hai GPU — chỉ đổi thứ tự và GPU chạy; cùng ô, cùng
    # seed, cùng cấu hình (STUDY3_PLAN không đổi). Tên tệp đầu ra giữ nguyên nên các ô đã xong ở phiên 1 được bỏ qua.
    ( s3_ 0 tabpfn_v3 pll --backend tabpfn_v3 --save_dir "$ND"
      s3_ 0 tabpfn_v3 llp --backend tabpfn_v3 --save_dir "$ND"
      s3_ 0 tabdpt llp --backend tabdpt --save_dir "$ND"
      s3_ 0 tabpfn_v2 pll --backend tabpfn_v2 --save_dir "$ND" --datasets 4538
      s3_ 0 tabpfn_v2_sh1 llp --backend tabpfn_v2 --save_dir "$ND" --datasets $PFN2 --shard 1/2
      s3_ 0 tabpfn_v2_sh0 llp --backend tabpfn_v2 --save_dir "$ND" --datasets $PFN2 --shard 0/2 ) &
    ( s3_ 1 tabicl_v2 pll --backend tabicl_v2 --save_dir "$ND"
      s3_ 1 tabicl_v2 llp --backend tabicl_v2 --save_dir "$ND"
      s3_ 1 tabdpt pll --backend tabdpt --save_dir "$ND" ) &
    wait ;;
  S3B|S3BC)
    # LƯỢT BỔ SUNG trên dataset Study 3 (de-tai-moi/S3/S3B_PLAN.md, ghi trước 12/11/2026; ngoài protocol Study 3).
    # S3BC (notebook CPU): baseline không nơ-ron (B1). S3B (notebook GPU): naive, hard-EM, oracle (B2) rồi chuỗi warm
    # 100 vòng quét trên PLL (B3). Hai lượt độc lập, chạy song song ở hai notebook.
    # Chạy tiếp được: chạy lại đúng lượt với output cũ (s3b_*.csv, s3b_npz/). Cần s2_code_v16
    python -c "from icgibbs.g7.methods import BASELINES_CPU, SUPP_TFM" || { echo "BẢN CODE CŨ: dùng s2_code_v16.zip"; exit 3; }
    ND="$OUT/s3b_npz"; mkdir -p "$ND"
    PFN2=$(python -c "from icgibbs.g7.registry import S3_TABPFN_V2, S3_BINARY; print(','.join(str(d) for d in S3_TABPFN_V2 if d in S3_BINARY))")
    b_ () {   # b_ <GPU ('' = CPU)> <tên> <setting> <đối số run_g7...>
      local g=$1 tag=$2 st=$3 lv; shift 3
      [ $st = llp ] && lv="llp=8" || lv="pll=0.6"
      CUDA_VISIBLE_DEVICES=$g python run_g7.py --split s3 --eval-ok --supp --seeds 0,1,2 --settings $st --levels "$lv" \
        "$@" --out "$OUT/s3b_${tag}_${st}.csv" >> "$OUT/s3b_${tag}_${st}.log" 2>&1
      if [ -n "$g" ]; then echo "GPU$g xong s3b_${tag}_${st}: $(date)"; else echo "CPU xong s3b_${tag}_${st}: $(date)"; fi
    }
    tfm_ () { local g=$1 be=$2 st=$3; shift 3; b_ $g tfm_$be $st --part tfm --only naive,hard_em_warm,oracle --backend $be "$@"; }
    long_ () { local g=$1 be=$2; b_ $g long_$be pll --part v0 --starts warm --sweeps 100 --backend $be --save_dir "$ND"; }
    if [ "$LOT" = S3BC ]; then            # notebook CPU riêng (không cần GPU): chỉ B1
      b_ "" cpu pll --part baselines
      b_ "" cpu llp --part baselines
      echo "S3BC xong: $(date)"; exit 0
    fi
    ( tfm_ 0 tabpfn_v3 pll
      tfm_ 0 tabpfn_v3 llp
      tfm_ 0 tabpfn_v2 pll --datasets 4538
      tfm_ 0 tabpfn_v2 llp --datasets $PFN2
      long_ 0 tabpfn_v3 ) &
    ( tfm_ 1 tabicl_v2 pll
      tfm_ 1 tabicl_v2 llp
      tfm_ 1 tabdpt pll
      tfm_ 1 tabdpt llp
      long_ 1 tabdpt
      long_ 1 tabicl_v2 ) &
    wait ;;
  V1E)
    # V2 (V2/V1E_PLAN.md, ghi trước 10/11): E1 hoàn tất V1D-2/V1D-4 sau sửa D-V1D-1; E2 đếm dataset mở rộng (CPU);
    # E3 TabPFN-3 và TabDPT trên 96 ô V1 chỉ có TabICL (P19–P21). Cần output V1B (v1_npz) và V1D (v1d_npz, l_cache,
    # v1d_*.csv) làm dữ liệu. Cần s2_code_v13
    [ -f v1d_choose_q.py ] && grep -q "all_openml" s3_dataset_census.py || { echo "BẢN CODE CŨ: dùng s2_code_v13.zip"; exit 3; }
    LC="$OUT/l_cache"; DD="$OUT/v1d_npz"; ND="$OUT/v1e_npz"; mkdir -p "$LC" "$DD" "$ND"
    PD=54,36,23; S2X_=(--split s2 --eval-ok --explore)
    r_ () {   # r_ <GPU ('' = CPU)> <tên> <đối số run_g7...>
      local g=$1 tag=$2; shift 2
      CUDA_VISIBLE_DEVICES=$g python run_g7.py --part v0 --l_cache "$LC" "$@" --out "$OUT/v1e_${tag}.csv" \
        >> "$OUT/v1e_${tag}.log" 2>&1
      if [ -n "$g" ]; then echo "GPU$g xong v1e_$tag: $(date)"; else echo "CPU xong v1e_$tag: $(date)"; fi
    }
    qstar_ () { python v1d_choose_q.py --calib_dir "$DD" --ref_dir "$OUT/v1_npz"; }
    inst_ () {    # inst_ <GPU> <backend>: E1, PLL phụ thuộc mẫu ở q*
      local g=$1 be=$2 q; q=$(qstar_)
      for k in $(seq 1 60); do [ -n "$q" ] && break; sleep 30; q=$(qstar_); done
      if [ -n "$q" ]; then
        r_ "$g" inst_${be} --backend $be --split dev --settings pll --levels "pll=$q" --datasets $PD --weak_variant inst \
          --starts warm,base,cold --save_dir "$ND"
      else echo "không có q* sau 30 phút: bỏ qua E1 cho $be"; fi
    }
    e3_ () {      # e3_ <GPU> <backend> <đối số thêm...>: 96 ô V1 chỉ có TabICL
      local g=$1 be=$2; shift 2
      r_ "$g" e3_${be}_pll_s2 --backend $be "${S2X_[@]}" --settings pll --levels "pll=0.6" --starts warm,base,cold --save_dir "$ND" "$@"
      r_ "$g" e3_${be}_llp_dev --backend $be --split dev --settings llp --levels "llp=8" --datasets 31,37 --starts warm,base,cold --save_dir "$ND" "$@"
      r_ "$g" e3_${be}_llp_eval --backend $be --split eval --eval-ok --settings llp --levels "llp=8" --datasets 23381,38 --starts warm,base,cold --save_dir "$ND" "$@"
    }
    ( CUDA_VISIBLE_DEVICES= python s3_dataset_census.py --all_openml --out "$OUT" > "$OUT/v1e_census.log" 2>&1
      echo "CPU xong đếm dataset mở rộng: $(date)"
      r_ "" lpre_llp_s2 --backend tabicl_v2 --l_only "${S2X_[@]}" --settings llp --levels "llp=8"
      r_ "" lpre_pll_s2 --backend tabicl_v2 --l_only "${S2X_[@]}" --settings pll --levels "pll=0.6"
      r_ "" lpre_llp_dev --backend tabicl_v2 --l_only --split dev --settings llp --levels "llp=8" --datasets 31,37
      r_ "" lpre_llp_eval --backend tabicl_v2 --l_only --split eval --eval-ok --settings llp --levels "llp=8" --datasets 23381,38 ) &
    ( inst_ 0 tabpfn_v3
      e3_ 0 tabpfn_v3
      r_ 0 e3_tabpfn_v3_llp_s2_sh0 --backend tabpfn_v3 "${S2X_[@]}" --settings llp --levels "llp=8" --starts warm,base,cold --save_dir "$ND" --shard 0/3
      r_ 0 e3_tabpfn_v3_llp_s2_sh1 --backend tabpfn_v3 "${S2X_[@]}" --settings llp --levels "llp=8" --starts warm,base,cold --save_dir "$ND" --shard 1/3 ) &
    ( CUDA_VISIBLE_DEVICES=1 python run_g7.py --part v0 --l_cache "$LC" --backend tabicl_v2 --split dev --settings pll \
        --levels "pll=0.2,0.3,0.4" --datasets $PD --weak_variant inst --starts warm,base,cold --save_dir "$DD" \
        --out "$OUT/v1d_calib_tabicl_inst.csv" >> "$OUT/v1e_calib_fix.log" 2>&1
      q=$(qstar_); echo "${q:-none}" > "$OUT/v1d_q.txt"; echo "q* (V1D_PLAN §2): ${q:-chưa chọn được}"
      inst_ 1 tabdpt
      inst_ 1 tabpfn_v2
      e3_ 1 tabdpt
      r_ 1 e3_tabdpt_llp_s2 --backend tabdpt "${S2X_[@]}" --settings llp --levels "llp=8" --starts warm,base,cold --save_dir "$ND"
      r_ 1 e3_tabpfn_v3_llp_s2_sh2 --backend tabpfn_v3 "${S2X_[@]}" --settings llp --levels "llp=8" --starts warm,base,cold --save_dir "$ND" --shard 2/3 ) &
    wait ;;
  V1D)
    # V2 sau V1C (thầy duyệt 08/11; V2/V1D_PLAN.md §1–§3): CPU đếm dataset Study 3 rồi tính trước learner L (--l_only);
    # GPU 1: hiệu chỉnh q của PLL phụ thuộc mẫu (TabICL) → chọn q* → TabDPT trên 69 ô chung của V1 → bối cảnh khó;
    # GPU 0: TabPFN-3 trên 69 ô chung → bối cảnh khó với TabPFN-3 và TabPFN v2. Cần output V1B (v1_npz) làm dữ liệu.
    [ -f v1d_choose_q.py ] && python run_g7.py --help 2>/dev/null | grep -q -- "--l_cache" \
      || { echo "BẢN CODE CŨ: dùng s2_code_v12.zip"; exit 3; }
    LC="$OUT/l_cache"; ND="$OUT/v1d_npz"; mkdir -p "$LC" "$ND"
    LLP_EP=3,29,44,1049,1489,1494; LLP_DP=1464,1462,1480,1510; LD=31,37,1464,1462,1480,1510
    PLL_E=11,46,182,188,458,469,1475,1497,40975,40982; PD=54,36,23
    r_ () {   # r_ <GPU ('' = CPU)> <tên> <đối số run_g7...>
      local g=$1 tag=$2; shift 2
      CUDA_VISIBLE_DEVICES=$g python run_g7.py --part v0 --l_cache "$LC" "$@" --out "$OUT/v1d_${tag}.csv" \
        >> "$OUT/v1d_${tag}.log" 2>&1
      if [ -n "$g" ]; then echo "GPU$g xong v1d_$tag: $(date)"; else echo "CPU xong v1d_$tag: $(date)"; fi
    }
    common_ () {  # common_ <GPU> <backend> <đối số thêm...>: 69 ô chung của V1 (23 dataset × 3 seed)
      local g=$1 be=$2; shift 2
      r_ "$g" c_${be}_llp_eval --backend $be --split eval --eval-ok --settings llp --levels "llp=8" --datasets $LLP_EP "$@"
      r_ "$g" c_${be}_llp_dev --backend $be --split dev --settings llp --levels "llp=8" --datasets $LLP_DP "$@"
      r_ "$g" c_${be}_pll_eval --backend $be --split eval --eval-ok --settings pll --levels "pll=0.6" --datasets $PLL_E "$@"
      r_ "$g" c_${be}_pll_dev --backend $be --split dev --settings pll --levels "pll=0.6" --datasets $PD "$@"
    }
    hard_ () {    # hard_ <GPU> <backend>: V1D-4 (túi 32; PLL phụ thuộc mẫu ở q*)
      local g=$1 be=$2
      r_ "$g" h_${be}_llp32 --backend $be --split dev --settings llp --levels "llp=32" --datasets $LD \
        --starts warm,base,cold --save_dir "$ND"
      local q; q=$(python v1d_choose_q.py --calib_dir "$ND" --ref_dir "$OUT/v1_npz")
      if [ -n "$q" ]; then
        echo "$q" > "$OUT/v1d_q.txt"
        r_ "$g" h_${be}_pll_inst --backend $be --split dev --settings pll --levels "pll=$q" --datasets $PD \
          --weak_variant inst --starts warm,base,cold --save_dir "$ND"
      else
        echo "chưa có q* (V1D-2 chưa đủ 27 ô): bỏ qua PLL phụ thuộc mẫu cho $be ở phiên này"
      fi
    }
    ( CUDA_VISIBLE_DEVICES= python s3_dataset_census.py --out "$OUT" > "$OUT/v1d_census.log" 2>&1
      echo "CPU xong đếm dataset: $(date)"
      common_ "" lprefetch_c --backend tabicl_v2 --l_only
      r_ "" lprefetch_llp32 --backend tabicl_v2 --l_only --split dev --settings llp --levels "llp=32" --datasets $LD ) &
    ( common_ 0 tabpfn_v3 --starts warm,base,cold --save_dir "$ND"
      hard_ 0 tabpfn_v3
      hard_ 0 tabpfn_v2 ) &
    ( r_ 1 calib_tabicl_inst --backend tabicl_v2 --split dev --settings pll --levels "pll=0.2,0.3,0.4" --datasets $PD \
        --weak_variant inst --starts warm,base,cold --save_dir "$ND"
      q=$(python v1d_choose_q.py --calib_dir "$ND" --ref_dir "$OUT/v1_npz" --verbose); echo "${q:-none}" > "$OUT/v1d_q.txt"
      echo "q* (V1D_PLAN §2): ${q:-chưa chọn được}"
      common_ 1 tabdpt --starts warm,base,cold --save_dir "$ND"
      hard_ 1 tabdpt ) &
    wait ;;
  V1C)
    [ -f v1c_chain_map.py ] && python run_g7.py --help 2>/dev/null | grep -q -- "--sweeps" \
      || { echo "BẢN CODE CŨ: dùng s2_code_v9.zip"; exit 3; }
    NPZ="$OUT/v1_npz"
    n_npz=$(ls "$NPZ"/v1_*.npz 2>/dev/null | wc -l)
    echo "số tệp npz của V1: $n_npz (cần 234: 69 TabPFN + 165 TabICL)"
    BE5=tabpfn_v2,tabicl_v2,tabpfn_v3,tabpfn_v3_5,tabdpt
    RHO=0.2,0.35,0.5,0.65,0.8,0.9,0.97
    F1P=12,1501,40966,28; F1L=50,6332; PD=54,36,23; LD=31,37,1464,1462,1480,1510
    LLP_E=3,29,44,1049,1489,1494,23381,38; PLL_E=11,46,182,188,458,469,1475,1497,40975,40982
    LLP_EP=3,29,44,1049,1489,1494; LLP_DP=1464,1462,1480,1510       # TabPFN v2: đúng các dataset có trong V1A
    run_ () {   # run_ <số GPU> <tên> <script> <đối số...>
      local g=$1 tag=$2 sc=$3; shift 3
      CUDA_VISIBLE_DEVICES=$g python $sc "$@" --out "$OUT/v1c_${tag}.csv" >> "$OUT/v1c_${tag}.log" 2>&1
      echo "GPU$g xong v1c_$tag: $(date)"
    }
    ch_ () {    # ch_ <số GPU> <tên> <backend> <thư mục npz> <đối số run_g7...>: 3 chuỗi warm, base, cold
      local g=$1 tag=$2 be=$3 d=$4; shift 4
      mkdir -p "$OUT/$d"
      run_ $g $tag run_g7.py --part v0 --backend $be --starts warm,base,cold --save_dir "$OUT/$d" "$@"
    }
    v1cells_ () {   # v1cells_ <số GPU> <tiền tố> <backend> <thư mục npz> <đối số thêm...>: đúng các ô của V1
      local g=$1 pre=$2 be=$3 d=$4; shift 4
      if [ $be = tabpfn_v2 ]; then
        ch_ $g ${pre}_llp_eval_pfn $be $d --split eval --eval-ok --settings llp --levels "llp=8" --datasets $LLP_EP "$@"
        ch_ $g ${pre}_llp_dev_pfn $be $d --split dev --settings llp --levels "llp=8" --datasets $LLP_DP "$@"
        ch_ $g ${pre}_pll_eval_pfn $be $d --split eval --eval-ok --settings pll --levels "pll=0.6" --datasets $PLL_E "$@"
        ch_ $g ${pre}_pll_dev_pfn $be $d --split dev --settings pll --levels "pll=0.6" --datasets $PD "$@"
      else
        ch_ $g ${pre}_llp_s2_icl $be $d --split s2 --eval-ok --explore --settings llp --levels "llp=8" "$@"
        ch_ $g ${pre}_pll_s2_icl $be $d --split s2 --eval-ok --explore --settings pll --levels "pll=0.6" "$@"
        ch_ $g ${pre}_llp_eval_icl $be $d --split eval --eval-ok --settings llp --levels "llp=8" --datasets $LLP_E "$@"
        ch_ $g ${pre}_llp_dev_icl $be $d --split dev --settings llp --levels "llp=8" --datasets $LD "$@"
        ch_ $g ${pre}_pll_eval_icl $be $d --split eval --eval-ok --settings pll --levels "pll=0.6" --datasets $PLL_E "$@"
        ch_ $g ${pre}_pll_dev_icl $be $d --split dev --settings pll --levels "pll=0.6" --datasets $PD "$@"
      fi
    }
    newtfm_ () {    # newtfm_ <số GPU> <backend>: phần E trên F1 và tập phát triển, seed 0
      local g=$1 be=$2
      ch_ $g e_${be}_pll_f1 $be v1c_e_npz --split eval --eval-ok --settings pll --levels "pll=0.6" --datasets $F1P --seeds 0
      ch_ $g e_${be}_llp_f1 $be v1c_e_npz --split eval --eval-ok --settings llp --levels "llp=8" --datasets $F1L --seeds 0
      ch_ $g e_${be}_pll_dev $be v1c_e_npz --split dev --settings pll --levels "pll=0.6" --datasets $PD --seeds 0
      ch_ $g e_${be}_llp_dev $be v1c_e_npz --split dev --settings llp --levels "llp=8" --datasets $LD --seeds 0
    }
    ( run_ 0 w4_timing w4_pilot.py --backends $BE5
      run_ 0 map_pll_uniform v0_response_map.py --setting pll --plant uniform --backends $BE5 \
        --datasets $F1P,188,40982 --seeds 0 --rho $RHO
      run_ 0 map_pll_struct v0_response_map.py --setting pll --plant struct --backends $BE5 \
        --datasets $F1P,188,40982 --seeds 0 --rho $RHO
      run_ 0 map_llp_uniform v0_response_map.py --setting llp --plant uniform --backends $BE5 \
        --datasets $F1L --seeds 0 --rho $RHO
      CH=$(python w4_pilot.py --choose "$OUT/v1c_w4_timing.csv"); echo "${CH:-none}" > "$OUT/v1c_choice.txt"
      echo "TabPFN-3.x được chọn (PHASE1B_PLAN §4): ${CH:-không bản nào hợp lệ}"
      [ -n "$CH" ] && newtfm_ 0 $CH
      ch_ 0 e_tabpfn_v2_pll_f1 tabpfn_v2 v1c_e_npz --split eval --eval-ok --settings pll --levels "pll=0.6" --datasets $F1P --seeds 0
      ch_ 0 e_tabpfn_v2_llp_f1 tabpfn_v2 v1c_e_npz --split eval --eval-ok --settings llp --levels "llp=8" --datasets $F1L --seeds 0
      run_ 0 chainmap_tabpfn v1c_chain_map.py --npz_glob "$NPZ/v1_tabpfn_v2_*.npz"
      v1cells_ 0 d tabpfn_v2 v1c_probe_npz --sweeps 3
      v1cells_ 0 f tabpfn_v2 v1c_f_npz --sweeps 10
      # 08/11 (phiên 2): chia lại việc giữa hai GPU cho cân — G và phần F của TabICL ngoài S2-LLP sang GPU 0;
      # F TabICL S2-LLP (learner L chậm, khoảng 7 giờ) chia 3 mảnh. Cùng ô, cùng seed, cùng code: không đổi protocol.
      ch_ 0 g_llp_feat8 tabicl_v2 v1c_g_npz --split dev --settings llp --levels "llp=8" --datasets $LD --weak_variant feat
      ch_ 0 g_llp_iid32 tabicl_v2 v1c_g_npz --split dev --settings llp --levels "llp=32" --datasets $LD
      ch_ 0 g_pll_inst tabicl_v2 v1c_g_npz --split dev --settings pll --levels "pll=0.6" --datasets $PD --weak_variant inst
      ch_ 0 f_pll_s2_icl tabicl_v2 v1c_f_npz --split s2 --eval-ok --explore --settings pll --levels "pll=0.6" --sweeps 10
      ch_ 0 f_llp_eval_icl tabicl_v2 v1c_f_npz --split eval --eval-ok --settings llp --levels "llp=8" --datasets $LLP_E --sweeps 10
      ch_ 0 f_llp_dev_icl tabicl_v2 v1c_f_npz --split dev --settings llp --levels "llp=8" --datasets $LD --sweeps 10
      ch_ 0 f_pll_eval_icl tabicl_v2 v1c_f_npz --split eval --eval-ok --settings pll --levels "pll=0.6" --datasets $PLL_E --sweeps 10
      ch_ 0 f_pll_dev_icl tabicl_v2 v1c_f_npz --split dev --settings pll --levels "pll=0.6" --datasets $PD --sweeps 10
      ch_ 0 f_llp_s2_icl_sh2 tabicl_v2 v1c_f_npz --split s2 --eval-ok --explore --settings llp --levels "llp=8" --sweeps 10 --shard 2/3 ) &
    ( run_ 1 chainmap_tabicl v1c_chain_map.py --npz_glob "$NPZ/v1_tabicl_v2_*.npz"
      v1cells_ 1 d tabicl_v2 v1c_probe_npz --sweeps 3
      ch_ 1 e_tabicl_v2_pll_f1 tabicl_v2 v1c_e_npz --split eval --eval-ok --settings pll --levels "pll=0.6" --datasets $F1P --seeds 0
      ch_ 1 e_tabicl_v2_llp_f1 tabicl_v2 v1c_e_npz --split eval --eval-ok --settings llp --levels "llp=8" --datasets $F1L --seeds 0
      newtfm_ 1 tabdpt
      ch_ 1 f_llp_s2_icl_sh0 tabicl_v2 v1c_f_npz --split s2 --eval-ok --explore --settings llp --levels "llp=8" --sweeps 10 --shard 0/3
      ch_ 1 f_llp_s2_icl_sh1 tabicl_v2 v1c_f_npz --split s2 --eval-ok --explore --settings llp --levels "llp=8" --sweeps 10 --shard 1/3 ) &
    wait ;;
  S2C1|S2C2)
    b=$(( (${LOT#S2C} - 1) * 4 ))
    for j in 0 1 2 3; do cpu "base_$((b+j))" --shard $((b+j))/8 & done; wait ;;
  *) echo "lượt không hợp lệ: $LOT"; exit 2 ;;
esac
echo "lượt $LOT kết thúc: $(date)"
