#!/usr/bin/env bash
# Nghiên cứu 2 (S2) — chạy tập đánh giá MỚI theo de-tai-moi/S2/S2_PLAN.md (chỉ sau khi S2 đã đóng băng).
# Dùng:  bash run_s2_lot.sh <LOT> /kaggle/working        LOT ∈ {S2X S2D S2T S2C1 S2C2 S2G S2R S2E V0A V0B V1A V1B V1C}
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
#                      phí (F); GPU 1 thêm D, TabICL trên F1, TabDPT, cơ chế nhãn yếu mới (G), F. Khoảng 2 phiên. Cần s2_code_v9
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
      v1cells_ 0 f tabpfn_v2 v1c_f_npz --sweeps 10 ) &
    ( run_ 1 chainmap_tabicl v1c_chain_map.py --npz_glob "$NPZ/v1_tabicl_v2_*.npz"
      v1cells_ 1 d tabicl_v2 v1c_probe_npz --sweeps 3
      ch_ 1 e_tabicl_v2_pll_f1 tabicl_v2 v1c_e_npz --split eval --eval-ok --settings pll --levels "pll=0.6" --datasets $F1P --seeds 0
      ch_ 1 e_tabicl_v2_llp_f1 tabicl_v2 v1c_e_npz --split eval --eval-ok --settings llp --levels "llp=8" --datasets $F1L --seeds 0
      newtfm_ 1 tabdpt
      ch_ 1 g_llp_feat8 tabicl_v2 v1c_g_npz --split dev --settings llp --levels "llp=8" --datasets $LD --weak_variant feat
      ch_ 1 g_llp_iid32 tabicl_v2 v1c_g_npz --split dev --settings llp --levels "llp=32" --datasets $LD
      ch_ 1 g_pll_inst tabicl_v2 v1c_g_npz --split dev --settings pll --levels "pll=0.6" --datasets $PD --weak_variant inst
      v1cells_ 1 f tabicl_v2 v1c_f_npz --sweeps 10 ) &
    wait ;;
  S2C1|S2C2)
    b=$(( (${LOT#S2C} - 1) * 4 ))
    for j in 0 1 2 3; do cpu "base_$((b+j))" --shard $((b+j))/8 & done; wait ;;
  *) echo "lượt không hợp lệ: $LOT"; exit 2 ;;
esac
echo "lượt $LOT kết thúc: $(date)"
