#!/usr/bin/env bash
set -euo pipefail

# Warm-start-only calibration of two stronger fixed-L2 perturbation radii.

PROJECT_ROOT=/home/imc/yzy/RA_LOOP
PYTHON_BIN=/home/imc/anaconda3/envs/ript_vla_openvla_oft/bin/python
EVALUATOR="${PROJECT_ROOT}/eval/ra_loop_checkpoint_pair_eval.py"
OUTPUT_ROOT="${PROJECT_ROOT}/logs/ra_loop_strength_calibration_20260726"
INIT_START=26
NUM_PAIRS=10
PERTURB_SEED=20260811
TASKS=(
  pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate
  pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate
  pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate
  pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate
  pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate
  pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate
  pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate
  pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate
  pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate
  pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate
)

strength_value() {
  case "$1" in
    0p15) echo 0.15 ;;
    0p20) echo 0.20 ;;
    *) return 1 ;;
  esac
}

build_command() {
  local strength_tag=$1 task_id=$2 gpu=$3
  local strength
  strength=$(strength_value "$strength_tag")
  command=(
    env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
    "$PYTHON_BIN" "$EVALUATOR"
    --checkpoint-source pilot --checkpoint-step 5
    --task-name "${TASKS[$task_id]}"
    --init-start "$INIT_START" --num-pairs "$NUM_PAIRS"
    --fixed-l2 "$strength" --perturb-seed "$PERTURB_SEED"
    --gpu-id "$gpu"
    --output-dir "${OUTPUT_ROOT}/${strength_tag}/task${task_id}"
  )
}

if [[ $# -eq 0 || "${1:-}" == --plan ]]; then
  [[ $# -le 1 ]] || exit 2
  export CUDA_VISIBLE_DEVICES=''
  echo "CPU-only calibration plans; no model, environment, or CUDA creation"
  for strength_tag in 0p15 0p20; do
    for task_id in "${!TASKS[@]}"; do
      build_command "$strength_tag" "$task_id" 0
      "${command[@]}"
    done
  done
  exit 0
fi

if [[ $# -ne 5 || "$1" != --run ]] \
  || ! strength_value "$2" >/dev/null \
  || [[ ! "$3" =~ ^[0-9]$ ]] || (( $3 < 0 || $3 >= ${#TASKS[@]} )) \
  || [[ ! "$4" =~ ^[2-7]$ ]] \
  || [[ ! "$5" =~ ^(calibration)$ ]]; then
  echo "Usage: bash $0 [--plan | --run <0p15|0p20> <TASK_ID:0-9> <GPU_ID:2-7> calibration]" >&2
  exit 2
fi
STRENGTH_TAG=$2
TASK_ID=$3
GPU_ID=$4
TASK_OUTPUT="${OUTPUT_ROOT}/${STRENGTH_TAG}/task${TASK_ID}"
[[ ! -e "$TASK_OUTPUT" ]] || { echo "Refusing existing output: $TASK_OUTPUT" >&2; exit 3; }

GPU_STATUS=$(nvidia-smi --id="$GPU_ID" \
  --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu \
  --format=csv,noheader,nounits)
IFS=',' read -r GPU_USED GPU_TOTAL GPU_UTIL GPU_TEMP <<< "$GPU_STATUS"
GPU_USED=${GPU_USED//[[:space:]]/}
GPU_TOTAL=${GPU_TOTAL//[[:space:]]/}
GPU_UTIL=${GPU_UTIL//[[:space:]]/}
GPU_TEMP=${GPU_TEMP//[[:space:]]/}
if (( GPU_USED > 1024 || GPU_UTIL > 10 || GPU_TEMP > 75 )); then
  echo "Refusing GPU ${GPU_ID}: used=${GPU_USED}/${GPU_TOTAL} MiB util=${GPU_UTIL}% temp=${GPU_TEMP}C" >&2
  exit 4
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LIBERO_CONFIG_PATH="${PROJECT_ROOT}/.libero_official"
export PYTHONPATH="${PROJECT_ROOT}:/home/imc/code/LIBERO-official:/home/imc/code/ript-vla"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export NUMBA_CACHE_DIR=/tmp/ra_loop_numba_cache
export MPLCONFIGDIR=/tmp/ra_loop_mpl_cache
build_command "$STRENGTH_TAG" "$TASK_ID" "$GPU_ID"
command+=(--execute)
"${command[@]}"
