#!/usr/bin/env bash
set -euo pipefail

# One held-out, paired evaluation job.  The protocol is frozen in the matching
# document; this wrapper refuses both existing outputs and busy GPUs.

PROJECT_ROOT=/home/imc/yzy/RA_LOOP
PYTHON_BIN=/home/imc/anaconda3/envs/ript_vla_openvla_oft/bin/python
EVALUATOR="${PROJECT_ROOT}/eval/ra_loop_checkpoint_pair_eval.py"
OUTPUT_ROOT="${PROJECT_ROOT}/logs/ra_loop_counterfactual_heldout_eval_20260725"
INIT_START=6
NUM_PAIRS=20
PERTURB_SEED=20260801
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

contains() { local wanted=$1; shift; local value; for value in "$@"; do [[ "$wanted" == "$value" ]] && return 0; done; return 1; }

valid_candidate() {
  local source=$1 step=$2
  [[ "$source" == pilot && "$step" == 5 ]] && return 0
  [[ "$source" == counterfactual_cra_only && "$step" == 40 ]] && return 0
  [[ "$source" == counterfactual_npc && "$step" == 40 ]] && return 0
  return 1
}

build_command() {
  local source=$1 step=$2 task_id=$3 gpu=$4
  command=(
    env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
    "$PYTHON_BIN" "$EVALUATOR"
    --checkpoint-source "$source" --checkpoint-step "$step"
    --task-name "${TASKS[$task_id]}"
    --init-start "$INIT_START" --num-pairs "$NUM_PAIRS"
    --fixed-l2 0.1 --perturb-seed "$PERTURB_SEED" --gpu-id "$gpu"
    --output-dir "${OUTPUT_ROOT}/${source}/step${step}/task${task_id}"
  )
}

if [[ $# -eq 0 || "${1:-}" == --plan ]]; then
  [[ $# -le 1 ]] || exit 2
  export CUDA_VISIBLE_DEVICES=''
  echo "CPU-only held-out plans; no model, environment, or CUDA creation"
  for source_step in pilot:5 counterfactual_cra_only:40 counterfactual_npc:40; do
    source=${source_step%%:*}; step=${source_step##*:}
    for task_id in "${!TASKS[@]}"; do build_command "$source" "$step" "$task_id" 0; "${command[@]}"; done
  done
  exit 0
fi

if [[ $# -ne 5 || "$1" != --run ]] \
  || ! contains "$2" pilot counterfactual_cra_only counterfactual_npc \
  || ! valid_candidate "$2" "$3" \
  || [[ ! "$4" =~ ^[0-9]$ ]] || (( $4 < 0 || $4 >= ${#TASKS[@]} )) \
  || [[ ! "$5" =~ ^[2-7]$ ]]; then
  echo "Usage: bash $0 [--plan | --run <SOURCE> <STEP> <TASK_ID:0-9> <GPU_ID:2-7>]" >&2
  exit 2
fi
SOURCE=$2; STEP=$3; TASK_ID=$4; GPU_ID=$5
TASK_OUTPUT="${OUTPUT_ROOT}/${SOURCE}/step${STEP}/task${TASK_ID}"
[[ ! -e "$TASK_OUTPUT" ]] || { echo "Refusing existing output: $TASK_OUTPUT" >&2; exit 3; }

GPU_STATUS=$(nvidia-smi --id="$GPU_ID" --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits)
IFS=',' read -r GPU_USED GPU_TOTAL GPU_UTIL GPU_TEMP <<< "$GPU_STATUS"
GPU_USED=${GPU_USED//[[:space:]]/}; GPU_TOTAL=${GPU_TOTAL//[[:space:]]/}; GPU_UTIL=${GPU_UTIL//[[:space:]]/}; GPU_TEMP=${GPU_TEMP//[[:space:]]/}
if (( GPU_USED > 1024 || GPU_UTIL > 10 || GPU_TEMP > 75 )); then
  echo "Refusing GPU ${GPU_ID}: used=${GPU_USED}/${GPU_TOTAL} MiB util=${GPU_UTIL}% temp=${GPU_TEMP}C" >&2
  exit 4
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID" MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export LIBERO_CONFIG_PATH="${PROJECT_ROOT}/.libero_official"
export PYTHONPATH="${PROJECT_ROOT}:/home/imc/code/LIBERO-official:/home/imc/code/ript-vla"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 NUMBA_CACHE_DIR=/tmp/ra_loop_numba_cache MPLCONFIGDIR=/tmp/ra_loop_mpl_cache
build_command "$SOURCE" "$STEP" "$TASK_ID" "$GPU_ID"
command+=(--execute)
"${command[@]}"
