#!/usr/bin/env bash
set -euo pipefail

# Evaluate one counterfactual gate checkpoint on the four matched Spatial tasks.

PROJECT_ROOT=/home/imc/yzy/RA_LOOP
PYTHON_BIN=/home/imc/anaconda3/envs/ript_vla_openvla_oft/bin/python
EVALUATOR="${PROJECT_ROOT}/eval/ra_loop_checkpoint_pair_eval.py"
OUTPUT_ROOT="${PROJECT_ROOT}/logs/ra_loop_counterfactual_checkpoint_eval_20260725"
VALID_SOURCES=(counterfactual_cra_only counterfactual_npc)
VALID_STEPS=(20 30 40)
TASKS=(
  pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate
  pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate
  pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate
  pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate
)
SEEDS=(20260720 20260730 20260730 20260730)

contains() {
  local requested=$1
  shift
  local value
  for value in "$@"; do
    [[ "${requested}" == "${value}" ]] && return 0
  done
  return 1
}

build_command() {
  local source=$1 task_id=$2 step=$3 gpu=$4
  command=(
    env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
    "${PYTHON_BIN}" "${EVALUATOR}"
    --checkpoint-source "${source}"
    --checkpoint-step "${step}"
    --task-name "${TASKS[$task_id]}"
    --num-pairs 6
    --fixed-l2 0.1
    --perturb-seed "${SEEDS[$task_id]}"
    --gpu-id "${gpu}"
    --output-dir "${OUTPUT_ROOT}/${source}/step${step}/task${task_id}"
  )
}

if [[ $# -eq 0 || "${1:-}" == "--plan" ]]; then
  [[ $# -le 1 ]] || exit 2
  export CUDA_VISIBLE_DEVICES=''
  echo "CPU-only plans; no model/env/CUDA creation"
  for source in "${VALID_SOURCES[@]}"; do
    for step in "${VALID_STEPS[@]}"; do
      for task_id in 0 1 2 3; do
        build_command "${source}" "${task_id}" "${step}" 0
        "${command[@]}"
      done
    done
  done
  exit 0
fi

if [[ $# -ne 4 || "$1" != "--run" ]] \
  || ! contains "$2" "${VALID_SOURCES[@]}" \
  || ! contains "$3" "${VALID_STEPS[@]}" \
  || [[ ! "$4" =~ ^[2-7]$ ]]; then
  echo "Usage: bash $0 [--plan | --run <SOURCE> <STEP> <GPU_ID:2-7>]" >&2
  exit 2
fi
SOURCE=$2
STEP=$3
GPU_ID=$4
STEP_OUTPUT="${OUTPUT_ROOT}/${SOURCE}/step${STEP}"

if [[ -e "${STEP_OUTPUT}" ]]; then
  echo "Refusing existing checkpoint output: ${STEP_OUTPUT}" >&2
  exit 3
fi

GPU_STATUS=$(nvidia-smi --id="${GPU_ID}" \
  --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu \
  --format=csv,noheader,nounits)
IFS=',' read -r GPU_USED GPU_TOTAL GPU_UTIL GPU_TEMP <<< "${GPU_STATUS}"
GPU_USED=${GPU_USED//[[:space:]]/}
GPU_TOTAL=${GPU_TOTAL//[[:space:]]/}
GPU_UTIL=${GPU_UTIL//[[:space:]]/}
GPU_TEMP=${GPU_TEMP//[[:space:]]/}
if (( GPU_USED > 1024 || GPU_UTIL > 10 || GPU_TEMP > 75 )); then
  echo "Refusing GPU ${GPU_ID}: used=${GPU_USED}/${GPU_TOTAL} MiB util=${GPU_UTIL}% temp=${GPU_TEMP}C" >&2
  exit 4
fi

mkdir -p "${STEP_OUTPUT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LIBERO_CONFIG_PATH="${PROJECT_ROOT}/.libero_official"
export PYTHONPATH="${PROJECT_ROOT}:/home/imc/code/LIBERO-official:/home/imc/code/ript-vla"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export NUMBA_CACHE_DIR=/tmp/ra_loop_numba_cache
export MPLCONFIGDIR=/tmp/ra_loop_mpl_cache

echo "Starting ${SOURCE} step=${STEP} GPU=${GPU_ID}: 4 tasks x 12 episodes"
for task_id in 0 1 2 3; do
  echo "[TASK_START] source=${SOURCE} task=${task_id} step=${STEP}"
  build_command "${SOURCE}" "${task_id}" "${STEP}" "${GPU_ID}"
  command+=(--execute)
  "${command[@]}"
  echo "[TASK_DONE] source=${SOURCE} task=${task_id} step=${STEP}"
done
echo "[CHECKPOINT_DONE] source=${SOURCE} step=${STEP} episodes=48"
