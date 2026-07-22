#!/usr/bin/env bash
set -euo pipefail

# Evaluate one stratified checkpoint on all four exactly matched Spatial tasks.
# Six invocations (steps 5..30) are intended to run in parallel on GPUs 1..6.

PROJECT_ROOT=/home/imc/yzy/RA_LOOP
PYTHON_BIN=/home/imc/anaconda3/envs/ript_vla_openvla_oft/bin/python
EVALUATOR="${PROJECT_ROOT}/eval/ra_loop_checkpoint_pair_eval.py"
OUTPUT_ROOT="${PROJECT_ROOT}/logs/ra_loop_stratified_checkpoint_eval_20260722"
VALID_STEPS=(5 10 15 20 25 30)
TASKS=(
  pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate
  pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate
  pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate
  pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate
)
SEEDS=(20260720 20260730 20260730 20260730)

is_valid_step() {
  local requested=$1 step
  for step in "${VALID_STEPS[@]}"; do
    if [[ "${requested}" == "${step}" ]]; then
      return 0
    fi
  done
  return 1
}

build_command() {
  local task_id=$1 step=$2 gpu=$3
  command=(
    env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
    "${PYTHON_BIN}" "${EVALUATOR}"
    --checkpoint-source stratified
    --checkpoint-step "${step}"
    --task-name "${TASKS[$task_id]}"
    --num-pairs 6
    --fixed-l2 0.1
    --perturb-seed "${SEEDS[$task_id]}"
    --gpu-id "${gpu}"
    --output-dir "${OUTPUT_ROOT}/step${step}/task${task_id}"
  )
}

if [[ $# -eq 0 || "${1:-}" == "--plan" ]]; then
  if [[ $# -gt 1 ]]; then
    exit 2
  fi
  export CUDA_VISIBLE_DEVICES=''
  echo "CPU-only plans; no model/env/CUDA creation"
  for step in "${VALID_STEPS[@]}"; do
    for task_id in 0 1 2 3; do
      build_command "${task_id}" "${step}" 0
      "${command[@]}"
    done
  done
  exit 0
fi

if [[ $# -ne 3 || "$1" != "--run" ]] || ! is_valid_step "$2" || [[ ! "$3" =~ ^[0-7]$ ]]; then
  echo "Usage: bash $0 [--plan | --run <STEP:5|10|15|20|25|30> <GPU_ID>]" >&2
  exit 2
fi
STEP=$2
GPU_ID=$3
STEP_OUTPUT="${OUTPUT_ROOT}/step${STEP}"

if [[ -e "${STEP_OUTPUT}" ]]; then
  echo "Refusing existing step output: ${STEP_OUTPUT}" >&2
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

echo "Starting stratified checkpoint step=${STEP} GPU=${GPU_ID}: 4 tasks x 12 episodes"
for task_id in 0 1 2 3; do
  echo "[TASK_START] task=${task_id} step=${STEP}"
  build_command "${task_id}" "${STEP}" "${GPU_ID}"
  command+=(--execute)
  "${command[@]}"
  echo "[TASK_DONE] task=${task_id} step=${STEP}"
done
echo "[STEP_DONE] step=${STEP} episodes=48"
