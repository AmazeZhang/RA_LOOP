#!/usr/bin/env bash
set -euo pipefail

# Three unseen Spatial tasks, step 0 versus step 5, six paired initial states.
# Safe default prints CPU-only plans; GPU execution requires --run <TASK_ID> <STEP> <GPU_ID>.

PROJECT_ROOT=/home/imc/yzy/RA_LOOP
PYTHON_BIN=/home/imc/anaconda3/envs/ript_vla_openvla_oft/bin/python
EVALUATOR="${PROJECT_ROOT}/eval/ra_loop_checkpoint_pair_eval.py"
OUTPUT_ROOT="${PROJECT_ROOT}/logs/ra_loop_disambiguation_eval_20260721"
TASKS=(
  pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate
  pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate
  pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate
)

build_command() {
  local task_id=$1 step=$2 gpu=$3
  local task_name=${TASKS[$task_id]}
  command=(
    env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
    "${PYTHON_BIN}" "${EVALUATOR}"
    --checkpoint-step "${step}"
    --task-name "${task_name}"
    --num-pairs 6
    --fixed-l2 0.1
    --perturb-seed 20260730
    --gpu-id "${gpu}"
    --output-dir "${OUTPUT_ROOT}/task${task_id}_step${step}"
  )
}

if [[ $# -eq 0 || "${1:-}" == "--plan" ]]; then
  if [[ $# -gt 1 ]]; then
    exit 2
  fi
  export CUDA_VISIBLE_DEVICES=''
  echo "CPU-only disambiguation plans; no model/env/CUDA creation"
  for task_id in 0 1 2; do
    for step in 0 5; do
      build_command "${task_id}" "${step}" 0
      "${command[@]}"
    done
  done
  exit 0
fi

if [[ $# -ne 4 || "$1" != "--run" || ! "$2" =~ ^[0-2]$ || ! "$3" =~ ^(0|5)$ || ! "$4" =~ ^[0-7]$ ]]; then
  echo "Usage: bash $0 [--plan | --run <TASK_ID:0-2> <STEP:0|5> <GPU_ID>]" >&2
  exit 2
fi

TASK_ID=$2
STEP=$3
GPU_ID=$4
TASK_NAME=${TASKS[$TASK_ID]}
OUTPUT_DIR="${OUTPUT_ROOT}/task${TASK_ID}_step${STEP}"

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing existing output: ${OUTPUT_DIR}" >&2
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

mkdir -p "${OUTPUT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LIBERO_CONFIG_PATH="${PROJECT_ROOT}/.libero_official"
export PYTHONPATH="${PROJECT_ROOT}:/home/imc/code/LIBERO-official:/home/imc/code/ript-vla"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export NUMBA_CACHE_DIR=/tmp/ra_loop_numba_cache
export MPLCONFIGDIR=/tmp/ra_loop_mpl_cache

build_command "${TASK_ID}" "${STEP}" "${GPU_ID}"
command+=(--execute)
echo "Starting task=${TASK_ID} step=${STEP} GPU=${GPU_ID}: ${TASK_NAME}"
exec "${command[@]}"
