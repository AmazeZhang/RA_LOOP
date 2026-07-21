#!/usr/bin/env bash
set -euo pipefail

# Safe default: validate all base/step5/10/15/20 paired-eval commands with CUDA hidden.
# GPU execution requires: --run <STEP> <GPU_ID>

PROJECT_ROOT=/home/imc/yzy/RA_LOOP
PYTHON_BIN=/home/imc/anaconda3/envs/ript_vla_openvla_oft/bin/python
EVALUATOR="${PROJECT_ROOT}/eval/ra_loop_checkpoint_pair_eval.py"
OUTPUT_ROOT="${PROJECT_ROOT}/logs/ra_loop_checkpoint_pair_eval_20260721"
VALID_STEPS=(0 5 10 15 20)

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
  local step=$1 gpu=$2
  command=(
    env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
    "${PYTHON_BIN}" "${EVALUATOR}"
    --checkpoint-step "${step}"
    --num-pairs 10
    --fixed-l2 0.1
    --perturb-seed 20260720
    --gpu-id "${gpu}"
    --output-dir "${OUTPUT_ROOT}/step${step}"
  )
}

if [[ $# -eq 0 || "${1:-}" == "--plan" ]]; then
  if [[ $# -gt 1 ]]; then
    echo "Usage: bash $0 [--plan | --print-commands | --run <STEP> <GPU_ID>]" >&2
    exit 2
  fi
  export CUDA_VISIBLE_DEVICES=''
  echo "CPU-only paired-eval plan; no model/env/CUDA creation"
  for step in "${VALID_STEPS[@]}"; do
    build_command "${step}" 0
    "${command[@]}"
  done
  exit 0
fi

if [[ "$1" == "--print-commands" ]]; then
  if [[ $# -ne 1 ]]; then
    exit 2
  fi
  for index in "${!VALID_STEPS[@]}"; do
    step=${VALID_STEPS[$index]}
    gpu=$((index + 1))
    build_command "${step}" "${gpu}"
    command+=(--execute)
    printf 'CUDA_VISIBLE_DEVICES=%s ' "${gpu}"
    printf '%q ' "${command[@]}"
    printf '\n'
  done
  exit 0
fi

if [[ $# -ne 3 || "$1" != "--run" ]] || ! is_valid_step "$2" || [[ ! "$3" =~ ^[0-7]$ ]]; then
  echo "Usage: bash $0 [--plan | --print-commands | --run <STEP> <GPU_ID>]" >&2
  exit 2
fi
STEP=$2
GPU_ID=$3

if [[ -e "${OUTPUT_ROOT}/step${STEP}" ]]; then
  echo "Refusing existing evaluation output: ${OUTPUT_ROOT}/step${STEP}" >&2
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

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LIBERO_CONFIG_PATH="${PROJECT_ROOT}/.libero_official"
export PYTHONPATH="${PROJECT_ROOT}:/home/imc/code/LIBERO-official:/home/imc/code/ript-vla"
export PYTHONNOUSERSITE=1
export NUMBA_CACHE_DIR=/tmp/ra_loop_numba_cache
export MPLCONFIGDIR=/tmp/ra_loop_mpl_cache

build_command "${STEP}" "${GPU_ID}"
command+=(--execute)
echo "Starting paired eval: step=${STEP} GPU=${GPU_ID} episodes=20 (10 anchor + 10 fixed-L2)"
exec "${command[@]}"
