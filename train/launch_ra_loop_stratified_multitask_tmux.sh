#!/usr/bin/env bash
set -euo pipefail

# Persistent launcher for the controlled mode-stratified overnight run.
# Safe default prints the plan; --run performs final checks and launches it.

PROJECT_ROOT=/home/imc/yzy/RA_LOOP
SESSION=ra_loop_stratified_multitask
WINDOW=run1
TRAINER="${PROJECT_ROOT}/train/ra_loop_stratified_multitask.sh"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/ra_loop_spatial_stratified_multitask"
LOG_ROOT="${PROJECT_ROOT}/logs/ra_loop_spatial_stratified_multitask_20260721"
LOG_FILE="${LOG_ROOT}/run1.log"

if [[ $# -eq 0 || "${1:-}" == "--plan" ]]; then
  if [[ $# -gt 1 ]]; then
    exit 2
  fi
  echo "Training is NOT started. Final command after confirmation:"
  echo "  bash $0 --run 7"
  echo "session=${SESSION} GPU=7 steps=35 K=8 tasks=4 expected=6-8h"
  exec bash "${TRAINER}" --print-command
fi

if [[ $# -ne 2 || "$1" != "--run" || ! "$2" =~ ^[0-7]$ ]]; then
  echo "Usage: bash $0 [--plan | --run <GPU_ID>]" >&2
  exit 2
fi
GPU_ID=$2

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "Refusing existing tmux session: ${SESSION}" >&2
  exit 3
fi
if [[ -e "${OUTPUT_ROOT}" || -e "${LOG_ROOT}" ]]; then
  echo "Refusing existing output/log root: ${OUTPUT_ROOT} or ${LOG_ROOT}" >&2
  exit 4
fi

mkdir -p "${LOG_ROOT}"
tmux new-session -d -s "${SESSION}" -n "${WINDOW}" \
  "cd ${PROJECT_ROOT} && PYTHONUNBUFFERED=1 bash ${TRAINER} --run ${GPU_ID} 2>&1 | tee ${LOG_FILE}; run_status=\${PIPESTATUS[0]}; echo '[RA_LOOP_EXIT]' \${run_status} | tee -a ${LOG_FILE}; exec bash -i"

echo "Started tmux=${SESSION}:${WINDOW} GPU=${GPU_ID} log=${LOG_FILE}"
