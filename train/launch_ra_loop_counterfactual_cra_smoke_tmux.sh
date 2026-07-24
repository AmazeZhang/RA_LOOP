#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/imc/yzy/RA_LOOP
SESSION=ra_loop_counterfactual_cra_smoke
TRAINER="${PROJECT_ROOT}/train/ra_loop_counterfactual_cra_smoke.sh"
LOG_ROOT="${PROJECT_ROOT}/logs/ra_loop_counterfactual_cra_smoke_20260724"

if [[ $# -eq 0 || "${1:-}" == "--plan" ]]; then
  [[ $# -le 1 ]] || exit 2
  echo "Training is NOT started."
  echo "Final command: bash $0 --run <GPU_ID:2-7>"
  echo "tmux=${SESSION} profile=counterfactual_cra_smoke steps=3 calibration=1 active_opportunities=2"
  exec bash "${TRAINER}" --print-command
fi

if [[ $# -ne 2 || "$1" != "--run" || ! "$2" =~ ^[2-7]$ ]]; then
  echo "Usage: bash $0 [--plan | --run <GPU_ID:2-7>]" >&2
  exit 2
fi
GPU_ID=$2

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "Refusing existing tmux session: ${SESSION}" >&2
  exit 3
fi
if [[ -e "${LOG_ROOT}" ]]; then
  echo "Refusing existing log directory: ${LOG_ROOT}" >&2
  exit 4
fi

mkdir -p "${LOG_ROOT}"
LOG_FILE="${LOG_ROOT}/gpu${GPU_ID}.log"
PANE_COMMAND="set -o pipefail; cd '${PROJECT_ROOT}'; bash '${TRAINER}' --run '${GPU_ID}' 2>&1 | tee '${LOG_FILE}'; code=\${PIPESTATUS[0]}; echo '[RA_LOOP_EXIT]' \${code} | tee -a '${LOG_FILE}'; exec bash -i"
tmux new-session -d -s "${SESSION}" -n "gpu${GPU_ID}" "${PANE_COMMAND}"
echo "Started tmux=${SESSION} gpu=${GPU_ID} log=${LOG_FILE}"
