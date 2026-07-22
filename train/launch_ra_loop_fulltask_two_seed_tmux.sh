#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/imc/yzy/RA_LOOP
SESSION=ra_loop_fulltask_two_seed
LOG_DIR="${PROJECT_ROOT}/logs/ra_loop_fulltask_two_seed_20260722"
TRAINER_10000="${PROJECT_ROOT}/train/ra_loop_fulltask_seed10000.sh"
TRAINER_20000="${PROJECT_ROOT}/train/ra_loop_fulltask_seed20000.sh"
OUTPUT_10000="${PROJECT_ROOT}/outputs/ra_loop_spatial_fulltask_seed10000"
OUTPUT_20000="${PROJECT_ROOT}/outputs/ra_loop_spatial_fulltask_seed20000"

if [[ $# -eq 0 || ( $# -eq 1 && "$1" == "--plan" ) ]]; then
  echo "Safe plan only; no tmux session, CUDA context, log, or output is created."
  echo "session=${SESSION} runs=2 tasks=10 demos=500 steps_per_run=100 expected=17-22h"
  echo "seed=10000 perturb_seed=20260720 physical_gpu=7"
  "${TRAINER_10000}" --print-command
  echo "seed=20000 perturb_seed=20270720 physical_gpu=6"
  "${TRAINER_20000}" --print-command
  exit 0
fi

if [[ $# -ne 1 || "$1" != "--run" ]]; then
  echo "Usage: bash $0 [--plan | --run]" >&2
  exit 2
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "Refusing to reuse tmux session: ${SESSION}" >&2
  exit 3
fi
for path in "${LOG_DIR}" "${OUTPUT_10000}" "${OUTPUT_20000}"; do
  if [[ -e "${path}" ]]; then
    echo "Refusing to reuse path: ${path}" >&2
    exit 4
  fi
done

AVAILABLE_BYTES=$(df --output=avail -B1 "${PROJECT_ROOT}" | tail -n 1)
AVAILABLE_BYTES=${AVAILABLE_BYTES//[[:space:]]/}
if [[ ! "${AVAILABLE_BYTES}" =~ ^[0-9]+$ ]] || (( AVAILABLE_BYTES < 60000000000 )); then
  echo "Refusing two-seed training: less than 60 GB available" >&2
  exit 6
fi

check_gpu() {
  local gpu_id=$1 status used total util temp
  status=$(nvidia-smi --id="${gpu_id}" \
    --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu \
    --format=csv,noheader,nounits)
  IFS=',' read -r used total util temp <<< "${status}"
  used=${used//[[:space:]]/}; total=${total//[[:space:]]/}
  util=${util//[[:space:]]/}; temp=${temp//[[:space:]]/}
  if (( used > 1024 || util > 10 || temp > 75 )); then
    echo "Refusing GPU ${gpu_id}: used=${used}/${total} MiB util=${util}% temp=${temp}C" >&2
    exit 5
  fi
  echo "GPU ${gpu_id} ready: used=${used}/${total} MiB util=${util}% temp=${temp}C"
}
check_gpu 7
check_gpu 6

mkdir -p "${LOG_DIR}"
command_10000="bash '${TRAINER_10000}' --run 7 2>&1 | tee '${LOG_DIR}/seed10000_gpu7.log'; code=\${PIPESTATUS[0]}; echo '[RA_LOOP_EXIT]' \${code} | tee -a '${LOG_DIR}/seed10000_gpu7.log'; exec bash"
command_20000="bash '${TRAINER_20000}' --run 6 2>&1 | tee '${LOG_DIR}/seed20000_gpu6.log'; code=\${PIPESTATUS[0]}; echo '[RA_LOOP_EXIT]' \${code} | tee -a '${LOG_DIR}/seed20000_gpu6.log'; exec bash"

tmux new-session -d -s "${SESSION}" -n seed10000_gpu7 "${command_10000}"
tmux new-window -t "${SESSION}" -n seed20000_gpu6 "${command_20000}"

echo "Started ${SESSION}."
echo "Attach: tmux attach -t ${SESSION}"
echo "Windows: seed10000_gpu7, seed20000_gpu6"
