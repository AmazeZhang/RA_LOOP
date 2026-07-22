#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/imc/yzy/RA_LOOP
SESSION=ra_loop_fulltask_lambda_ablation
LOG_DIR="${PROJECT_ROOT}/logs/ra_loop_fulltask_lambda_ablation_20260722"
TRAINERS=(
  "${PROJECT_ROOT}/train/ra_loop_fulltask_seed10000.sh"
  "${PROJECT_ROOT}/train/ra_loop_fulltask_seed20000.sh"
  "${PROJECT_ROOT}/train/ra_loop_fulltask_lambda0_seed10000.sh"
  "${PROJECT_ROOT}/train/ra_loop_fulltask_lambda0_seed20000.sh"
)
LABELS=(lambda05_seed10000 lambda05_seed20000 lambda0_seed10000 lambda0_seed20000)
GPUS=(7 6 5 4)
OUTPUTS=(
  "${PROJECT_ROOT}/outputs/ra_loop_spatial_fulltask_seed10000"
  "${PROJECT_ROOT}/outputs/ra_loop_spatial_fulltask_seed20000"
  "${PROJECT_ROOT}/outputs/ra_loop_spatial_fulltask_lambda0_seed10000"
  "${PROJECT_ROOT}/outputs/ra_loop_spatial_fulltask_lambda0_seed20000"
)

if [[ $# -eq 0 || ( $# -eq 1 && "$1" == "--plan" ) ]]; then
  echo "Safe plan only; no tmux session, CUDA context, log, or output is created."
  echo "session=${SESSION} runs=4 design=lambda{0,0.5}xseed{10000,20000} tasks=10 demos=500 steps=100 expected=17-22h"
  for index in "${!TRAINERS[@]}"; do
    echo "${LABELS[$index]} physical_gpu=${GPUS[$index]}"
    "${TRAINERS[$index]}" --print-command
  done
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
if [[ -e "${LOG_DIR}" ]]; then
  echo "Refusing to reuse log directory: ${LOG_DIR}" >&2
  exit 4
fi
for output in "${OUTPUTS[@]}"; do
  if [[ -e "${output}" ]]; then
    echo "Refusing to reuse output: ${output}" >&2
    exit 4
  fi
done

AVAILABLE_BYTES=$(df --output=avail -B1 "${PROJECT_ROOT}" | tail -n 1)
AVAILABLE_BYTES=${AVAILABLE_BYTES//[[:space:]]/}
if [[ ! "${AVAILABLE_BYTES}" =~ ^[0-9]+$ ]] || (( AVAILABLE_BYTES < 80000000000 )); then
  echo "Refusing four-run training: less than 80 GB available" >&2
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
for gpu in "${GPUS[@]}"; do check_gpu "${gpu}"; done

mkdir -p "${LOG_DIR}"
for index in "${!TRAINERS[@]}"; do
  trainer=${TRAINERS[$index]}; label=${LABELS[$index]}; gpu=${GPUS[$index]}
  log_file="${LOG_DIR}/${label}_gpu${gpu}.log"
  pane_command="bash '${trainer}' --run '${gpu}' 2>&1 | tee '${log_file}'; code=\${PIPESTATUS[0]}; echo '[RA_LOOP_EXIT]' \${code} | tee -a '${log_file}'; exec bash"
  if [[ ${index} -eq 0 ]]; then
    tmux new-session -d -s "${SESSION}" -n "${label}_gpu${gpu}" "${pane_command}"
  else
    tmux new-window -t "${SESSION}" -n "${label}_gpu${gpu}" "${pane_command}"
  fi
done

echo "Started ${SESSION}."
echo "Attach: tmux attach -t ${SESSION}"
