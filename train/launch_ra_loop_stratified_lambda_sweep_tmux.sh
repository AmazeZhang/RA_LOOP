#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/imc/yzy/RA_LOOP
SESSION=ra_loop_stratified_lambda_sweep
LOG_DIR="${PROJECT_ROOT}/logs/ra_loop_stratified_lambda_sweep_20260722"
LAMBDA0_TRAINER="${PROJECT_ROOT}/train/ra_loop_stratified_lambda0_multitask.sh"
LAMBDA025_TRAINER="${PROJECT_ROOT}/train/ra_loop_stratified_lambda025_multitask.sh"
LAMBDA0_OUTPUT="${PROJECT_ROOT}/outputs/ra_loop_spatial_stratified_lambda0_multitask"
LAMBDA025_OUTPUT="${PROJECT_ROOT}/outputs/ra_loop_spatial_stratified_lambda025_multitask"

if [[ $# -eq 0 || ( $# -eq 1 && "$1" == "--plan" ) ]]; then
  echo "Safe plan only; no tmux session, CUDA context, log, or output is created."
  echo "session=${SESSION}"
  echo "lambda=0.0  physical_gpu=7 log=${LOG_DIR}/lambda0_gpu7.log"
  "${LAMBDA0_TRAINER}" --print-command
  echo "lambda=0.25 physical_gpu=6 log=${LOG_DIR}/lambda025_gpu6.log"
  "${LAMBDA025_TRAINER}" --print-command
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
for path in "${LOG_DIR}" "${LAMBDA0_OUTPUT}" "${LAMBDA025_OUTPUT}"; do
  if [[ -e "${path}" ]]; then
    echo "Refusing to reuse path: ${path}" >&2
    exit 4
  fi
done

check_gpu() {
  local gpu_id=$1 status used total util temp
  status=$(nvidia-smi --id="${gpu_id}" \
    --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu \
    --format=csv,noheader,nounits)
  IFS=',' read -r used total util temp <<< "${status}"
  used=${used//[[:space:]]/}
  total=${total//[[:space:]]/}
  util=${util//[[:space:]]/}
  temp=${temp//[[:space:]]/}
  if (( used > 1024 || util > 10 || temp > 75 )); then
    echo "Refusing GPU ${gpu_id}: used=${used}/${total} MiB util=${util}% temp=${temp}C" >&2
    exit 5
  fi
  echo "GPU ${gpu_id} ready: used=${used}/${total} MiB util=${util}% temp=${temp}C"
}

check_gpu 7
check_gpu 6
mkdir -p "${LOG_DIR}"

lambda0_command="bash '${LAMBDA0_TRAINER}' --run 7 2>&1 | tee '${LOG_DIR}/lambda0_gpu7.log'; code=\${PIPESTATUS[0]}; echo '[RA_LOOP_EXIT]' \${code} | tee -a '${LOG_DIR}/lambda0_gpu7.log'; exec bash"
lambda025_command="bash '${LAMBDA025_TRAINER}' --run 6 2>&1 | tee '${LOG_DIR}/lambda025_gpu6.log'; code=\${PIPESTATUS[0]}; echo '[RA_LOOP_EXIT]' \${code} | tee -a '${LOG_DIR}/lambda025_gpu6.log'; exec bash"

tmux new-session -d -s "${SESSION}" -n lambda0_gpu7 "${lambda0_command}"
tmux new-window -t "${SESSION}" -n lambda025_gpu6 "${lambda025_command}"

echo "Started ${SESSION}."
echo "Attach: tmux attach -t ${SESSION}"
echo "Windows: lambda0_gpu7, lambda025_gpu6"
