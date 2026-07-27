#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/imc/yzy/RA_LOOP
SESSION=ra_loop_counterfactual_soft_two_seed
LOG_ROOT="${PROJECT_ROOT}/logs/ra_loop_counterfactual_soft_two_seed_20260726"
GPUS=(6 7)
SEEDS=(10000 20000)
TRAINERS=(
  "${PROJECT_ROOT}/train/ra_loop_counterfactual_soft_gate_seed10000.sh"
  "${PROJECT_ROOT}/train/ra_loop_counterfactual_soft_gate_seed20000.sh"
)
OUTPUTS=(
  "${PROJECT_ROOT}/outputs/ra_loop_spatial_counterfactual_soft_gate_seed10000"
  "${PROJECT_ROOT}/outputs/ra_loop_spatial_counterfactual_soft_gate_seed20000"
)

if [[ $# -eq 0 || "${1:-}" == --plan ]]; then
  [[ $# -le 1 ]] || exit 2
  echo "Training is NOT started. Final command: bash $0 --run"
  echo "tmux=${SESSION} workers=2 steps=51 checkpoint=50 GPUs=6,7 expected=9-10h"
  for index in "${!GPUS[@]}"; do
    echo "GPU ${GPUS[$index]}: seed=${SEEDS[$index]} trainer=${TRAINERS[$index]}"
    bash "${TRAINERS[$index]}" --print-command
  done
  exit 0
fi

if [[ $# -ne 1 || "$1" != --run ]]; then
  echo "Usage: bash $0 [--plan | --run]" >&2
  exit 2
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Refusing existing tmux session: $SESSION" >&2
  exit 3
fi
if [[ -e "$LOG_ROOT" ]]; then
  echo "Refusing existing log directory: $LOG_ROOT" >&2
  exit 4
fi
for output in "${OUTPUTS[@]}"; do
  if [[ -e "$output" ]]; then
    echo "Refusing existing training output: $output" >&2
    exit 5
  fi
done

check_gpu() {
  local gpu_id=$1 status used total util temp
  status=$(nvidia-smi --id="$gpu_id" \
    --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu \
    --format=csv,noheader,nounits)
  IFS=',' read -r used total util temp <<< "$status"
  used=${used//[[:space:]]/}
  total=${total//[[:space:]]/}
  util=${util//[[:space:]]/}
  temp=${temp//[[:space:]]/}
  if (( used > 1024 || util > 10 || temp > 75 )); then
    echo "Refusing GPU ${gpu_id}: used=${used}/${total} MiB util=${util}% temp=${temp}C" >&2
    exit 6
  fi
  echo "GPU ${gpu_id} ready: used=${used}/${total} MiB util=${util}% temp=${temp}C"
}
for gpu in "${GPUS[@]}"; do check_gpu "$gpu"; done

mkdir -p "$LOG_ROOT"
for index in "${!GPUS[@]}"; do
  gpu=${GPUS[$index]}
  seed=${SEEDS[$index]}
  trainer=${TRAINERS[$index]}
  log_file="${LOG_ROOT}/seed${seed}_gpu${gpu}.log"
  pane_command="set -o pipefail; cd '${PROJECT_ROOT}'; bash '${trainer}' --run '${gpu}' 2>&1 | tee '${log_file}'; code=\${PIPESTATUS[0]}; echo '[SOFT_TWO_SEED_EXIT]' 'seed${seed}' \${code} | tee -a '${log_file}'; exec bash -i"
  if [[ $index -eq 0 ]]; then
    tmux new-session -d -s "$SESSION" -n "gpu${gpu}" "$pane_command"
  else
    tmux new-window -d -t "${SESSION}:" -n "gpu${gpu}" "$pane_command"
  fi
done
echo "Started tmux=${SESSION} logs=${LOG_ROOT}"
