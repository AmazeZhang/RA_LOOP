#!/usr/bin/env bash
set -euo pipefail

# Six GPU workers evaluate A4/A5 steps 20/30/40 in one wave.

PROJECT_ROOT=/home/imc/yzy/RA_LOOP
SESSION=ra_loop_counterfactual_checkpoint_eval
LAUNCHER="${PROJECT_ROOT}/eval/launch_ra_loop_counterfactual_checkpoint_eval.sh"
OUTPUT_ROOT="${PROJECT_ROOT}/logs/ra_loop_counterfactual_checkpoint_eval_20260725"
GPUS=(2 3 4 5 6 7)
SOURCES=(
  counterfactual_cra_only
  counterfactual_cra_only
  counterfactual_cra_only
  counterfactual_npc
  counterfactual_npc
  counterfactual_npc
)
STEPS=(20 30 40 20 30 40)

if [[ $# -eq 0 || "${1:-}" == "--plan" ]]; then
  [[ $# -le 1 ]] || exit 2
  echo "Evaluation is NOT started. Final command: bash $0 --run"
  echo "session=${SESSION} checkpoints=6 GPUs=2..7 episodes=288 expected=45-60m"
  for index in "${!GPUS[@]}"; do
    echo "GPU ${GPUS[$index]}: ${SOURCES[$index]}/step${STEPS[$index]}"
  done
  exec bash "${LAUNCHER}" --plan
fi

if [[ $# -ne 1 || "$1" != "--run" ]]; then
  echo "Usage: bash $0 [--plan | --run]" >&2
  exit 2
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "Refusing existing tmux session: ${SESSION}" >&2
  exit 3
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing existing output root: ${OUTPUT_ROOT}" >&2
  exit 4
fi

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
for gpu in "${GPUS[@]}"; do
  check_gpu "${gpu}"
done

mkdir -p "${OUTPUT_ROOT}"
for index in "${!GPUS[@]}"; do
  gpu=${GPUS[$index]}
  source=${SOURCES[$index]}
  step=${STEPS[$index]}
  log_file="${OUTPUT_ROOT}/console_${source}_step${step}_gpu${gpu}.log"
  pane_command="set -o pipefail; cd '${PROJECT_ROOT}'; bash '${LAUNCHER}' --run '${source}' '${step}' '${gpu}' 2>&1 | tee '${log_file}'; code=\${PIPESTATUS[0]}; echo '[COUNTERFACTUAL_EVAL_EXIT]' '${source}' 'step${step}' \${code} | tee -a '${log_file}'; exec bash -i"
  if [[ ${index} -eq 0 ]]; then
    tmux new-session -d -s "${SESSION}" -n "gpu${gpu}" "${pane_command}"
  else
    tmux new-window -d -t "${SESSION}:" -n "gpu${gpu}" "${pane_command}"
  fi
done

echo "Started tmux=${SESSION} output=${OUTPUT_ROOT}"
