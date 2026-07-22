#!/usr/bin/env bash
set -euo pipefail

# Six GPU workers, each evaluating two checkpoints sequentially (12 total).

PROJECT_ROOT=/home/imc/yzy/RA_LOOP
SESSION=ra_loop_stratified_lambda_checkpoint_eval
LAUNCHER="${PROJECT_ROOT}/eval/launch_ra_loop_lambda_checkpoint_eval.sh"
OUTPUT_ROOT="${PROJECT_ROOT}/logs/ra_loop_stratified_lambda_checkpoint_eval_20260722"
GPUS=(2 3 4 5 6 7)
FIRST_SOURCES=(stratified_lambda0 stratified_lambda025 stratified_lambda0 stratified_lambda025 stratified_lambda0 stratified_lambda025)
FIRST_STEPS=(5 5 10 10 15 15)
SECOND_SOURCES=(stratified_lambda025 stratified_lambda0 stratified_lambda025 stratified_lambda0 stratified_lambda025 stratified_lambda0)
SECOND_STEPS=(20 20 25 25 30 30)

if [[ $# -eq 0 || "${1:-}" == "--plan" ]]; then
  [[ $# -le 1 ]] || exit 2
  echo "Evaluation is NOT started. Final command: bash $0 --run"
  echo "session=${SESSION} checkpoints=12 GPUs=2..7 waves=2 episodes=576 expected=90-120m"
  for index in "${!GPUS[@]}"; do
    echo "GPU ${GPUS[$index]}: ${FIRST_SOURCES[$index]}/step${FIRST_STEPS[$index]} -> ${SECOND_SOURCES[$index]}/step${SECOND_STEPS[$index]}"
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
  used=${used//[[:space:]]/}; total=${total//[[:space:]]/}
  util=${util//[[:space:]]/}; temp=${temp//[[:space:]]/}
  if (( used > 1024 || util > 10 || temp > 75 )); then
    echo "Refusing GPU ${gpu_id}: used=${used}/${total} MiB util=${util}% temp=${temp}C" >&2
    exit 5
  fi
  echo "GPU ${gpu_id} ready: used=${used}/${total} MiB util=${util}% temp=${temp}C"
}
for gpu in "${GPUS[@]}"; do check_gpu "${gpu}"; done

mkdir -p "${OUTPUT_ROOT}"
for index in "${!GPUS[@]}"; do
  gpu=${GPUS[$index]}
  source1=${FIRST_SOURCES[$index]}; step1=${FIRST_STEPS[$index]}
  source2=${SECOND_SOURCES[$index]}; step2=${SECOND_STEPS[$index]}
  log1="${OUTPUT_ROOT}/console_${source1}_step${step1}_gpu${gpu}.log"
  log2="${OUTPUT_ROOT}/console_${source2}_step${step2}_gpu${gpu}.log"
  pane_command="set -o pipefail; cd '${PROJECT_ROOT}'; bash '${LAUNCHER}' --run '${source1}' '${step1}' '${gpu}' 2>&1 | tee '${log1}'; code1=\${PIPESTATUS[0]}; echo '[LAMBDA_EVAL_EXIT]' '${source1}' 'step${step1}' \${code1} | tee -a '${log1}'; code2=99; if [[ \${code1} -eq 0 ]]; then bash '${LAUNCHER}' --run '${source2}' '${step2}' '${gpu}' 2>&1 | tee '${log2}'; code2=\${PIPESTATUS[0]}; echo '[LAMBDA_EVAL_EXIT]' '${source2}' 'step${step2}' \${code2} | tee -a '${log2}'; fi; echo '[LAMBDA_EVAL_WINDOW_EXIT]' \${code1} \${code2}; exec bash -i"
  window_name="gpu${gpu}_s${step1}_s${step2}"
  if [[ ${index} -eq 0 ]]; then
    tmux new-session -d -s "${SESSION}" -n "${window_name}" "${pane_command}"
  else
    tmux new-window -d -t "${SESSION}:" -n "${window_name}" "${pane_command}"
  fi
done

echo "Started tmux=${SESSION} output=${OUTPUT_ROOT}"
