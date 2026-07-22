#!/usr/bin/env bash
set -euo pipefail

# Launch six independent checkpoint evaluations on GPUs 1..6 in one shared tmux.

PROJECT_ROOT=/home/imc/yzy/RA_LOOP
SESSION=ra_loop_stratified_checkpoint_eval
LAUNCHER="${PROJECT_ROOT}/eval/launch_ra_loop_stratified_checkpoint_eval.sh"
OUTPUT_ROOT="${PROJECT_ROOT}/logs/ra_loop_stratified_checkpoint_eval_20260722"
STEPS=(5 10 15 20 25 30)
GPUS=(1 2 3 4 5 6)

if [[ $# -eq 0 || "${1:-}" == "--plan" ]]; then
  if [[ $# -gt 1 ]]; then
    exit 2
  fi
  echo "Evaluation is NOT started. Final command: bash $0 --run"
  echo "session=${SESSION} checkpoints=6 GPUs=1..6 episodes=288 expected=45-60m"
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

mkdir -p "${OUTPUT_ROOT}"
for index in "${!STEPS[@]}"; do
  step=${STEPS[$index]}
  gpu=${GPUS[$index]}
  log_file="${OUTPUT_ROOT}/console_step${step}.log"
  pane_command="cd ${PROJECT_ROOT} && PYTHONUNBUFFERED=1 bash ${LAUNCHER} --run ${step} ${gpu} 2>&1 | tee ${log_file}; run_status=\${PIPESTATUS[0]}; echo '[STRATIFIED_EVAL_EXIT]' \${run_status} | tee -a ${log_file}; exec bash -i"
  if [[ ${index} -eq 0 ]]; then
    tmux new-session -d -s "${SESSION}" -n "step${step}_gpu${gpu}" "${pane_command}"
  else
    tmux new-window -d -t "${SESSION}:" -n "step${step}_gpu${gpu}" "${pane_command}"
  fi
done

echo "Started tmux=${SESSION} output=${OUTPUT_ROOT}"
