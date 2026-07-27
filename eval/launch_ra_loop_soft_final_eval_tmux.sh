#!/usr/bin/env bash
set -euo pipefail

# Six workers for the pre-registered soft-CRA final go/no-go evaluation.

PROJECT_ROOT=/home/imc/yzy/RA_LOOP
SESSION=ra_loop_soft_final_eval
LAUNCHER="${PROJECT_ROOT}/eval/launch_ra_loop_soft_final_eval.sh"
OUTPUT_ROOT="${PROJECT_ROOT}/logs/ra_loop_soft_final_eval_20260727"
GPUS=(2 3 4 5 6 7)
WORKER_JOBS=(
  "counterfactual_soft_seed10000:0p1:0 counterfactual_soft_seed10000:0p1:1 counterfactual_soft_seed10000:0p1:2 counterfactual_soft_seed10000:0p1:3 counterfactual_soft_seed10000:0p1:4"
  "counterfactual_soft_seed10000:0p1:5 counterfactual_soft_seed10000:0p1:6 counterfactual_soft_seed10000:0p1:7 counterfactual_soft_seed10000:0p1:8 counterfactual_soft_seed10000:0p1:9"
  "counterfactual_soft_seed20000:0p1:0 counterfactual_soft_seed20000:0p1:1 counterfactual_soft_seed20000:0p1:2 counterfactual_soft_seed20000:0p1:3 counterfactual_soft_seed20000:0p1:4"
  "counterfactual_soft_seed20000:0p1:5 counterfactual_soft_seed20000:0p1:6 counterfactual_soft_seed20000:0p1:7 counterfactual_soft_seed20000:0p1:8 counterfactual_soft_seed20000:0p1:9"
  "counterfactual_soft_seed10000:0p2:0 counterfactual_soft_seed10000:0p2:1 counterfactual_soft_seed10000:0p2:2 counterfactual_soft_seed10000:0p2:3 counterfactual_soft_seed10000:0p2:4 counterfactual_soft_seed10000:0p2:5 counterfactual_soft_seed10000:0p2:6 counterfactual_soft_seed10000:0p2:7 counterfactual_soft_seed10000:0p2:8 counterfactual_soft_seed10000:0p2:9"
  "counterfactual_soft_seed20000:0p2:0 counterfactual_soft_seed20000:0p2:1 counterfactual_soft_seed20000:0p2:2 counterfactual_soft_seed20000:0p2:3 counterfactual_soft_seed20000:0p2:4 counterfactual_soft_seed20000:0p2:5 counterfactual_soft_seed20000:0p2:6 counterfactual_soft_seed20000:0p2:7 counterfactual_soft_seed20000:0p2:8 counterfactual_soft_seed20000:0p2:9"
)

if [[ $# -eq 0 || "${1:-}" == --plan ]]; then
  [[ $# -le 1 ]] || exit 2
  echo "Evaluation is NOT started. Final command: bash $0 --run"
  echo "session=${SESSION} seeds=2 protocols=2 episodes=1360 GPUs=2..7 expected=3-5h"
  for index in "${!GPUS[@]}"; do echo "GPU ${GPUS[$index]}: ${WORKER_JOBS[$index]}"; done
  exec bash "$LAUNCHER" --plan
fi

if [[ $# -ne 1 || "$1" != --run ]]; then
  echo "Usage: bash $0 [--plan | --run]" >&2
  exit 2
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Refusing existing tmux session: $SESSION" >&2
  exit 3
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Refusing existing output root: $OUTPUT_ROOT" >&2
  exit 4
fi

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
    exit 5
  fi
  echo "GPU ${gpu_id} ready: used=${used}/${total} MiB util=${util}% temp=${temp}C"
}
for gpu in "${GPUS[@]}"; do check_gpu "$gpu"; done

mkdir -p "$OUTPUT_ROOT"
for index in "${!GPUS[@]}"; do
  gpu=${GPUS[$index]}
  pane_command="set -o pipefail; cd '${PROJECT_ROOT}'; overall=0;"
  read -ra jobs <<< "${WORKER_JOBS[$index]}"
  for job in "${jobs[@]}"; do
    IFS=: read -r source protocol task_id <<< "$job"
    log_file="${OUTPUT_ROOT}/console_${source}_${protocol}_task${task_id}_gpu${gpu}.log"
    pane_command+=" if [[ \${overall} -eq 0 ]]; then sleep 10; bash '${LAUNCHER}' --run '${source}' '${protocol}' '${task_id}' '${gpu}' final 2>&1 | tee '${log_file}'; code=\${PIPESTATUS[0]}; echo '[SOFT_FINAL_EVAL_EXIT]' '${source}' '${protocol}' 'task${task_id}' \${code} | tee -a '${log_file}'; if [[ \${code} -ne 0 ]]; then overall=\${code}; fi; fi;"
  done
  pane_command+=" echo '[SOFT_FINAL_EVAL_WORKER_EXIT]' \${overall}; exec bash -i"
  if [[ $index -eq 0 ]]; then
    tmux new-session -d -s "$SESSION" -n "gpu${gpu}" "$pane_command"
  else
    tmux new-window -d -t "${SESSION}:" -n "gpu${gpu}" "$pane_command"
  fi
done
echo "Started tmux=${SESSION} output=${OUTPUT_ROOT}"
