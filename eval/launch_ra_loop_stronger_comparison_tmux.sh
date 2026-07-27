#!/usr/bin/env bash
set -euo pipefail

# Six workers: two task shards for each frozen model candidate.

PROJECT_ROOT=/home/imc/yzy/RA_LOOP
SESSION=ra_loop_stronger_comparison
LAUNCHER="${PROJECT_ROOT}/eval/launch_ra_loop_stronger_comparison.sh"
OUTPUT_ROOT="${PROJECT_ROOT}/logs/ra_loop_stronger_comparison_20260726"
GPUS=(2 3 4 5 6 7)
WORKER_JOBS=(
  "pilot:5:0 pilot:5:1 pilot:5:2 pilot:5:3 pilot:5:4"
  "pilot:5:5 pilot:5:6 pilot:5:7 pilot:5:8 pilot:5:9"
  "counterfactual_cra_only:40:0 counterfactual_cra_only:40:1 counterfactual_cra_only:40:2 counterfactual_cra_only:40:3 counterfactual_cra_only:40:4"
  "counterfactual_cra_only:40:5 counterfactual_cra_only:40:6 counterfactual_cra_only:40:7 counterfactual_cra_only:40:8 counterfactual_cra_only:40:9"
  "counterfactual_npc:40:0 counterfactual_npc:40:1 counterfactual_npc:40:2 counterfactual_npc:40:3 counterfactual_npc:40:4"
  "counterfactual_npc:40:5 counterfactual_npc:40:6 counterfactual_npc:40:7 counterfactual_npc:40:8 counterfactual_npc:40:9"
)

if [[ $# -eq 0 || "${1:-}" == --plan ]]; then
  [[ $# -le 1 ]] || exit 2
  echo "Comparison is NOT started. Final command: bash $0 --run"
  echo "session=${SESSION} models=3 tasks=10 pairs/task=14 episodes=840 GPUs=2..7"
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
    IFS=: read -r source step task_id <<< "$job"
    log_file="${OUTPUT_ROOT}/console_${source}_step${step}_task${task_id}_gpu${gpu}.log"
    pane_command+=" if [[ \${overall} -eq 0 ]]; then sleep 10; bash '${LAUNCHER}' --run '${source}' '${step}' '${task_id}' '${gpu}' comparison 2>&1 | tee '${log_file}'; code=\${PIPESTATUS[0]}; echo '[STRONGER_COMPARISON_EXIT]' '${source}' 'step${step}' 'task${task_id}' \${code} | tee -a '${log_file}'; if [[ \${code} -ne 0 ]]; then overall=\${code}; fi; fi;"
  done
  pane_command+=" echo '[STRONGER_COMPARISON_WORKER_EXIT]' \${overall}; exec bash -i"
  if [[ $index -eq 0 ]]; then
    tmux new-session -d -s "$SESSION" -n "gpu${gpu}" "$pane_command"
  else
    tmux new-window -d -t "${SESSION}:" -n "gpu${gpu}" "$pane_command"
  fi
done
echo "Started tmux=${SESSION} output=${OUTPUT_ROOT}"
