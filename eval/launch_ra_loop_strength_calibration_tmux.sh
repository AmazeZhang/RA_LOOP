#!/usr/bin/env bash
set -euo pipefail

# Four workers: two task shards for each stronger perturbation radius.

PROJECT_ROOT=/home/imc/yzy/RA_LOOP
SESSION=ra_loop_strength_calibration
LAUNCHER="${PROJECT_ROOT}/eval/launch_ra_loop_strength_calibration.sh"
OUTPUT_ROOT="${PROJECT_ROOT}/logs/ra_loop_strength_calibration_20260726"
GPUS=(2 3 4 5)
WORKER_JOBS=(
  "0p15:0 0p15:1 0p15:2 0p15:3 0p15:4"
  "0p15:5 0p15:6 0p15:7 0p15:8 0p15:9"
  "0p20:0 0p20:1 0p20:2 0p20:3 0p20:4"
  "0p20:5 0p20:6 0p20:7 0p20:8 0p20:9"
)

if [[ $# -eq 0 || "${1:-}" == --plan ]]; then
  [[ $# -le 1 ]] || exit 2
  echo "Calibration is NOT started. Final command: bash $0 --run"
  echo "session=${SESSION} strengths=2 tasks=10 pairs/task=10 episodes=400 GPUs=2..5"
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
    strength_tag=${job%%:*}
    task_id=${job##*:}
    log_file="${OUTPUT_ROOT}/console_${strength_tag}_task${task_id}_gpu${gpu}.log"
    pane_command+=" if [[ \${overall} -eq 0 ]]; then sleep 10; bash '${LAUNCHER}' --run '${strength_tag}' '${task_id}' '${gpu}' calibration 2>&1 | tee '${log_file}'; code=\${PIPESTATUS[0]}; echo '[STRENGTH_CALIBRATION_EXIT]' '${strength_tag}' 'task${task_id}' \${code} | tee -a '${log_file}'; if [[ \${code} -ne 0 ]]; then overall=\${code}; fi; fi;"
  done
  pane_command+=" echo '[STRENGTH_CALIBRATION_WORKER_EXIT]' \${overall}; exec bash -i"
  if [[ $index -eq 0 ]]; then
    tmux new-session -d -s "$SESSION" -n "gpu${gpu}" "$pane_command"
  else
    tmux new-window -d -t "${SESSION}:" -n "gpu${gpu}" "$pane_command"
  fi
done
echo "Started tmux=${SESSION} output=${OUTPUT_ROOT}"
