#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [--plan | --execute | --resume]"
  echo "  --plan     Validate all seven evaluator commands without GPU imports (default)."
  echo "  --execute  Launch the seven GPU evaluations in a detached tmux session."
  echo "  --resume   Resume an interrupted run, validating existing per-shard metadata."
}

mode="plan"
case "${1:-}" in
  ""|--plan)
    ;;
  --execute)
    mode="execute"
    ;;
  --resume)
    mode="resume"
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

launcher_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${launcher_dir}/.." && pwd -P)"
evaluator="${repo_root}/eval/run_libero_plus_bounded.py"
python_bin="/home/imc/anaconda3/envs/ript_vla_openvla_oft/bin/python"
checkpoint="/home/imc/models/ra-loop/openvla-oft-union"
openvla_root="/home/imc/code/openvla-oft"
libero_config="${repo_root}/.libero"
output_root="${repo_root}/logs/openvla_oft_union_libero10_robot_full393_7gpu_20260718"
session="ra_union_robot393"

for required_path in "$evaluator" "$python_bin" "$checkpoint" "$openvla_root" "$libero_config"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required path is missing: $required_path" >&2
    exit 1
  fi
done
for shard in 0 1 2 3 4 5 6; do
  manifest="${repo_root}/eval/manifests/libero_10_robot_init_full_shard${shard}_of_7.jsonl"
  if [[ ! -f "$manifest" ]]; then
    echo "Manifest is missing: $manifest" >&2
    exit 1
  fi
done

build_command() {
  local shard="$1"
  local gpu="$2"
  local output_dir="${output_root}/shard${shard}_gpu${gpu}"
  local manifest="${repo_root}/eval/manifests/libero_10_robot_init_full_shard${shard}_of_7.jsonl"
  shard_command=(
    env PYTHONNOUSERSITE=1 "$python_bin" "$evaluator"
    --manifest "$manifest"
    --checkpoint "$checkpoint"
    --openvla-root "$openvla_root"
    --libero-config "$libero_config"
    --output-dir "$output_dir"
    --seed 7
    --gpu-id "$gpu"
    --max-tasks 1000
    --render-backend osmesa
  )
}

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

if [[ "$mode" == "plan" ]]; then
  plan_resume=false
  if [[ -d "$output_root" ]]; then
    plan_resume=true
    echo "Existing output detected; validating the resume form of all commands"
  fi
  echo "CPU-only seven-shard validation; CUDA is hidden and no model or simulator is loaded"
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="" "$python_bin" -c \
    "import draccus, torch; from transformers import AutoImageProcessor; print('ript environment preflight: PASS; CUDA visible:', torch.cuda.is_available())"
  for shard in 0 1 2 3 4 5 6; do
    gpu=$((shard + 1))
    build_command "$shard" "$gpu"
    if [[ "$plan_resume" == true ]]; then
      shard_command+=(--resume)
    fi
    echo "shard=$shard physical_gpu=$gpu"
    print_command "${shard_command[@]}"
    "${shard_command[@]}"
  done
  echo "Plan validation complete. No tmux session was created and no CUDA workload ran."
  exit 0
fi

if [[ "$mode" == "execute" && -e "$output_root" ]]; then
  echo "Refusing to overwrite or mix with existing output: $output_root" >&2
  exit 1
fi
if [[ "$mode" == "resume" && ! -d "$output_root" ]]; then
  echo "Cannot resume because the output root does not exist: $output_root" >&2
  exit 1
fi
if tmux has-session -t "$session" 2>/dev/null; then
  echo "Refusing to reuse existing tmux session: $session" >&2
  exit 1
fi

for shard in 0 1 2 3 4 5 6; do
  gpu=$((shard + 1))
  delay=$((shard * 30))
  build_command "$shard" "$gpu"
  if [[ "$mode" == "resume" ]]; then
    shard_command+=(--resume)
  fi
  shard_command+=(--execute)
  printf -v command_text '%q ' "${shard_command[@]}"
  printf -v repo_text '%q' "$repo_root"
  pane_command="cd ${repo_text}; echo 'shard ${shard} -> physical GPU ${gpu}; OSMesa; starts after ${delay}s'; trap 'echo shard ${shard} cancelled before start; exec bash -i' INT; sleep ${delay}; trap - INT; ${command_text}; status=\$?; echo; echo 'shard ${shard} finished with exit code' \$status; exec bash -i"
  if [[ "$shard" -eq 0 ]]; then
    tmux new-session -d -s "$session" -n "shard${shard}_gpu${gpu}" "$pane_command"
  else
    tmux new-window -d -t "${session}:" -n "shard${shard}_gpu${gpu}" "$pane_command"
  fi
done

for window in 0 1 2 3 4 5 6; do
  tmux set-window-option -t "${session}:${window}" remain-on-exit on >/dev/null
done
echo "Launched tmux session: $session"
echo "Attach with: tmux attach -t $session"
echo "Choose a shard window with: Ctrl-b w"
echo "The seven starts are staggered by 30 seconds; there is no evaluator timeout."
echo "To stop one shard, attach to its window and press Ctrl-C."
echo "After the session is gone, resume flushed per-task results with: $0 --resume"
