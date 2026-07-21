#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [--plan | --execute | --resume]"
  echo "  --plan     Validate the checkpoint and all seven manifests with CUDA hidden (default)."
  echo "  --execute  Start a new seven-GPU spatial Robot-init evaluation."
  echo "  --resume   Resume an existing output tree, skipping completed tasks."
}

launcher_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${launcher_dir}/.." && pwd -P)"
launcher="${launcher_dir}/$(basename -- "${BASH_SOURCE[0]}")"
evaluator="${repo_root}/eval/run_libero_plus_bounded.py"
python_bin="/home/imc/anaconda3/envs/ript_vla_openvla_oft/bin/python"
checkpoint="/home/imc/models/ra-loop/openvla-oft-spatial"
openvla_root="/home/imc/code/openvla-oft"
libero_config="${repo_root}/.libero"
classification="/home/imc/code/LIBERO-plus/libero/libero/benchmark/task_classification.json"
output_root="${repo_root}/logs/openvla_oft_spatial_robot_full350_7gpu_20260720"
session="ra_oft_spatial_robot350"
suite="libero_spatial"

validate_paths() {
  local required_path shard manifest
  for required_path in \
    "$evaluator" "$python_bin" "$checkpoint" "$openvla_root" \
    "$libero_config" "$classification"; do
    if [[ ! -e "$required_path" ]]; then
      echo "Required path is missing: $required_path" >&2
      exit 1
    fi
  done
  for shard in 0 1 2 3 4 5 6; do
    manifest="${repo_root}/eval/manifests/${suite}_robot_init_full_shard${shard}_of_7.jsonl"
    if [[ ! -f "$manifest" ]]; then
      echo "Manifest is missing: $manifest" >&2
      exit 1
    fi
  done
}

validate_manifest_union() {
  REPO_ROOT="$repo_root" CLASSIFICATION="$classification" \
    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="" "$python_bin" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["REPO_ROOT"])
classification = json.loads(Path(os.environ["CLASSIFICATION"]).read_text())
expected_rows = [
    (row["id"], row["name"], row["difficulty_level"])
    for row in classification["libero_spatial"]
    if row["category"] == "Robot Initial States"
]
expected_by_shard = {
    shard: {row for index, row in enumerate(expected_rows) if index % 7 == shard}
    for shard in range(7)
}
seen = []
for shard in range(7):
    path = root / "eval" / "manifests" / f"libero_spatial_robot_init_full_shard{shard}_of_7.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 50:
        raise SystemExit(f"{path}: expected 50 rows, found {len(rows)}")
    observed = set()
    for row in rows:
        if row["suite"] != "libero_spatial" or row["category"] != "Robot Initial States":
            raise SystemExit(f"{path}: unexpected suite/category in task {row.get('task_id')}")
        identity = (row["classification_id"], row["task_name"], row["difficulty_level"])
        observed.add(identity)
        seen.append(identity)
    if observed != expected_by_shard[shard]:
        raise SystemExit(f"{path}: rows do not match official filtered-order modulo shard")
if len(seen) != len(set(seen)):
    raise SystemExit("Duplicate Robot-init rows across spatial manifests")
if set(seen) != set(expected_rows):
    raise SystemExit(
        f"Spatial manifest union mismatch: selected={len(set(seen))}, expected={len(expected_rows)}"
    )
print("manifest union: PASS; 350/350 official spatial Robot-init tasks; 7 x 50; no duplicates")
PY
}

build_command() {
  local shard="$1"
  local gpu="$2"
  local manifest="${repo_root}/eval/manifests/${suite}_robot_init_full_shard${shard}_of_7.jsonl"
  local output_dir="${output_root}/shard${shard}_gpu${gpu}"
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

run_worker() {
  local shard="$1"
  local gpu="$2"
  local resume_mode="$3"
  local delay=$((shard * 30))
  local status

  trap 'echo; echo "worker interrupted before or during evaluation"; exec bash -i' INT
  echo "spatial shard ${shard} -> physical GPU ${gpu}; OSMesa; starts after ${delay}s"
  sleep "$delay"
  build_command "$shard" "$gpu"
  if [[ "$resume_mode" == true ]]; then
    shard_command+=(--resume)
  fi
  shard_command+=(--execute)
  print_command "${shard_command[@]}"
  if "${shard_command[@]}"; then
    echo "completed spatial shard ${shard}"
  else
    status=$?
    echo "FAILED spatial shard ${shard} with exit code ${status}"
  fi
  exec bash -i
}

if [[ "${1:-}" == "--worker" ]]; then
  if [[ $# -ne 4 ]]; then
    echo "Invalid internal worker invocation" >&2
    exit 2
  fi
  worker_shard="$2"
  worker_gpu="$3"
  worker_resume="$4"
  if [[ ! "$worker_shard" =~ ^[0-6]$ ]] || [[ ! "$worker_gpu" =~ ^[1-7]$ ]]; then
    echo "Invalid internal shard/GPU mapping" >&2
    exit 2
  fi
  if (( worker_gpu != worker_shard + 1 )); then
    echo "Invalid internal shard/GPU mapping" >&2
    exit 2
  fi
  if [[ "$worker_resume" != true && "$worker_resume" != false ]]; then
    echo "Invalid internal resume flag" >&2
    exit 2
  fi
  validate_paths
  run_worker "$worker_shard" "$worker_gpu" "$worker_resume"
fi

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

validate_paths

if [[ "$mode" == "plan" ]]; then
  plan_resume=false
  if [[ -d "$output_root" ]]; then
    plan_resume=true
    echo "Existing output detected; validating the resume form of all commands"
  fi
  echo "CPU-only seven-command validation; CUDA is hidden and no model or simulator is loaded"
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="" "$python_bin" -c \
    "import draccus, torch; from transformers import AutoImageProcessor; print('ript environment preflight: PASS; CUDA visible:', torch.cuda.is_available())"
  validate_manifest_union
  for shard in 0 1 2 3 4 5 6; do
    gpu=$((shard + 1))
    build_command "$shard" "$gpu"
    if [[ "$plan_resume" == true ]]; then
      shard_command+=(--resume)
    fi
    echo "suite=$suite shard=$shard physical_gpu=$gpu"
    print_command "${shard_command[@]}"
    "${shard_command[@]}"
  done
  echo "Plan validation complete: 350 tasks across 7 manifests."
  echo "No output directory or tmux session was created and no CUDA workload ran."
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

resume_worker=false
if [[ "$mode" == "resume" ]]; then
  resume_worker=true
fi

for shard in 0 1 2 3 4 5 6; do
  gpu=$((shard + 1))
  worker_command=(bash "$launcher" --worker "$shard" "$gpu" "$resume_worker")
  printf -v worker_text '%q ' "${worker_command[@]}"
  printf -v repo_text '%q' "$repo_root"
  pane_command="cd ${repo_text}; exec ${worker_text}"
  if [[ "$shard" -eq 0 ]]; then
    window_id="$(tmux new-session -d -P -F '#{window_id}' -s "$session" -n "shard${shard}_gpu${gpu}" "$pane_command")"
  else
    window_id="$(tmux new-window -d -P -F '#{window_id}' -t "${session}:" -n "shard${shard}_gpu${gpu}" "$pane_command")"
  fi
  tmux set-window-option -t "$window_id" automatic-rename off >/dev/null
  tmux set-window-option -t "$window_id" remain-on-exit on >/dev/null
done

echo "Launched tmux session: $session"
echo "Attach with: tmux attach -t $session"
echo "Choose a shard window with: Ctrl-b w"
echo "Each shard evaluates exactly 50 spatial Robot-init tasks."
echo "The seven starts are staggered by 30 seconds; there is no evaluator timeout."
echo "Ctrl-C stops the current shard and flushed per-task results can be resumed."
echo "After the session is gone, resume with: $0 --resume"
