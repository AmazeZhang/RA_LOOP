#!/usr/bin/env bash
set -euo pipefail

# One-task, one-step RA-LOOP connectivity smoke.
# Safe default prints the command. GPU execution requires: --run <GPU_ID>

PROJECT_ROOT=/home/imc/yzy/RA_LOOP
RIPT_ROOT=/home/imc/code/ript-vla
OFFICIAL_LIBERO_ROOT=/home/imc/code/LIBERO-official
TORCHRUN_BIN=/home/imc/anaconda3/envs/ript_vla_openvla_oft/bin/torchrun
MODEL_PATH=/home/imc/yzy/RA_LOOP/runtime/openvla-oft-spatial-smoke
HEADER_PATH=/home/imc/models/ra-loop/ript-vla/openvla_oft/scale_header/LIBERO_SPATIAL_scale_header.pth
DATA_PATH=/home/imc/data/ra-loop/libero-datasets
TASK_NAME=pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate
OUTPUT_PATH="${PROJECT_ROOT}/outputs/ra_loop_spatial_connectivity_smoke"

OVERRIDES=(
  exp_name=RA-LOOP_spatial_robot_init_connectivity_smoke
  variant_name=one_task_one_step_k4_h10_robot_init_recovery
  make_unique_experiment_dir=true
  paths.data_prefix="${DATA_PATH}"
  paths.output_prefix="${OUTPUT_PATH}"
  task.suite_name=libero_spatial
  task.dataset.suite_name=.
  task.task_names_to_use="[${TASK_NAME}]"
  task.demos_per_env=1
  task.rollouts_per_env=1
  train_dataloader.batch_size=1
  train_dataloader.shuffle=false
  train_dataloader.num_workers=0
  train_dataloader.persistent_workers=false
  train_dataloader.pin_memory=false
  train_dataloader.multiprocessing_context=null
  training.gradient_accumulation_steps=1
  training.n_steps=1
  training.rollout_steps=1
  training.save_interval=9999
  training.log_interval=1
  training.use_tqdm=true
  rollout.enabled=false
  algo.env_runner._target_=ra_loop.ript_compat.InProcessOpenVLAOFTLiberoRunner
  algo.rollout_generator_factory._target_=ra_loop.ript_recovery.RobotInitRecoveryRolloutGenerator
  +algo.rollout_generator_factory.robot_init_strength=0.001
  +algo.rollout_generator_factory.perturb_seed=20260720
  algo.rl_optimizer_factory._target_=ra_loop.ript_recovery.RobotInitRecoveryOptimizer
  reward_function._target_=ra_loop.ript_recovery.RobotInitRecoveryReward
  +reward_function.lambda_recovery=0.5
  algo.rloo_batch_size=4
  algo.rollouts_per_env=4
  algo.num_parallel_envs=1
  algo.max_episode_length=10
  algo.enable_dynamic_sampling=false
  algo.use_val_init=false
  algo.mix_val_init_in_rloo=false
  algo.gradient_accumulation_steps=1
  algo.num_ppo_epochs=1
  algo.max_step_batch_size=1
  algo.scale_factor=2.0
  algo.fix_scale_head=true
  algo.checkpoint_path="${MODEL_PATH}"
  algo.header_checkpoint="${HEADER_PATH}"
  algo.lora_adaptor_ckpt=null
  logging.mode=disabled
  logging.resume=false
  logging.save_code=false
)

COMMAND=(
  "${TORCHRUN_BIN}"
  --standalone
  --nproc_per_node=1
  train_ript_openvla_oft.py
  --config-name=train_rl_openvla_oft_all_task_spatial.yaml
  "${OVERRIDES[@]}"
)

print_command() {
  echo "CUDA_VISIBLE_DEVICES=<GPU_ID> ${COMMAND[*]}"
}

if [[ $# -eq 0 || "${1:-}" == "--print-command" ]]; then
  if [[ $# -gt 1 ]]; then
    echo "Usage: bash $0 [--print-command | --run <GPU_ID>]" >&2
    exit 2
  fi
  echo "Dry command only; no CUDA initialization"
  print_command
  exit 0
fi

if [[ $# -ne 2 || "$1" != "--run" || ! "$2" =~ ^[0-7]$ ]]; then
  echo "Usage: bash $0 [--print-command | --run <GPU_ID>]" >&2
  exit 2
fi
GPU_ID=$2

for required in \
  "${TORCHRUN_BIN}" \
  "${RIPT_ROOT}/train_ript_openvla_oft.py" \
  "${MODEL_PATH}/config.json" \
  "${HEADER_PATH}" \
  "${DATA_PATH}/libero_spatial" \
  "${PROJECT_ROOT}/ra_loop/ript_recovery.py"; do
  if [[ ! -e "${required}" ]]; then
    echo "Required path is missing: ${required}" >&2
    exit 3
  fi
done

GPU_STATUS=$(nvidia-smi --id="${GPU_ID}" \
  --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu \
  --format=csv,noheader,nounits)
IFS=',' read -r GPU_USED GPU_TOTAL GPU_UTIL GPU_TEMP <<< "${GPU_STATUS}"
GPU_USED=${GPU_USED//[[:space:]]/}
GPU_TOTAL=${GPU_TOTAL//[[:space:]]/}
GPU_UTIL=${GPU_UTIL//[[:space:]]/}
GPU_TEMP=${GPU_TEMP//[[:space:]]/}
if (( GPU_USED > 1024 || GPU_UTIL > 10 || GPU_TEMP > 75 )); then
  echo "Refusing GPU ${GPU_ID}: used=${GPU_USED}/${GPU_TOTAL} MiB util=${GPU_UTIL}% temp=${GPU_TEMP}C" >&2
  exit 4
fi

export LIBERO_CONFIG_PATH="${PROJECT_ROOT}/.libero_official"
export PYTHONPATH="${PROJECT_ROOT}:${OFFICIAL_LIBERO_ROOT}:${RIPT_ROOT}"
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
export PYTHONNOUSERSITE=1
export NUMBA_CACHE_DIR=/tmp/ra_loop_numba_cache
export MPLCONFIGDIR=/tmp/ra_loop_mpl_cache
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export NCCL_TIMEOUT=108000

echo "Starting bounded RA-LOOP connectivity smoke on physical GPU ${GPU_ID}"
echo "GPU before start: used=${GPU_USED}/${GPU_TOTAL} MiB util=${GPU_UTIL}% temp=${GPU_TEMP}C"
echo "task=${TASK_NAME} steps=1 K=4 pairs=2 horizon=10 strength=0.001 lambda_r=0.5"
echo "W&B=disabled periodic_eval=disabled checkpoint_save=disabled"

cd "${RIPT_ROOT}"
exec "${COMMAND[@]}"
