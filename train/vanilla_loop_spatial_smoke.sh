#!/usr/bin/env bash
set -euo pipefail

# Bounded vanilla LOOP smoke for OpenVLA-OFT on one LIBERO-Spatial task.
# Safe default: compose and print the resolved Hydra config without using CUDA.
# GPU execution requires the explicit form: --run <GPU_ID>

PROJECT_ROOT=/home/imc/yzy/RA_LOOP
RIPT_ROOT=/home/imc/code/ript-vla
PYTHON_BIN=/home/imc/anaconda3/envs/ript_vla_openvla_oft/bin/python
TORCHRUN_BIN=/home/imc/anaconda3/envs/ript_vla_openvla_oft/bin/torchrun
MODEL_PATH=/home/imc/yzy/RA_LOOP/runtime/openvla-oft-spatial-smoke
HEADER_PATH=/home/imc/models/ra-loop/ript-vla/openvla_oft/scale_header/LIBERO_SPATIAL_scale_header.pth
DATA_PATH=/home/imc/data/ra-loop/libero-datasets
TASK_NAME=pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate
OFFICIAL_LIBERO_ROOT=/home/imc/code/LIBERO-official
RUN_PROFILE="${RA_LOOP_PROFILE:-smoke}"

case "${RUN_PROFILE}" in
  smoke)
    RUN_LABEL=smoke
    EXP_NAME=OpenVLA-OFT_spatial_vanilla_smoke
    VARIANT_NAME=one_task_one_step_k2_h10
    OUTPUT_PATH="${PROJECT_ROOT}/outputs/vanilla_loop_spatial_smoke"
    DEMOS_PER_ENV=1
    TASK_ROLLOUTS_PER_ENV=1
    RLOO_K=2
    POLICY_ROLLOUTS=2
    MAX_EPISODE_LENGTH=10
    SCALE_FACTOR=2.0
    DATALOADER_SHUFFLE=true
    ;;
  learning_probe)
    RUN_LABEL=learning_probe
    EXP_NAME=OpenVLA-OFT_spatial_vanilla_learning_probe
    VARIANT_NAME=one_task_one_step_k4_h220
    OUTPUT_PATH="${PROJECT_ROOT}/outputs/vanilla_loop_spatial_learning_probe"
    DEMOS_PER_ENV=4
    TASK_ROLLOUTS_PER_ENV=4
    RLOO_K=4
    POLICY_ROLLOUTS=4
    MAX_EPISODE_LENGTH=220
    SCALE_FACTOR=2.0
    DATALOADER_SHUFFLE=false
    ;;
  mixed_reward_probe)
    RUN_LABEL=mixed_reward_probe
    EXP_NAME=OpenVLA-OFT_spatial_vanilla_mixed_reward_probe
    VARIANT_NAME=one_task_one_step_k8_h220_scale5
    OUTPUT_PATH="${PROJECT_ROOT}/outputs/vanilla_loop_spatial_mixed_reward_probe"
    DEMOS_PER_ENV=4
    TASK_ROLLOUTS_PER_ENV=4
    RLOO_K=8
    POLICY_ROLLOUTS=8
    MAX_EPISODE_LENGTH=220
    SCALE_FACTOR=5.0
    DATALOADER_SHUFFLE=false
    ;;
  *)
    echo "Unknown RA_LOOP_PROFILE: ${RUN_PROFILE}" >&2
    exit 2
    ;;
esac

export LIBERO_CONFIG_PATH="${PROJECT_ROOT}/.libero_official"
export PYTHONPATH="${PROJECT_ROOT}:${OFFICIAL_LIBERO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
# Prevent ~/.local packages from shadowing the pinned RIPT conda environment.
export PYTHONNOUSERSITE=1
export NUMBA_CACHE_DIR=/tmp/ra_loop_numba_cache
export MPLCONFIGDIR=/tmp/ra_loop_mpl_cache

OVERRIDES=(
  exp_name="${EXP_NAME}"
  variant_name="${VARIANT_NAME}"
  make_unique_experiment_dir=true
  paths.data_prefix="${DATA_PATH}"
  paths.output_prefix="${OUTPUT_PATH}"
  task.suite_name=libero_spatial
  task.dataset.suite_name=.
  task.task_names_to_use="[${TASK_NAME}]"
  task.demos_per_env="${DEMOS_PER_ENV}"
  task.rollouts_per_env="${TASK_ROLLOUTS_PER_ENV}"
  train_dataloader.batch_size=1
  train_dataloader.shuffle="${DATALOADER_SHUFFLE}"
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
  algo.rl_optimizer_factory._target_=ra_loop.ript_compat.RLOptimizerOpenVLAOFTCompat
  algo.env_runner._target_=ra_loop.ript_compat.InProcessOpenVLAOFTLiberoRunner
  algo.rloo_batch_size="${RLOO_K}"
  algo.rollouts_per_env="${POLICY_ROLLOUTS}"
  algo.num_parallel_envs=1
  algo.max_episode_length="${MAX_EPISODE_LENGTH}"
  algo.enable_dynamic_sampling=false
  algo.gradient_accumulation_steps=1
  algo.num_ppo_epochs=1
  algo.max_step_batch_size=1
  algo.scale_factor="${SCALE_FACTOR}"
  algo.fix_scale_head=true
  algo.checkpoint_path="${MODEL_PATH}"
  algo.header_checkpoint="${HEADER_PATH}"
  algo.lora_adaptor_ckpt=null
  logging.mode=disabled
  logging.resume=false
  logging.save_code=false
)

cd "${RIPT_ROOT}"

if [[ "${1:-}" != "--run" ]]; then
  echo "CPU-only Hydra compose; no model loading and no CUDA initialization"
  exec "${PYTHON_BIN}" train_ript_openvla_oft.py \
    --config-name=train_rl_openvla_oft_all_task_spatial.yaml \
    --cfg job --resolve "${OVERRIDES[@]}"
fi

if [[ $# -ne 2 || ! "${2}" =~ ^[0-7]$ ]]; then
  echo "Usage for GPU ${RUN_LABEL}: bash $0 --run <GPU_ID 0-7>" >&2
  exit 2
fi

GPU_ID="${2}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export NCCL_TIMEOUT=108000

echo "Starting bounded vanilla LOOP ${RUN_LABEL} on physical GPU ${GPU_ID}"
echo "task=${TASK_NAME} n_steps=1 K=${RLOO_K} max_episode_length=${MAX_EPISODE_LENGTH} scale_factor=${SCALE_FACTOR} wandb=disabled"
exec "${TORCHRUN_BIN}" --standalone --nproc_per_node=1 \
  train_ript_openvla_oft.py \
  --config-name=train_rl_openvla_oft_all_task_spatial.yaml \
  "${OVERRIDES[@]}"
